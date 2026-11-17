import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { cp, mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const repository = resolve( fileURLToPath( new URL( '../../', import.meta.url ) ) );
const args = process.argv.slice( 2 );
const value = name => args.find( arg => arg.startsWith( `--${ name }=` ) )?.slice( name.length + 3 );
const help = `DS3 same-build pipeline warm-up measurement
Usage: node results/probes/ds3-warmup.mjs [options]
  --power=high|low           Choose hardware GPU (optional)
  --adapter=swiftshader      Use software Vulkan WebGPU
  --repeats=3               Alternating A/B pairs per output path
  --paths=storage,buffer,render
  --dist=dist                Build to snapshot before launching Chrome
  --output=PATH              Result JSON (default results/probes/ds3-default.json)
  --chrome=PATH              Browser override; Windows uses installed Chrome
  --trace                    Collect compiler events and raw traces
  --trace-dir=PATH           Retain raw traces here (default temporary trace directory)
  --keep-temp                Retain build snapshot and private profiles for diagnosis
  --dry-run                  Validate options, snapshot/serve build, then clean up; no browser/GPU
  --help                    Print this help
`;

if ( args.includes( '--help' ) ) {

	console.log( help );
	process.exit( 0 );

}

const allowedOptions = new Set( [ 'power', 'adapter', 'repeats', 'paths', 'dist', 'output', 'chrome', 'trace', 'trace-dir', 'keep-temp', 'dry-run' ] );
for ( const arg of args ) {

	if ( ! arg.startsWith( '--' ) || ! allowedOptions.has( arg.slice( 2 ).split( '=' )[ 0 ] ) ) throw new Error( `Unknown option: ${ arg }` );

}

const power = value( 'power' );
const software = value( 'adapter' ) === 'swiftshader';
const repeats = Number( value( 'repeats' ) ?? 3 );
const paths = ( value( 'paths' ) ?? 'storage,buffer,render' ).split( ',' );
const traceEnabled = args.includes( '--trace' );
const keepTemp = args.includes( '--keep-temp' );
const dryRun = args.includes( '--dry-run' );
const output = resolve( value( 'output' ) ?? 'results/probes/ds3-default.json' );
const startedAt = new Date().toISOString();

if ( power && ! [ 'high', 'low' ].includes( power ) ) throw new Error( '--power must be high or low' );
if ( value( 'adapter' ) && ! software ) throw new Error( '--adapter must be swiftshader' );
if ( software && power ) throw new Error( 'Choose --adapter or --power, not both' );
if ( ! Number.isInteger( repeats ) || repeats < 1 ) throw new Error( '--repeats must be a positive integer' );
if ( ! paths.length || paths.some( path => ! [ 'storage', 'buffer', 'render' ].includes( path ) ) || new Set( paths ).size !== paths.length ) throw new Error( 'Invalid or duplicate --paths' );
if ( value( 'trace-dir' ) && ! traceEnabled ) throw new Error( '--trace-dir requires --trace' );

const windowsChrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const executablePath = value( 'chrome' ) ?? ( process.platform === 'win32' && existsSync( windowsChrome ) ? windowsChrome : undefined );
if ( executablePath && ! existsSync( executablePath ) ) throw new Error( `Browser does not exist: ${ executablePath }` );
const flags = [ '--enable-unsafe-webgpu' ];
if ( software ) flags.push( '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' );
// A dry run must not query adapters or start a GPU process.
else if ( ! dryRun ) flags.push( ...gpuChoiceArgs( power ) );

// Keep paths in a shared result usable from another checkout.
function reportPath( file ) {

	const local = relative( repository, file );
	return isAbsolute( local ) || local === '..' || local.startsWith( '..' + sep ) ? file : local.split( sep ).join( '/' );

}

async function manifest( root ) {

	const files = [];
	async function visit( directory ) {

		for ( const entry of ( await readdir( directory, { withFileTypes: true } ) ).sort( ( a, b ) => a.name.localeCompare( b.name ) ) ) {

			const path = join( directory, entry.name );
			if ( entry.isDirectory() ) await visit( path );
			else if ( entry.isFile() ) {

				const bytes = await readFile( path );
				files.push( { path: relative( root, path ).split( sep ).join( '/' ), bytes: bytes.length, sha256: createHash( 'sha256' ).update( bytes ).digest( 'hex' ) } );

			}

		}

	}
	await visit( root );
	return { sha256: createHash( 'sha256' ).update( JSON.stringify( files ) ).digest( 'hex' ), files };

}

// Only remove exact directories created by this invocation, beneath the resolved temp root.
async function removeOwnedTemp( path, tempRoot ) {

	const actual = await realpath( path );
	if ( ! actual.startsWith( tempRoot + sep ) || ! actual.split( sep ).at( - 1 ).startsWith( 'ds3-' ) ) throw new Error( `Refusing unsafe temp cleanup: ${ actual }` );
	await rm( actual, { recursive: true, force: true } );

}

function staticServer( snapshot ) {

	return createServer( async ( request, response ) => {

		try {

			const pathname = decodeURIComponent( new URL( request.url, 'http://localhost' ).pathname );
			if ( pathname === '/favicon.ico' ) return response.writeHead( 204 ).end();
			const fromDist = pathname.startsWith( '/dist/' );
			const root = fromDist ? snapshot : repository;
			const file = resolve( root, pathname.slice( fromDist ? 6 : 1 ) );
			if ( ! file.startsWith( root + sep ) ) return response.writeHead( 403 ).end();
			const body = await readFile( file );
			const type = { '.js': 'text/javascript', '.html': 'text/html' }[ extname( file ) ] ?? 'application/octet-stream';
			response.writeHead( 200, { 'Content-Type': type, 'Cache-Control': 'no-store' } ).end( body );

		} catch {

			response.writeHead( 404 ).end();

		}

	} );

}

async function collectTrace( page, traceDirectory, name ) {

	try {

		const bytes = Buffer.from( await page.tracing.stop() );
		const trace = JSON.parse( bytes.toString() );
		if ( ! Array.isArray( trace.traceEvents ) ) throw new Error( 'Missing traceEvents array' );
		const file = join( traceDirectory, `${ name }.json` );
		await writeFile( file, bytes );
		const compilerEvents = trace.traceEvents.filter( event => /^CompileShader(DXC|FXC)$/.test( event.name ) ).map( event => ( {
			name: event.name, phase: event.ph, durationUs: event.dur ?? null, timestampUs: event.ts, process: event.pid, thread: event.tid,
		} ) );
		const compilerSpans = [];
		const stacks = new Map();
		for ( const event of [ ...trace.traceEvents ].sort( ( a, b ) => ( a.ts ?? 0 ) - ( b.ts ?? 0 ) ) ) {

			const key = `${ event.pid }:${ event.tid }`;
			if ( event.ph === 'X' && /^CompileShader(DXC|FXC)$/.test( event.name ) ) compilerSpans.push( { name: event.name, durationUs: event.dur ?? 0 } );
			else if ( event.ph === 'B' ) {

				if ( ! stacks.has( key ) ) stacks.set( key, [] );
				stacks.get( key ).push( event );

			} else if ( event.ph === 'E' ) {

				const start = stacks.get( key )?.pop();
				if ( start && /^CompileShader(DXC|FXC)$/.test( start.name ) ) compilerSpans.push( { name: start.name, durationUs: event.ts - start.ts } );

			}

		}
		return { status: 'collected', file: reportPath( file ), eventCount: trace.traceEvents.length, compilerEvents, compilerSpans };

	} catch ( error ) {

		throw new Error( `Trace collection failed for ${ name }: ${ error.message }`, { cause: error } );

	}

}

const tempRoot = await realpath( tmpdir() );
const ownedTemp = [];
const runs = [];
let server;
let build;
let traceDirectory;
let failure;
const report = {
	schema: 2, startedAt, platform: process.platform, command: [ reportPath( process.argv[ 1 ] ), ...args ], flags,
	options: { power, software, repeats, paths, traceEnabled, keepTemp, dryRun, executablePath: executablePath ?? 'puppeteer-default' },
	cacheCaveat: 'Fresh private profile per path/arm/pair, followed by same-browser reload at one origin. OS/driver shader caches remain. Arms alternate by pair.',
	runs,
};

try {

	const snapshot = await mkdtemp( join( tempRoot, 'ds3-build-' ) );
	ownedTemp.push( snapshot );
	await cp( resolve( value( 'dist' ) ?? 'dist' ), snapshot, { recursive: true } );
	build = await manifest( snapshot );
	if ( ! build.files.some( file => file.path === 'webgpu/index.js' ) ) throw new Error( 'Build lacks webgpu/index.js; run npm run build first' );
	report.build = { ...build, capturedAt: new Date().toISOString(), source: reportPath( resolve( value( 'dist' ) ?? 'dist' ) ), snapshot: keepTemp ? snapshot : null };
	if ( traceEnabled ) {

		traceDirectory = value( 'trace-dir' ) ? resolve( value( 'trace-dir' ) ) : await mkdtemp( join( tempRoot, 'ds3-traces-' ) );
		await mkdir( traceDirectory, { recursive: true } );
		report.traceDirectory = reportPath( traceDirectory );
		console.log( `Raw traces retained: ${ traceDirectory }` );

	}
	server = staticServer( snapshot );
	await new Promise( resolveListen => server.listen( 0, '127.0.0.1', resolveListen ) );
	const origin = `http://127.0.0.1:${ server.address().port }`;
	if ( dryRun ) {

		const fetched = await fetch( `${ origin }/dist/webgpu/index.js` );
		if ( ! fetched.ok ) throw new Error( 'Snapshot server smoke failed' );
		for ( const path of [ '/results/probes/ds3-warmup.html', '/results/probes/ds3-warmup.js' ] ) {

			if ( ! ( await fetch( origin + path ) ).ok ) throw new Error( `Probe server smoke failed: ${ path }` );

		}
		console.log( `Offline smoke passed: ${ build.files.length } files, SHA256 ${ build.sha256 }` );

	} else {

		for ( let pair = 0; pair < repeats; pair ++ ) {

			for ( const path of paths ) {

				for ( const arm of pair % 2 ? [ 'warmup', 'sync' ] : [ 'sync', 'warmup' ] ) {

					const profile = await mkdtemp( join( tempRoot, 'ds3-profile-' ) );
					ownedTemp.push( profile );
					let browser;
					try {

						browser = await puppeteer.launch( { headless: true, executablePath, args: flags, userDataDir: profile, timeout: 120000, protocolTimeout: 300000 } );
						const page = await browser.newPage();
						const logs = [];
						const session = await page.createCDPSession();
						await session.send( 'Log.enable' );
						session.on( 'Log.entryAdded', event => {

							if ( event.entry.level === 'error' ) logs.push( event.entry.text );

						} );
						for ( const cache of [ 'fresh-profile', 'same-browser-reload' ] ) {

							const entry = { pair, cache, output: path, arm, browser: await browser.version(), logs: [], trace: { status: 'disabled' }, error: null };
							runs.push( entry );
							if ( keepTemp ) entry.profile = profile;
							try {

								if ( traceEnabled ) {

									try {

										await page.tracing.start( { categories: [ 'gpu', 'gpu.dawn', 'dawn', 'disabled-by-default-gpu.dawn', 'disabled-by-default-dawn', 'gpu.service', 'toplevel' ] } );

									} catch ( error ) {

										throw new Error( `Trace start failed: ${ error.message }`, { cause: error } );

									}

								}
								await page.goto( `${ origin }/results/probes/ds3-warmup.html?output=${ path }&arm=${ arm }` );
								await page.waitForFunction( () => window.__probeResult || window.__probeError, { timeout: 300000 } );
								const response = await page.evaluate( () => ( { result: window.__probeResult, error: window.__probeError } ) );
								if ( traceEnabled ) entry.trace = await collectTrace( page, traceDirectory, `p${ pair }-${ path }-${ arm }-${ cache }` );
								if ( response.error ) throw new Error( response.error );
								Object.assign( entry, response.result );
								entry.logs = [ ...logs ];
								console.log( `${ pair + 1 }/${ repeats } ${ path } ${ arm } ${ cache }: ${ entry.first.recordToDoneMs.toFixed( 1 ) } ms first record-to-done` );

							} catch ( error ) {

								entry.error = error.message;
								if ( traceEnabled && entry.trace.status === 'disabled' ) entry.trace = { status: 'failed', error: error.message };
								throw error;

							}

						}

					} finally {

						if ( browser ) await browser.close();
						if ( ! keepTemp ) {

							await removeOwnedTemp( profile, tempRoot );
							ownedTemp.splice( ownedTemp.indexOf( profile ), 1 );

						}

					}

				}

			}

		}

	}

} catch ( error ) {

	failure = error;
	report.error = error.stack;

} finally {

	if ( server ) await new Promise( resolveClose => server.close( resolveClose ) );
	report.finishedAt = new Date().toISOString();
	try {

		if ( ! keepTemp ) for ( const path of ownedTemp ) await removeOwnedTemp( path, tempRoot );

	} catch ( error ) {

		report.cleanupError = error.message;
		failure ??= error;

	}
	await mkdir( dirname( output ), { recursive: true } );
	await writeFile( output, JSON.stringify( report ) + '\n' );
	console.log( `Result JSON: ${ output }` );

}

if ( failure ) throw failure;
