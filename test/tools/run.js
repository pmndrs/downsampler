// Headless runner for the tool fixtures (test/tools/README.md).
//   node test/tools/run.js                       check + benchmark, all sizes
//   node test/tools/run.js --check               oracles only, no timing
//   node test/tools/run.js --mutate              perturb the oracles' inputs: every check must fail
//   node test/tools/run.js --sizes=1920x1080 --rounds=11 --repeats=16
//   node test/tools/run.js --only=fsr/luma,upscaler/exposure
//   node test/tools/run.js --output=results/tools/m5-pro.json --verbose
//   node test/tools/run.js --adapter=swiftshader   software adapter (no timestamps: checks only)
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import os from 'node:os';
import puppeteer from 'puppeteer';
import { createServer } from '../serve.js';
import { gpuChoiceArgs } from '../gpu-choice.js';

const args = process.argv.slice( 2 );
const value = name => args.find( arg => arg.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const options = { bench: ! args.includes( '--check' ) && ! args.includes( '--mutate' ), mutation: args.includes( '--mutate' ) };
if ( value( 'sizes' ) ) options.sizes = value( 'sizes' ).split( ',' ).map( size => size.split( 'x' ).map( Number ) );
if ( value( 'rounds' ) ) options.rounds = Number( value( 'rounds' ) );
if ( value( 'repeats' ) ) options.repeats = Number( value( 'repeats' ) );
if ( value( 'only' ) ) options.only = value( 'only' ).split( ',' );

const browserArgs = [ '--enable-unsafe-webgpu' ];
if ( process.env.CI ) browserArgs.push( '--no-sandbox' );
if ( value( 'adapter' ) === 'swiftshader' ) browserArgs.push( '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' );
// --power=high|low picks the GPU on machines with two (e.g. a laptop's NVIDIA and Intel; see
// test/gpu-choice.js for Windows); the printed adapter info confirms which one ran.
browserArgs.push( ...gpuChoiceArgs( value( 'power' ) ) );

const server = createServer();
let browser;

function formatCheck( check ) {

	if ( ! check ) return '-';
	const failed = check.checks.filter( entry => ! entry.ok ).map( entry => entry.output );
	const outside = check.checks.reduce( ( sum, entry ) => sum + ( entry.outside || 0 ), 0 );
	return check.ok ? ( outside ? `ok (${ outside } outside, in budget)` : 'ok' ) : `FAILED: ${ failed.join( ', ' ) }`;

}

try {

	await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
	browser = await puppeteer.launch( { headless: ! args.includes( '--headful' ), args: browserArgs, protocolTimeout: 60 * 60 * 1000 } );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' ) errors.push( message.text() );
		else if ( args.includes( '--verbose' ) && message.text().startsWith( 'PROGRESS ' ) ) console.log( message.text().slice( 9 ) );

	} );
	page.setDefaultTimeout( 3600000 );
	await page.goto( `http://127.0.0.1:${ server.address().port }/test/tools/` );
	await page.waitForFunction( () => typeof window.runTools === 'function' );
	const loadBefore = os.loadavg();
	const report = await page.evaluate( options => window.runTools( options ), options );
	const loadAfter = os.loadavg();
	if ( errors.length ) throw new Error( errors.join( '\n' ) );
	report.host = { platform: `${ os.platform() } ${ os.release() }`, cpus: os.cpus().length, loadAverage: { before: loadBefore, after: loadAfter } };

	console.log( JSON.stringify( report.adapter ) );
	console.log( `load average (1/5/15 min): before ${ loadBefore.map( x => x.toFixed( 2 ) ).join( ' ' ) }, after ${ loadAfter.map( x => x.toFixed( 2 ) ).join( ' ' ) }` );
	for ( const row of report.results ) {

		console.log( `\n${ row.width }×${ row.height }  (scene ${ row.scene.ms } ms, oracles ${ row.checkMs } ms; moving ${ ( row.scene.moving * 100 ).toFixed( 1 ) } %, disoccluded ${ ( row.scene.disoccluded * 100 ).toFixed( 2 ) } %)` );
		console.table( Object.fromEntries( Object.entries( row.fixtures ).map( ( [ name, fixture ] ) => [ name, {
			dispatches: fixture.dispatches,
			'gpu ms': fixture.gpuMs?.toFixed( 4 ) ?? '-',
			check: formatCheck( fixture.check )
		} ] ) ) );
		if ( row.ratios ) for ( const [ name, ratio ] of Object.entries( row.ratios ) ) console.log( `  ${ name }: ${ ratio.toFixed( 2 ) }×` );
		if ( row.pending.length ) console.log( `  pending plug-ins: ${ row.pending.join( ', ' ) }` );

	}

	if ( report.timestamps === false ) console.log( 'timestamp-query unavailable: checks only' );
	const output = value( 'output' ) || `results/tools/${ report.date.slice( 0, 10 ) }.json`;
	await mkdir( dirname( output ), { recursive: true } );
	await writeFile( output, JSON.stringify( { ...report, browserArgs }, null, 2 ) + '\n' );
	console.log( `\nwrote ${ output }` );
	if ( report.mutation ) console.log( report.ok ? 'mutation: every check failed, as it must' : `mutation MISSED by: ${ report.missed.join( ', ' ) }` );
	else if ( ! report.ok ) console.error( 'some oracle checks FAILED' );
	if ( ! report.ok ) process.exitCode = 1;

} finally {

	if ( browser ) await browser.close();
	server.close();

}
