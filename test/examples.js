// Loads every example page in headless Chrome on the real GPU and checks that
// it ran: no console errors or page errors, and its on-page readout passed
// (`window.exampleResult`, set by examples/common.js).
//   npm run test:examples                     build, then run every example
//   node test/examples.js --only=hi-z,scalar  some of them
//   node test/examples.js --adapter=swiftshader --power=low --headful --verbose
// three.html loads three.js from cdn.jsdelivr.net, so it needs network access.
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { gpuChoiceArgs } from './gpu-choice.js';
import { createServer } from './serve.js';

const args = process.argv.slice( 2 );
const value = name => args.find( arg => arg.startsWith( `--${ name }=` ) )?.slice( name.length + 3 );
const directory = fileURLToPath( new URL( '../examples/', import.meta.url ) );
const only = value( 'only' )?.split( ',' );
// Every page, plus option presets (a page's query string sets its controls).
const variants = [
	'mip-chain.html?odd=spd', 'mip-chain.html?size=1129x635&odd=spd', 'mip-chain.html?size=255x257&tail=0', 'mip-chain.html?size=31x31',
	'partial-update.html?odd=box', 'partial-update.html?size=257x255&odd=spd', 'partial-update.html?size=1920x1080',
	'odd-sizes.html?size=31x31', 'odd-sizes.html?size=1477x831',
	'hi-z-pow2.html?hiz=2048x1024', 'hi-z-pow2.html?size=1129x635&hiz=512x256', 'hi-z-pow2.html?hiz=1024x1024',
];
const pages = [ ...readdirSync( directory ).filter( file => file.endsWith( '.html' ) ).sort(), ...variants ].filter( file => only === undefined || only.some( name => file.startsWith( name ) ) );

const browserArgs = [ '--enable-unsafe-webgpu' ];
if ( process.env.CI ) browserArgs.push( '--no-sandbox' );
if ( value( 'adapter' ) === 'swiftshader' ) browserArgs.push( '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' );
browserArgs.push( ...gpuChoiceArgs( value( 'power' ) ) );

const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const browser = await puppeteer.launch( { headless: ! args.includes( '--headful' ), args: browserArgs } );
let failed = 0;

try {

	// Name the GPU the pages run on (dual-GPU laptops: check that --power picked the right one).
	const probe = await browser.newPage();
	await probe.goto( `http://127.0.0.1:${ server.address().port }/examples/index.html`, { waitUntil: 'load' } );
	const info = await probe.evaluate( async () => {

		const adapter = await navigator.gpu?.requestAdapter();
		return adapter ? { ...Object.fromEntries( [ 'vendor', 'architecture', 'device', 'description' ].map( key => [ key, adapter.info[ key ] ] ) ), browser: navigator.userAgent.match( /Chrome\/[\d.]+/ )?.[ 0 ] } : null;

	} );
	console.log( `adapter: ${ info ? `${ info.vendor } ${ info.architecture } ${ info.description || info.device } (${ info.browser })` : 'none' }` );
	await probe.close();

	for ( const file of pages ) {

		const page = await browser.newPage();
		await page.setViewport( { width: 1100, height: 900 } );
		const errors = [];
		page.on( 'pageerror', error => errors.push( `page error: ${ error.message }` ) );
		page.on( 'console', message => {

			if ( message.type() === 'error' ) errors.push( `console: ${ message.text() }` );
			else if ( args.includes( '--verbose' ) ) console.log( `  [${ file }] ${ message.text() }` );

		} );
		page.on( 'requestfailed', request => errors.push( `request failed: ${ request.url() } (${ request.failure()?.errorText })` ) );
		const started = Date.now();
		let result;

		try {

			await page.goto( `http://127.0.0.1:${ server.address().port }/examples/${ file }`, { waitUntil: 'load' } );
			// index.html has no readout: it only has to load cleanly, with links that resolve.
			if ( file === 'index.html' ) {

				const links = await page.$$eval( 'a[href$=".html"]', anchors => anchors.map( anchor => anchor.href ) );
				const missing = [];
				for ( const link of links ) if ( ! ( await fetch( link ) ).ok ) missing.push( link );
				result = { ok: missing.length === 0, checks: [ { name: `${ links.length } example links resolve`, ok: missing.length === 0, detail: missing.join( ', ' ) } ] };

			} else {

				await page.waitForFunction( () => window.exampleResult !== undefined, { timeout: 120000 } );
				result = await page.evaluate( () => window.exampleResult );

			}

		} catch ( error ) {

			result = { ok: false, error: error.message };

		}

		const ok = result.ok && errors.length === 0;
		if ( ! ok ) failed ++;
		console.log( `${ ok ? 'PASS' : 'FAIL' } ${ file } (${ ( ( Date.now() - started ) / 1000 ).toFixed( 1 ) } s)` );
		for ( const check of result.checks ?? [] ) if ( ! check.ok || args.includes( '--verbose' ) ) console.log( `  ${ check.ok ? 'ok  ' : 'FAIL' } ${ check.name }${ check.detail ? ` — ${ check.detail }` : '' }` );
		if ( result.error ) console.log( `  error: ${ result.error }` );
		for ( const error of errors ) console.log( `  ${ error }` );
		await page.close();

	}

} finally {

	await browser.close();
	server.close();

}

console.log( failed ? `${ failed } of ${ pages.length } example pages failed` : `all ${ pages.length } example pages passed` );
process.exitCode = failed ? 1 : 0;
