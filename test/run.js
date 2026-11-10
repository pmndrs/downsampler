// Headless GPU suite runner.
//   node test/run.js                     validation suite
//   node test/run.js --quick             small shapes only
//   node test/run.js --bench             validation + benchmark
//   node test/run.js --bench --sizes=1024x1024,512x512x6 --rounds=11 --repeats=16
//   node test/run.js --output=results/report.json --headful
//   node test/run.js --adapter=swiftshader   software adapter (CI, no GPU)
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import puppeteer from 'puppeteer';
import { createServer } from './serve.js';

const args = process.argv.slice( 2 );
const value = name => args.find( arg => arg.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const options = { bench: args.includes( '--bench' ), quick: args.includes( '--quick' ) };
if ( value( 'sizes' ) ) options.sizes = value( 'sizes' ).split( ',' ).map( size => size.split( 'x' ).map( Number ) );
if ( value( 'rounds' ) ) options.rounds = Number( value( 'rounds' ) );
if ( value( 'repeats' ) ) options.repeats = Number( value( 'repeats' ) );

const browserArgs = [ '--enable-unsafe-webgpu' ];
if ( process.env.CI ) browserArgs.push( '--no-sandbox' );
if ( value( 'adapter' ) === 'swiftshader' ) browserArgs.push( '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' );

const server = createServer();
let browser;

try {

	await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
	// The full matrix takes minutes; the default 180 s protocol timeout would cut it off.
	browser = await puppeteer.launch( { headless: ! args.includes( '--headful' ), args: browserArgs, protocolTimeout: 30 * 60 * 1000 } );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' ) errors.push( message.text() );
		else if ( args.includes( '--verbose' ) && message.text().startsWith( 'PROGRESS ' ) ) console.log( message.text().slice( 9 ) );

	} );
	page.setDefaultTimeout( 600000 );
	await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
	await page.waitForFunction( () => typeof window.runSuite === 'function' );
	const report = await page.evaluate( options => window.runSuite( options ), options );
	if ( errors.length ) throw new Error( errors.join( '\n' ) );

	const { validation, timing, adapter } = report;
	console.log( JSON.stringify( adapter ) );
	console.log( `validation passed: ${ validation.cases } cases, ${ validation.checks } pattern runs, ${ validation.ms } ms` );

	if ( timing?.timestamps === false ) console.log( 'timestamp-query unavailable: benchmark skipped' );

	if ( timing?.results.length ) {

		const names = timing.strategies;
		const best = names.filter( name => name.startsWith( 'compute' ) ).pop();
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }x${ row.layers }`, format: row.format,
			passes: names.map( name => row.passes[ name ] ).join( ' / ' ),
			...Object.fromEntries( names.map( name => [ `${ name } ms`, row.gpuMs[ name ].toFixed( 4 ) ] ) ),
			[ `${ best } vs render` ]: `${ ( row.gpuMs.render / row.gpuMs[ best ] ).toFixed( 2 ) }×`,
			[ `${ best } vs compute4` ]: `${ ( row.gpuMs.compute4 / row.gpuMs[ best ] ).toFixed( 2 ) }×`
		} ) ) );

	}

	if ( value( 'output' ) ) {

		await mkdir( dirname( value( 'output' ) ), { recursive: true } );
		await writeFile( value( 'output' ), JSON.stringify( { ...report, browserArgs }, null, 2 ) + '\n' );

	}

} finally {

	if ( browser ) await browser.close();
	server.close();

}
