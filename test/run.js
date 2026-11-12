// Headless GPU suite runner.
//   node test/run.js                     validation suite
//   node test/run.js --quick             small shapes only
//   node test/run.js --only=scalar       one part of the suite: downsampler, reducer, scalar
//                                        (a partition; CI runs them in parallel), or mips (downsampler + reducer)
//   node test/run.js --bench             validation + benchmark
//   node test/run.js --bench --sizes=1024x1024,512x512x6 --formats=rgba8unorm --rounds=11 --repeats=16
//   node test/run.js --output=results/report.json --headful
//   node test/run.js --adapter=swiftshader   software adapter (CI, no GPU)
//   node test/run.js --bench --compare=spd --skip-validation   M2T.4: exact / tail / spd / atomic probe / render
//   node test/run.js --quality --skip-validation              M2T.4: 'spd' vs 'exact' per level, Hi-Z violations
//   node test/run.js --compat            on a compatibility-mode device (featureLevel 'compatibility')
//   node test/run.js --power=low         force the low-power GPU (default high); for laptops with two GPUs
import { mkdir, writeFile } from 'node:fs/promises';
import { loadavg } from 'node:os';
import { dirname } from 'node:path';
import puppeteer from 'puppeteer';
import { createServer } from './serve.js';

const args = process.argv.slice( 2 );
const value = name => args.find( arg => arg.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const options = { bench: args.includes( '--bench' ), quick: args.includes( '--quick' ), compat: args.includes( '--compat' ) };
if ( value( 'only' ) ) options.only = value( 'only' );
if ( value( 'sizes' ) ) options.sizes = value( 'sizes' ).split( ',' ).map( size => size.split( 'x' ).map( Number ) );
if ( value( 'formats' ) ) options.formats = value( 'formats' ).split( ',' );
if ( value( 'rounds' ) ) options.rounds = Number( value( 'rounds' ) );
if ( value( 'repeats' ) ) options.repeats = Number( value( 'repeats' ) );
if ( value( 'compare' ) ) options.compare = value( 'compare' );
if ( args.includes( '--quality' ) ) options.quality = true;
if ( args.includes( '--skip-validation' ) ) options.skipValidation = true;
const loadBefore = loadavg();

const browserArgs = [ '--enable-unsafe-webgpu' ];
if ( process.env.CI ) browserArgs.push( '--no-sandbox' );
if ( value( 'adapter' ) === 'swiftshader' ) browserArgs.push( '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' );
// --power=high|low picks the GPU on machines with two (e.g. a laptop's NVIDIA and Intel); the printed adapter info confirms which one ran.
if ( value( 'power' ) ) browserArgs.push( `--use-webgpu-power-preference=${ value( 'power' ) === 'low' ? 'force-low-power' : 'force-high-performance' }` );

const server = createServer();
let browser;

try {

	await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
	// The full matrix takes minutes; the default 180 s protocol timeout would cut it off.
	browser = await puppeteer.launch( { headless: ! args.includes( '--headful' ), args: browserArgs, protocolTimeout: ( args.includes( '--adapter=swiftshader' ) ? 60 : 30 ) * 60 * 1000 } );
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
	report.loadAverage = { before: loadBefore, after: loadavg() };
	console.log( JSON.stringify( adapter ) );
	console.log( `load average (1, 5, 15 min): before ${ loadBefore.map( v => v.toFixed( 1 ) ).join( ' ' ) }, after ${ report.loadAverage.after.map( v => v.toFixed( 1 ) ).join( ' ' ) }` );
	console.log( `validation passed: ${ validation.cases } cases, ${ validation.checks } pattern runs, ${ validation.ms } ms` );
	if ( validation.scalarPrecision ) console.log( `scalar float sums: worst error ${ ( validation.scalarPrecision.worst * 100 ).toFixed( 2 ) } % of the oracle's bound; ${ validation.scalarPrecision.pairing } results equal the oracle's float32 pairing bit for bit` );
	if ( validation.skipped.length ) console.log( `skipped (feature unavailable): ${ validation.skipped.join( ', ' ) }` );

	if ( timing?.timestamps === false ) console.log( 'timestamp-query unavailable: benchmark skipped' );

	if ( timing?.results.length && options.compare === 'spd' ) {

		// Same-run ratios to 'exact' (average) and 'exactMax' (max).
		const names = timing.strategies;
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }`, format: row.format,
			passes: names.map( name => `${ name } ${ row.passes[ name ] }` ).join( ', ' ),
			...Object.fromEntries( names.map( name => [ name, `${ row.gpuMs[ name ].toFixed( 4 ) } (${ ( row.gpuMs[ name ] / row.gpuMs[ name.endsWith( 'Max' ) ? 'exactMax' : 'exact' ] ).toFixed( 2 ) })` ] ) )
		} ) ) );

	} else if ( timing?.results.length ) {

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

	if ( report.quality ) {

		console.table( report.quality.flatMap( row => row.levels.filter( level => level.width * level.height > 1 || level.level === row.levels.length ).map( level => ( {
			size: `${ row.width }x${ row.height }`, format: row.format, pattern: row.pattern, level: `${ level.level } (${ level.width }x${ level.height })`,
			'spd-exact max': level.spdVsExact.max.toFixed( 4 ), 'spd-exact mean': level.spdVsExact.mean.toFixed( 5 ),
			'exact-ideal mean': level.exactVsIdeal.mean.toFixed( 5 ), 'spd-ideal mean': level.spdVsIdeal.mean.toFixed( 5 ),
			'tail-exact max': level.tailVsExact.max.toFixed( 4 ),
			'max violations e/s/t': `${ level.maxViolations.exact }/${ level.maxViolations.spd }/${ level.maxViolations.tail }`
		} ) ) ) );

	}

	if ( value( 'output' ) ) {

		await mkdir( dirname( value( 'output' ) ), { recursive: true } );
		await writeFile( value( 'output' ), JSON.stringify( { ...report, browserArgs }, null, 2 ) + '\n' );

	}

} finally {

	if ( browser ) await browser.close();
	server.close();

}
