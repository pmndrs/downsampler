// Probe (Windows round 2): pipeline creation time of sRGB kernels on D3D12.
// The GPU suite's sRGB groups took 70–90 s for 2×2 and 16×16 textures on the RTX 3060
// (Chrome 148, D3D12), where every other format takes a second or two: the cost is in
// compiling, not running. This times createComputePipelineAsync for the same exact kernel
// as rgba8unorm and as rgba8unorm-srgb, and the sRGB kernel with its two lookup tables
// (DS_SRGB_DECODE, DS_SRGB_BOUND) declared as module-scope `const` (as shipped),
// `var<private>`, or read from a storage buffer.
// Run: node results/probes/srgb-compile-time.mjs [--power=high|low] [--output=file.json]
import { writeFileSync } from 'node:fs';
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const arg = name => process.argv.find( a => a.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const browser = await puppeteer.launch( { headless: true, protocolTimeout: 900000, args: [ '--enable-unsafe-webgpu', ...gpuChoiceArgs( arg( 'power' ) ) ] } );

try {

	const page = await browser.newPage();
	page.on( 'console', message => console.log( `  ${ message.text() }` ) );
	await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
	const result = await page.evaluate( async () => {

		const { createKernel } = await import( '/dist/webgpu/index.js' );
		const adapter = await navigator.gpu.requestAdapter();
		const device = await adapter.requestDevice( { requiredLimits: { maxStorageTexturesPerShaderStage: Math.min( 6, adapter.limits.maxStorageTexturesPerShaderStage ) } } );
		const out = { adapter: `${ adapter.info.vendor } ${ adapter.info.architecture } ${ adapter.info.description }`, browser: navigator.userAgent.match( /Chrome\/[\d.]+/ )?.[ 0 ], rows: [] };

		// A unique constant per compile, so no shader cache (Dawn's or the driver's) can hit.
		let nonce = 0;
		const unique = code => code.replace( /@compute/, `const DS_NONCE = ${ ++ nonce }u;\n@compute` ).replace( /let size0 = /, 'let ds_n = DS_NONCE; let size0 = ' );

		const variants = {
			const: code => code,
			private: code => code.replace( 'const DS_SRGB_DECODE', 'var<private> DS_SRGB_DECODE' ).replace( 'const DS_SRGB_BOUND', 'var<private> DS_SRGB_BOUND' ),
			storage: code => code
				.replace( /const DS_SRGB_DECODE = array<f32, 256>\([^)]*\);/, '@group(2) @binding(0) var<storage, read> DS_SRGB_DECODE: array<f32, 256>;' )
				.replace( /const DS_SRGB_BOUND = array<f32, 255>\([^)]*\);/, '@group(2) @binding(1) var<storage, read> DS_SRGB_BOUND: array<f32, 255>;' ),
		};

		const time = async ( label, code ) => {

			device.pushErrorScope( 'validation' );
			const module = device.createShaderModule( { code: unique( code ) } );
			const start = performance.now();
			let error;
			try { await device.createComputePipelineAsync( { layout: 'auto', compute: { module, entryPoint: 'main' } } ); } catch ( e ) { error = e.message; }
			const ms = performance.now() - start;
			const scoped = await device.popErrorScope();
			const row = { label, bytes: code.length, ms: Math.round( ms ), error: error ?? scoped?.message };
			console.log( JSON.stringify( row ) );
			out.rows.push( row );

		};

		for ( const count of [ 1, 4, 6 ] ) {

			for ( const tail of [ false, true ] ) {

				const base = { dimension: '2d', count, tail, tile: tail ? 32 : undefined };
				await time( `rgba8unorm count ${ count }${ tail ? ' tail' : '' }`, createKernel( { ...base, format: 'rgba8unorm', output: 'storage' } ) );
				const srgb = createKernel( { ...base, format: 'rgba8unorm-srgb', output: 'buffer' } );
				for ( const [ name, change ] of Object.entries( variants ) ) await time( `rgba8unorm-srgb count ${ count }${ tail ? ' tail' : '' } tables ${ name }`, change( srgb ) );

			}

		}

		return out;

	} );
	console.log( `${ result.adapter } (${ result.browser })` );
	if ( arg( 'output' ) ) writeFileSync( arg( 'output' ), JSON.stringify( result, null, '\t' ) + '\n' );

} finally {

	await browser.close();
	server.close();

}
