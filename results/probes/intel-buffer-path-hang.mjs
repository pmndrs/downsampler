// Probe (Windows round 2): which step of the suite's `packed/buffer-path` group loses the
// Intel UHD device (DXGI_ERROR_DEVICE_HUNG, reproducible with the shard alone, 2026-11-14).
// Runs the group's steps one at a time, each followed by onSubmittedWorkDone, and logs the
// time of each and the moment the device is lost.
// Run: node results/probes/intel-buffer-path-hang.mjs [--power=low|high] [--step=name]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const arg = name => process.argv.find( a => a.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const browser = await puppeteer.launch( { headless: true, protocolTimeout: 900000, args: [ '--enable-unsafe-webgpu', ...gpuChoiceArgs( arg( 'power' ) ?? 'low' ) ] } );

try {

	const page = await browser.newPage();
	page.on( 'console', message => console.log( `  ${ message.text() }` ) );
	await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
	await page.evaluate( async only => {

		const { Downsampler } = await import( '/dist/webgpu/index.js' );
		const { createTexture, makeData, upload, readMips } = await import( '/test/browser/gpu.js' );
		const adapter = await navigator.gpu.requestAdapter();
		console.log( `adapter ${ adapter.info.vendor } ${ adapter.info.architecture }` );
		const t0 = performance.now();
		const stamp = () => `${ ( ( performance.now() - t0 ) / 1000 ).toFixed( 2 ) } s`;

		const run = async ( name, device, fn ) => {

			if ( only && only !== name ) return;
			const start = performance.now();
			let error = null;
			try { await fn(); await device.queue.onSubmittedWorkDone(); } catch ( e ) { error = e.message; }
			console.log( `${ stamp() } ${ name }: ${ ( performance.now() - start ).toFixed( 0 ) } ms${ error ? ` ERROR ${ error }` : '' }` );

		};

		// The suite's own device (raised limits) and a plain one (default limits), as the group uses.
		const raised = await adapter.requestDevice( { requiredLimits: { maxStorageTexturesPerShaderStage: Math.min( 6, adapter.limits.maxStorageTexturesPerShaderStage ), maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } } );
		raised.lost.then( info => console.log( `${ stamp() } raised device lost: ${ info.reason } ${ info.message.split( '\n' )[ 0 ] }` ) );
		const plain = await ( await navigator.gpu.requestAdapter() ).requestDevice();
		plain.lost.then( info => console.log( `${ stamp() } plain device lost: ${ info.reason } ${ info.message.split( '\n' )[ 0 ] }` ) );

		for ( const [ label, device ] of [ [ 'raised', raised ], [ 'plain', plain ] ] ) {

			for ( const format of [ 'rgb9e5ufloat', 'rg11b10ufloat', 'rgba16float', 'rgba32float' ] ) {

				for ( const [ w, h ] of [ [ 256, 128 ], [ 64, 64 ], [ 61, 37 ] ] ) {

					for ( const reduction of [ 'average', 'max' ] ) {

						await run( `${ label } ${ format } ${ w }x${ h } ${ reduction } buffer`, device, async () => {

							const ds = new Downsampler( device, { output: 'buffer', levelsPerDispatch: 6 } );
							const texture = createTexture( device, { width: w, height: h, format } );
							if ( ds.unsupportedReason( texture ) ) { texture.destroy(); return; }
							upload( device, texture, makeData( w, h, 1, format, 'random', 3 ) );
							ds.generate( texture, { reduction } );
							await readMips( device, texture, 1 );
							texture.destroy();

						} );

					}

				}

			}

		}

		plain.destroy();
		raised.destroy();

	}, arg( 'step' ) ?? null );

} finally {

	await browser.close();
	server.close();

}
