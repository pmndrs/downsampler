// Probe: what the library's sampled first level reads, per axis, on the real GPU. A W×H texture
// holds x mod 2 in red and y mod 2 in green, so mip 1's red and green give the bilinear weight the
// kernel's tap used per axis (w, or 1 − w after an odd base texel), in [0, 1] where even f16 resolves
// 1/2048; compared with the exact position ((2x + 1) · W − w) / 2w. rgba16float and rgba32float
// (needs float32-filterable): does the sampler's precision depend on the format?
// Run: node results/probes/kernel-ramp.mjs [--power=high|low] [--sizes=65x129,255x257]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const value = name => process.argv.find( arg => arg.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const sizes = ( value( 'sizes' ) ?? '65x129,255x257,15x17,192x108,1129x635' ).split( ',' ).map( size => size.split( 'x' ).map( Number ) );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...gpuChoiceArgs( value( 'power' ) ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async sizes => {

	const { Downsampler } = await import( '/dist/webgpu/index.js' );
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice( { requiredFeatures: [ 'float32-filterable' ].filter( name => adapter.features.has( name ) ) } );
	const report = { adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture }, sizes: {} };
	for ( const format of [ 'rgba32float', 'rgba16float' ] ) for ( const [ W, H ] of sizes ) {

		const texture = device.createTexture( { size: [ W, H ], format, mipLevelCount: 2, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC } );
		const data = new Float32Array( W * H * 4 );
		for ( let y = 0; y < H; y ++ ) for ( let x = 0; x < W; x ++ ) data.set( [ x % 2, y % 2, 0, 1 ], ( y * W + x ) * 4 );
		const half = format === 'rgba16float';
		device.queue.writeTexture( { texture }, half ? new Uint16Array( Array.from( data, v => v === 1 ? 0x3c00 : 0 ) ) : data, { bytesPerRow: W * ( half ? 8 : 16 ) }, [ W, H ] );
		new Downsampler( device ).generate( texture );
		const w = Math.max( 1, W >> 1 ), h = Math.max( 1, H >> 1 ), texel = half ? 8 : 16, bytesPerRow = Math.ceil( w * texel / 256 ) * 256;
		const read = device.createBuffer( { size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		const encoder = device.createCommandEncoder();
		encoder.copyTextureToBuffer( { texture, mipLevel: 1 }, { buffer: read, bytesPerRow }, [ w, h ] );
		device.queue.submit( [ encoder.finish() ] );
		await read.mapAsync( GPUMapMode.READ );
		const raw = read.getMappedRange().slice( 0 );
		const fromHalf = h16 => { const e = ( h16 >> 10 ) & 31, m = h16 & 1023; return ( h16 >> 15 ? - 1 : 1 ) * ( e ? 2 ** ( e - 15 ) * ( 1 + m / 1024 ) : 2 ** - 14 * m / 1024 ); };
		const got = half ? Float32Array.from( new Uint16Array( raw ), fromHalf ) : new Float32Array( raw );
		const stride = bytesPerRow / ( half ? 2 : 4 );
		// The filtered value of x mod 2 at position p: frac(p) after an even base texel, 1 − frac(p) after an odd one.
		const axis = ( size, n, i ) => { const p = Math.min( size - 1, Math.max( 0, ( ( 2 * i + 1 ) * size - n ) / ( 2 * n ) ) ), base = Math.floor( p ), f = p - base; return base === size - 1 ? base % 2 : base % 2 ? 1 - f : f; };
		const worst = { x: 0, y: 0 };
		for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const o = y * stride + x * 4;
			const ex = got[ o ] - axis( W, w, x ), ey = got[ o + 1 ] - axis( H, h, y );
			if ( Math.abs( ex ) > Math.abs( worst.x ) ) worst.x = ex;
			if ( Math.abs( ey ) > Math.abs( worst.y ) ) worst.y = ey;

		}
		report.sizes[ `${ format } ${ W }x${ H }` ] = { maxErrorX256: worst.x * 256, maxErrorY256: worst.y * 256 };
		texture.destroy(); read.destroy();

	}
	return report;

}, sizes ), null, 1 ) );
await browser.close();
server.close();
