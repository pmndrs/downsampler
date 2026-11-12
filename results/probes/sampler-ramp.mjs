// Probe: the texel position a GPU's sampler really filters at, at the coordinates the library's
// 'exact' kernel uses. A size×1 rgba32float texture holds texel k = k (a ramp), so a bilinear
// sample returns base texel + weight in f32. The shader samples at uv = (x + 0.5) / n, as
// src/webgpu/kernel.ts `first()` does for an n-wide output, and the result is compared with the
// exact position (x + 0.5) · size / n − 0.5. Needs float32-filterable.
// Run: node results/probes/sampler-ramp.mjs [--power=high|low]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const power = process.argv.find( arg => arg.startsWith( '--power=' ) )?.split( '=' )[ 1 ];
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...gpuChoiceArgs( power ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {

	const adapter = await navigator.gpu.requestAdapter();
	if ( ! adapter.features.has( 'float32-filterable' ) ) return { error: 'no float32-filterable' };
	const device = await adapter.requestDevice( { requiredFeatures: [ 'float32-filterable' ] } );
	const module = device.createShaderModule( { code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var linear: sampler;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> n: vec4u;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= n.x) { return; }
	let uv = (vec2f(f32(id.x), 0.0) + 0.5) / vec2f(f32(n.x), 1.0);
	out[id.x] = textureSampleLevel(image, linear, uv, 0.0).r;
}` } );
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const sampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear' } );
	const report = { adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture }, sizes: {} };

	for ( const size of [ 7, 15, 17, 31, 33, 65, 129, 255, 257, 1129, 1477, 192, 108, 1920 ] ) {

		const n = Math.max( 1, size >> 1 );
		const texture = device.createTexture( { size: [ size, 1 ], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		const ramp = new Float32Array( size * 4 );
		for ( let k = 0; k < size; k ++ ) ramp[ 4 * k ] = k;
		device.queue.writeTexture( { texture }, ramp, { bytesPerRow: size * 16 }, [ size, 1 ] );
		const out = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		const read = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		const uniform = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( uniform, 0, new Uint32Array( [ n, 0, 0, 0 ] ) );
		const bind = device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: sampler }, { binding: 2, resource: { buffer: out } }, { binding: 3, resource: { buffer: uniform } } ] } );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline( pipeline ); pass.setBindGroup( 0, bind ); pass.dispatchWorkgroups( Math.ceil( n / 64 ) ); pass.end();
		encoder.copyBufferToBuffer( out, 0, read, 0, n * 4 );
		device.queue.submit( [ encoder.finish() ] );
		await read.mapAsync( GPUMapMode.READ );
		const got = new Float32Array( read.getMappedRange().slice( 0 ) );
		let worst = 0, worstAt = null, off = 0;
		for ( let x = 0; x < n; x ++ ) {

			const exact = Math.min( size - 1, Math.max( 0, ( x + 0.5 ) * size / n - 0.5 ) );
			const error = got[ x ] - exact;
			if ( Math.abs( error ) > 1e-6 ) off ++;
			if ( Math.abs( error ) > Math.abs( worst ) ) { worst = error; worstAt = { x, exact, got: got[ x ] }; }

		}
		report.sizes[ size ] = { n, maxError: worst, maxErrorIn256ths: worst * 256, off, worstAt };
		[ texture, out, read, uniform ].forEach( resource => resource.destroy() );

	}
	return report;

} ), null, 1 ) );
await browser.close();
server.close();
