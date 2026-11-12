// Probe: the 2D bilinear tap weights a GPU's sampler applies. A 2×2 texture with a single 1 at one
// corner, sampled on a 33×33 grid of (fx, fy) in 1/32 steps (each axis weight exact in 8 bits), returns
// that corner's 2D weight: exact products (1 − fx)(1 − fy), or products rounded to 1/256 per tap.
// 2026-11-12: the RTX 3060 rounds per tap (every result on a 1/256 grid, 512 of 1089 off by up to
// 0.25/256); Intel UHD gives exact products (its axis weights truncate instead: sampler-weights.mjs).
// Run: node results/probes/sampler-2d-weights.mjs [high|low]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';
const server = createServer();
await new Promise( r => server.listen( 0, '127.0.0.1', r ) );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...gpuChoiceArgs( process.argv[ 2 ] ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {
	const { toHalf } = await import( '/dist/index.js' );
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice( { requiredFeatures: [ 'float32-filterable' ] } );
	const module = device.createShaderModule( { code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<storage, read> us: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	if (2u * id.x >= arrayLength(&us)) { return; }
	out[id.x] = textureSampleLevel(image, s, vec2f(us[2u * id.x], us[2u * id.x + 1u]), 0.0).r;
}` } );
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const sampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear' } );
	const res = {};
	const grid = [];
	for ( let i = 0; i <= 32; i ++ ) for ( let j = 0; j <= 32; j ++ ) grid.push( [ i / 32, j / 32 ] );
	for ( const format of [ 'rgba16float', 'rgba32float' ] ) for ( const corner of [ 0, 3 ] ) {
		const tex = device.createTexture( { size: [ 2, 2 ], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		const half = format === 'rgba16float';
		const vals = new Array( 16 ).fill( 0 ); vals[ corner * 4 ] = 1;
		device.queue.writeTexture( { texture: tex }, half ? new Uint16Array( vals.map( toHalf ) ) : new Float32Array( vals ), { bytesPerRow: half ? 16 : 32 }, [ 2, 2 ] );
		const us = new Float32Array( grid.flatMap( ( [ fx, fy ] ) => [ 0.25 + fx / 2, 0.25 + fy / 2 ] ) );
		const n = grid.length;
		const inp = device.createBuffer( { size: us.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( inp, 0, us );
		const o = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		const r = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		const bg = device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: tex.createView() }, { binding: 1, resource: sampler }, { binding: 2, resource: { buffer: inp } }, { binding: 3, resource: { buffer: o } } ] } );
		const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline( pipeline ); p.setBindGroup( 0, bg ); p.dispatchWorkgroups( Math.ceil( n / 64 ) ); p.end();
		e.copyBufferToBuffer( o, 0, r, 0, n * 4 ); device.queue.submit( [ e.finish() ] );
		await r.mapAsync( GPUMapMode.READ );
		const got = new Float32Array( r.getMappedRange().slice( 0 ) );
		let worst = { err: 0 }, off = 0, on256 = 0;
		grid.forEach( ( [ fx, fy ], i ) => { const exact = corner === 0 ? ( 1 - fx ) * ( 1 - fy ) : fx * fy; const err = got[ i ] - exact; if ( Math.abs( err ) > 1e-6 ) off ++; if ( Math.abs( got[ i ] * 256 - Math.round( got[ i ] * 256 ) ) < 1e-3 ) on256 ++; if ( Math.abs( err ) > Math.abs( worst.err ) ) worst = { err256: err * 256, fx, fy, got: got[ i ], exact }; } );
		res[ format + ' corner ' + corner ] = { off, of: n, resultsOnA256Grid: on256, worst };
	}
	return res;
} ), null, 0 ) );
await browser.close(); server.close();
