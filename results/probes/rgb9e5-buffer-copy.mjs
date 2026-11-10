// Probe: rgb9e5ufloat is neither storage nor renderable, but a compute kernel can
// pack it into a buffer and copyBufferToTexture it into a mip level. Run: node results/probes/rgb9e5-buffer-copy.mjs
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu' ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {
	const device = await ( await navigator.gpu.requestAdapter() ).requestDevice();
	const errors = [];
	const scope = async ( name, fn ) => { device.pushErrorScope( 'validation' ); let r; try { r = await fn(); } catch ( e ) { errors.push( `${ name } threw: ${ e.message }` ); } const e = await device.popErrorScope(); if ( e ) errors.push( `${ name }: ${ e.message }` ); return r; };
	const usage = ( u ) => scope( `usage ${ u }`, () => device.createTexture( { size: [ 8, 8 ], format: 'rgb9e5ufloat', usage: GPUTextureUsage[ u ] | GPUTextureUsage.TEXTURE_BINDING } ).destroy() );
	for ( const u of [ 'STORAGE_BINDING', 'RENDER_ATTACHMENT', 'COPY_DST' ] ) await usage( u );
	const texture = device.createTexture( { size: [ 8, 8 ], format: 'rgb9e5ufloat', mipLevelCount: 4, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
	// A compute kernel packs (1, 0.5, 0.25) and (12, 3, 0.125) as rgb9e5 into a buffer laid out with 256-byte rows.
	const buffer = device.createBuffer( { size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
	const pack = device.createShaderModule( { code: `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
fn pack9e5(v: vec3f) -> u32 {
	let c = clamp(v, vec3f(0), vec3f(65408.0));
	let m = max(c.x, max(c.y, c.z));
	var e = max(-16, i32(floor(log2(m))) + 1) ;
	var scale = exp2(f32(e - 9));
	if (u32(round(m / scale)) == 512u) { e += 1; scale *= 2.0; }
	let q = vec3u(round(c / scale));
	return q.x | (q.y << 9u) | (q.z << 18u) | (u32(e + 15) << 27u);
}
@compute @workgroup_size(4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
	let v = select(vec3f(12, 3, 0.125), vec3f(1, 0.5, 0.25), id.x < 2u);
	out[id.y * 64u + id.x] = pack9e5(v);
}` } );
	const pipe = device.createComputePipeline( { layout: 'auto', compute: { module: pack } } );
	const read = device.createTexture( { size: [ 4, 4 ], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC } );
	const decode = device.createComputePipeline( { layout: 'auto', compute: { module: device.createShaderModule( { code: `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(4, 4) fn main(@builtin(global_invocation_id) id: vec3u) { textureStore(dst, id.xy, textureLoad(src, id.xy, 1)); }` } ) } } );
	const out = device.createBuffer( { size: 256 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	await scope( 'pack + copyBufferToTexture + sample', async () => {
		const encoder = device.createCommandEncoder();
		let pass = encoder.beginComputePass();
		pass.setPipeline( pipe ); pass.setBindGroup( 0, device.createBindGroup( { layout: pipe.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: { buffer } } ] } ) ); pass.dispatchWorkgroups( 1 ); pass.end();
		encoder.copyBufferToTexture( { buffer, bytesPerRow: 256 }, { texture, mipLevel: 1 }, [ 4, 4 ] );
		pass = encoder.beginComputePass();
		pass.setPipeline( decode ); pass.setBindGroup( 0, device.createBindGroup( { layout: decode.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: read.createView() } ] } ) ); pass.dispatchWorkgroups( 1 ); pass.end();
		encoder.copyTextureToBuffer( { texture: read }, { buffer: out, bytesPerRow: 256 }, [ 4, 4 ] );
		device.queue.submit( [ encoder.finish() ] );
		await out.mapAsync( GPUMapMode.READ );
	} );
	const values = Array.from( new Float32Array( out.getMappedRange() ).slice( 0, 16 ) );
	return { errors, firstRow: values };
} ) ) );
await browser.close(); server.close();
