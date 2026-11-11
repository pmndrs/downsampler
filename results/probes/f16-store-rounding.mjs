// Probe: how f32 → f16 rounds in textureStore to rgba16float, against WGSL
// pack2x16float and round-to-nearest-even (core toHalf). Values sit between two
// halves at 1/4, 1/2 (ties) and 3/4 of a step, both signs.
// Run: node results/probes/f16-store-rounding.mjs [--swiftshader]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const swift = process.argv.includes( '--swiftshader' );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...( swift ? [ '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' ] : [] ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {
	const { toHalf } = await import( '/dist/index.js' );
	const device = await ( await navigator.gpu.requestAdapter() ).requestDevice();
	const values = [];
	for ( const base of [ 1, 1.5, 3, 1000 / 1024, 0.1 ] ) for ( const fraction of [ 0.25, 0.5, 0.75 ] ) for ( const sign of [ 1, - 1 ] ) {
		const step = 2 ** ( Math.floor( Math.log2( base ) ) - 10 );
		const lower = Math.floor( base / step ) * step;
		values.push( Math.fround( sign * ( lower + fraction * step ) ), Math.fround( sign * ( lower + step + fraction * step ) ) );
	}
	const n = values.length;
	const input = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( input, 0, new Float32Array( values ) );
	const packed = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
	const texture = device.createTexture( { size: [ n, 1 ], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC } );
	const module = device.createShaderModule( { code: `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var image: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
	let v = input[id.x];
	packed[id.x] = pack2x16float(vec2f(v, 0.0));
	textureStore(image, vec2i(i32(id.x), 0), vec4f(v));
}` } );
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const bytesPerRow = Math.ceil( n * 8 / 256 ) * 256;
	const readTexture = device.createBuffer( { size: bytesPerRow, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const readPacked = device.createBuffer( { size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline( pipeline );
	pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: packed } }, { binding: 2, resource: texture.createView() } ] } ) );
	pass.dispatchWorkgroups( n );
	pass.end();
	encoder.copyTextureToBuffer( { texture }, { buffer: readTexture, bytesPerRow }, [ n, 1 ] );
	encoder.copyBufferToBuffer( packed, 0, readPacked, 0, n * 4 );
	device.queue.submit( [ encoder.finish() ] );
	await Promise.all( [ readTexture.mapAsync( GPUMapMode.READ ), readPacked.mapAsync( GPUMapMode.READ ) ] );
	const stored = new Uint16Array( readTexture.getMappedRange() ), pack = new Uint32Array( readPacked.getMappedRange() );
	const rows = values.map( ( value, i ) => ( { value, rtne: toHalf( value ), pack2x16float: pack[ i ] & 0xffff, textureStore: stored[ i * 4 ] } ) );
	return {
		textureStoreEqualsRtne: rows.filter( row => row.textureStore === row.rtne ).length,
		packEqualsRtne: rows.filter( row => row.pack2x16float === row.rtne ).length,
		count: rows.length,
		differences: rows.filter( row => row.textureStore !== row.rtne || row.pack2x16float !== row.rtne )
	};
} ), null, 1 ) );
await browser.close(); server.close();
