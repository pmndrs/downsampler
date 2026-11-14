// Probe: which integer formats and rgb10a2unorm allow STORAGE_BINDING (write-only), with and
// without texture-formats-tier1, on a core and a compatibility-mode device; and how a
// sampled read of each fills the channels it doesn't store.
// Run: node results/probes/int-format-caps.mjs [--swiftshader]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const swift = process.argv.includes( '--swiftshader' );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...( swift ? [ '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' ] : [] ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {
	const formats = [ 'rgb10a2unorm', 'rgb10a2uint', 'r8uint', 'r8sint', 'rg8uint', 'rg8sint', 'rgba8uint', 'rgba8sint', 'r16uint', 'r16sint', 'rg16uint', 'rg16sint', 'rgba16uint', 'rgba16sint', 'r32uint', 'r32sint', 'rg32uint', 'rg32sint', 'rgba32uint', 'rgba32sint' ];
	const out = { userAgent: navigator.userAgent };
	for ( const [ name, compat, tier1 ] of [ [ 'core', false, false ], [ 'tier1', false, true ], [ 'compat', true, false ], [ 'compat tier1', true, true ] ] ) {
		const adapter = await navigator.gpu.requestAdapter( compat ? { featureLevel: 'compatibility' } : {} );
		if ( ! adapter || ( tier1 && ! adapter.features.has( 'texture-formats-tier1' ) ) ) continue;
		const device = await adapter.requestDevice( { requiredFeatures: tier1 ? [ 'texture-formats-tier1' ] : [] } );
		if ( compat && device.features.has( 'core-features-and-limits' ) ) { out[ name ] = 'core device'; continue; }
		const result = {};
		for ( const format of formats ) {
			device.pushErrorScope( 'validation' );
			device.createTexture( { size: [ 4, 4 ], format, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING } ).destroy();
			result[ format ] = ( await device.popErrorScope() ) ? 'no' : 'storage';
		}
		out[ name ] = result;
		if ( name === 'core' ) {
			// Channels a sampled read returns for an r32uint and an rg8sint texel.
			const reads = {};
			for ( const [ format, data, type ] of [ [ 'r32uint', new Uint32Array( [ 7 ] ), 'u32' ], [ 'rg32sint', new Int32Array( [ - 7, 9 ] ), 'i32' ] ] ) {
				const texture = device.createTexture( { size: [ 1, 1 ], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
				device.queue.writeTexture( { texture }, data, { bytesPerRow: 256 }, [ 1, 1 ] );
				const buffer = device.createBuffer( { size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
				const read = device.createBuffer( { size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST } );
				const module = device.createShaderModule( { code: `@group(0) @binding(0) var t: texture_2d<${ type }>; @group(0) @binding(1) var<storage, read_write> o: vec4<${ type }>;
@compute @workgroup_size(1) fn main() { o = textureLoad(t, vec2i(0), 0); }` } );
				const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
				const encoder = device.createCommandEncoder();
				const pass = encoder.beginComputePass();
				pass.setPipeline( pipeline );
				pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer } } ] } ) );
				pass.dispatchWorkgroups( 1 );
				pass.end();
				encoder.copyBufferToBuffer( buffer, 0, read, 0, 16 );
				device.queue.submit( [ encoder.finish() ] );
				await read.mapAsync( GPUMapMode.READ );
				reads[ format ] = [ ...( type === 'u32' ? new Uint32Array( read.getMappedRange() ) : new Int32Array( read.getMappedRange() ) ) ];
				read.unmap();
			}
			out.reads = reads;
		}
		device.destroy();
	}
	return out;
} ), null, 1 ) );
await browser.close(); server.close();
