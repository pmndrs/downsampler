// Probe (M2.3.5): what sRGB destinations can rely on.
// 1. Hardware decode: textureLoad of every 8-bit code through an rgba8unorm-srgb view,
//    against the f32 of the exact IEC 61966-2-1 decode (core srgbDecodeTable).
// 2. Filtering happens in linear: a bilinear sample halfway between codes 0 and 255.
// 3. Copies: rgba8unorm → rgba8unorm-srgb (copyTextureToTexture) and a buffer into
//    rgba8unorm-srgb and bgra8unorm-srgb (copyBufferToTexture) validate and keep bytes.
// 4. A storage texture can't be sRGB, even through a view (expected: validation errors).
// 5. A module-scope const array indexed at run time compiles and reads correctly.
// Run: node results/probes/srgb-probe.mjs [--swiftshader] [--power=high|low]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const swift = process.argv.includes( '--swiftshader' );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...( swift ? [ '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' ] : gpuChoiceArgs( process.argv.find( arg => arg.startsWith( '--power=' ) )?.split( '=' )[ 1 ] ) ) ] } );
try {

	const page = await browser.newPage();
	await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
	console.log( JSON.stringify( await page.evaluate( async () => {

		const { srgbDecodeTable } = await import( '/dist/index.js' );
		const adapter = await navigator.gpu.requestAdapter();
		const device = await adapter.requestDevice();
		const out = { adapter: adapter.info?.description || adapter.info?.vendor };
		const read = async ( texture, bytesPerTexel = 4 ) => {

			const bytesPerRow = Math.ceil( texture.width * bytesPerTexel / 256 ) * 256;
			const buffer = device.createBuffer( { size: bytesPerRow * texture.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
			const encoder = device.createCommandEncoder();
			encoder.copyTextureToBuffer( { texture }, { buffer, bytesPerRow }, [ texture.width, texture.height ] );
			device.queue.submit( [ encoder.finish() ] );
			await buffer.mapAsync( GPUMapMode.READ );
			const bytes = new Uint8Array( buffer.getMappedRange().slice( 0 ) );
			buffer.unmap();
			return bytes;

		};

		// 1 + 5: decode every code with textureLoad; the same values through a runtime-indexed const array.
		const codes = new Uint8Array( 256 * 4 );
		for ( let i = 0; i < 256; i ++ ) codes.set( [ i, 255 - i, i, i ], i * 4 );
		const source = device.createTexture( { size: [ 256, 1 ], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC } );
		device.queue.writeTexture( { texture: source }, codes, { bytesPerRow: 1024 }, [ 256, 1 ] );
		const table = Array.from( srgbDecodeTable, value => value.toPrecision( 9 ) ).join( ', ' );
		device.pushErrorScope( 'validation' );
		const module = device.createShaderModule( { code: `
const DS_TABLE = array<f32, 256>(${ table });
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> lookup: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= 256u) { return; }
	output[id.x] = textureLoad(image, vec2i(i32(id.x), 0), 0);
	lookup[id.x] = DS_TABLE[255u - id.x];
}` } );
		const info = await module.getCompilationInfo();
		out.constArrayMessages = info.messages.map( message => message.message );
		const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
		const output = device.createBuffer( { size: 256 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		const lookup = device.createBuffer( { size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		const readOut = device.createBuffer( { size: 256 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		const readLookup = device.createBuffer( { size: 256 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		let encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline( pipeline );
		pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: source.createView() }, { binding: 1, resource: { buffer: output } }, { binding: 2, resource: { buffer: lookup } } ] } ) );
		pass.dispatchWorkgroups( 4 );
		pass.end();
		encoder.copyBufferToBuffer( output, 0, readOut, 0, 256 * 16 );
		encoder.copyBufferToBuffer( lookup, 0, readLookup, 0, 256 * 4 );
		device.queue.submit( [ encoder.finish() ] );
		await Promise.all( [ readOut.mapAsync( GPUMapMode.READ ), readLookup.mapAsync( GPUMapMode.READ ) ] );
		const decoded = new Float32Array( readOut.getMappedRange() ), looked = new Float32Array( readLookup.getMappedRange() );
		const decodeDiffs = [], alphaDiffs = [], lookupDiffs = [];
		for ( let i = 0; i < 256; i ++ ) {

			if ( decoded[ i * 4 ] !== srgbDecodeTable[ i ] ) decodeDiffs.push( { code: i, gpu: decoded[ i * 4 ], table: srgbDecodeTable[ i ], ulps: ( decoded[ i * 4 ] - srgbDecodeTable[ i ] ) / ( srgbDecodeTable[ i ] * 2 ** - 23 || 2 ** - 149 ) } );
			if ( decoded[ i * 4 + 3 ] !== Math.fround( i / 255 ) ) alphaDiffs.push( { code: i, gpu: decoded[ i * 4 + 3 ] } );
			if ( looked[ i ] !== srgbDecodeTable[ 255 - i ] ) lookupDiffs.push( i );

		}

		out.decode = { mismatches: decodeDiffs.length, first: decodeDiffs.slice( 0, 4 ), alphaLinear: alphaDiffs.length === 0, values: Array.from( { length: 256 }, ( _, i ) => decoded[ i * 4 ] ) };
		out.constArrayLookupMismatches = lookupDiffs.length;

		// 2: sample halfway between codes 0 and 255 (linear filtering gives 0.5, sRGB-space would give ~0.214).
		const pair = device.createTexture( { size: [ 2, 1 ], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		device.queue.writeTexture( { texture: pair }, new Uint8Array( [ 0, 0, 0, 0, 255, 255, 255, 255 ] ), { bytesPerRow: 8 }, [ 2, 1 ] );
		const sampleModule = device.createShaderModule( { code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: vec4f;
@compute @workgroup_size(1) fn main() { output = textureSampleLevel(image, linearSampler, vec2f(0.5, 0.5), 0.0); }` } );
		const samplePipeline = device.createComputePipeline( { layout: 'auto', compute: { module: sampleModule, entryPoint: 'main' } } );
		const sampled = device.createBuffer( { size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		const readSampled = device.createBuffer( { size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		encoder = device.createCommandEncoder();
		const samplePass = encoder.beginComputePass();
		samplePass.setPipeline( samplePipeline );
		samplePass.setBindGroup( 0, device.createBindGroup( { layout: samplePipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: pair.createView() }, { binding: 1, resource: device.createSampler( { minFilter: 'linear', magFilter: 'linear' } ) }, { binding: 2, resource: { buffer: sampled } } ] } ) );
		samplePass.dispatchWorkgroups( 1 );
		samplePass.end();
		encoder.copyBufferToBuffer( sampled, 0, readSampled, 0, 16 );
		device.queue.submit( [ encoder.finish() ] );
		await readSampled.mapAsync( GPUMapMode.READ );
		out.halfwaySample = Array.from( new Float32Array( readSampled.getMappedRange() ) );

		// 3: copies keep bytes.
		const unorm = device.createTexture( { size: [ 256, 1 ], format: 'rgba8unorm', usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		device.queue.writeTexture( { texture: unorm }, codes, { bytesPerRow: 1024 }, [ 256, 1 ] );
		const srgb = device.createTexture( { size: [ 256, 1 ], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		const bgra = device.createTexture( { size: [ 256, 1 ], format: 'bgra8unorm-srgb', usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		const staging = device.createBuffer( { size: 1024, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( staging, 0, codes );
		encoder = device.createCommandEncoder();
		encoder.copyTextureToTexture( { texture: unorm }, { texture: srgb }, [ 256, 1 ] );
		encoder.copyBufferToTexture( { buffer: staging, bytesPerRow: 1024 }, { texture: bgra }, [ 256, 1 ] );
		device.queue.submit( [ encoder.finish() ] );
		const t2t = await read( srgb ), b2t = await read( bgra );
		out.copyTextureToTextureKeepsBytes = t2t.slice( 0, 1024 ).every( ( byte, i ) => byte === codes[ i ] );
		out.copyBufferToTextureKeepsBytes = b2t.slice( 0, 1024 ).every( ( byte, i ) => byte === codes[ i ] );
		const copyErrors = await device.popErrorScope();
		out.validationBeforeStorageChecks = copyErrors?.message ?? null;

		// 4: sRGB storage, directly and through a view, must fail.
		const storageErrors = [];
		for ( const attempt of [
			() => device.createTexture( { size: [ 4, 4 ], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.STORAGE_BINDING } ),
			() => device.createTexture( { size: [ 4, 4 ], format: 'rgba8unorm', viewFormats: [ 'rgba8unorm-srgb' ], usage: GPUTextureUsage.STORAGE_BINDING } ).createView( { format: 'rgba8unorm-srgb' } ),
			() => device.createShaderModule( { code: '@group(0) @binding(0) var image: texture_storage_2d<rgba8unorm-srgb, write>; @compute @workgroup_size(1) fn main() {}' } )
		] ) {

			device.pushErrorScope( 'validation' );
			let thrown = null;
			try { attempt(); } catch ( error ) { thrown = error.message; }
			const error = await device.popErrorScope();
			storageErrors.push( thrown ?? error?.message?.split( '\n' )[ 0 ] ?? 'no error' );

		}

		out.srgbStorage = storageErrors;
		return out;

	} ), null, 1 ) );

} finally {

	await browser.close();
	server.close();

}
