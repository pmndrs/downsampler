// Shared browser-side helpers for the GPU suite, benchmark and device lab.
import { formats, fromHalf, integerDomains, isIntegerDomain, isPackedQuantization, isSupportedFormat, packTexel, storageWritable, toHalf, unpackTexel } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';

/** `compat`: a compatibility-mode device (featureLevel 'compatibility', without 'core-features-and-limits'). */
export async function requestDevice( { compat = false } = {} ) {

	if ( ! navigator.gpu ) throw new Error( 'WebGPU is not available in this browser.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance', ...( compat ? { featureLevel: 'compatibility' } : {} ) } );
	if ( ! adapter ) throw new Error( 'No WebGPU adapter.' );
	const timestamps = adapter.features.has( 'timestamp-query' );
	// Format features are requested when available; float32-filterable is not,
	// so float32 formats keep exercising the load path.
	const optional = [ 'texture-formats-tier1', 'bgra8unorm-storage' ].filter( feature => adapter.features.has( feature ) );
	const device = await adapter.requestDevice( {
		requiredFeatures: [ ...optional, ...( timestamps ? [ 'timestamp-query' ] : [] ) ],
		requiredLimits: Downsampler.requiredLimits( adapter )
	} );
	if ( compat && device.features.has( 'core-features-and-limits' ) ) throw new Error( 'compat: the browser gave a core device (it cannot enforce compatibility-mode validation).' );
	const info = adapter.info || {};
	return {
		device,
		adapter: {
			vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
			isFallbackAdapter: info.isFallbackAdapter, timestamps, features: optional, compatibility: ! device.features.has( 'core-features-and-limits' ),
			maxStorageTexturesPerShaderStage: device.limits.maxStorageTexturesPerShaderStage,
			subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize
		},
		userAgent: navigator.userAgent
	};

}

export function mipLevels( width, height ) {

	return Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;

}

/**
 * `render` adds RENDER_ATTACHMENT (the benchmark's baseline and markers need it; not every format is renderable).
 * `storage` defaults to whether the device can write the format as a storage texture (not rgb9e5ufloat, nor rg11b10ufloat without tier1, nor rg32float in compatibility mode).
 */
export function createTexture( device, { width, height, layers = 1, format, dimension = layers > 1 ? '2d-array' : '2d', storage = ! isSupportedFormat( format ) || ( storageWritable( format, device.features ) && ! ( formats[ format ].coreStorage && ! device.features.has( 'core-features-and-limits' ) ) ), render = false } ) {

	return device.createTexture( {
		label: `test ${ width }x${ height }x${ layers } ${ format }`,
		size: [ width, height, layers ], format, mipLevelCount: mipLevels( width, height ), textureBindingViewDimension: dimension,
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | ( storage ? GPUTextureUsage.STORAGE_BINDING : 0 ) | ( render ? GPUTextureUsage.RENDER_ATTACHMENT : 0 )
	} );

}

/** Stored values per texel and the typed array that holds them (packed formats: one u32 word per texel). */
function storage( format ) {

	const { quantization, channels } = formats[ format ];
	if ( isPackedQuantization( quantization ) ) return { channels: 1, Array: Uint32Array, quantization, packed: true };
	const arrays = { unorm8: Uint8Array, snorm8: Int8Array, unorm16: Uint16Array, snorm16: Int16Array, float16: Uint16Array, float32: Float32Array };
	return { channels, Array: arrays[ quantization ], quantization };

}

/** A value in a format's encoding (the inverse of `decode`), rounding to nearest. */
function encode( value, quantization ) {

	if ( isIntegerDomain( quantization ) ) {

		const { scale, min, max } = integerDomains[ quantization ];
		return Math.min( max, Math.max( min, Math.round( value * scale ) ) );

	}

	return quantization === 'float16' ? toHalf( value ) : value;

}

function decode( stored, quantization ) {

	if ( isIntegerDomain( quantization ) ) return Math.max( - 1, stored / integerDomains[ quantization ].scale );
	return quantization === 'float16' ? fromHalf( stored ) : stored;

}

/**
 * Deterministic fixtures, `channels` values per texel. 'random' covers all
 * values (HDR inputs span [-4, 12] for float formats unless `unit`, as depth
 * needs), 'edge' isolates last-row/last-column handling, 'constant' catches
 * weight errors, 'tiles' changes per 64×64 tile and frame.
 */
export function makeData( width, height, layers, format, pattern = 'random', seed = 1, { unit = false } = {} ) {

	const { channels, Array, quantization, packed } = storage( format );
	if ( packed ) return makePackedData( width, height, layers, quantization, pattern, seed );
	const data = new Array( width * height * layers * channels );
	let state = seed;

	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let c = 0; c < channels; c ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		let value = ( state >>> 24 ) / 255;
		if ( pattern === 'random' && quantization === 'float16' && ! unit ) value = ( ( state >>> 24 ) - 64 ) / 16;
		if ( pattern === 'random' && quantization === 'float32' ) value = unit ? ( state >>> 8 ) / 2 ** 24 : ( state >>> 8 ) / 2 ** 20 - 4;
		if ( pattern === 'random' && quantization === 'unorm16' ) value = ( state >>> 16 ) / 65535;
		if ( pattern === 'random' && quantization.startsWith( 'snorm' ) ) value = ( state >>> 16 ) / 32767.5 - 1;
		if ( pattern === 'edge' ) value = x === width - 1 || y === height - 1 ? ( c + layer + 1 ) / ( layers + 4 ) : 0;
		if ( pattern === 'constant' ) value = ( c + layer + 1 ) / ( layers + 4 );
		if ( pattern === 'tiles' ) value = ( ( ( x >> 6 ) * 53 + ( y >> 6 ) * 29 + seed * 97 + c * 37 + layer * 11 ) % 256 ) / 255;
		const index = ( ( layer * height + y ) * width + x ) * channels + c;
		data[ index ] = encode( value, quantization );

	}

	return data;

}

/**
 * Packed float fixtures (rgb9e5ufloat, rg11b10ufloat): r, g, b per texel, packed
 * with the core encoder. 'random' mixes HDR values in [−4, 12) (negatives clamp
 * to 0) with values spread over 2^−14 … 2^10 per channel, so texels mix small
 * and large channels (shared-exponent flushing, subnormal 11/10-bit values).
 * The other patterns are as in `makeData`.
 */
function makePackedData( width, height, layers, quantization, pattern, seed ) {

	const data = new Uint32Array( width * height * layers );
	let state = seed;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 );
	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		const wide = ( next() >>> 31 ) === 1;
		const rgb = [ 0, 1, 2 ].map( c => {

			const bits = next();
			let value = wide ? ( ( bits & 0xffff ) / 65536 + 1 ) * 2 ** ( ( bits >>> 24 ) % 25 - 14 ) : ( ( bits >>> 24 ) - 64 ) / 16;
			if ( pattern === 'edge' ) value = x === width - 1 || y === height - 1 ? ( c + layer + 1 ) / ( layers + 4 ) : 0;
			if ( pattern === 'constant' ) value = ( c + layer + 1 ) / ( layers + 4 );
			if ( pattern === 'tiles' ) value = ( ( ( x >> 6 ) * 53 + ( y >> 6 ) * 29 + seed * 97 + c * 37 + layer * 11 ) % 256 ) / 32;
			return value;

		} );
		data[ ( layer * height + y ) * width + x ] = packTexel( ...rgb, quantization );

	}

	return data;

}

/** Stored values to RGBA floats, as a texture read returns them (missing channels read 0, 0, 1). */
export function toFloats( data, format ) {

	const { channels, quantization, packed } = storage( format );
	if ( packed ) {

		const output = new Float32Array( data.length * 4 );
		for ( let i = 0; i < data.length; i ++ ) output.set( [ ...unpackTexel( data[ i ], quantization ), 1 ], i * 4 );
		return output;

	}
	const texels = data.length / channels;
	const output = new Float32Array( texels * 4 );
	for ( let i = 0; i < texels; i ++ ) for ( let c = 0; c < 4; c ++ ) {

		const value = c < channels ? data[ i * channels + c ] : c === 3 ? 1 : 0;
		output[ i * 4 + c ] = c >= channels ? value : decode( value, quantization );

	}

	return output;

}

export function upload( device, texture, data ) {

	device.queue.writeTexture( { texture }, data, {
		bytesPerRow: texture.width * formats[ texture.format ].bytesPerTexel, rowsPerImage: texture.height
	}, [ texture.width, texture.height, texture.depthOrArrayLayers ] );

}

/**
 * Overwrite levels so a skipped write cannot pass on stale data. Uses
 * writeTexture rather than a render-pass clear: not every format is renderable.
 */
export function poison( device, texture, fromLevel = 1 ) {

	const { channels, Array, quantization, packed } = storage( texture.format );
	const colour = packed ? [ packTexel( 0.9, 0.1, 0.7, quantization ) ] : [ 0.9, 0.1, 0.7, 0.3 ].slice( 0, channels ).map( value => encode( value, quantization ) );

	for ( let level = fromLevel; level < texture.mipLevelCount; level ++ ) {

		const width = Math.max( 1, texture.width >> level ), height = Math.max( 1, texture.height >> level ), layers = texture.depthOrArrayLayers;
		const data = new Array( width * height * layers * channels );
		for ( let i = 0; i < data.length; i ++ ) data[ i ] = colour[ i % channels ];
		device.queue.writeTexture( { texture, mipLevel: level }, data, { bytesPerRow: width * formats[ texture.format ].bytesPerTexel, rowsPerImage: height }, [ width, height, layers ] );

	}

}

export async function readMip( device, texture, level ) {

	return ( await readMips( device, texture, level, level + 1 ) )[ 0 ];

}

/** Every level from `fromLevel` on, read back with a single submit and map. */
export async function readMips( device, texture, fromLevel = 0, toLevel = texture.mipLevelCount ) {

	const bytes = formats[ texture.format ].bytesPerTexel, layers = texture.depthOrArrayLayers;
	const levels = [];
	let size = 0;

	for ( let level = fromLevel; level < toLevel; level ++ ) {

		const width = Math.max( 1, texture.width >> level ), height = Math.max( 1, texture.height >> level );
		const bytesPerRow = Math.ceil( width * bytes / 256 ) * 256;
		levels.push( { level, width, height, bytesPerRow, offset: size } );
		size += bytesPerRow * height * layers;

	}

	if ( levels.length === 0 ) return [];
	const buffer = device.createBuffer( { size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	for ( const { level, width, height, bytesPerRow, offset } of levels ) encoder.copyTextureToBuffer( { texture, mipLevel: level }, { buffer, offset, bytesPerRow, rowsPerImage: height }, [ width, height, layers ] );
	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const mapped = buffer.getMappedRange();
	const output = levels.map( ( { width, height, bytesPerRow, offset } ) => {

		const { channels, Array } = storage( texture.format );
		const data = new Array( width * height * layers * channels );
		for ( let row = 0; row < height * layers; row ++ ) data.set( new Array( mapped, offset + row * bytesPerRow, width * channels ), row * width * channels );
		return toFloats( data, texture.format );

	} );
	buffer.unmap();
	buffer.destroy();
	return output;

}

/**
 * A depth texture holding `values` (one float in [0, 1] per texel), and what it
 * actually stores, read back as RGBA (d, 0, 0, 1). depth16unorm is written with a
 * copy; depth32float can't be, so a render pass writes frag_depth from an
 * r32float staging texture.
 */
export async function createDepthSource( device, { width, height, layers = 1, format = 'depth32float' }, values ) {

	const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | ( format === 'depth16unorm' ? GPUTextureUsage.COPY_DST : 0 );
	const depth = device.createTexture( { label: `test ${ format } ${ width }x${ height }x${ layers }`, size: [ width, height, layers ], format, usage } );

	if ( format === 'depth16unorm' ) {

		device.queue.writeTexture( { texture: depth }, Uint16Array.from( values, value => Math.round( value * 65535 ) ), { bytesPerRow: width * 2, rowsPerImage: height }, [ width, height, layers ] );

	} else {

		// One staging texture per layer: compatibility mode binds whole textures only.
		const stagings = Array.from( { length: layers }, ( _, layer ) => {

			const staging = device.createTexture( { size: [ width, height ], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
			device.queue.writeTexture( { texture: staging }, values.subarray( layer * width * height, ( layer + 1 ) * width * height ), { bytesPerRow: width * 4, rowsPerImage: height }, [ width, height ] );
			return staging;

		} );
		const module = device.createShaderModule( { code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	return vec4f(array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3))[index], 0, 1);
}
@fragment fn fs(@builtin(position) position: vec4f) -> @builtin(frag_depth) f32 { return textureLoad(image, vec2i(position.xy), 0).r; }` } );
		const pipeline = device.createRenderPipeline( { layout: 'auto', vertex: { module }, fragment: { module, targets: [] }, depthStencil: { format, depthWriteEnabled: true, depthCompare: 'always' } } );
		const encoder = device.createCommandEncoder();

		for ( let layer = 0; layer < layers; layer ++ ) {

			const layerView = stagings[ layer ].createView();
			const pass = encoder.beginRenderPass( { colorAttachments: [], depthStencilAttachment: {
				view: depth.createView( { dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 } ), depthLoadOp: 'clear', depthClearValue: 0.5, depthStoreOp: 'store'
			} } );
			pass.setPipeline( pipeline );
			pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: layerView } ] } ) );
			pass.draw( 3 );
			pass.end();

		}

		device.queue.submit( [ encoder.finish() ] );
		for ( const staging of stagings ) staging.destroy();

	}

	// Read back what was stored: the oracle uses actual contents, not intent.
	const bytes = format === 'depth16unorm' ? 2 : 4;
	const bytesPerRow = Math.ceil( width * bytes / 256 ) * 256;
	const buffer = device.createBuffer( { size: bytesPerRow * height * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyTextureToBuffer( { texture: depth, aspect: 'depth-only' }, { buffer, bytesPerRow, rowsPerImage: height }, [ width, height, layers ] );
	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const stored = new Float32Array( width * height * layers * 4 );
	const mapped = buffer.getMappedRange();
	for ( let row = 0; row < height * layers; row ++ ) {

		const rowValues = bytes === 2 ? new Uint16Array( mapped, row * bytesPerRow, width ) : new Float32Array( mapped, row * bytesPerRow, width );
		for ( let x = 0; x < width; x ++ ) stored.set( [ bytes === 2 ? rowValues[ x ] / 65535 : rowValues[ x ], 0, 0, 1 ], ( row * width + x ) * 4 );

	}

	buffer.unmap();
	buffer.destroy();
	return { texture: depth, floats: stored };

}

export function median( values ) {

	const sorted = values.slice().sort( ( a, b ) => a - b );
	return sorted[ Math.floor( sorted.length / 2 ) ];

}
