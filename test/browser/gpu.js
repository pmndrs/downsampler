// Shared browser-side helpers for the GPU suite, benchmark and device lab.
import { channelDomain, formats, fromHalf, intRange, isIntegerDomain, isIntQuantization, isPackedQuantization, isSupportedFormat, packTexel, srgbDecodeTable, storageWritable, toHalf, unpackTexel } from '/dist/index.js';
import { requiredLimits } from '/dist/webgpu/index.js';

/**
 * `compat`: a compatibility-mode device (featureLevel 'compatibility', without 'core-features-and-limits').
 * `maxStorageTextures`: request the adapter's storage texture limit (the atomic probe's tail levels), not just six.
 * `features`: extra optional features to request when the adapter has them (webgpu-spd uses 'subgroups' and 'shader-f16').
 * `timestamps: false`: don't request timestamp-query even when the adapter has it (the throughput bench's no-timestamp path).
 */
export async function requestDevice( { compat = false, maxStorageTextures = false, features = [], timestamps: wantTimestamps = true } = {} ) {

	if ( ! navigator.gpu ) throw new Error( 'WebGPU is not available in this browser.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance', ...( compat ? { featureLevel: 'compatibility' } : {} ) } );
	if ( ! adapter ) throw new Error( 'No WebGPU adapter.' );
	const timestamps = wantTimestamps && adapter.features.has( 'timestamp-query' );
	// Format features are requested when available; float32-filterable is not,
	// so float32 formats keep exercising the load path.
	const optional = [ 'texture-formats-tier1', 'bgra8unorm-storage', ...features ].filter( feature => adapter.features.has( feature ) );
	const device = await adapter.requestDevice( {
		requiredFeatures: [ ...optional, ...( timestamps ? [ 'timestamp-query' ] : [] ) ],
		requiredLimits: maxStorageTextures ? { maxStorageTexturesPerShaderStage: adapter.limits.maxStorageTexturesPerShaderStage } : requiredLimits( adapter )
	} );
	if ( compat && device.features.has( 'core-features-and-limits' ) ) throw new Error( 'compat: the browser gave a core device (it cannot enforce compatibility-mode validation).' );
	const info = adapter.info || {};
	return {
		device,
		adapter: {
			vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
			isFallbackAdapter: info.isFallbackAdapter, timestamps, features: optional, compatibility: ! device.features.has( 'core-features-and-limits' ),
			maxStorageTexturesPerShaderStage: device.limits.maxStorageTexturesPerShaderStage,
			subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize,
			adapterFeatures: [ ...adapter.features ].sort(), maxTextureDimension2D: device.limits.maxTextureDimension2D
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

/**
 * Stored values per texel and the typed array that holds them (packed formats: one u32 word per texel;
 * rgb10a2unorm and rgb10a2uint too, `word10a2`, their four channels packed with `pack10a2`).
 */
function storage( format ) {

	const { quantization, channels } = formats[ format ];
	if ( isPackedQuantization( quantization ) ) return { channels: 1, Array: Uint32Array, quantization, packed: true };
	if ( quantization === 'unorm10a2' || quantization === 'uint10a2' ) return { channels: 1, Array: Uint32Array, quantization, word10a2: true };
	const arrays = { unorm8: Uint8Array, snorm8: Int8Array, unorm16: Uint16Array, snorm16: Int16Array, float16: Uint16Array, float32: Float32Array, srgb8: Uint8Array, uint8: Uint8Array, sint8: Int8Array, uint16: Uint16Array, sint16: Int16Array, uint32: Uint32Array, sint32: Int32Array };
	return { channels, Array: arrays[ quantization ], quantization };

}

/** Four 10-, 10-, 10- and 2-bit channel integers as one rgb10a2 word, and back. */
export const pack10a2 = ( r, g, b, a ) => ( r | ( g << 10 ) | ( b << 20 ) | ( a << 30 ) ) >>> 0;
export const unpack10a2 = word => [ word & 0x3ff, ( word >>> 10 ) & 0x3ff, ( word >>> 20 ) & 0x3ff, word >>> 30 ];

/**
 * A value in a format's encoding (the inverse of `decode`), rounding to nearest. `channel` matters for
 * rgb10a2unorm's 2-bit alpha. sRGB fixtures are made as codes: the value is the encoded one (code / 255)
 * in every channel.
 */
function encode( value, quantization, channel = 0 ) {

	if ( quantization === 'srgb8' ) return Math.min( 255, Math.max( 0, Math.round( value * 255 ) ) );
	if ( isIntegerDomain( quantization ) ) {

		const { scale, min, max } = channelDomain( quantization, channel );
		return Math.min( max, Math.max( min, Math.round( value * scale ) ) );

	}

	return quantization === 'float16' ? toHalf( value ) : value;

}

/** A stored value as a read returns it; sRGB colour channels through the exact decode table (alpha is linear). */
function decode( stored, quantization, channel = 0 ) {

	if ( quantization === 'srgb8' ) return channel === 3 ? Math.fround( stored / 255 ) : srgbDecodeTable[ stored ];
	if ( isIntegerDomain( quantization ) ) return Math.max( - 1, stored / channelDomain( quantization, channel ).scale );
	if ( isIntQuantization( quantization ) ) return stored;
	return quantization === 'float16' ? fromHalf( stored ) : stored;

}

/**
 * Integer fixtures: 'random' spans each channel's range, a quarter of the values at or next to its
 * ends (overflow and sign handling); 'small' takes values in [min, min + 3] and [max − 3, max] (frequent
 * exact ties next to the extremes); the other patterns are as in `makeData`, scaled to the range.
 */
function intValue( quantization, pattern, state, x, y, c, layer, width, height, layers, seed ) {

	const [ min, max ] = intRange( quantization, c );
	const span = max - min + 1;
	// Every bit of a second hash, so low bits (tie parity) vary at every width.
	const bits = Math.imul( state ^ ( state >>> 15 ), 0x2c1b3c6d ) >>> 0;
	const end = () => ( bits >>> 2 ) & 1 ? max - ( bits & 3 ) : min + ( bits & 3 );
	switch ( pattern ) {

		case 'random': return state >>> 30 === 0 ? end() : min + ( bits % span );
		case 'small': return end();
		case 'edge': return x === width - 1 || y === height - 1 ? max - c - layer : min + c;
		case 'constant': return min + Math.floor( span * ( c + layer + 1 ) / ( layers + 5 ) );
		default: return min + ( ( ( ( x >> 6 ) * 53 + ( y >> 6 ) * 29 + seed * 97 + c * 37 + layer * 11 ) % 256 ) * Math.floor( span / 256 ) );

	}

}

/**
 * Deterministic fixtures, `channels` values per texel. 'random' covers all
 * values (HDR inputs span [-4, 12] for float formats unless `unit`, as depth
 * needs), 'edge' isolates last-row/last-column handling, 'constant' catches
 * weight errors, 'tiles' changes per 64×64 tile and frame.
 */
export function makeData( width, height, layers, format, pattern = 'random', seed = 1, { unit = false } = {} ) {

	const { channels, Array, quantization, packed, word10a2 } = storage( format );
	if ( packed ) return makePackedData( width, height, layers, quantization, pattern, seed );
	if ( word10a2 ) {

		// Four channels per word: the rgb10a2unorm values of 'random' etc., or rgb10a2uint's integers.
		const words = new Uint32Array( width * height * layers );
		let state = seed;
		for ( let i = 0; i < words.length; i ++ ) {

			const x = i % width, y = Math.floor( i / width ) % height, layer = Math.floor( i / ( width * height ) );
			words[ i ] = pack10a2( ...[ 0, 1, 2, 3 ].map( c => {

				state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
				if ( quantization === 'uint10a2' ) return intValue( quantization, pattern, state, x, y, c, layer, width, height, layers, seed );
				let value = ( state >>> 22 ) / 1023;
				if ( pattern === 'edge' ) value = x === width - 1 || y === height - 1 ? ( c + layer + 1 ) / ( layers + 4 ) : 0;
				if ( pattern === 'constant' ) value = ( c + layer + 1 ) / ( layers + 4 );
				if ( pattern === 'tiles' ) value = ( ( ( x >> 6 ) * 53 + ( y >> 6 ) * 29 + seed * 97 + c * 37 + layer * 11 ) % 256 ) / 255;
				return encode( value, quantization, c );

			} ) );

		}

		return words;

	}

	const data = new Array( width * height * layers * channels );
	let state = seed;

	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let c = 0; c < channels; c ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		if ( isIntQuantization( quantization ) ) {

			data[ ( ( layer * height + y ) * width + x ) * channels + c ] = intValue( quantization, pattern, state, x, y, c, layer, width, height, layers, seed );
			continue;

		}

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

/**
 * Stored values to RGBA floats, as a texture read returns them (missing channels read 0, 0, 1).
 * Integer formats give their exact integers, in a Float64Array (a Float32Array can't hold every u32).
 */
export function toFloats( data, format ) {

	const { channels, quantization, packed, word10a2 } = storage( format );
	if ( packed ) {

		const output = new Float32Array( data.length * 4 );
		for ( let i = 0; i < data.length; i ++ ) output.set( [ ...unpackTexel( data[ i ], quantization ), 1 ], i * 4 );
		return output;

	}

	const int = isIntQuantization( quantization );
	if ( word10a2 ) {

		const output = new ( int ? Float64Array : Float32Array )( data.length * 4 );
		for ( let i = 0; i < data.length; i ++ ) output.set( unpack10a2( data[ i ] ).map( ( value, c ) => decode( value, quantization, c ) ), i * 4 );
		return output;

	}

	const texels = data.length / channels;
	const output = new ( int ? Float64Array : Float32Array )( texels * 4 );
	for ( let i = 0; i < texels; i ++ ) for ( let c = 0; c < 4; c ++ ) {

		const value = c < channels ? data[ i * channels + c ] : c === 3 ? 1 : 0;
		output[ i * 4 + c ] = c >= channels ? value : decode( value, quantization, c );

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

	const { channels, Array, quantization, packed, word10a2 } = storage( texture.format );
	// Integer formats: a value near each channel's top, bottom and middle.
	const tint = c => { const [ min, max ] = intRange( quantization, c ); return [ max - 1, min + 3, min + Math.floor( ( max - min ) / 3 ), max ][ c ]; };
	const rgba = [ 0.9, 0.1, 0.7, 0.3 ].map( ( value, c ) => isIntQuantization( quantization ) ? tint( c ) : encode( value, quantization, c ) );
	const colour = packed ? [ packTexel( 0.9, 0.1, 0.7, quantization ) ] : word10a2 ? [ pack10a2( ...rgba ) ] : rgba.slice( 0, channels );

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

/**
 * Records the buffers created on `device` from now on, and which of them are
 * destroyed, by wrapping `createBuffer` (and each new buffer's `destroy`) until
 * `restore()`. `settle()` waits for the queue and a task, so deferred
 * destruction (cache evictions, after onSubmittedWorkDone) has happened.
 */
export function trackBuffers( device ) {

	const created = [], destroyed = new Set();
	const createBuffer = device.createBuffer;
	device.createBuffer = descriptor => {

		const buffer = createBuffer.call( device, descriptor );
		const destroy = buffer.destroy;
		buffer.destroy = () => { destroyed.add( buffer ); destroy.call( buffer ); };
		created.push( buffer );
		return buffer;

	};

	return {
		created, destroyed,
		/** Buffers created since tracking started, by label substring. */
		labelled: text => created.filter( buffer => buffer.label.includes( text ) ),
		settle: async () => { await device.queue.onSubmittedWorkDone(); await new Promise( resolve => setTimeout( resolve, 0 ) ); },
		restore: () => { delete device.createBuffer; }
	};

}
