// Shared browser-side helpers for the GPU suite, benchmark and device lab.
import { formats, fromHalf, toHalf } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';

export async function requestDevice() {

	if ( ! navigator.gpu ) throw new Error( 'WebGPU is not available in this browser.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
	if ( ! adapter ) throw new Error( 'No WebGPU adapter.' );
	const timestamps = adapter.features.has( 'timestamp-query' );
	const device = await adapter.requestDevice( {
		requiredFeatures: timestamps ? [ 'timestamp-query' ] : [],
		requiredLimits: Downsampler.requiredLimits( adapter )
	} );
	const info = adapter.info || {};
	return {
		device,
		adapter: {
			vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
			isFallbackAdapter: info.isFallbackAdapter, timestamps,
			maxStorageTexturesPerShaderStage: device.limits.maxStorageTexturesPerShaderStage,
			subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize
		},
		userAgent: navigator.userAgent
	};

}

export function mipLevels( width, height ) {

	return Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;

}

export function createTexture( device, { width, height, layers = 1, format, dimension = layers > 1 ? '2d-array' : '2d', storage = true } ) {

	return device.createTexture( {
		label: `test ${ width }x${ height }x${ layers } ${ format }`,
		size: [ width, height, layers ], format, mipLevelCount: mipLevels( width, height ), textureBindingViewDimension: dimension,
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | ( storage ? GPUTextureUsage.STORAGE_BINDING : 0 )
	} );

}

/**
 * Deterministic fixtures. 'random' covers all values (HDR inputs span [-4, 12]
 * for float formats), 'edge' isolates last-row/last-column handling,
 * 'constant' catches weight errors, 'tiles' changes per 64×64 tile and frame.
 */
export function makeData( width, height, layers, format, pattern = 'random', seed = 1 ) {

	const unorm = formats[ format ].quantization === 'unorm8';
	const data = unorm ? new Uint8Array( width * height * layers * 4 ) : new Uint16Array( width * height * layers * 4 );
	let state = seed;

	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let c = 0; c < 4; c ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		let value = ( state >>> 24 ) / 255;
		if ( pattern === 'random' && ! unorm ) value = ( ( state >>> 24 ) - 64 ) / 16;
		if ( pattern === 'edge' ) value = x === width - 1 || y === height - 1 ? ( c + layer + 1 ) / ( layers + 4 ) : 0;
		if ( pattern === 'constant' ) value = ( c + layer + 1 ) / ( layers + 4 );
		if ( pattern === 'tiles' ) value = ( ( ( x >> 6 ) * 53 + ( y >> 6 ) * 29 + seed * 97 + c * 37 + layer * 11 ) % 256 ) / 255;
		const index = ( ( layer * height + y ) * width + x ) * 4 + c;
		data[ index ] = unorm ? Math.round( value * 255 ) : toHalf( value );

	}

	return data;

}

export function toFloats( data, format ) {

	return Float32Array.from( data, value => formats[ format ].quantization === 'unorm8' ? value / 255 : fromHalf( value ) );

}

export function upload( device, texture, data ) {

	device.queue.writeTexture( { texture }, data, {
		bytesPerRow: texture.width * formats[ texture.format ].bytesPerTexel, rowsPerImage: texture.height
	}, [ texture.width, texture.height, texture.depthOrArrayLayers ] );

}

/** Overwrite levels so a skipped write cannot pass on stale data. */
export function poison( device, texture, fromLevel = 1 ) {

	const encoder = device.createCommandEncoder();
	for ( let level = fromLevel; level < texture.mipLevelCount; level ++ ) for ( let layer = 0; layer < texture.depthOrArrayLayers; layer ++ ) {

		encoder.beginRenderPass( { colorAttachments: [ {
			view: texture.createView( { dimension: '2d', baseMipLevel: level, mipLevelCount: 1, baseArrayLayer: layer, arrayLayerCount: 1 } ),
			loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.9, g: 0.1, b: 0.7, a: 0.3 }
		} ] } ).end();

	}

	device.queue.submit( [ encoder.finish() ] );

}

export async function readMip( device, texture, level ) {

	const width = Math.max( 1, texture.width >> level ), height = Math.max( 1, texture.height >> level ), layers = texture.depthOrArrayLayers;
	const bytes = formats[ texture.format ].bytesPerTexel;
	const bytesPerRow = Math.ceil( width * bytes / 256 ) * 256;
	const buffer = device.createBuffer( { size: bytesPerRow * height * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyTextureToBuffer( { texture, mipLevel: level }, { buffer, bytesPerRow, rowsPerImage: height }, [ width, height, layers ] );
	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const mapped = buffer.getMappedRange();
	const output = new Float32Array( width * height * layers * 4 );

	for ( let row = 0; row < height * layers; row ++ ) {

		const values = bytes === 4 ? new Uint8Array( mapped, row * bytesPerRow, width * 4 ) : new Uint16Array( mapped, row * bytesPerRow, width * 4 );
		for ( let i = 0; i < values.length; i ++ ) output[ row * width * 4 + i ] = bytes === 4 ? values[ i ] / 255 : fromHalf( values[ i ] );

	}

	buffer.unmap();
	buffer.destroy();
	return output;

}

/** Every level from `fromLevel` on, read back with a single submit and map. */
export async function readMips( device, texture, fromLevel = 0 ) {

	const bytes = formats[ texture.format ].bytesPerTexel, layers = texture.depthOrArrayLayers;
	const levels = [];
	let size = 0;

	for ( let level = fromLevel; level < texture.mipLevelCount; level ++ ) {

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
	const output = levels.map( ( { width, height, bytesPerRow, offset } ) => decodeRows( mapped, offset, bytesPerRow, width, height * layers, bytes ) );
	buffer.unmap();
	buffer.destroy();
	return output;

}

function decodeRows( mapped, offset, bytesPerRow, width, rows, bytes ) {

	const output = new Float32Array( width * rows * 4 );

	for ( let row = 0; row < rows; row ++ ) {

		const values = bytes === 4 ? new Uint8Array( mapped, offset + row * bytesPerRow, width * 4 ) : new Uint16Array( mapped, offset + row * bytesPerRow, width * 4 );
		for ( let i = 0; i < values.length; i ++ ) output[ row * width * 4 + i ] = bytes === 4 ? values[ i ] / 255 : fromHalf( values[ i ] );

	}

	return output;

}

export function median( values ) {

	const sorted = values.slice().sort( ( a, b ) => a - b );
	return sorted[ Math.floor( sorted.length / 2 ) ];

}
