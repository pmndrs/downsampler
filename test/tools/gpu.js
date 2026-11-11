// WebGPU helpers for the tool fixtures: device, uploads, readback, formats.
import { fromHalfBits, halfArray } from './half.js';

export async function requestToolsDevice() {

	if ( ! navigator.gpu ) throw new Error( 'WebGPU is not available in this browser.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
	if ( ! adapter ) throw new Error( 'No WebGPU adapter.' );
	const optional = [ 'timestamp-query', 'texture-formats-tier1' ].filter( feature => adapter.features.has( feature ) );
	const device = await adapter.requestDevice( {
		requiredFeatures: optional,
		requiredLimits: { maxStorageTexturesPerShaderStage: adapter.limits.maxStorageTexturesPerShaderStage }
	} );
	const info = adapter.info || {};
	return {
		device,
		adapter: {
			vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
			isFallbackAdapter: info.isFallbackAdapter, features: optional,
			maxStorageTexturesPerShaderStage: device.limits.maxStorageTexturesPerShaderStage
		},
		userAgent: navigator.userAgent
	};

}

/**
 * Storage formats for the FSR outputs. AMD uses R16F, RG16F and R8; WebGPU
 * needs `texture-formats-tier1` to write those, so without it the fixtures
 * write the four-channel formats (same quantization, more bytes).
 */
export function outputFormats( device ) {

	const tier1 = device.features.has( 'texture-formats-tier1' );
	return { half1: tier1 ? 'r16float' : 'rgba16float', half2: tier1 ? 'rg16float' : 'rgba16float', unorm1: tier1 ? 'r8unorm' : 'rgba8unorm' };

}

const layouts = {
	r16float: { channels: 1, bytes: 2, kind: 'half' }, rg16float: { channels: 2, bytes: 4, kind: 'half' }, rgba16float: { channels: 4, bytes: 8, kind: 'half' },
	r32float: { channels: 1, bytes: 4, kind: 'float' }, rgba32float: { channels: 4, bytes: 16, kind: 'float' },
	r8unorm: { channels: 1, bytes: 1, kind: 'unorm' }, rgba8unorm: { channels: 4, bytes: 4, kind: 'unorm' },
	rgba32uint: { channels: 4, bytes: 16, kind: 'uint' }
};

export function channelsOf( format ) {

	return layouts[ format ].channels;

}

/** A texture filled from floats (one value per channel of `format`; uint formats take a Uint32Array). */
export function createTexture( device, { label, width, height, format, data = null, usage = 0, mipLevelCount = 1 } ) {

	const texture = device.createTexture( {
		label, size: [ width, height ], format, mipLevelCount,
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | usage
	} );
	if ( data ) writeTexture( device, texture, data );
	return texture;

}

export function writeTexture( device, texture, data, mipLevel = 0 ) {

	const { bytes, kind } = layouts[ texture.format ];
	const width = Math.max( 1, texture.width >> mipLevel ), height = Math.max( 1, texture.height >> mipLevel );
	const raw = kind === 'half' ? halfArray( data ) : kind === 'unorm' ? Uint8Array.from( data, value => Math.round( Math.min( 1, Math.max( 0, value ) ) * 255 ) ) : data;
	device.queue.writeTexture( { texture, mipLevel }, raw, { bytesPerRow: width * bytes, rowsPerImage: height }, [ width, height ] );

}

/** One level read back as floats (uint formats: a Uint32Array), `channels` values per texel. */
export async function readTexture( device, texture, mipLevel = 0 ) {

	const { bytes, kind, channels } = layouts[ texture.format ];
	const width = Math.max( 1, texture.width >> mipLevel ), height = Math.max( 1, texture.height >> mipLevel );
	const bytesPerRow = Math.ceil( width * bytes / 256 ) * 256;
	const buffer = device.createBuffer( { size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyTextureToBuffer( { texture, mipLevel }, { buffer, bytesPerRow, rowsPerImage: height }, [ width, height ] );
	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const mapped = buffer.getMappedRange();
	const output = kind === 'uint' ? new Uint32Array( width * height * channels ) : new Float32Array( width * height * channels );

	for ( let y = 0; y < height; y ++ ) {

		const row = kind === 'half' ? new Uint16Array( mapped, y * bytesPerRow, width * channels )
			: kind === 'unorm' ? new Uint8Array( mapped, y * bytesPerRow, width * channels )
				: kind === 'uint' ? new Uint32Array( mapped, y * bytesPerRow, width * channels )
					: new Float32Array( mapped, y * bytesPerRow, width * channels );
		for ( let i = 0; i < row.length; i ++ ) output[ y * width * channels + i ] = kind === 'half' ? fromHalfBits( row[ i ] ) : kind === 'unorm' ? row[ i ] / 255 : row[ i ];

	}

	buffer.unmap();
	buffer.destroy();
	return { width, height, channels, data: output };

}

export async function readBuffer( device, source, size = source.size ) {

	const buffer = device.createBuffer( { size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer( source, 0, buffer, 0, size );
	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const copy = buffer.getMappedRange().slice( 0 );
	buffer.unmap();
	buffer.destroy();
	return copy;

}

export function uniformBuffer( device, label, bytes ) {

	const buffer = device.createBuffer( { label, size: Math.ceil( bytes.byteLength / 16 ) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( buffer, 0, bytes );
	return buffer;

}

/** Compiles a module and fails loudly on WGSL errors (getCompilationInfo), naming the fixture. */
export async function compile( device, label, code ) {

	const module = device.createShaderModule( { label, code } );
	const info = await module.getCompilationInfo();
	const errors = info.messages.filter( message => message.type === 'error' );
	if ( errors.length ) {

		const lines = code.split( '\n' );
		throw new Error( `${ label }: ${ errors.map( error => `${ error.lineNum }:${ error.linePos } ${ error.message }\n  ${ lines[ error.lineNum - 1 ] }` ).join( '\n' ) }` );

	}

	return module;

}

export function median( values ) {

	const sorted = values.slice().sort( ( a, b ) => a - b );
	return sorted[ Math.floor( sorted.length / 2 ) ];

}
