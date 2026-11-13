// Helpers shared by the examples: device setup, readback, a level viewer and
// the on-page readout. Nothing here is part of the library; it is plain
// WebGPU and DOM code you would write yourself.
import { formats, fromHalf } from '../dist/index.js';
import { Downsampler } from '../dist/webgpu/index.js';

/**
 * An adapter and a device with the limits the library can use: six storage
 * textures (six levels per dispatch) and 256 invocations per workgroup where
 * the adapter has them. Without them it still works, with four levels per
 * dispatch.
 */
export async function initDevice() {

	if ( ! navigator.gpu ) throw new Error( 'WebGPU is not available in this browser.' );
	const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
	if ( ! adapter ) throw new Error( 'No WebGPU adapter.' );
	const device = await adapter.requestDevice( { requiredLimits: Downsampler.requiredLimits( adapter ) } );
	device.lost.then( info => { if ( info.reason !== 'destroyed' ) fail( new Error( `GPU device lost: ${ info.message }` ) ); } );
	device.addEventListener( 'uncapturederror', event => fail( new Error( `WebGPU error: ${ event.error.message }` ) ) );
	const info = adapter.info ?? {};
	const name = [ info.vendor, info.architecture, info.description ].filter( Boolean ).join( ' ' ) || 'unknown adapter';
	setText( 'adapter', `${ name } · ${ device.limits.maxStorageTexturesPerShaderStage } storage textures per stage` );
	return { adapter, device };

}

/** Size of mip `level` of a `size`-texel dimension. */
export const mipSize = ( size, level ) => Math.max( 1, size >> level );

/** Full mip count for a width × height texture. */
export const mipLevels = ( width, height ) => Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;

/**
 * Read levels `from` … `to − 1` of a 2D texture (layer 0) back as RGBA floats,
 * the way a shader's textureLoad sees them (missing channels read 0, 0, 1).
 * Formats: rgba8unorm, rgba16float, r32float, rg32float, rgba32float, depth32float.
 */
export async function readLevels( device, texture, from = 0, to = texture.mipLevelCount ) {

	const format = texture.format;
	const bytes = format === 'depth32float' ? 4 : formats[ format ].bytesPerTexel;
	const levels = [];
	let size = 0;
	for ( let level = from; level < to; level ++ ) {

		const width = mipSize( texture.width, level ), height = mipSize( texture.height, level );
		const bytesPerRow = Math.ceil( width * bytes / 256 ) * 256;
		levels.push( { level, width, height, bytesPerRow, offset: size } );
		size += bytesPerRow * height;

	}

	const buffer = device.createBuffer( { label: 'example readback', size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	for ( const { level, width, height, bytesPerRow, offset } of levels ) {

		const aspect = format === 'depth32float' ? 'depth-only' : 'all';
		encoder.copyTextureToBuffer( { texture, mipLevel: level, aspect }, { buffer, offset, bytesPerRow, rowsPerImage: height }, [ width, height, 1 ] );

	}

	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const mapped = buffer.getMappedRange();
	const result = levels.map( ( { width, height, bytesPerRow, offset } ) => {

		const values = new Float32Array( width * height * 4 );
		for ( let y = 0; y < height; y ++ ) {

			const row = offset + y * bytesPerRow;
			for ( let x = 0; x < width; x ++ ) {

				const out = ( y * width + x ) * 4;
				values.set( decodeTexel( mapped, row + x * bytes, format ), out );

			}

		}

		return { width, height, values };

	} );
	buffer.unmap();
	buffer.destroy();
	return result;

}

function decodeTexel( buffer, offset, format ) {

	switch ( format ) {

		case 'rgba8unorm': return Array.from( new Uint8Array( buffer, offset, 4 ), value => value / 255 );
		case 'rgba16float': return Array.from( new Uint16Array( buffer, offset, 4 ), fromHalf );
		case 'rgba32float': return new Float32Array( buffer, offset, 4 );
		case 'rg32float': { const [ r, g ] = new Float32Array( buffer, offset, 2 ); return [ r, g, 0, 1 ]; }
		case 'r32float': case 'depth32float': return [ new Float32Array( buffer, offset, 1 )[ 0 ], 0, 0, 1 ];
		default: throw new Error( `readLevels: ${ format } is not handled by the examples` );

	}

}

/**
 * Draw RGBA float values (`width` × `height`) into a canvas. `toColour( values,
 * i )` returns [ r, g, b ] in 0…1 for the texel at value index `i`; default:
 * the texel's own rgb.
 */
export function drawLevel( canvas, { width, height, values }, toColour = ( v, i ) => [ v[ i ], v[ i + 1 ], v[ i + 2 ] ] ) {

	canvas.width = width;
	canvas.height = height;
	const context = canvas.getContext( '2d' );
	const image = context.createImageData( width, height );
	for ( let i = 0; i < width * height; i ++ ) {

		const [ r, g, b ] = toColour( values, i * 4 );
		image.data.set( [ r, g, b ].map( c => Math.round( Math.min( 1, Math.max( 0, c ) ) * 255 ) ), i * 4 );
		image.data[ i * 4 + 3 ] = 255;

	}

	context.putImageData( image, 0, 0 );

}

/** One button per level; `onSelect( level )` runs on click, and once for `initial`. */
export function levelButtons( container, count, onSelect, initial = 0 ) {

	container.replaceChildren();
	const buttons = Array.from( { length: count }, ( _, level ) => {

		const button = document.createElement( 'button' );
		button.textContent = String( level );
		button.addEventListener( 'click', () => select( level ) );
		container.append( button );
		return button;

	} );
	const select = level => {

		buttons.forEach( ( button, i ) => button.classList.toggle( 'selected', i === level ) );
		onSelect( level );

	};

	select( initial );

}

/**
 * A procedural rgba8unorm test image: smooth gradients, fine rings that alias
 * without mipmaps, coloured discs, and one-texel bright lines every 97 texels and
 * on the last row and column (they show how odd sizes handle the edges).
 */
export function makeImage( width, height ) {

	const data = new Uint8Array( width * height * 4 );
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		const u = x / width, v = y / height;
		const dx = u - 0.5, dy = ( v - 0.5 ) * height / width;
		const rings = 0.5 + 0.5 * Math.cos( Math.sqrt( dx * dx + dy * dy ) * width * 0.9 );
		let r = 0.15 + 0.6 * u, g = 0.2 + 0.5 * v, b = 0.35 + 0.4 * rings;
		for ( const [ cx, cy, radius, colour ] of discs ) {

			if ( ( u - cx ) ** 2 + ( ( v - cy ) * height / width ) ** 2 < radius * radius ) [ r, g, b ] = colour;

		}

		if ( x % 97 === 0 || y % 97 === 0 || x === width - 1 || y === height - 1 ) r = g = b = 1;
		data.set( [ r, g, b ].map( c => Math.round( Math.min( 1, Math.max( 0, c ) ) * 255 ) ), ( y * width + x ) * 4 );
		data[ ( y * width + x ) * 4 + 3 ] = 255;

	}

	return data;

}

const discs = [ [ 0.2, 0.3, 0.08, [ 0.95, 0.3, 0.2 ] ], [ 0.75, 0.65, 0.12, [ 0.2, 0.8, 0.35 ] ], [ 0.55, 0.2, 0.05, [ 1, 0.85, 0.1 ] ] ];

/** RGBA floats of an rgba8unorm upload, as a shader reads them. */
export const unormFloats = data => Float32Array.from( data, value => value / 255 );

/** Write `data` into level 0 of a 2D texture. */
export function upload( device, texture, data, bytesPerTexel ) {

	device.queue.writeTexture( { texture }, data, { bytesPerRow: texture.width * bytesPerTexel, rowsPerImage: texture.height }, [ texture.width, texture.height, 1 ] );

}

// --- Readout ---------------------------------------------------------------

let finished = false;

/**
 * Show the checks (`{ name, ok, detail }`) and publish the result for the
 * headless example test (`window.exampleResult`, test/examples.js).
 */
export function report( checks ) {

	const ok = checks.every( check => check.ok );
	const readout = document.getElementById( 'readout' );
	readout.replaceChildren( ...checks.map( ( { name, ok, detail } ) => {

		const row = document.createElement( 'li' );
		row.className = ok ? 'pass' : 'fail';
		row.textContent = `${ ok ? 'PASS' : 'FAIL' } · ${ name }${ detail ? ` — ${ detail }` : '' }`;
		return row;

	} ) );
	readout.dataset.status = ok ? 'pass' : 'fail';
	if ( ! finished ) window.exampleResult = { ok, checks };
	finished = true;
	return ok;

}

/** Show an error and publish a failed result. */
export function fail( error ) {

	console.error( error );
	const readout = document.getElementById( 'readout' );
	if ( readout ) {

		const row = document.createElement( 'li' );
		row.className = 'fail';
		row.textContent = `ERROR · ${ error.message ?? error }`;
		readout.append( row );
		readout.dataset.status = 'fail';

	}

	if ( ! finished ) window.exampleResult = { ok: false, error: String( error.message ?? error ) };
	finished = true;

}

export function setText( id, text ) {

	const element = document.getElementById( id );
	if ( element ) element.textContent = text;

}

/**
 * Compare GPU values with a reference. `allowed( expected, i )` is the
 * tolerance for value `i`. Returns the worst error and how many values failed.
 */
export function compare( actual, expected, allowed ) {

	let maxError = 0, failing = 0;
	for ( let i = 0; i < expected.length; i ++ ) {

		const error = Math.abs( actual[ i ] - expected[ i ] );
		if ( ! ( error <= allowed( expected[ i ], i ) ) ) failing ++;
		maxError = Math.max( maxError, error );

	}

	return { maxError, failing };

}
