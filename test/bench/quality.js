// Quality of `oddSizes: 'spd'` against 'exact' (SPEC.md §2.1, M2T.4), on the GPU.
//
// Per level of a full chain, for averages:
// - max / mean |spd − exact| (stored values, channels r, g, b), and
// - each against the ideal: the area-weighted mean of the base texels under
//   the level texel's normalized footprint [x, x+1) · W / w (fractional at the
//   ends). Neither mode is area-correct at odd sizes; this says which is closer.
// For max chains: Hi-Z conservativeness violations, the level texels smaller
// than some base texel inside their footprint (`maxViolations`), per mode.
// The tail kernel's outputs are compared with 'exact' too (same rule; only
// shader vs sampler bilinear weights differ).
import { maxViolations, toHalf } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, readMips } from '../browser/gpu.js';

export const qualitySizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 3840, 2160 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ] ];

/**
 * Test images, RGBA in [0, 1] (rgba8) or HDR (rgba16float):
 * - 'noise': uniform per-texel noise, the worst case for any filter.
 * - 'scene': smooth gradients and soft waves, thin bright lines every 97 texels
 *   (and one in the last column and row), and small bright highlights.
 */
function image( width, height, pattern, hdr ) {

	const data = new Float32Array( width * height * 4 );
	let state = 1337;
	const random = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		const i = ( y * width + x ) * 4;
		if ( pattern === 'noise' ) {

			for ( let c = 0; c < 4; c ++ ) data[ i + c ] = hdr ? random() * 8 : random();
			continue;

		}

		const u = x / width, v = y / height;
		const line = x % 97 === 0 || y % 97 === 0 || x === width - 1 || y === height - 1;
		const highlight = ( ( x * 7919 + y * 104729 ) % 4099 ) === 0;
		const base = [ 0.2 + 0.6 * u, 0.3 + 0.4 * Math.sin( 9 * u + 5 * v ) ** 2, 0.5 + 0.4 * Math.cos( 13 * v - 3 * u ), 1 ];
		for ( let c = 0; c < 4; c ++ ) data[ i + c ] = c === 3 ? 1 : highlight ? ( hdr ? 16 : 1 ) : line ? ( hdr ? 4 : 0.95 ) : base[ c ] * ( hdr ? 2 : 1 );

	}

	return data;

}

function encode( floats, format ) {

	if ( format === 'rgba8unorm' ) return Uint8Array.from( floats, v => Math.round( Math.min( 1, Math.max( 0, v ) ) * 255 ) );
	return Uint16Array.from( floats, v => toHalf( v ) );

}

/** Area-weighted means of `source` (RGBA) over each texel's footprint at w × h, separably. */
function areaMean( source, width, height, w, h ) {

	const weights = ( n, out ) => Array.from( { length: out }, ( _, x ) => {

		const lo = x * n / out, hi = ( x + 1 ) * n / out, taps = [];
		for ( let s = Math.floor( lo ); s < Math.ceil( hi ); s ++ ) taps.push( [ s, ( Math.min( hi, s + 1 ) - Math.max( lo, s ) ) / ( hi - lo ) ] );
		return taps;

	} );
	const wx = weights( width, w ), wy = weights( height, h );
	const rows = new Float64Array( height * w * 4 );
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < w; x ++ ) for ( const [ s, k ] of wx[ x ] ) for ( let c = 0; c < 4; c ++ ) rows[ ( y * w + x ) * 4 + c ] += k * source[ ( y * width + s ) * 4 + c ];
	const out = new Float64Array( w * h * 4 );
	for ( let y = 0; y < h; y ++ ) for ( const [ s, k ] of wy[ y ] ) for ( let x = 0; x < w; x ++ ) for ( let c = 0; c < 4; c ++ ) out[ ( y * w + x ) * 4 + c ] += k * rows[ ( s * w + x ) * 4 + c ];
	return out;

}

/** max and mean |a − b| over channels r, g, b. */
function difference( a, b ) {

	let max = 0, sum = 0, n = 0;
	for ( let i = 0; i < a.length; i ++ ) {

		if ( i % 4 === 3 ) continue;
		const d = Math.abs( a[ i ] - b[ i ] );
		max = Math.max( max, d ); sum += d; n ++;

	}

	return { max, mean: sum / n };

}

export async function quality( device, { sizes = qualitySizes, formats = [ 'rgba8unorm', 'rgba16float' ], patterns = [ 'scene', 'noise' ], onProgress = () => {} } = {} ) {

	const exact = new Downsampler( device ), tail = new Downsampler( device, { tail: true } );
	const runs = { exact: [ exact, {} ], spd: [ exact, { oddSizes: 'spd' } ], tail: [ tail, {} ] };
	const results = [];

	for ( const format of formats ) for ( const [ width, height ] of sizes ) {

		const texture = createTexture( device, { width, height, format } );

		try {

			for ( const pattern of patterns ) {

				const floats = image( width, height, pattern, format === 'rgba16float' );
				const data = encode( floats, format );
				const base = format === 'rgba8unorm' ? Float32Array.from( data, v => v / 255 ) : floats.map( v => Math.fround( v ) );
				const chains = {};
				for ( const [ name, [ downsampler, options ] ] of Object.entries( runs ) ) {

					device.queue.writeTexture( { texture }, data, { bytesPerRow: width * ( format === 'rgba8unorm' ? 4 : 8 ) }, [ width, height ] );
					downsampler.generate( texture, options );
					chains[ name ] = await readMips( device, texture, 1 );
					downsampler.generate( texture, { ...options, reduction: 'max' } );
					chains[ `${ name }Max` ] = await readMips( device, texture, 1 );

				}

				const levels = chains.exact.map( ( level, i ) => {

					const w = Math.max( 1, width >> ( i + 1 ) ), h = Math.max( 1, height >> ( i + 1 ) );
					const ideal = areaMean( base, width, height, w, h );
					return {
						level: i + 1, width: w, height: h,
						spdVsExact: difference( chains.spd[ i ], level ),
						tailVsExact: difference( chains.tail[ i ], level ),
						exactVsIdeal: difference( level, ideal ),
						spdVsIdeal: difference( chains.spd[ i ], ideal ),
						maxViolations: {
							exact: maxViolations( base, width, height, chains.exactMax[ i ], w, h ),
							spd: maxViolations( base, width, height, chains.spdMax[ i ], w, h ),
							tail: maxViolations( base, width, height, chains.tailMax[ i ], w, h )
						}
					};

				} );

				const result = { width, height, format, pattern, dispatches: Object.fromEntries( Object.entries( runs ).map( ( [ name, [ downsampler, options ] ] ) => [ name, downsampler.prepare( texture, options ).length ] ) ), levels };
				results.push( result );
				onProgress( `quality ${ width }×${ height } ${ format } ${ pattern }` );

			}

		} finally {

			texture.destroy();

		}

	}

	return results;

}
