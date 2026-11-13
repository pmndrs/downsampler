/**
 * Integer formats (r32uint, rgba8sint, rgb10a2uint, …; SPEC.md §6.1 Integer formats):
 * the CPU oracle of `Downsampler` chains on them. Values are exact integers (a
 * Float64Array holds every u32 and i32), never normalized.
 *
 * - Folds ('min', 'max', 'minmax', custom) combine every source texel the output
 *   covers, as for every other format: `footprint()` at 2:1, the area span at any
 *   other ratio (`source.resample: 'area'`). Signed formats compare as signed.
 * - 'average' is the exact weighted mean, rounded to the nearest integer with ties
 *   to even. With `weights: 'bilinear'` (`oddSizes: 'exact'`) the weights are the
 *   render-pass bilinear weights at the output texel centre, as rationals: per axis
 *   the sample position ((2x + 1)·S − n) / 2n splits into a base texel and a
 *   remainder r, and the two taps weigh (2n − r) / 2n and r / 2n. With `weights:
 *   'area'` (`oddSizes: 'box'`, resampled sources) each source texel weighs its
 *   overlap / S (see `areaReference`). A 2:1 step of an even size is the 2×2 mean
 *   under both. Nothing is rounded but the result, so the GPU must match bit for
 *   bit; the kernels do it in u32 arithmetic (`createIntKernel`).
 * - Every result is clamped to the destination's range, which only matters for a
 *   separate source of a wider integer format.
 */
import { intDomains, intRange, type IntQuantization } from './formats.js';
import { areaOverlap, areaSpan } from './box.js';
import { footprint, foldChannel, type BuiltinReduction } from './reduction.js';

/** A reduction of integer values: a builtin, or a JS fold `( a, b, channel ) => value` on the integers the WGSL `reduce` sees. */
export type IntReduction = BuiltinReduction | ( ( a: number, b: number, channel: number ) => number );

export interface IntReferenceOptions {
	/** 'bilinear' (default; `oddSizes: 'exact'`): render-pass weights, 2:1 or same size only. 'area': overlap weights at any size. */
	weights?: 'bilinear' | 'area';
	/** Output size; default the next mip level's (⌊n/2⌋, at least 1). */
	width?: number;
	height?: number;
	/** The source's integer format, when it isn't the destination's (a separate source): only its signedness must agree. */
	source?: IntQuantization;
}

export interface IntLevel {
	width: number;
	height: number;
	layers: number;
	/** RGBA integers, layer-major then row-major. */
	values: Float64Array;
}

/** ⌊n / d⌋ and n mod d for non-negative integers below 2^53 (the float quotient corrected). */
function divmod( n: number, d: number ): [ number, number ] {

	let q = Math.floor( n / d );
	while ( q * d > n ) q --;
	while ( ( q + 1 ) * d <= n ) q ++;
	return [ q, n - q * d ];

}

/**
 * Σ weights[i] · values[i] / denominator, rounded to nearest with ties to even, exactly.
 * Values are non-negative integers below 2^32 and weights non-negative integers below 2^31
 * summing to `denominator`. Each value splits into 16-bit halves, so every partial sum
 * stays below 2^53.
 */
export function roundedMean( values: readonly number[], weights: readonly number[], denominator: number ): number {

	let high = 0, low = 0;
	values.forEach( ( value, i ) => {

		high += weights[ i ] * Math.floor( value / 65536 );
		low += weights[ i ] * ( value % 65536 );

	} );
	// (high · 2^16 + low) / d = ⌊high / d⌋ · 2^16 + ((high mod d) · 2^16 + low) / d.
	const [ qh, rh ] = divmod( high, denominator );
	const [ ql, rl ] = divmod( rh * 65536 + low, denominator );
	const floor = qh * 65536 + ql;
	const twice = 2 * rl;
	return twice > denominator || ( twice === denominator && floor % 2 === 1 ) ? floor + 1 : floor;

}

/** One axis of an average: source texels and their integer weights, over `denominator`. */
interface Taps { texels: number[]; weights: number[]; denominator: number }

/** Render-pass bilinear taps of output texel x of n from a level of S (S = n: the texel itself). */
function bilinearTaps( x: number, sourceSize: number, size: number ): Taps {

	const numerator = ( 2 * x + 1 ) * sourceSize - size, denominator = 2 * size;
	const last = sourceSize - 1;
	const lo = Math.min( Math.floor( numerator / denominator ), last ), r = numerator % denominator;
	return { texels: [ lo, Math.min( lo + 1, last ) ], weights: [ denominator - r, r ], denominator };

}

function areaTaps( x: number, sourceSize: number, size: number ): Taps {

	const [ first, last ] = areaSpan( x, sourceSize, size );
	const texels: number[] = [], weights: number[] = [];
	for ( let t = first; t <= last; t ++ ) {

		texels.push( t );
		weights.push( areaOverlap( t, x, sourceSize, size ) );

	}

	return { texels, weights, denominator: sourceSize };

}

/** Fold texels along one axis: the footprint at 2:1 (bilinear rule), the area span otherwise. */
function foldSpan( x: number, sourceSize: number, size: number, area: boolean ): [ number, number ] {

	if ( sourceSize === size ) return [ x, x ];
	return area ? areaSpan( x, sourceSize, size ) : footprint( x, sourceSize );

}

/**
 * One step of an integer chain on the CPU: the level of `width`×`height` (RGBA
 * integers, layer-major then row-major; signed formats as signed values) reduced to
 * the next level, or to `options.width`×`options.height`, of `quantization`.
 */
export function intReference( source: ArrayLike<number>, width: number, height: number, layers: number, quantization: IntQuantization, reduction: IntReduction = 'average', options: IntReferenceOptions = {} ): IntLevel {

	const { weights = 'bilinear' } = options;
	const w = options.width ?? Math.max( 1, width >> 1 ), h = options.height ?? Math.max( 1, height >> 1 );
	if ( ! Number.isInteger( w ) || ! Number.isInteger( h ) || w < 1 || h < 1 ) throw new Error( `downsampler: integer reference size ${ w }×${ h } is not a texture size.` );
	if ( weights !== 'bilinear' && weights !== 'area' ) throw new Error( `downsampler: integer reference weights must be 'bilinear' or 'area', got ${ JSON.stringify( weights ) }.` );
	const area = weights === 'area';
	const step = ( s: number, n: number ) => n === s || n === Math.max( 1, s >> 1 );
	if ( ! area && ( ! step( width, w ) || ! step( height, h ) ) ) throw new Error( `downsampler: bilinear integer steps are 2:1 or same-size; ${ width }×${ height } → ${ w }×${ h } needs weights 'area'.` );
	const { signed } = intDomains[ quantization ];
	if ( options.source !== undefined && intDomains[ options.source ].signed !== signed ) throw new Error( `downsampler: a ${ options.source } source can't feed ${ quantization } (signed and unsigned integers don't mix).` );
	const ranges = [ 0, 1, 2, 3 ].map( c => intRange( quantization, c ) );
	// Averages work on non-negative values: signed ones are offset by 2^31, which is even, so ties keep their parity.
	const bias = signed ? 2 ** 31 : 0;
	const values = new Float64Array( w * h * layers * 4 );
	const at = ( layer: number, sx: number, sy: number, c: number ) => source[ ( ( layer * height + sy ) * width + sx ) * 4 + c ];

	const columns = Array.from( { length: w }, ( _, x ) => reduction === 'average' ? ( area ? areaTaps : bilinearTaps )( x, width, w ) : foldSpan( x, width, w, area ) );
	const rows = Array.from( { length: h }, ( _, y ) => reduction === 'average' ? ( area ? areaTaps : bilinearTaps )( y, height, h ) : foldSpan( y, height, h, area ) );
	const fold = reduction === 'average' ? null : typeof reduction === 'function' ? reduction : ( a: number, b: number, c: number ) => foldChannel( reduction, a, b, c );

	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) for ( let c = 0; c < 4; c ++ ) {

		let value: number;
		if ( fold !== null ) {

			const [ x0, x1 ] = columns[ x ] as [ number, number ], [ y0, y1 ] = rows[ y ] as [ number, number ];
			value = at( layer, x0, y0, c );
			for ( let sy = y0; sy <= y1; sy ++ ) for ( let sx = x0; sx <= x1; sx ++ ) value = fold( value, at( layer, sx, sy, c ), c );

		} else {

			const cx = columns[ x ] as Taps, cy = rows[ y ] as Taps;
			const taps: number[] = [], products: number[] = [];
			cy.texels.forEach( ( sy, j ) => cx.texels.forEach( ( sx, i ) => {

				taps.push( at( layer, sx, sy, c ) + bias );
				products.push( cx.weights[ i ] * cy.weights[ j ] );

			} ) );
			value = roundedMean( taps, products, cx.denominator * cy.denominator ) - bias;

		}

		const [ min, max ] = ranges[ c ];
		values[ ( ( layer * h + y ) * w + x ) * 4 + c ] = Math.min( max, Math.max( min, value ) );

	}

	return { width: w, height: h, layers, values };

}
