/**
 * Area-weighted reduction: `oddSizes: 'box'` (SPEC.md §2.1) and a separate source
 * resampled into a destination of another size (`source.resample: 'area'`, §4
 * Separate source). The CPU oracle for both.
 *
 * Output texel x of a level n texels wide covers the source interval
 * [x·S/n, (x+1)·S/n) of a level S texels wide. Each source texel t it overlaps
 * contributes in proportion to that overlap:
 *
 *   overlap(t) = min(n·(t+1), (x+1)·S) − max(n·t, x·S)     (in units of 1/n texel)
 *   weight(t)  = overlap(t) / S                             (the weights sum to 1)
 *
 * Per axis, separably in 2D. For a 2:1 step this is nvpro_pyramid's kernel
 * (NVIDIA's 2005 non-power-of-two mipmap report): an even size takes 2 taps of
 * 1/2, a size of 1 one tap, and an odd size 2n+1 three taps 2x … 2x+2 with weights
 * (n − x, n, x + 1) / (2n + 1). Other ratios take ⌈S/n⌉ + 1 taps at most.
 *
 * Folds (min, max, minmax, custom) combine every texel with a positive overlap:
 * the smallest conservative footprint at any ratio. For a 2:1 step it is
 * `footprint()`, so under `oddSizes: 'box'` folds are the 'exact' folds.
 */
import { integerDomains, isIntegerDomain, isPackedQuantization, quantize, quantizeTexel, roundHalfEven, type Quantization } from './formats.js';
import { pyramidTolerance } from './pyramid.js';
import { foldChannel, type ReferenceReduction } from './reduction.js';
import { reduceReference, type ReferenceLevel, type ReferenceOptions } from './reference.js';

/** First and last source texel (inclusive) that output texel `x` of `size` overlaps, from a level of `sourceSize`. */
export function areaSpan( x: number, sourceSize: number, size: number ): [ number, number ] {

	return [ Math.floor( x * sourceSize / size ), Math.floor( ( ( x + 1 ) * sourceSize - 1 ) / size ) ];

}

/** The overlap of source texel `t` with output texel `x`, in units of 1/size texel: an integer; the weight is overlap / sourceSize. */
export function areaOverlap( t: number, x: number, sourceSize: number, size: number ): number {

	return Math.max( 0, Math.min( size * ( t + 1 ), ( x + 1 ) * sourceSize ) - Math.max( size * t, x * sourceSize ) );

}

/** Per-axis taps of output texel `x`: source texels and their integer overlaps (summing to `sourceSize`). */
function taps( x: number, sourceSize: number, size: number ): { texels: number[]; overlaps: number[] } {

	const [ first, last ] = areaSpan( x, sourceSize, size );
	const texels: number[] = [], overlaps: number[] = [];
	for ( let t = first; t <= last; t ++ ) {

		texels.push( t );
		overlaps.push( areaOverlap( t, x, sourceSize, size ) );

	}

	return { texels, overlaps };

}

/** A 2:1 step along one axis: the destination is the floored half (at least 1) of the source. */
const halves = ( source: number, size: number ) => size === Math.max( 1, Math.floor( source / 2 ) );
/** A step whose 'exact' rule is already area-weighted: even or unit sizes halving. */
const exactBox = ( source: number, size: number ) => halves( source, size ) && ( source === 1 || source % 2 === 0 );

/**
 * Reference for an area-weighted step from a `width`×`height` level (RGBA,
 * layer-major then row-major) to `outWidth`×`outHeight`, quantized to the storage
 * format. `options` as for {@link reduceReference}: `quantizedInput` (normalized
 * formats: average exact integer units, rounded half to even from the exact
 * rational value), and `weights`/`pairing`, which apply where the step is the
 * 'exact' rule's own 2×2 box (even or unit sizes halving, sampled or reduced in
 * workgroup memory) and the reference delegates to `reduceReference`. Folds at
 * 2:1 delegate too (the same footprint). `filterTolerance` of other averages is
 * the f32 error allowance of shader weights ({@link pyramidTolerance} of Σ|w·v|),
 * 0 for normalized formats, whose declared tolerance already covers it.
 */
export function areaReference( source: ArrayLike<number>, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction, outWidth: number, outHeight: number, options: ReferenceOptions = {} ): ReferenceLevel {

	if ( ! Number.isInteger( outWidth ) || ! Number.isInteger( outHeight ) || outWidth < 1 || outHeight < 1 ) throw new Error( `downsampler: area reference size ${ outWidth }×${ outHeight } is not a texture size.` );
	const twoToOne = halves( width, outWidth ) && halves( height, outHeight );
	if ( twoToOne && ( reduction !== 'average' || ( exactBox( width, outWidth ) && exactBox( height, outHeight ) ) ) ) return reduceReference( Float32Array.from( source ), width, height, layers, quantization, reduction, options );

	const { quantizedInput = true } = options;
	const w = outWidth, h = outHeight;
	const values = new Float32Array( w * h * layers * 4 );
	const filterTolerance = new Float32Array( values.length );
	const packed = isPackedQuantization( quantization );
	const columns = Array.from( { length: w }, ( _, x ) => taps( x, width, w ) );
	const rows = Array.from( { length: h }, ( _, y ) => taps( y, height, h ) );
	const at = ( layer: number, sx: number, sy: number, c: number ) => source[ ( ( layer * height + sy ) * width + sx ) * 4 + c ];

	if ( reduction !== 'average' ) {

		const fold = typeof reduction === 'function' ? reduction : ( a: number, b: number, c: number ) => foldChannel( reduction, a, b, c );
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) for ( let c = 0; c < 4; c ++ ) {

			const ys = rows[ y ].texels, xs = columns[ x ].texels;
			let value = at( layer, xs[ 0 ], ys[ 0 ], c );
			for ( const sy of ys ) for ( const sx of xs ) value = fold( value, at( layer, sx, sy, c ), c );
			values[ ( ( layer * h + y ) * w + x ) * 4 + c ] = packed ? value : quantize( value, quantization );

		}

	} else {

		const domain = isIntegerDomain( quantization ) && quantizedInput ? integerDomains[ quantization ] : null;
		const denominator = width * height;
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) for ( let c = 0; c < 4; c ++ ) {

			const { texels: ys, overlaps: oy } = rows[ y ], { texels: xs, overlaps: ox } = columns[ x ];
			const index = ( ( layer * h + y ) * w + x ) * 4 + c;
			if ( domain ) {

				// Exact: Σ overlap_x · overlap_y · units is an integer below 2^53, over width · height.
				let numerator = 0;
				ys.forEach( ( sy, j ) => xs.forEach( ( sx, i ) => { numerator += ox[ i ] * oy[ j ] * Math.round( at( layer, sx, sy, c ) * domain.scale ); } ) );
				values[ index ] = Math.min( domain.max, Math.max( domain.min, roundRational( numerator, denominator ) ) ) / domain.scale;
				continue;

			}

			let sum = 0, magnitude = 0;
			ys.forEach( ( sy, j ) => xs.forEach( ( sx, i ) => {

				const term = ox[ i ] / width * ( oy[ j ] / height ) * at( layer, sx, sy, c );
				sum += term;
				magnitude += Math.abs( term );

			} ) );
			values[ index ] = packed ? sum : quantize( sum, quantization );
			filterTolerance[ index ] = isIntegerDomain( quantization ) ? 0 : pyramidTolerance( magnitude );

		}

	}

	if ( packed ) for ( let i = 0; i < values.length; i += 4 ) quantizeTexel( values, i, quantization );
	return { width: w, height: h, layers, values, filterTolerance };

}

/** numerator / denominator (integers, denominator > 0) rounded half to even, exactly. */
function roundRational( numerator: number, denominator: number ): number {

	let q = Math.floor( numerator / denominator );
	while ( q * denominator > numerator ) q --;
	while ( ( q + 1 ) * denominator <= numerator ) q ++;
	const twice = 2 * ( numerator - q * denominator );
	if ( twice > denominator ) return q + 1;
	if ( twice < denominator ) return q;
	return roundHalfEven( q + 0.5 );

}

/**
 * Reference for one `oddSizes: 'box'` step: {@link areaReference} to the next
 * mip level's size (⌊n/2⌋, at least 1).
 */
export function boxReference( source: ArrayLike<number>, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction = 'average', options: ReferenceOptions = {} ): ReferenceLevel {

	return areaReference( source, width, height, layers, quantization, reduction, Math.max( 1, width >> 1 ), Math.max( 1, height >> 1 ), options );

}

/**
 * Every level of an `oddSizes: 'box'` chain of `levels` output levels, each from
 * the previous quantized level. Level i of the result is mip i + 1.
 */
export function boxChainReference( source: ArrayLike<number>, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction, levels: number, options: ReferenceOptions = {} ): ReferenceLevel[] {

	const chain: ReferenceLevel[] = [];
	let input: ArrayLike<number> = source, w = width, h = height;
	for ( let i = 0; i < levels; i ++ ) {

		const level = boxReference( input, w, h, layers, quantization, reduction, i === 0 ? options : { weights: options.weights } );
		chain.push( level );
		input = level.values; w = level.width; h = level.height;

	}

	return chain;

}
