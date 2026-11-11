/**
 * CPU reference for one downsampling step. This is the correctness oracle for
 * every backend. 'average' is a normalized-coordinate bilinear sample at each
 * output texel centre (an exact 2×2 box for even sizes), clamped at edges. Fold
 * reductions (min, max, minmax, custom) combine every source texel the output
 * covers (see `footprint`). Results are quantized to the storage format.
 */
import { integerDomains, isIntegerDomain, isPackedQuantization, packedStep, quantize, quantizeTexel, roundHalfEven, type Quantization } from './formats.js';
import { foldChannel, footprint, type ReferenceReduction } from './reduction.js';

export interface ReferenceLevel {
	width: number;
	height: number;
	layers: number;
	/** RGBA values, layer-major then row-major. */
	values: Float32Array;
	/**
	 * Extra per-value tolerance for hardware bilinear weights, which may be
	 * quantized to 8 fractional bits when a weight is not exactly representable.
	 */
	filterTolerance: Float32Array;
}

export interface ReferenceOptions {
	/**
	 * Whether `source` already holds values of the destination's encoding (true
	 * for every level of a chain). For normalized formats the average is then
	 * computed in exact integer units, as the kernels do: values read back as
	 * k/255 floats would otherwise land a hair off exact ties and round the wrong
	 * way. False for a separate source of another format.
	 */
	quantizedInput?: boolean;
	/**
	 * 'float32': compute an exact 2:1 average (each dimension even or 1) as the
	 * kernels do in workgroup memory, `(((a + b) + c) + d) · 0.25` with float32
	 * rounding after every add, so the result can be required to match bit for
	 * bit (packed float formats, whose sums aren't always exact in f32). Other
	 * sizes and integer formats ignore it. Default: exact (f64) arithmetic.
	 */
	pairing?: 'float32';
}

export function reduceReference( source: Float32Array, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction = 'average', { quantizedInput = true, pairing }: ReferenceOptions = {} ): ReferenceLevel {

	const w = Math.max( 1, Math.floor( width / 2 ) ), h = Math.max( 1, Math.floor( height / 2 ) );
	const values = new Float32Array( w * h * layers * 4 );
	const filterTolerance = new Float32Array( values.length );
	// Packed formats quantize whole texels once all four values are known.
	const packed = isPackedQuantization( quantization );
	const perValue = ( value: number ) => packed ? value : quantize( value, quantization );
	const finish = () => {

		if ( packed ) for ( let i = 0; i < values.length; i += 4 ) quantizeTexel( values, i, quantization );
		return { width: w, height: h, layers, values, filterTolerance };

	};

	if ( reduction !== 'average' ) {

		const fold = typeof reduction === 'function' ? reduction : ( a: number, b: number, c: number ) => foldChannel( reduction, a, b, c );
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const [ y0, y1 ] = footprint( y, height ), [ x0, x1 ] = footprint( x, width );
			for ( let c = 0; c < 4; c ++ ) {

				let value = source[ ( ( layer * height + y0 ) * width + x0 ) * 4 + c ];
				for ( let sy = y0; sy <= y1; sy ++ ) for ( let sx = x0; sx <= x1; sx ++ ) value = fold( value, source[ ( ( layer * height + sy ) * width + sx ) * 4 + c ], c );
				values[ ( ( layer * h + y ) * w + x ) * 4 + c ] = perValue( value );

			}

		}

		return finish();

	}

	const domainOf = isIntegerDomain( quantization ) ? integerDomains[ quantization ] : null;
	const halves = ( n: number ) => n === 1 || n % 2 === 0;
	if ( pairing === 'float32' && domainOf === null && halves( width ) && halves( height ) ) {

		// The kernels' in-tile reduction: clamped 2×2 reads, summed left to right in f32.
		const f = Math.fround;
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const y0 = Math.min( 2 * y, height - 1 ), y1 = Math.min( 2 * y + 1, height - 1 );
			const x0 = Math.min( 2 * x, width - 1 ), x1 = Math.min( 2 * x + 1, width - 1 );
			const at = ( sx: number, sy: number, c: number ) => source[ ( ( layer * height + sy ) * width + sx ) * 4 + c ];
			for ( let c = 0; c < 4; c ++ ) values[ ( ( layer * h + y ) * w + x ) * 4 + c ] = perValue( f( f( f( at( x0, y0, c ) + at( x1, y0, c ) ) + at( x0, y1, c ) ) + at( x1, y1, c ) ) * 0.25 );

		}

		return finish();

	}

	const clamp = ( x: number, size: number ) => Math.min( size - 1, Math.max( 0, x ) );
	const exact = ( weight: number ) => weight * 256 === Math.round( weight * 256 );
	const domain = isIntegerDomain( quantization ) && quantizedInput ? integerDomains[ quantization ] : null;
	const scale = domain ? domain.scale : 1;
	const units = domain ? source.map( value => Math.round( value * scale ) ) : source;

	for ( let layer = 0; layer < layers; layer ++ ) {

		for ( let y = 0; y < h; y ++ ) {

			const sy = ( y + 0.5 ) * height / h - 0.5;
			const y0 = Math.floor( sy ), fy = sy - y0;

			const r0 = ( layer * height + clamp( y0, height ) ) * width, r1 = ( layer * height + clamp( y0 + 1, height ) ) * width;
			const exactY = exact( fy );

			for ( let x = 0; x < w; x ++ ) {

				const sx = ( x + 0.5 ) * width / w - 0.5;
				const x0 = Math.floor( sx ), fx = sx - x0;
				const c0 = clamp( x0, width ), c1 = clamp( x0 + 1, width ), exactX = exact( fx );
				const i00 = ( r0 + c0 ) * 4, i10 = ( r0 + c1 ) * 4, i01 = ( r1 + c0 ) * 4, i11 = ( r1 + c1 ) * 4;

				for ( let c = 0; c < 4; c ++ ) {

					const a = units[ i00 + c ], b = units[ i10 + c ], d = units[ i01 + c ], e = units[ i11 + c ];
					const top = a * ( 1 - fx ) + b * fx;
					const bottom = d * ( 1 - fx ) + e * fx;
					const index = ( ( layer * h + y ) * w + x ) * 4 + c;
					const gx = Math.max( Math.abs( b - a ), Math.abs( e - d ) );
					const gy = Math.max( Math.abs( d - a ), Math.abs( e - b ) );
					filterTolerance[ index ] = ( ( exactX ? 0 : gx / 256 ) + ( exactY ? 0 : gy / 256 ) ) / scale;
					const value = top * ( 1 - fy ) + bottom * fy;
					values[ index ] = domain ? roundHalfEven( Math.min( domain.max, Math.max( domain.min, value ) ) ) / scale : perValue( value );

				}

			}

		}

	}

	return finish();

}

/**
 * Reference for a same-size converting copy (a separate source whose level is
 * the destination's first level): each value (packed formats: each texel) quantized to the destination.
 */
export function copyReference( source: Float32Array, width: number, height: number, layers: number, quantization: Quantization ): ReferenceLevel {

	const values = Float32Array.from( source );
	for ( let i = 0; i < values.length; i += 4 ) quantizeTexel( values, i, quantization );
	return { width, height, layers, values, filterTolerance: new Float32Array( source.length ) };

}

export interface ToleranceTexel {
	/** Channel of the value (rg11b10ufloat: 2 is the 10-bit one). */
	channel: number;
	/** Largest of the texel's r, g, b (rgb9e5ufloat: sets the shared exponent). */
	texelMax: number;
}

/**
 * Declared comparison tolerance for one value: about two steps for normalized formats,
 * or max(0.002, 0.3 %) for float16, or 1e-5 relative for float32, or two steps of
 * the stored value plus 2^-13 for the packed float formats (rgb9e5ufloat: steps of the texel's
 * shared exponent, so `texel` is required), plus the reference's filter allowance. Fold
 * reductions select existing values, so they must match exactly. These are test
 * tolerances for the fixtures in this repository, not a formal error bound.
 */
export function tolerance( expected: number, quantization: Quantization, filterTolerance = 0, reduction: ReferenceReduction = 'average', texel?: ToleranceTexel ): number {

	if ( reduction !== 'average' ) return 0;
	if ( isPackedQuantization( quantization ) ) {

		if ( texel === undefined ) throw new Error( `downsampler: ${ quantization } tolerances need the texel (channel and largest value).` );
		// Plus 2^-13 absolute for filtering arithmetic near 0, where a sampled HDR source can cancel to a
		// few subnormal steps (SwiftShader: 1.4e-5 for an rgba16float source whose exact average is 0).
		return 2 * packedStep( quantization, expected, texel.channel, texel.texelMax ) + 2 ** - 13 + filterTolerance;

	}
	if ( isIntegerDomain( quantization ) ) return 2.01 / integerDomains[ quantization ].scale + filterTolerance;
	// float32 kernels compute in f32 and the reference in f64: a few ulps.
	if ( quantization === 'float32' ) return Math.max( 1e-6, Math.abs( expected ) * 1e-5 ) + filterTolerance;
	return Math.max( 0.002, Math.abs( expected ) * 0.003 ) + filterTolerance;

}
