/**
 * CPU reference for one downsampling step. This is the correctness oracle for
 * every backend. 'average' is a normalized-coordinate bilinear sample at each
 * output texel centre (an exact 2×2 box for even sizes), clamped at edges. Fold
 * reductions (min, max, minmax, custom) combine every source texel the output
 * covers (see `footprint`). Results are quantized to the storage format.
 */
import { quantize, type Quantization } from './formats.js';
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

export function reduceReference( source: Float32Array, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction = 'average' ): ReferenceLevel {

	const w = Math.max( 1, Math.floor( width / 2 ) ), h = Math.max( 1, Math.floor( height / 2 ) );
	const values = new Float32Array( w * h * layers * 4 );
	const filterTolerance = new Float32Array( values.length );
	if ( reduction !== 'average' ) {

		const fold = typeof reduction === 'function' ? reduction : ( a: number, b: number, c: number ) => foldChannel( reduction, a, b, c );
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const [ y0, y1 ] = footprint( y, height ), [ x0, x1 ] = footprint( x, width );
			for ( let c = 0; c < 4; c ++ ) {

				let value = source[ ( ( layer * height + y0 ) * width + x0 ) * 4 + c ];
				for ( let sy = y0; sy <= y1; sy ++ ) for ( let sx = x0; sx <= x1; sx ++ ) value = fold( value, source[ ( ( layer * height + sy ) * width + sx ) * 4 + c ], c );
				values[ ( ( layer * h + y ) * w + x ) * 4 + c ] = quantize( value, quantization );

			}

		}

		return { width: w, height: h, layers, values, filterTolerance };

	}

	const clamp = ( x: number, size: number ) => Math.min( size - 1, Math.max( 0, x ) );
	const exact = ( weight: number ) => weight * 256 === Math.round( weight * 256 );

	for ( let layer = 0; layer < layers; layer ++ ) {

		for ( let y = 0; y < h; y ++ ) {

			const sy = ( y + 0.5 ) * height / h - 0.5;
			const y0 = Math.floor( sy ), fy = sy - y0;

			for ( let x = 0; x < w; x ++ ) {

				const sx = ( x + 0.5 ) * width / w - 0.5;
				const x0 = Math.floor( sx ), fx = sx - x0;

				for ( let c = 0; c < 4; c ++ ) {

					const get = ( dx: number, dy: number ) => source[ ( ( layer * height + clamp( y0 + dy, height ) ) * width + clamp( x0 + dx, width ) ) * 4 + c ];
					const top = get( 0, 0 ) * ( 1 - fx ) + get( 1, 0 ) * fx;
					const bottom = get( 0, 1 ) * ( 1 - fx ) + get( 1, 1 ) * fx;
					const index = ( ( layer * h + y ) * w + x ) * 4 + c;
					const gx = Math.max( Math.abs( get( 1, 0 ) - get( 0, 0 ) ), Math.abs( get( 1, 1 ) - get( 0, 1 ) ) );
					const gy = Math.max( Math.abs( get( 0, 1 ) - get( 0, 0 ) ), Math.abs( get( 1, 1 ) - get( 1, 0 ) ) );
					filterTolerance[ index ] = ( exact( fx ) ? 0 : gx / 256 ) + ( exact( fy ) ? 0 : gy / 256 );
					values[ index ] = quantize( top * ( 1 - fy ) + bottom * fy, quantization );

				}

			}

		}

	}

	return { width: w, height: h, layers, values, filterTolerance };

}

/**
 * Declared comparison tolerance for one value: about two 8-bit steps for unorm8,
 * or max(0.002, 0.3 %) for float16, plus the reference's filter allowance. Fold
 * reductions select existing values, so they must match exactly. These are test
 * tolerances for the fixtures in this repository, not a formal error bound.
 */
export function tolerance( expected: number, quantization: Quantization, filterTolerance = 0, reduction: ReferenceReduction = 'average' ): number {

	if ( reduction !== 'average' ) return 0;
	return ( quantization === 'unorm8' ? 2.01 / 255 : Math.max( 0.002, Math.abs( expected ) * 0.003 ) ) + filterTolerance;

}
