/**
 * sRGB 8-bit colour (rgba8unorm-srgb, bgra8unorm-srgb; SPEC.md §6.1 sRGB).
 *
 * Levels are reduced in linear light and stored sRGB-encoded. A read of an sRGB
 * texture view decodes each 8-bit code to linear (hardware), so filtering and
 * averaging see linear values; storing encodes them back to 8-bit codes (ours: WebGPU
 * has no sRGB storage textures). Alpha is linear in these formats: plain unorm8.
 *
 * - Decode (IEC 61966-2-1): `c ≤ 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4`, c = code / 255.
 *   Exact per code, so it is a 256-entry table ({@link srgbDecodeTable}, each the f32 nearest
 *   the exact value, as a GPU's decode returns it).
 * - Encode: round to nearest. The code of a linear value v is the k whose rounding
 *   interval holds it: v lies between the thresholds `t(k − 1)` and `t(k)`, with
 *   `t(k) = decode((k + ½) / 255)` ({@link srgbThresholds}), i.e. `round(255 · decode⁻¹(v))`.
 *   Ties can't occur: no threshold is a binary fraction (they are `(k + ½) / (255 · 12.92)`
 *   for k ≤ 9, and 2.4th powers of non-dyadic rationals above), so no f32 or f64 value
 *   lies exactly on one. Values below 0 (and NaN) encode to 0, above 1 to 255.
 * - Quantize as stored: `decode(encode(v))`, a table value. Encoding a table value
 *   returns its code, so min/max of decoded values select stored codes exactly.
 *
 * Kernels encode with the same thresholds, as the smallest f32 above each
 * ({@link srgbBounds}): for an f32 v, v > t(k) ⇔ v ≥ bound(k), so the shader's
 * encoder is correctly rounded for its f32 input (`ds_srgb_code`).
 */

/** The exact sRGB decode of an encoded value in [0, 1]. */
export function srgbDecode( encoded: number ): number {

	return encoded <= 0.04045 ? encoded / 12.92 : ( ( encoded + 0.055 ) / 1.055 ) ** 2.4;

}

/** The exact sRGB encode of a linear value in [0, 1] (the inverse of {@link srgbDecode}), not rounded. */
export function srgbEncode( linear: number ): number {

	return linear <= 0.04045 / 12.92 ? linear * 12.92 : 1.055 * linear ** ( 1 / 2.4 ) - 0.055;

}

/** Linear value of each 8-bit code: the f32 nearest the exact decode. */
export const srgbDecodeTable = Float32Array.from( { length: 256 }, ( _, code ) => srgbDecode( code / 255 ) );

/** Rounding thresholds: `t(k) = decode((k + ½) / 255)`, k = 0…254; values above t(k) encode to k + 1 or more. */
export const srgbThresholds = Float64Array.from( { length: 255 }, ( _, k ) => srgbDecode( ( k + 0.5 ) / 255 ) );

/** The smallest f32 above each threshold (k = 0…254): for an f32 v, v > t(k) exactly when v ≥ bound(k). */
export const srgbBounds = Float32Array.from( srgbThresholds, nextFloat32Above );

/** The smallest float32 strictly above `value` (a positive, finite f64). */
function nextFloat32Above( value: number ): number {

	const rounded = Math.fround( value );
	if ( rounded > value ) return rounded;
	const bits = new Uint32Array( new Float32Array( [ rounded ] ).buffer );
	bits[ 0 ] ++;
	return new Float32Array( bits.buffer )[ 0 ];

}

/** The 8-bit sRGB code of a linear value, rounded to nearest (see the module comment). NaN → 0. */
export function srgbCode( linear: number ): number {

	if ( ! ( linear > srgbThresholds[ 0 ] ) ) return 0;
	// The number of thresholds below `linear`.
	let lo = 0, hi = 255;
	while ( hi - lo > 1 ) {

		const mid = ( lo + hi ) >> 1;
		if ( linear > srgbThresholds[ mid ] ) lo = mid; else hi = mid;

	}

	return hi;

}

/** A linear value quantized as an sRGB texture stores it: `decode(encode(v))`. */
export function quantizeSrgb( linear: number ): number {

	return srgbDecodeTable[ srgbCode( linear ) ];

}

/**
 * The linear distance from `linear` to the value `steps` codes away from its code, the
 * larger of both directions (clamped to the code range). Used for test tolerances.
 */
export function srgbSteps( linear: number, steps: number ): number {

	const code = srgbCode( linear );
	return Math.max( srgbDecodeTable[ Math.min( 255, code + steps ) ] - linear, linear - srgbDecodeTable[ Math.max( 0, code - steps ) ] );

}

/** A reference level: RGBA values and per-value filter allowances (see `ReferenceLevel`). */
interface Level {
	width: number;
	height: number;
	layers: number;
	values: Float32Array;
	filterTolerance: Float32Array;
}

/**
 * An sRGB reference level from one of the reference functions (`reduceReference`,
 * `areaReference`, …), passed as `step( quantization, quantizedInput )`: colour is
 * reduced in linear light as float32 values (the kernels' f32 working values) and
 * then quantized as stored, alpha as unorm8 in integer units (ties to even).
 * `quantizedInput` is the caller's: whether alpha already holds unorm8 values.
 */
export function srgbLevel<T extends Level>( step: ( quantization: 'float32' | 'unorm8', quantizedInput: boolean ) => T, quantizedInput: boolean ): T {

	const colour = step( 'float32', false ), alpha = step( 'unorm8', quantizedInput );
	const { values, filterTolerance } = colour;
	for ( let i = 0; i < values.length; i += 4 ) {

		for ( let c = 0; c < 3; c ++ ) values[ i + c ] = quantizeSrgb( values[ i + c ] );
		values[ i + 3 ] = alpha.values[ i + 3 ];
		filterTolerance[ i + 3 ] = alpha.filterTolerance[ i + 3 ];

	}

	return colour;

}
