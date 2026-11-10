/**
 * Formats and how intermediate levels are quantized so that every level matches
 * what would be read back from the stored texture.
 */

/** How values are rounded between levels. */
export type Quantization = 'unorm8' | 'float16';

export interface FormatInfo {
	quantization: Quantization;
	/** Bytes per texel, for readback and planning. */
	bytesPerTexel: number;
	channels: 4;
	/** Whether a filtering sampler can read it without optional features. */
	filterable: boolean;
}

/** Formats supported by the current kernels. See SPEC.md for the planned matrix. */
export const formats = {
	rgba8unorm: { quantization: 'unorm8', bytesPerTexel: 4, channels: 4, filterable: true },
	rgba16float: { quantization: 'float16', bytesPerTexel: 8, channels: 4, filterable: true }
} as const satisfies Record<string, FormatInfo>;

export type SupportedFormat = keyof typeof formats;

export function isSupportedFormat( format: string ): format is SupportedFormat {

	return Object.prototype.hasOwnProperty.call( formats, format );

}

/** Round half to even, as used for exact 8-bit ties. */
export function roundHalfEven( value: number ): number {

	const floor = Math.floor( value );
	const fraction = value - floor;
	if ( fraction > 0.5 ) return floor + 1;
	if ( fraction < 0.5 ) return floor;
	return floor % 2 === 0 ? floor : floor + 1;

}

const float32 = new Float32Array( 1 );
const uint32 = new Uint32Array( float32.buffer );

/** Float to IEEE half bits, round to nearest even (matches WGSL pack2x16float on tested GPUs). */
export function toHalf( value: number ): number {

	float32[ 0 ] = value;
	const x = uint32[ 0 ];
	const sign = ( x >>> 16 ) & 0x8000;
	const exponent = ( x >>> 23 ) & 0xff;
	let mantissa = x & 0x7fffff;

	if ( exponent === 0xff ) return sign | 0x7c00 | ( mantissa ? 0x200 : 0 );
	const e = exponent - 127 + 15;
	if ( e >= 0x1f ) return sign | 0x7c00;

	if ( e <= 0 ) {

		if ( e < - 10 ) return sign;
		mantissa |= 0x800000;
		const shift = 14 - e;
		let half = mantissa >>> shift;
		const remainder = mantissa & ( ( 1 << shift ) - 1 );
		const halfway = 1 << ( shift - 1 );
		if ( remainder > halfway || ( remainder === halfway && ( half & 1 ) ) ) half ++;
		return sign | half;

	}

	// A rounding carry out of the mantissa correctly increments the exponent.
	let half = ( e << 10 ) | ( mantissa >>> 13 );
	const remainder = mantissa & 0x1fff;
	if ( remainder > 0x1000 || ( remainder === 0x1000 && ( half & 1 ) ) ) half ++;
	return sign | half;

}

export function fromHalf( bits: number ): number {

	const sign = bits & 0x8000 ? - 1 : 1;
	const exponent = ( bits >>> 10 ) & 0x1f;
	const mantissa = bits & 0x3ff;
	if ( exponent === 0 ) return sign * mantissa * 2 ** - 24;
	if ( exponent === 0x1f ) return mantissa ? NaN : sign * Infinity;
	return sign * ( 1 + mantissa / 1024 ) * 2 ** ( exponent - 15 );

}

/** Quantize a value the way the GPU kernels do between levels. */
export function quantize( value: number, quantization: Quantization ): number {

	if ( quantization === 'unorm8' ) return roundHalfEven( Math.min( 1, Math.max( 0, value ) ) * 255 ) / 255;
	return fromHalf( toHalf( value ) );

}
