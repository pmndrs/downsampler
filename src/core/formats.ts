/**
 * Formats and how intermediate levels are quantized so that every level matches
 * what would be read back from the stored texture.
 */

/** How values are rounded between levels. */
export type Quantization = 'unorm8' | 'snorm8' | 'unorm16' | 'snorm16' | 'float16' | 'float32';

/**
 * Normalized formats are reduced in integer units (value × scale), so 2:1 sums
 * are exact in f32 (4 × 65535 < 2²⁴) and ties round to even. snorm clamps to
 * ±scale: −128 and −32768 also decode to −1, so they are never produced.
 */
export const integerDomains = {
	unorm8: { scale: 255, min: 0, max: 255 },
	snorm8: { scale: 127, min: - 127, max: 127 },
	unorm16: { scale: 65535, min: 0, max: 65535 },
	snorm16: { scale: 32767, min: - 32767, max: 32767 }
} as const;

export type IntegerQuantization = keyof typeof integerDomains;

export function isIntegerDomain( quantization: Quantization ): quantization is IntegerQuantization {

	return Object.prototype.hasOwnProperty.call( integerDomains, quantization );

}

export interface FormatInfo {
	quantization: Quantization;
	/** Bytes per texel, for readback and planning. */
	bytesPerTexel: number;
	/** Stored channels. Kernels work on vec4 and store the first `channels`. */
	channels: 1 | 2 | 4;
	/** Whether a filtering sampler can read it without optional features. */
	filterable: boolean;
	/** Device feature needed to use it as a storage texture, if any. */
	feature?: string;
}

/** Formats supported by the current kernels. See SPEC.md for the planned matrix. */
export const formats = {
	rgba8unorm: { quantization: 'unorm8', bytesPerTexel: 4, channels: 4, filterable: true },
	rgba16float: { quantization: 'float16', bytesPerTexel: 8, channels: 4, filterable: true },
	rgba8snorm: { quantization: 'snorm8', bytesPerTexel: 4, channels: 4, filterable: true },
	// Storage for these needs optional features. bgra8unorm is swizzled by the
	// hardware on read and write, so kernels see RGBA like any other format.
	bgra8unorm: { quantization: 'unorm8', bytesPerTexel: 4, channels: 4, filterable: true, feature: 'bgra8unorm-storage' },
	r8unorm: { quantization: 'unorm8', bytesPerTexel: 1, channels: 1, filterable: true, feature: 'texture-formats-tier1' },
	rg8unorm: { quantization: 'unorm8', bytesPerTexel: 2, channels: 2, filterable: true, feature: 'texture-formats-tier1' },
	r16float: { quantization: 'float16', bytesPerTexel: 2, channels: 1, filterable: true, feature: 'texture-formats-tier1' },
	rg16float: { quantization: 'float16', bytesPerTexel: 4, channels: 2, filterable: true, feature: 'texture-formats-tier1' },
	r8snorm: { quantization: 'snorm8', bytesPerTexel: 1, channels: 1, filterable: true, feature: 'texture-formats-tier1' },
	rg8snorm: { quantization: 'snorm8', bytesPerTexel: 2, channels: 2, filterable: true, feature: 'texture-formats-tier1' },
	// 16-bit normalized formats sample as unfilterable-float (Dawn, Apple), so they load.
	r16unorm: { quantization: 'unorm16', bytesPerTexel: 2, channels: 1, filterable: false, feature: 'texture-formats-tier1' },
	rg16unorm: { quantization: 'unorm16', bytesPerTexel: 4, channels: 2, filterable: false, feature: 'texture-formats-tier1' },
	rgba16unorm: { quantization: 'unorm16', bytesPerTexel: 8, channels: 4, filterable: false, feature: 'texture-formats-tier1' },
	r16snorm: { quantization: 'snorm16', bytesPerTexel: 2, channels: 1, filterable: false, feature: 'texture-formats-tier1' },
	rg16snorm: { quantization: 'snorm16', bytesPerTexel: 4, channels: 2, filterable: false, feature: 'texture-formats-tier1' },
	rgba16snorm: { quantization: 'snorm16', bytesPerTexel: 8, channels: 4, filterable: false, feature: 'texture-formats-tier1' },
	// Filtering float32 needs the optional `float32-filterable` feature, so these
	// always use the load path.
	r32float: { quantization: 'float32', bytesPerTexel: 4, channels: 1, filterable: false },
	rg32float: { quantization: 'float32', bytesPerTexel: 8, channels: 2, filterable: false },
	rgba32float: { quantization: 'float32', bytesPerTexel: 16, channels: 4, filterable: false }
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

	if ( isIntegerDomain( quantization ) ) {

		const { scale, min, max } = integerDomains[ quantization ];
		return roundHalfEven( Math.min( max, Math.max( min, value * scale ) ) ) / scale;

	}

	if ( quantization === 'float32' ) return Math.fround( value );
	return fromHalf( toHalf( value ) );

}

/**
 * Channels a sampled texture of this format returns data in (the rest read as
 * 0, 0, 1). Depth formats and r* formats have one. Used to splat single-channel
 * sources so that, for example, minmax of depth yields (d, d).
 */
export function sampledChannels( format: string ): number {

	if ( format.startsWith( 'depth' ) || format === 'stencil8' ) return 1;
	// 'rgb10a2unorm' -> 'rgb10a2' -> r, g, b, a; 'rg11b10ufloat' -> r, g, b.
	const layout = format.replace( /(unorm|snorm|uint|sint|ufloat|float)(-srgb)?$/, '' );
	return layout.match( /[rgba]/g )?.length ?? 4;

}
