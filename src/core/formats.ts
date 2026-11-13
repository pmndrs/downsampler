/**
 * Formats and how intermediate levels are quantized so that every level matches
 * what would be read back from the stored texture.
 */
import { quantizeSrgb } from './srgb.js';

/**
 * How values are rounded between levels. The packed float formats quantize a
 * whole texel ({@link quantizeTexel}): rgb9e5ufloat shares one exponent across
 * r, g and b, and rg11b10ufloat has 11-, 11- and 10-bit channels.
 */
export type Quantization = 'unorm8' | 'snorm8' | 'unorm16' | 'snorm16' | 'float16' | 'float32' | 'srgb8' | PackedQuantization;

/** Quantizations of formats that pack r, g and b into one 32-bit word (no alpha). */
export type PackedQuantization = 'rgb9e5ufloat' | 'rg11b10ufloat';

export function isPackedQuantization( quantization: Quantization ): quantization is PackedQuantization {

	return quantization === 'rgb9e5ufloat' || quantization === 'rg11b10ufloat';

}

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
	/** Stored channels. Kernels work on vec4 and store the first `channels`; a sampled read returns 0, 0, 1 for the rest. */
	channels: 1 | 2 | 3 | 4;
	/** Whether a filtering sampler can read it without optional features. */
	filterable: boolean;
	/** Device feature needed to use it as a storage texture, if any. */
	feature?: string;
	/** False for formats that can never be storage textures (rgb9e5ufloat, the sRGB formats): only the buffer output path writes them. */
	storage?: false;
	/** Storage needs 'core-features-and-limits': not available in compatibility mode (rg32float). */
	coreStorage?: true;
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
	rg32float: { quantization: 'float32', bytesPerTexel: 8, channels: 2, filterable: false, coreStorage: true },
	rgba32float: { quantization: 'float32', bytesPerTexel: 16, channels: 4, filterable: false },
	// Packed HDR formats (GI caches). rgb9e5ufloat is never a storage texture, so
	// it is always written through the buffer output path; rg11b10ufloat takes
	// that path without texture-formats-tier1 (or without STORAGE_BINDING usage).
	rgb9e5ufloat: { quantization: 'rgb9e5ufloat', bytesPerTexel: 4, channels: 3, filterable: true, storage: false },
	rg11b10ufloat: { quantization: 'rg11b10ufloat', bytesPerTexel: 4, channels: 3, filterable: true, feature: 'texture-formats-tier1' },
	// sRGB colour (src/core/srgb.ts): read through the sRGB view (decoded to linear by the
	// hardware), reduced in linear, encoded in the shader and written through the buffer
	// output path: WebGPU has no sRGB storage textures, even through a view. Alpha is linear.
	'rgba8unorm-srgb': { quantization: 'srgb8', bytesPerTexel: 4, channels: 4, filterable: true, storage: false },
	'bgra8unorm-srgb': { quantization: 'srgb8', bytesPerTexel: 4, channels: 4, filterable: true, storage: false }
} as const satisfies Record<string, FormatInfo>;

export type SupportedFormat = keyof typeof formats;

/**
 * Whether the buffer output path can write a format: kernels write whole 32-bit
 * words, so a texel must be a multiple of 4 bytes.
 */
export function bufferWritable( format: SupportedFormat ): boolean {

	return formats[ format ].bytesPerTexel % 4 === 0;

}

/** Whether a format can be a storage texture on a device with `features`. */
export function storageWritable( format: SupportedFormat, features: { has( feature: string ): boolean } ): boolean {

	const info: FormatInfo = formats[ format ];
	return info.storage !== false && ( info.feature === undefined || features.has( info.feature ) );

}

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

/** Quantize a value the way the GPU kernels do between levels. Packed and sRGB formats quantize whole texels: use {@link quantizeTexel}. */
export function quantize( value: number, quantization: Quantization ): number {

	if ( isPackedQuantization( quantization ) || quantization === 'srgb8' ) throw new Error( `downsampler: ${ quantization } quantizes whole texels; use quantizeTexel.` );
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

// ---------------------------------------------------------------------------
// Packed float formats (SPEC.md §6.1).
//
// rgb9e5ufloat follows EXT_texture_shared_exponent §3.8 (Vulkan's
// E5B9G9R9_UFLOAT_PACK32): N = 9 mantissa bits, bias B = 15, Emax = 31.
//   c' = max(0, min(sharedexp_max, c)), NaN → 0; sharedexp_max = 511/512 · 2^16 = 65408
//   e' = max(−B − 1, ⌊log2 max(c')⌋) + 1 + B, with ⌊log2⌋ from the float's exponent bits
//   m  = ⌊max(c') / 2^(e' − B − N) + 0.5⌋, and e = e' + 1 when m = 2^N, else e'
//   cs = ⌊c' / 2^(e − B − N) + 0.5⌋ (round half up, as the extension specifies)
// Bits: r 0–8, g 9–17, b 18–26, e 27–31. Decoding, cs · 2^(e − B − N), is exact.
//
// rg11b10ufloat (Khronos Data Format §10.2–10.3): unsigned floats with a 5-bit
// exponent (bias 15) and 6-bit (r, g) or 5-bit (b) mantissas; exponent 31 is
// Inf (mantissa 0) or NaN. Bits: r 0–10, g 11–21, b 22–31. Vulkan (§3.9.1
// Floating-Point Format Conversions) leaves the rounding mode open, converts
// negatives to 0, and lets overflow go to the largest finite value or to
// infinity. We round to nearest even, clamp negatives (and −Inf) to 0, send
// finite values above the largest finite value (65024 for 11 bits, 64512 for
// 10) to it, and keep +Inf and NaN, as DirectXMath's XMStoreFloat3PK does.
// Kernels quantize before every store, so stored values never depend on the
// GPU's own conversion (storage path) or rounding mode.

/** Largest rgb9e5 channel value: 511/512 · 2^16. */
export const rgb9e5Max = 65408;

/** Largest finite rg11b10ufloat value of the 11-bit channels; the 10-bit one's is 64512. */
export const rg11b10Max = 65024;

function roundHalfUp( value: number ): number {

	const floor = Math.floor( value );
	return value - floor >= 0.5 ? floor + 1 : floor;

}

/** ⌊log2 v⌋ of a non-negative float32 from its exponent bits: −127 for 0 and subnormals. */
function floorLog2( value: number ): number {

	float32[ 0 ] = value;
	return ( ( uint32[ 0 ] >>> 23 ) & 0xff ) - 127;

}

/** As float32, clamped to [0, 65408]; NaN → 0. */
const clampShared = ( value: number ) => value > 0 ? Math.min( rgb9e5Max, Math.fround( value ) ) : 0;

/** Pack r, g, b (rounded to float32 first, as kernels hold them) as rgb9e5ufloat. */
export function packRgb9e5( r: number, g: number, b: number ): number {

	const cr = clampShared( r ), cg = clampShared( g ), cb = clampShared( b );
	const max = Math.max( cr, cg, cb );
	let exponent = Math.max( - 16, floorLog2( max ) ) + 16;
	if ( roundHalfUp( max * 2 ** ( 24 - exponent ) ) === 512 ) exponent ++;
	const scale = 2 ** ( 24 - exponent );
	return ( roundHalfUp( cr * scale ) | ( roundHalfUp( cg * scale ) << 9 ) | ( roundHalfUp( cb * scale ) << 18 ) | ( exponent << 27 ) ) >>> 0;

}

export function unpackRgb9e5( bits: number ): [ number, number, number ] {

	const scale = 2 ** ( ( bits >>> 27 ) - 24 );
	return [ ( bits & 0x1ff ) * scale, ( ( bits >>> 9 ) & 0x1ff ) * scale, ( ( bits >>> 18 ) & 0x1ff ) * scale ];

}

/** A float32 value as an unsigned float with a 5-bit exponent and `mantissaBits` (6 or 5) of mantissa. */
export function toUfloat( value: number, mantissaBits: number ): number {

	float32[ 0 ] = value;
	const x = uint32[ 0 ];
	const inf = 0x1f << mantissaBits;
	if ( ( x & 0x7fffffff ) > 0x7f800000 ) return inf | 1; // NaN
	if ( x >>> 31 ) return 0; // negative, −0, −Inf
	if ( x === 0x7f800000 ) return inf;
	const shift = 23 - mantissaBits;
	const exponent = ( x >>> 23 ) - 127 + 15;
	let code: number, remainder: number, halfway: number;

	if ( exponent <= 0 ) {

		// Subnormal: the implicit bit joins the mantissa, shifted 1 − exponent further.
		const s = shift + 1 - exponent;
		if ( s > 24 ) return 0;
		const mantissa = ( x & 0x7fffff ) | 0x800000;
		code = mantissa >>> s;
		remainder = mantissa & ( ( 1 << s ) - 1 );
		halfway = 1 << ( s - 1 );

	} else {

		code = ( exponent << mantissaBits ) | ( ( x & 0x7fffff ) >>> shift );
		remainder = x & ( ( 1 << shift ) - 1 );
		halfway = 1 << ( shift - 1 );

	}

	// A carry out of the mantissa correctly increments the exponent.
	if ( remainder > halfway || ( remainder === halfway && ( code & 1 ) ) ) code ++;
	return Math.min( code, inf - 1 );

}

export function fromUfloat( code: number, mantissaBits: number ): number {

	const exponent = code >>> mantissaBits, mantissa = code & ( ( 1 << mantissaBits ) - 1 );
	if ( exponent === 0 ) return mantissa * 2 ** ( - 14 - mantissaBits );
	if ( exponent === 0x1f ) return mantissa ? NaN : Infinity;
	return ( 1 + mantissa / 2 ** mantissaBits ) * 2 ** ( exponent - 15 );

}

export function packRg11b10( r: number, g: number, b: number ): number {

	return ( toUfloat( r, 6 ) | ( toUfloat( g, 6 ) << 11 ) | ( toUfloat( b, 5 ) << 22 ) ) >>> 0;

}

export function unpackRg11b10( bits: number ): [ number, number, number ] {

	return [ fromUfloat( bits & 0x7ff, 6 ), fromUfloat( ( bits >>> 11 ) & 0x7ff, 6 ), fromUfloat( bits >>> 22, 5 ) ];

}

/** A packed format's 32-bit word for an RGB texel. */
export function packTexel( r: number, g: number, b: number, quantization: PackedQuantization ): number {

	return quantization === 'rgb9e5ufloat' ? packRgb9e5( r, g, b ) : packRg11b10( r, g, b );

}

export function unpackTexel( bits: number, quantization: PackedQuantization ): [ number, number, number ] {

	return quantization === 'rgb9e5ufloat' ? unpackRgb9e5( bits ) : unpackRg11b10( bits );

}

/**
 * Quantize one RGBA texel as stored, in place at `offset` of `values`. Packed
 * formats pack and unpack r, g and b (a sampled read returns alpha 1); sRGB
 * formats encode and decode linear r, g and b, and alpha is unorm8; the
 * others quantize each value.
 */
export function quantizeTexel( values: { [ index: number ]: number }, offset: number, quantization: Quantization ): void {

	if ( quantization === 'srgb8' ) {

		for ( let c = 0; c < 3; c ++ ) values[ offset + c ] = quantizeSrgb( values[ offset + c ] );
		values[ offset + 3 ] = quantize( values[ offset + 3 ], 'unorm8' );
		return;

	}

	if ( isPackedQuantization( quantization ) ) {

		const [ r, g, b ] = unpackTexel( packTexel( values[ offset ], values[ offset + 1 ], values[ offset + 2 ], quantization ), quantization );
		values[ offset ] = r; values[ offset + 1 ] = g; values[ offset + 2 ] = b; values[ offset + 3 ] = 1;
		return;

	}

	for ( let c = 0; c < 4; c ++ ) values[ offset + c ] = quantize( values[ offset + c ], quantization );

}

/**
 * The quantization step of a stored packed value: for rgb9e5ufloat the step of
 * the shared exponent of a texel whose largest channel is `texelMax`; for
 * rg11b10ufloat the step at `value` of channel `channel` (2 is the 10-bit one).
 * Used for test tolerances.
 */
export function packedStep( quantization: PackedQuantization, value: number, channel: number, texelMax = value ): number {

	if ( quantization === 'rgb9e5ufloat' ) return 2 ** ( ( packRgb9e5( texelMax, 0, 0 ) >>> 27 ) - 24 );
	const bits = channel === 2 ? 5 : 6;
	const v = Math.min( Math.abs( value ), rg11b10Max );
	return v < 2 ** - 14 ? 2 ** ( - 14 - bits ) : 2 ** ( floorLog2( v ) - bits );

}
