import { channelDomain, formats, intDomains, isIntegerDomain, isIntQuantization, isPackedQuantization, isSupportedFormat, sampledChannels, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { tailTile, tileSize, workgroupSide } from '../core/plan.js';
import { checkLevels } from '../core/pyramid.js';
import { isBuiltinReduction, needsTileEdges, reductionKey, resolveReduction, type Channels, type CustomReduction, type EdgePolicy, type Reduction } from '../core/reduction.js';
import { lanes, payloadCombine } from './lanes.js';
import { createIntKernel, intSampleType } from './integer.js';
import { checkHooks } from './tool.js';

/** Texture view dimension of kernel sources and outputs: '2d', or '2d-array' (every layer; cube maps as 6 layers). */
export type ViewDimension = '2d' | '2d-array';

/**
 * How the first output of a batch reads its source level:
 * - 'sample': one hardware bilinear sample (average only; needs a filterable format)
 * - 'load': `textureLoad` of the footprint, with bilinear weights computed in the
 *   shader for 'average'. Required for fold reductions and unfilterable formats.
 * Tile kernels ('zero-weight', 'clamp') load by default. Under 'clamp', an
 * all-'average' reduction without a load hook can 'sample' instead: one
 * bilinear sample at the shared corner of each 2×2 block (as SPD does).
 */
export type SourceAccess = 'sample' | 'load';

/**
 * How float16 outputs are rounded from the f32 working values before they are stored:
 * - 'nearest' (default): round to nearest even in the shader with integer math
 *   ({@link float16Wgsl}), so stored halves are the same on every GPU and match the
 *   CPU oracles. The store then writes an exact half, which no conversion changes.
 * - 'device': WGSL `pack2x16float`, the device's own conversion. Cheaper, but its
 *   rounding varies: D3D12 (NVIDIA, Intel) truncates toward zero, Metal rounds
 *   (results/probes/f16-store-rounding.mjs, results/windows/correctness.md).
 */
export type F16Rounding = 'nearest' | 'device';

/** Options of the pure WGSL generator {@link createKernel} and {@link kernelLayoutEntries}. `Downsampler` and `Reducer` build these for you. */
export interface KernelOptions {
	/** Storage format of the output levels. Required unless a `store` hook writes them. */
	format?: SupportedFormat;
	/** View dimension of the source and output textures. Default '2d'. */
	dimension?: ViewDimension;
	/** Levels reduced by one dispatch (1–6). Tile kernels default to the deepest of `levels`. */
	count?: number;
	/** Default 'average'. Channel ops and custom `kind: 'sum'` need a tile edge policy. */
	reduction?: Reduction;
	/**
	 * 'exact': default 'sample' for 'average', otherwise 'load' (the only option
	 * for folds). Tile kernels: default 'load'; 'sample' only under 'clamp' for
	 * an all-'average' reduction read from a texture (no load hook).
	 */
	access?: SourceAccess;
	/**
	 * 'exact' only: a tail kernel. One workgroup (dispatch 1×1×layers) writes all
	 * `count` levels; the first output (at most `tile`²) reads the source level as
	 * usual, and every further level is reduced from the previous one in
	 * workgroup memory by the same 'exact' rule, odd sizes included.
	 */
	tail?: boolean;
	/**
	 * 'exact' only: an area-weighted first output (`source.resample: 'area'`; `areaReference`
	 * in core). It covers [x·S/n, (x+1)·S/n) of its source level at any size ratio S/n:
	 * averages weight each texel it overlaps by the overlap, folds combine every texel it
	 * overlaps. Needs access 'load'.
	 */
	area?: boolean;
	/**
	 * 'exact' only: `oddSizes: 'box'`. The first output is area-weighted (as `area`), and so
	 * are a tail's further levels (three taps at odd sizes). Needs access 'load'.
	 */
	box?: boolean;
	/**
	 * The tile edge, 8, 16 or 32. Tail kernels: at least the first output's larger
	 * side, default 32. Exact kernels: at least what `count` levels need (8 up to four
	 * levels, then 16 and 32), which is the default.
	 */
	tile?: number;
	/**
	 * Format of the texture the first output reads, when it isn't `format` (a
	 * separate source, e.g. depth32float). Single-channel sources are splatted to
	 * all four channels; loads are rounded to exact integer units only when the
	 * source has the destination's normalized encoding ('exact' only).
	 */
	sourceFormat?: string;
	/**
	 * Default 'exact' (mip chains: render-pass parity, a batch per odd size).
	 * 'zero-weight' and 'clamp' select the tile kernel: every level of the dispatch
	 * in one tile, no intermediate rounding, sums unnormalised until emitted.
	 */
	edges?: EdgePolicy;
	/**
	 * Tile kernels: the levels to emit, source-relative (1 = half size), strictly
	 * increasing, at most 6. Default every level 1…count. Unemitted levels are
	 * reduced in workgroup memory but never stored, and get no binding.
	 */
	levels?: readonly number[];
	/** Tile kernels: WGSL defining `fn load(texel: vec2i, layer: u32) -> vec4f`, replacing the source texture. */
	load?: string;
	/** Tile kernels: WGSL defining `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f)`, replacing the output textures. */
	store?: string;
	/** Tile kernels with a `load` hook: WGSL defining `fn size() -> vec2u`, the source size, replacing the size uniform. */
	size?: string;
	/** Tile kernels: WGSL declarations of the tool's `@group(1)` bindings, used by the hooks. */
	bindings?: string;
	/** Kernels without a store hook: write storage textures (default) or packed texels into a buffer ({@link OutputPath}). */
	output?: OutputPath;
	/** float16 output formats: how values are rounded before the store ({@link F16Rounding}). Default 'nearest'. */
	f16Rounding?: F16Rounding;
	/**
	 * Tile kernels: accumulator channels per texel, 4 (a `vec4f`, default) or 8
	 * (`DsWide { lo: vec4f, hi: vec4f }`, which the load and store hooks and a
	 * custom reduce then take instead). Eight channels need both hooks.
	 */
	channels?: Channels;
	/**
	 * 'exact' kernels and 'clamp' tile kernels without hooks: a partial update
	 * (`Downsampler` `rect` and `layers`). Adds the region uniform ({@link regionBinding}):
	 * the dispatch's first workgroup and layer, and per level the texels to store.
	 * Every texel is computed as in a full dispatch; only stores are restricted.
	 */
	region?: boolean;
}

/** Group 0 binding of the region uniform of partial updates ({@link KernelOptions.region}). */
export const regionBinding = 11;
/**
 * Bytes of the region uniform: `struct { origin: vec4u, bounds: array<vec4u, 6> }`. `origin`
 * is the first workgroup (x, y, in tiles) and the first layer (z); `bounds[i]` the texels
 * level i of the dispatch stores, (x0, y0, x1, y1) with x0 ≤ x < x1, or all 0 for none.
 */
export const regionBytes = 112;

/** WGSL of the region uniform and `ds_inside(i, texel)`, shared by the kernels that take it. */
export function regionWgsl(): string {

	return `struct DsRegion { origin: vec4u, bounds: array<vec4u, 6> }
@group(0) @binding(${ regionBinding }) var<uniform> ds_region: DsRegion;
fn ds_inside(i: u32, texel: vec2u) -> bool {
	let b = ds_region.bounds[i];
	return all(texel >= b.xy) && all(texel < b.zw);
}
`;

}

/** Group 0 binding of the size uniform (`struct { size: vec2u }`) of tile kernels with a load hook. */
export const sizeBinding = 8;

/**
 * Where a kernel writes its levels:
 * - 'storage': one write-only storage texture per level (bindings 2…).
 * - 'buffer': packed texels into one storage buffer (binding {@link outputBufferBinding}),
 *   each level and layer at its own offset with 256-byte rows, described by a
 *   uniform (binding {@link outputLayoutBinding}, {@link outputLayoutBytes} bytes). The
 *   caller copies each level into its texture with `copyBufferToTexture`. One
 *   binding, so not limited by `maxStorageTexturesPerShaderStage`, and it writes
 *   formats that can't be storage textures (rgb9e5ufloat). Exact kernels only for now.
 */
export type OutputPath = 'storage' | 'buffer';

/** Group 0 binding of the buffer output's layout uniform. */
export const outputLayoutBinding = 9;
/** Group 0 binding of the buffer output's storage buffer (`array<u32>`). */
export const outputBufferBinding = 10;
/**
 * Bytes of the layout uniform: `struct { size: vec2u, offset, stride, rows: array<vec4u, 2> }`
 * with, per level i (0 = the first output), its offset and row stride in u32
 * words and its rows per layer. `size` is the first output's size.
 */
export const outputLayoutBytes = 112;

/**
 * WGSL for the packed float formats, shared by every kernel that quantizes or
 * writes them (mirrors `packRgb9e5` and `toUfloat` in core, bit for bit):
 * - `ds_pack9e5(v: vec3f) -> u32`, `ds_unpack9e5(bits: u32) -> vec3f`
 * - `ds_ufloat(v: f32, bits: u32) -> u32`, `ds_from_ufloat(code: u32, bits: u32) -> f32`
 *   (5-bit exponent, `bits` of mantissa: 6 for r and g, 5 for b of rg11b10ufloat)
 * Exponents come from the float's bits, never `log2` or `exp2`, which aren't exact.
 */
export function packedFloatWgsl( quantization: 'rgb9e5ufloat' | 'rg11b10ufloat' ): string {

	const pow2 = `fn ds_pow2(e: i32) -> f32 { return bitcast<f32>(u32(e + 127) << 23u); }
`;
	return quantization === 'rgb9e5ufloat' ? pow2 + `fn ds_round_half_up(v: vec3f) -> vec3f {
	let f = floor(v);
	return select(f, f + vec3f(1.0), v - f >= vec3f(0.5));
}
// EXT_texture_shared_exponent: clamp to [0, 65408] (NaN → 0), exponent from the
// largest channel's bits, bumped when its mantissa rounds to 512; round half up.
fn ds_pack9e5(value: vec3f) -> u32 {
	let nan = (bitcast<vec3u>(value) & vec3u(0x7fffffffu)) > vec3u(0x7f800000u);
	let c = select(clamp(value, vec3f(0.0), vec3f(65408.0)), vec3f(0.0), nan);
	let m = max(c.x, max(c.y, c.z));
	var e = max(-16, i32((bitcast<u32>(m) >> 23u) & 0xffu) - 127) + 16;
	if (ds_round_half_up(vec3f(m * ds_pow2(24 - e))).x == 512.0) { e += 1; }
	let q = vec3u(ds_round_half_up(c * ds_pow2(24 - e)));
	return q.x | (q.y << 9u) | (q.z << 18u) | (u32(e) << 27u);
}
fn ds_unpack9e5(bits: u32) -> vec3f {
	return vec3f((vec3u(bits, bits >> 9u, bits >> 18u) & vec3u(0x1ffu))) * ds_pow2(i32(bits >> 27u) - 24);
}
` : pow2 + `// Unsigned float, 5-bit exponent: round to nearest even, negatives → 0, finite
// overflow → the largest finite value, +Inf and NaN kept.
fn ds_ufloat(value: f32, bits: u32) -> u32 {
	let x = bitcast<u32>(value);
	let inf = 31u << bits;
	if ((x & 0x7fffffffu) > 0x7f800000u) { return inf | 1u; }
	if ((x >> 31u) != 0u) { return 0u; }
	if (x == 0x7f800000u) { return inf; }
	let shift = 23u - bits;
	let exponent = i32(x >> 23u) - 112;
	var code: u32;
	var remainder: u32;
	var halfway: u32;
	if (exponent <= 0) {
		let s = shift + 1u + u32(-exponent);
		if (s > 24u) { return 0u; }
		let mantissa = (x & 0x7fffffu) | 0x800000u;
		code = mantissa >> s;
		remainder = mantissa & ((1u << s) - 1u);
		halfway = 1u << (s - 1u);
	} else {
		code = (u32(exponent) << bits) | ((x & 0x7fffffu) >> shift);
		remainder = x & ((1u << shift) - 1u);
		halfway = 1u << (shift - 1u);
	}
	if (remainder > halfway || (remainder == halfway && (code & 1u) == 1u)) { code += 1u; }
	return min(code, inf - 1u);
}
fn ds_from_ufloat(code: u32, bits: u32) -> f32 {
	let exponent = code >> bits;
	let mantissa = code & ((1u << bits) - 1u);
	if (exponent == 0u) { return f32(mantissa) * ds_pow2(-14 - i32(bits)); }
	if (exponent == 31u) { return bitcast<f32>(select(0x7f800000u, 0x7fc00000u, mantissa != 0u)); }
	return bitcast<f32>(((exponent + 112u) << 23u) | (mantissa << (23u - bits)));
}
fn ds_pack11_11_10(v: vec3f) -> u32 { return ds_ufloat(v.x, 6u) | (ds_ufloat(v.y, 6u) << 11u) | (ds_ufloat(v.z, 5u) << 22u); }
fn ds_unpack11_11_10(bits: u32) -> vec3f { return vec3f(ds_from_ufloat(bits & 0x7ffu, 6u), ds_from_ufloat((bits >> 11u) & 0x7ffu, 6u), ds_from_ufloat(bits >> 22u, 5u)); }
`;

}

/**
 * The u32 words of one texel of `format`, as WGSL expressions of `v`, little-endian
 * in the format's byte order. `v` is either the exact kernels' working value
 * (`normalized` false: integer units for normalized formats, already rounded and
 * quantized as stored) or a normalized value (tile kernels), which normalized
 * formats round with WGSL's pack builtins.
 */
function packedWords( format: SupportedFormat, normalized = false ): string[] {

	const { quantization, channels } = formats[ format ];
	const ch = [ 'x', 'y', 'z', 'w' ];
	const order = format === 'bgra8unorm' ? [ 'z', 'y', 'x', 'w' ] : ch;
	// 10-bit r, g, b and 2-bit a: integer units (rgb10a2unorm normalized values rounded to them, ties to even).
	if ( quantization === 'unorm10a2' || quantization === 'uint10a2' ) {

		const unit = ( c: string, scale: number ) => normalized ? `u32(round(clamp(v.${ c }, 0.0, 1.0) * ${ scale }.0))` : `u32(v.${ c })`;
		return [ `${ unit( 'x', 1023 ) } | (${ unit( 'y', 1023 ) } << 10u) | (${ unit( 'z', 1023 ) } << 20u) | (${ unit( 'w', 3 ) } << 30u)` ];

	}

	// Integer formats: `v` holds the stored integers' bits (vec4u), sign-extended for sint.
	if ( isIntQuantization( quantization ) ) {

		const bits = intDomains[ quantization ].bits[ 0 ];
		if ( bits === 32 ) return ch.slice( 0, channels ).map( c => `v.${ c }` );
		const mask = bits === 8 ? '0xffu' : '0xffffu';
		const perWord = 32 / bits;
		if ( channels % perWord !== 0 ) throw new Error( `downsampler: the buffer output path writes 32-bit words; ${ format } texels are ${ formats[ format ].bytesPerTexel } bytes.` );
		return Array.from( { length: channels / perWord }, ( _, w ) => Array.from( { length: perWord }, ( _, i ) => {

			const value = `(v.${ ch[ w * perWord + i ] } & ${ mask })`;
			return i === 0 ? value : `(${ value } << ${ bits * i }u)`;

		} ).join( ' | ' ) );

	}

	if ( normalized && isIntegerDomain( quantization ) ) {

		const kind = quantization.startsWith( 'snorm' ) ? 'snorm' : 'unorm';
		if ( quantization.endsWith( '8' ) && channels === 4 ) return [ `pack4x8${ kind }(v.${ order.join( '' ) })` ];
		if ( quantization.endsWith( '16' ) && channels >= 2 ) return Array.from( { length: channels / 2 }, ( _, w ) => `pack2x16${ kind }(v.${ ch[ 2 * w ] }${ ch[ 2 * w + 1 ] })` );

	}

	const unsigned = ( c: string, mask: string ) => quantization.startsWith( 'snorm' ) ? `(bitcast<u32>(i32(v.${ c })) & ${ mask })` : `u32(v.${ c })`;
	switch ( quantization ) {

		case 'unorm8': case 'snorm8':
			if ( channels !== 4 ) break;
			return [ order.map( ( c, i ) => i === 0 ? unsigned( c, '0xffu' ) : `(${ unsigned( c, '0xffu' ) } << ${ 8 * i }u)` ).join( ' | ' ) ];
		case 'unorm16': case 'snorm16':
			if ( channels < 2 ) break;
			return Array.from( { length: channels / 2 }, ( _, w ) => `${ unsigned( ch[ 2 * w ], '0xffffu' ) } | (${ unsigned( ch[ 2 * w + 1 ], '0xffffu' ) } << 16u)` );
		case 'float16':
			if ( channels < 2 ) break;
			return Array.from( { length: channels / 2 }, ( _, w ) => `ds_pack_f16(v.${ ch[ 2 * w ] }${ ch[ 2 * w + 1 ] })` );
		case 'float32':
			return ch.slice( 0, channels ).map( c => `bitcast<u32>(v.${ c })` );
		case 'rgb9e5ufloat':
			return [ 'ds_pack9e5(v.xyz)' ];
		case 'rg11b10ufloat':
			return [ 'ds_pack11_11_10(v.xyz)' ];

	}

	throw new Error( `downsampler: the buffer output path writes 32-bit words; ${ format } texels are ${ formats[ format ].bytesPerTexel } bytes.` );

}

/**
 * WGSL for float16 rounding ({@link F16Rounding}), shared by every kernel that stores float16:
 * - `ds_pack_f16(v: vec2f) -> u32`: two halves, as `pack2x16float` packs them
 * - `ds_round_f16(v: vec4f) -> vec4f`: each channel rounded to a half, as f32
 * 'nearest' mirrors core `toHalf` bit for bit (NaN → 0x7e00, ≥ 65520 → infinity, subnormals
 * rounded): `ds_pack_f16` with integer math on the float's bits; `ds_round_f16`, every level's
 * hot path, branch-free: in the half normal range, round the f32 bits to 10 mantissa bits with
 * integer math (a carry into the exponent is correct); below it, `round( x · 2^24 ) · 2^−24`.
 * Not the shorter `( x + c ) − c` with c = 2^(exponent + 13): Metal compiles WGSL with fast
 * math, which folds that to x, so nothing was rounded on Apple GPUs (2026-11-13).
 * 'device' uses `pack2x16float`.
 */
export function float16Wgsl( rounding: F16Rounding = 'nearest' ): string {

	if ( rounding !== 'nearest' && rounding !== 'device' ) throw new Error( `downsampler: f16Rounding must be 'nearest' or 'device', got ${ JSON.stringify( rounding ) }.` );
	if ( rounding === 'device' ) return `fn ds_pack_f16(v: vec2f) -> u32 { return pack2x16float(v); }
fn ds_round_f16(v: vec4f) -> vec4f { return vec4f(unpack2x16float(pack2x16float(v.xy)), unpack2x16float(pack2x16float(v.zw))); }
`;
	return `fn ds_half_bits(x: f32) -> u32 {
	let b = bitcast<u32>(x);
	let sign = (b >> 16u) & 0x8000u;
	let a = b & 0x7fffffffu;
	if (a > 0x7f800000u) { return sign | 0x7e00u; }
	if (a >= 0x477ff000u) { return sign | 0x7c00u; }
	if (a >= 0x38800000u) {
		let m = a - 0x38000000u;
		return sign | ((m + 0xfffu + ((m >> 13u) & 1u)) >> 13u);
	}
	let e = a >> 23u;
	if (e < 102u) { return sign; }
	let mantissa = (a & 0x7fffffu) | 0x800000u;
	let shift = 126u - e;
	let halfway = 1u << (shift - 1u);
	let remainder = mantissa & ((1u << shift) - 1u);
	var q = mantissa >> shift;
	if (remainder > halfway || (remainder == halfway && (q & 1u) == 1u)) { q += 1u; }
	return sign | q;
}
fn ds_pack_f16(v: vec2f) -> u32 { return ds_half_bits(v.x) | (ds_half_bits(v.y) << 16u); }
fn ds_round_f16(v: vec4f) -> vec4f {
	let bits = bitcast<vec4u>(v);
	let a = bits & vec4u(0x7fffffffu);
	let normal = (a + vec4u(0xfffu) + ((a >> vec4u(13u)) & vec4u(1u))) & vec4u(0xffffe000u);
	let subnormal = bitcast<vec4u>(round(bitcast<vec4f>(a) * 16777216.0) * (1.0 / 16777216.0));
	let r = select(subnormal, normal, a >= vec4u(0x38800000u));
	let q = select(select(r, vec4u(0x7f800000u), a >= vec4u(0x477ff000u)), vec4u(0x7fc00000u), a > vec4u(0x7f800000u));
	return bitcast<vec4f>(q | (bits & vec4u(0x80000000u)));
}
`;

}

/**
 * WGSL for the buffer output path (see {@link OutputPath}): the layout uniform,
 * the buffer, and `fn ds_put(i: u32, texel: vec2u, layer: u32, v: vec4f)`, which
 * writes level i's texel. Shared by every kernel family that writes buffers.
 * float16 formats include {@link float16Wgsl} with `f16Rounding`, unless it is
 * false (the caller defines `ds_pack_f16` itself).
 */
export function bufferOutputWgsl( format: SupportedFormat, normalized = false, f16Rounding: F16Rounding | false = 'nearest' ): string {

	const words = packedWords( format, normalized );
	// Integer formats put their integers' bits (vec4u).
	const type = isIntQuantization( formats[ format ].quantization ) ? 'vec4u' : 'vec4f';
	const half = formats[ format ].quantization === 'float16' && f16Rounding !== false ? float16Wgsl( f16Rounding ) : '';
	return `${ half }struct DsOutput { size: vec2u, offset: array<vec4u, 2>, stride: array<vec4u, 2>, rows: array<vec4u, 2> }
@group(0) @binding(${ outputLayoutBinding }) var<uniform> ds_output: DsOutput;
@group(0) @binding(${ outputBufferBinding }) var<storage, read_write> ds_texels: array<u32>;
fn ds_put(i: u32, texel: vec2u, layer: u32, v: ${ type }) {
	let index = ds_output.offset[i / 4u][i % 4u] + (layer * ds_output.rows[i / 4u][i % 4u] + texel.y) * ds_output.stride[i / 4u][i % 4u] + texel.x * ${ words.length }u;
${ words.map( ( word, w ) => `	ds_texels[index${ w ? ` + ${ w }u` : '' }] = ${ word };\n` ).join( '' ) }}
`;

}

/** Validates tile kernel options and resolves their defaults. */
function tileOptions( options: KernelOptions ) {

	const { format, dimension = '2d', reduction = 'average', edges, load, store, size, bindings = '', sourceFormat, f16Rounding = 'nearest' } = options;
	if ( f16Rounding !== 'nearest' && f16Rounding !== 'device' ) throw new Error( `downsampler: f16Rounding must be 'nearest' or 'device', got ${ JSON.stringify( f16Rounding ) }.` );
	if ( edges !== 'zero-weight' && edges !== 'clamp' ) throw new Error( `downsampler: unknown edge policy ${ JSON.stringify( edges ) }; expected 'exact', 'zero-weight' or 'clamp'.` );
	const levels = options.levels ?? Array.from( { length: options.count ?? 0 }, ( _, i ) => i + 1 );
	checkLevels( levels );
	const count = levels[ levels.length - 1 ];
	if ( options.count !== undefined && options.count !== count ) throw new Error( `downsampler: count ${ options.count } must be the deepest emitted level (${ count }).` );
	const access = options.access ?? 'load';
	if ( access !== 'load' && access !== 'sample' ) throw new Error( `downsampler: unknown access ${ JSON.stringify( access ) }.` );
	if ( access === 'sample' && ( edges !== 'clamp' || load !== undefined || resolveReduction( reduction, options.channels ).ops?.some( op => op !== 'average' ) !== false ) ) throw new Error( 'downsampler: tile kernels sample (access "sample") only under \'clamp\', for an all-\'average\' reduction read from a texture; otherwise they read with textureLoad.' );
	if ( options.tail !== undefined || options.tile !== undefined ) throw new Error( 'downsampler: tail and tile are \'exact\' kernel options.' );
	checkHooks( { load, store, size } );
	if ( options.region === true && ( load !== undefined || store !== undefined ) ) throw new Error( 'downsampler: region (partial update) tile kernels read a source texture and write output textures or the buffer output; hooks aren\'t supported.' );
	if ( size !== undefined && load === undefined ) throw new Error( 'downsampler: a size hook replaces the size uniform of a load hook; without a load hook the size is the source texture\'s.' );
	if ( typeof bindings !== 'string' ) throw new Error( 'downsampler: bindings must be WGSL declarations for @group(1).' );
	if ( store === undefined && ( format === undefined || ! isSupportedFormat( format ) ) ) throw new Error( `downsampler: output format ${ JSON.stringify( format ) } is not supported; pass a supported format or a store hook.` );
	// Tile kernels accumulate f32 sums, which can't hold 32-bit integers; integer chains are Downsampler's 'exact' and 'box'.
	if ( store === undefined && isIntQuantization( formats[ format! ].quantization ) ) throw new Error( `downsampler: tile kernels (a Reducer, oddSizes 'spd') accumulate floats, so they don't write integer formats (${ format }); use a store hook, or a Downsampler with oddSizes 'exact' or 'box'.` );
	if ( load === undefined && sourceFormat !== undefined && /int$/.test( sourceFormat ) ) throw new Error( `downsampler: tile kernels read float-sampleable sources, not ${ sourceFormat }; read it in a load hook.` );
	if ( dimension !== '2d' && dimension !== '2d-array' ) throw new Error( `downsampler: view dimension ${ JSON.stringify( dimension ) } is not supported.` );
	const output = options.output ?? 'storage';
	if ( output !== 'storage' && output !== 'buffer' ) throw new Error( `downsampler: unknown output path ${ JSON.stringify( output ) }; expected 'storage' or 'buffer'.` );
	if ( store !== undefined && output === 'buffer' ) throw new Error( 'downsampler: a store hook replaces the output textures; output \'buffer\' writes them through a buffer.' );
	if ( store === undefined && output === 'storage' && ( formats[ format! ] as FormatInfo ).storage === false ) throw new Error( `downsampler: ${ format } can't be a storage texture; use output 'buffer'.` );
	const { channels } = resolveReduction( reduction, options.channels );
	// Eight channels never reach output textures (or the buffer output, which writes them).
	if ( channels === 8 && ( load === undefined || store === undefined ) ) throw new Error( 'downsampler: 8-channel accumulators need a load hook and a store hook (textures hold at most 4 channels, so neither output textures nor output \'buffer\' can take them).' );
	return { format, dimension, reduction, edges, access, levels, count, load, store, size, bindings, output, channels, sourceFormat: sourceFormat ?? format ?? 'rgba32float', f16Rounding };

}

/** Bind group layout entries (group 0) matching {@link createKernel}. */
export function kernelLayoutEntries( options: KernelOptions ): GPUBindGroupLayoutEntry[] {

	const visibility = GPUShaderStage.COMPUTE;
	const entries: GPUBindGroupLayoutEntry[] = [];

	if ( ( options.edges ?? 'exact' ) !== 'exact' ) {

		const { format, dimension, access, levels, load, store, size, output } = tileOptions( options );
		if ( access === 'sample' ) entries.push( { binding: 0, visibility, sampler: { type: 'filtering' } } );
		if ( load === undefined ) entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
		if ( store === undefined && output === 'buffer' ) entries.push( { binding: outputLayoutBinding, visibility, buffer: { type: 'uniform', minBindingSize: outputLayoutBytes } }, { binding: outputBufferBinding, visibility, buffer: { type: 'storage' } } );
		else if ( store === undefined ) levels.forEach( ( _, i ) => entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format: format!, viewDimension: dimension } } ) );
		if ( load !== undefined && size === undefined ) entries.push( { binding: sizeBinding, visibility, buffer: { type: 'uniform', minBindingSize: 8 } } );
		if ( options.region === true ) entries.push( { binding: regionBinding, visibility, buffer: { type: 'uniform', minBindingSize: regionBytes } } );
		return entries;

	}

	const { format, dimension = '2d', count, reduction = 'average', output = 'storage' } = options;
	// Integer formats always load, through a 'uint' or 'sint' binding.
	const int = format !== undefined && isSupportedFormat( format ) && isIntQuantization( formats[ format ].quantization );
	const access = options.access ?? ( reduction === 'average' && ! int ? 'sample' : 'load' );
	if ( access === 'sample' ) entries.push( { binding: 0, visibility, sampler: { type: 'filtering' } } );
	entries.push( { binding: 1, visibility, texture: { sampleType: int ? intSampleType( format! ) : access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
	if ( output === 'buffer' ) {

		entries.push( { binding: outputLayoutBinding, visibility, buffer: { type: 'uniform', minBindingSize: outputLayoutBytes } } );
		entries.push( { binding: outputBufferBinding, visibility, buffer: { type: 'storage' } } );

	} else for ( let i = 0; i < count!; i ++ ) entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format: format!, viewDimension: dimension } } );
	if ( options.region === true ) entries.push( { binding: regionBinding, visibility, buffer: { type: 'uniform', minBindingSize: regionBytes } } );
	return entries;

}

/**
 * WGSL for area-weighted steps (see `areaReference` in core): the span of source
 * texels output texel x of n overlaps in a level of S, `[x·S/n, (x+1)·S/n)`, and
 * (averages) each texel's weight, overlap / S. The overlap is an integer in units
 * of 1/n texel. Halving and same-size axes take exact weights (1/2, 1): GPU float
 * division isn't correctly rounded (WGSL allows 2.5 ulp), and an even step must
 * stay the 'exact' rule's 2×2 box bit for bit.
 */
function areaWgsl( average: boolean ): string {

	return `fn ds_area_first(x: vec2u, s: vec2u, n: vec2u) -> vec2u { return (x * s) / n; }
fn ds_area_last(x: vec2u, s: vec2u, n: vec2u) -> vec2u { return ((x + vec2u(1u)) * s - vec2u(1u)) / n; }
${ average ? `fn ds_area_weight(t: u32, x: u32, s: u32, n: u32) -> f32 {
	if (s == 2u * n) { return 0.5; }
	if (s == n) { return 1.0; }
	return f32(min(n * (t + 1u), (x + 1u) * s) - max(n * t, x * s)) / f32(s);
}
` : '' }`;

}

const builtinFolds = {
	min: 'min(a, b)',
	max: 'max(a, b)',
	minmax: 'vec4f(min(a.x, b.x), max(a.y, b.y), min(a.z, b.z), max(a.w, b.w))'
};

/**
 * WGSL for one dispatch. `edges` selects the kernel family:
 * - 'exact' (default): a mip chain batch, as `Downsampler` plans them.
 * - 'zero-weight' or 'clamp': a tile reduction, as `Reducer` runs it (SPEC §14.1).
 */
export function createKernel( options: KernelOptions ): string {

	if ( ( options.edges ?? 'exact' ) !== 'exact' ) return createTileKernel( options );
	const { reduction = 'average' } = options;
	if ( needsTileEdges( reduction ) ) throw new Error( 'downsampler: sums, per-channel reductions and custom sums need edges \'zero-weight\' or \'clamp\' (a Reducer); \'exact\' only has meaning for averages and folds.' );
	for ( const name of [ 'load', 'store', 'size', 'bindings', 'levels', 'channels' ] as const ) {

		if ( options[ name ] !== undefined ) throw new Error( `downsampler: ${ name } needs edges 'zero-weight' or 'clamp' (a Reducer).` );

	}

	if ( options.format === undefined || options.count === undefined ) throw new Error( 'downsampler: exact kernels need a format and a level count.' );
	if ( isSupportedFormat( options.format ) && isIntQuantization( formats[ options.format ].quantization ) ) return createIntKernel( { ...options, format: options.format, dimension: options.dimension ?? '2d', count: options.count } );
	return createExactKernel( { ...options, format: options.format, dimension: options.dimension ?? '2d', count: options.count } );

}

/**
 * WGSL for one batch. Bindings: 0 sampler ('sample' access only), 1 source level,
 * 2… output levels (write-only storage). Workgroup z selects the array layer.
 *
 * Each workgroup writes a tile×tile block of the first output (32×32 for six
 * levels, i.e. a 64×64 source tile as in SPD). Adjacent invocations read and
 * store adjacent texels — the layout matters: assigning each invocation a 2×2
 * block measured ~30–45 % slower. Later levels are 2:1 reductions of the tile in
 * workgroup memory, with clamped reads for unit dimensions and partial tiles.
 *
 * A tail kernel (`tail`) is one workgroup with a `tile`² tile (8, 16 or 32): the
 * first output as above, then every further level from the previous one in the
 * tile by the 'exact' rule at any parity (shader bilinear weights for 'average',
 * the `2x … 2x+2` footprint for folds), each quantized as stored.
 */
function createExactKernel( { format, dimension, count, reduction = 'average', access, sourceFormat = format, output = 'storage', tail = false, box = false, area = box, tile: tileOption, f16Rounding = 'nearest', region = false }: KernelOptions & { format: SupportedFormat, dimension: ViewDimension, count: number } ): string {

	reductionKey( reduction );
	const average = reduction === 'average';
	if ( box && ! area ) throw new Error( 'downsampler: a box kernel\'s first output is area-weighted (area true).' );
	access ??= average && ! area ? 'sample' : 'load';
	if ( ! average && access !== 'load' ) throw new Error( 'downsampler: fold reductions read their source with textureLoad (access "load").' );
	if ( area && access !== 'load' ) throw new Error( 'downsampler: area-weighted (box) kernels read their source with textureLoad (access "load").' );

	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const quantization = formats[ format ].quantization;
	const integer = isIntegerDomain( quantization );
	const exactLoads = integer && isSupportedFormat( sourceFormat ) && formats[ sourceFormat ].quantization === quantization;
	const splat = sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '';
	const tile = tileOption ?? ( tail ? tailTile : tileSize( count ) );
	if ( tile !== 8 && tile !== 16 && tile !== 32 ) throw new Error( `downsampler: a tile is 8, 16 or 32, got ${ tile }.` );
	if ( ! tail && tile < tileSize( count ) ) throw new Error( `downsampler: ${ count } levels need a tile of at least ${ tileSize( count ) }, got ${ tile }.` );
	const side = workgroupSide( tile );
	const samples = tile / side;

	let code = access === 'sample' ? '@group(0) @binding(0) var imageSampler: sampler;\n' : '';
	code += `@group(0) @binding(1) var source: texture_2d${ suffix }<f32>;
var<workgroup> tile: array<vec4f, ${ tile * tile }>;
`;
	const buffer = output === 'buffer';
	if ( output !== 'storage' && ! buffer ) throw new Error( `downsampler: unknown output path ${ JSON.stringify( output ) }; expected 'storage' or 'buffer'.` );
	if ( ! buffer && ( formats[ format ] as FormatInfo ).storage === false ) throw new Error( `downsampler: ${ format } can't be a storage texture; use output 'buffer'.` );
	if ( quantization === 'float16' ) code += float16Wgsl( f16Rounding );
	if ( buffer ) code += bufferOutputWgsl( format, false, false );
	else for ( let i = 0; i < count; i ++ ) code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;
	if ( region ) code += regionWgsl();

	// Intermediate levels are quantized exactly as they would be stored.
	// Normalized values stay in integer units (value × scale) so 2:1 averages are
	// exact and ties round to even; with k/255 floats, float error turns ties into
	// round-ups that drift by several steps along long 1D tails. The parity test
	// uses i32: converting a negative float to u32 is undefined (snorm).
	if ( integer ) {

		// Per-channel units: a scalar, or a vector for rgb10a2unorm (2-bit alpha).
		const domains = [ 0, 1, 2, 3 ].map( c => channelDomain( quantization, c ) );
		const units = ( key: 'scale' | 'min' | 'max' ) => domains.every( domain => domain[ key ] === domains[ 0 ][ key ] ) ? `${ domains[ 0 ][ key ] }.0` : `vec4f(${ domains.map( domain => `${ domain[ key ] }.0` ).join( ', ' ) })`;
		const vector = ( key: 'min' | 'max' ) => units( key ).startsWith( 'vec4f' ) ? units( key ) : `vec4f(${ units( key ) })`;
		code += `
const SCALE = ${ units( 'scale' ) };
fn quantize(value: vec4f) -> vec4f {
	let scaled = clamp(value, ${ vector( 'min' ) }, ${ vector( 'max' ) });
	let lower = floor(scaled);
	let fraction = scaled - lower;
	let nearest = select(lower, lower + vec4f(1), fraction > vec4f(0.5));
	return select(nearest, lower + vec4f(vec4i(lower) & vec4i(1)), fraction == vec4f(0.5));
}
`;

	} else if ( quantization === 'float16' ) {

		code += `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return ds_round_f16(value); }
`;

	} else if ( isPackedQuantization( quantization ) ) {

		// Packed floats: r, g and b packed and unpacked as stored; alpha reads as 1.
		const [ pack, unpack ] = quantization === 'rgb9e5ufloat' ? [ 'ds_pack9e5', 'ds_unpack9e5' ] : [ 'ds_pack11_11_10', 'ds_unpack11_11_10' ];
		code += `
${ packedFloatWgsl( quantization ) }const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return vec4f(${ unpack }(${ pack }(value.xyz)), 1.0); }
`;

	} else {

		code += `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return value; }
`;

	}

	if ( ! average ) {

		// Folds see normalized values; the tile holds scaled ones.
		if ( isBuiltinReduction( reduction ) ) code += `fn fold(a: vec4f, b: vec4f) -> vec4f { return ${ builtinFolds[ reduction as keyof typeof builtinFolds ] }; }\n`;
		else code += `${ ( reduction as CustomReduction ).wgsl }\nfn fold(a: vec4f, b: vec4f) -> vec4f { return ${ integer ? 'reduce(a / SCALE, b / SCALE) * SCALE' : 'reduce(a, b)' }; }\n`;

	}

	if ( access === 'load' ) {

		// Normalized texels are rounded back to exact integer units so later sums are exact.
		code += `
fn load(p: vec2u, layer: u32) -> vec4f {
	let v = textureLoad(source, vec2i(p)${ array ? ', i32(layer)' : '' }, 0)${ splat } * SCALE;
	return ${ exactLoads ? 'round(v)' : 'v' };
}
`;

	}

	if ( area ) code += areaWgsl( average );

	code += `
fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
`;

	if ( area ) {

		// Every texel of [pos·S/n, (pos+1)·S/n) per axis, at any ratio S/n (a same-size
		// source reads its texel alone, with weight 1).
		code += `	let sourceSize = textureDimensions(source);
	let lo = ds_area_first(pos, sourceSize, size);
	let hi = ds_area_last(pos, sourceSize, size);
${ average ? `	var sum = vec4f(0.0);
	for (var y = lo.y; y <= hi.y; y++) {
		var row = vec4f(0.0);
		for (var x = lo.x; x <= hi.x; x++) { row += ds_area_weight(x, pos.x, sourceSize.x, size.x) * load(vec2u(x, y), layer); }
		sum += ds_area_weight(y, pos.y, sourceSize.y, size.y) * row;
	}
	return quantize(sum);
` : `	var v = load(lo, layer);
	for (var y = lo.y; y <= hi.y; y++) {
		for (var x = lo.x; x <= hi.x; x++) { v = fold(v, load(vec2u(x, y), layer)); }
	}
	return quantize(v);
` }`;

	} else if ( access === 'sample' ) {

		code += `	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv${ array ? ', i32(layer)' : '' }, 0.0)${ splat } * SCALE);
`;

	} else if ( average ) {

		// Bilinear weights at the output texel centre, as the sampler would apply
		// them but without its fixed-point weight rounding. The sample position
		// (x + 0.5) * S / n - 0.5 is ((2x + 1) * S - n) / 2n: integer math keeps
		// the base texel exact (float division isn't correctly rounded on GPUs),
		// and gives weights of exactly 0 or 0.5 for copies and even sizes. A same-size
		// level (a converting copy) reads its texel alone: a zero weight on an Inf
		// neighbour would still give NaN.
		code += `	let sourceSize = textureDimensions(source);
	if (all(sourceSize == size)) { return quantize(load(pos, layer)); }
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(load(lo, layer), load(vec2u(hi.x, lo.y), layer), f.x);
	let bottom = mix(load(vec2u(lo.x, hi.y), layer), load(hi, layer), f.x);
	return quantize(mix(top, bottom, f.y));
`;

	} else {

		// The footprint: 2x and 2x + 1, plus 2x + 2 along odd source dimensions.
		// A same-size source (separate source copied into the first level) reads
		// one texel per axis.
		code += `	let sourceSize = textureDimensions(source);
	let last = sourceSize - vec2u(1);
	let same = sourceSize == size;
	let lo = select(min(pos * 2u, last), pos, same);
	let hi = select(min(lo + vec2u(1), last), lo, same);
	var v = fold(fold(load(lo, layer), load(vec2u(hi.x, lo.y), layer)), fold(load(vec2u(lo.x, hi.y), layer), load(hi, layer)));
	let odd = ((sourceSize & vec2u(1)) == vec2u(1)) & (sourceSize > vec2u(1)) & !same;
	let far = min(lo + vec2u(2), last);
	if (odd.x) { v = fold(v, fold(load(vec2u(far.x, lo.y), layer), load(vec2u(far.x, hi.y), layer))); }
	if (odd.y) { v = fold(v, fold(load(vec2u(lo.x, far.y), layer), load(vec2u(hi.x, far.y), layer))); }
	if (odd.x && odd.y) { v = fold(v, load(far, layer)); }
	return quantize(v);
`;

	}

	code += '}\n';

	// Level i of the batch: its size, and the store of one texel. The buffer path
	// derives sizes from the first output's (exact batches halve, floored, at least 1).
	if ( buffer ) code += `
fn ds_level_size(i: u32) -> vec2u { return max(ds_output.size >> vec2u(i), vec2u(1u)); }
`;
	// Later sizes are derived from the first (floored halves, at least 1), not queried:
	// textureDimensions on a storage texture is a resource query per call (5–25 % of the
	// chain on Intel UHD, rgba8; results/windows/timings.md).
	const dims = ( i: number ) => buffer ? `ds_level_size(${ i }u)` : i === 0 ? 'textureDimensions(mip0)' : `max(size0 >> vec2u(${ i }u), vec2u(1u))`;
	const put = ( i: number, pos: string ) => buffer ? `ds_put(${ i }u, ${ pos }, group.z, value);` : `textureStore(mip${ i }, vec2i(${ pos })${ layer }, value / SCALE);`;
	// Partial updates store only the region's texels (which lie inside the level).
	const stored = ( i: number, pos: string, size: string ) => region ? `ds_inside(${ i }u, ${ pos })` : `all(${ pos } < ${ size })`;
	// Partial updates offset the workgroup grid and the layer.
	const entry = region ? `@builtin(workgroup_id) ds_group: vec3u) {
	let group = ds_group + ds_region.origin.xyz;` : '@builtin(workgroup_id) group: vec3u) {';

	if ( tail ) {

		// The 'exact' rule from a level held in the tile (scaled, quantized units),
		// as `first` applies it to a texture level with 'load' access.
		code += `
fn tileAt(p: vec2u) -> vec4f { return tile[p.y * ${ tile }u + p.x]; }
fn tailStep(pos: vec2u, sourceSize: vec2u, size: vec2u) -> vec4f {
	let last = sourceSize - vec2u(1);
`;
		code += average && box ? `	let lo = ds_area_first(pos, sourceSize, size);
	let hi = ds_area_last(pos, sourceSize, size);
	var sum = vec4f(0.0);
	for (var y = lo.y; y <= hi.y; y++) {
		var row = vec4f(0.0);
		for (var x = lo.x; x <= hi.x; x++) { row += ds_area_weight(x, pos.x, sourceSize.x, size.x) * tileAt(vec2u(x, y)); }
		sum += ds_area_weight(y, pos.y, sourceSize.y, size.y) * row;
	}
	return quantize(sum);
}
` : average ? `	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(tileAt(lo), tileAt(vec2u(hi.x, lo.y)), f.x);
	let bottom = mix(tileAt(vec2u(lo.x, hi.y)), tileAt(hi), f.x);
	return quantize(mix(top, bottom, f.y));
}
` : `	let lo = min(pos * 2u, last);
	let hi = min(lo + vec2u(1), last);
	var v = fold(fold(tileAt(lo), tileAt(vec2u(hi.x, lo.y))), fold(tileAt(vec2u(lo.x, hi.y)), tileAt(hi)));
	let odd = ((sourceSize & vec2u(1)) == vec2u(1)) & (sourceSize > vec2u(1));
	let far = min(lo + vec2u(2), last);
	if (odd.x) { v = fold(v, fold(tileAt(vec2u(far.x, lo.y)), tileAt(vec2u(far.x, hi.y)))); }
	if (odd.y) { v = fold(v, fold(tileAt(vec2u(lo.x, far.y)), tileAt(vec2u(hi.x, far.y)))); }
	if (odd.x && odd.y) { v = fold(v, tileAt(far)); }
	return quantize(v);
}
`;

	}

	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, ${ entry }
	let size0 = ${ dims( 0 ) };
	var value: vec4f;
`;

	for ( let i = 0; i < samples * samples; i ++ ) {

		code += `
	{
		let p = local.xy + vec2u(${ ( i % samples ) * side }u, ${ Math.floor( i / samples ) * side }u);
		let pos0 = group.xy * ${ tile }u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * ${ tile }u + p.x] = value;
		if (${ stored( 0, 'pos0', 'size0' ) }) { ${ put( 0, 'pos0' ) } }
	}
`;

	}

	if ( tail ) {

		// The first output fits the tile, so every later level fits half of it, at
		// most the workgroup's side: one texel per invocation.
		for ( let i = 1; i < count; i ++ ) {

			code += `
	workgroupBarrier();
	{
		let size = ${ dims( i ) };
		if (all(local.xy < size)) { value = tailStep(local.xy, ${ dims( i - 1 ) }, size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < ${ dims( i ) })) {
		tile[local.y * ${ tile }u + local.x] = value;
		${ region ? `if (ds_inside(${ i }u, local.xy)) { ${ put( i, 'local.xy' ) } }` : put( i, 'local.xy' ) }
	}
`;

		}

		return code + '}\n';

	}

	const combine = ( a: string, b: string, c: string, d: string ) => average ? `quantize((${ a } + ${ b } + ${ c } + ${ d }) * 0.25)` : `quantize(fold(fold(${ a }, ${ b }), fold(${ c }, ${ d })))`;

	for ( let i = 1; i < count; i ++ ) {

		const extent = tile >> i;
		const at = ( x: string, y: string ) => `tile[${ y }.y * ${ tile }u + ${ x }.x]`;
		code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = ${ dims( i - 1 ) } - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = ${ combine( at( 'lo', 'lo' ), at( 'hi', 'lo' ), at( 'lo', 'hi' ), at( 'hi', 'hi' ) ) };
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		tile[local.y * ${ tile }u + local.x] = value;
		let pos = group.xy * ${ extent }u + local.xy;
		if (${ stored( i, 'pos', dims( i ) ) }) { ${ put( i, 'pos' ) } }
	}
`;

	}

	return code + '}\n';

}

/**
 * WGSL for a tile reduction ('zero-weight' or 'clamp' edges): every level up to
 * the deepest emitted one, reduced in one workgroup tile, in one dispatch.
 *
 * Group 0 (ours): 1 source texture (no load hook), 2… one storage texture per
 * emitted level (no store hook), 8 the size uniform (load hook without a size
 * hook). Group 1 is the tool's (`bindings`). Workgroup z is the array layer.
 * Identifiers starting with `ds_` are reserved, as are the entry point `main`
 * and the hook names.
 *
 * Workgroup memory holds unnormalised accumulations per channel (sums, weighted
 * sums, min/max or a custom combine); a level is normalised only when it is
 * emitted. Nothing is rounded between levels: each emitted level is the
 * reduction of its source block (`reducePyramid` in core is the oracle).
 */
function createTileKernel( options: KernelOptions ): string {

	const { format, dimension, reduction, edges, access, levels, count, load, store, size, bindings, output, channels, sourceFormat, f16Rounding } = tileOptions( options );
	const buffer = store === undefined && output === 'buffer';
	const half = store === undefined && formats[ format! ].quantization === 'float16';
	const packed = store === undefined && isPackedQuantization( formats[ format! ].quantization ) ? formats[ format! ].quantization as 'rgb9e5ufloat' | 'rg11b10ufloat' : null;
	const resolved = resolveReduction( reduction, channels );
	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const tile = tileSize( count );
	const side = workgroupSide( tile );
	const samples = tile / side;
	const zeroWeight = edges === 'zero-weight';
	// The value type (vec4f, or DsWide for 8 channels) and its channel accessors.
	const lane = lanes( channels );
	const { type: T, channel: ch, build, declarations } = lane;

	let code = declarations;
	if ( access === 'sample' ) code += '@group(0) @binding(0) var ds_sampler: sampler;\n';
	if ( load === undefined ) code += `@group(0) @binding(1) var ds_source: texture_2d${ suffix }<f32>;\n`;
	if ( half ) code += float16Wgsl( f16Rounding );
	if ( buffer ) code += bufferOutputWgsl( format!, true, false );
	// Packed floats are quantized by our encoder on either path, so stored values never depend on the GPU's conversion.
	if ( packed !== null ) code += packedFloatWgsl( packed );
	if ( store === undefined && ! buffer ) levels.forEach( ( level, i ) => code += `@group(0) @binding(${ i + 2 }) var ds_out${ level }: texture_storage_2d${ suffix }<${ format }, write>;\n` );
	if ( load !== undefined && size === undefined ) code += `struct DsParams { size: vec2u }\n@group(0) @binding(${ sizeBinding }) var<uniform> ds_params: DsParams;\n`;
	const region = options.region === true;
	if ( region ) code += regionWgsl();
	if ( bindings !== '' ) code += `\n// Tool bindings (group 1)\n${ bindings }\n`;
	if ( load !== undefined ) code += `\n// load hook\n${ load }\n`;
	if ( store !== undefined ) code += `\n// store hook\n${ store }\n`;
	if ( size !== undefined ) code += `\n// size hook\n${ size }\n`;

	code += `
var<workgroup> ds_tile: array<${ T }, ${ tile * tile }>;
`;

	if ( load === undefined ) {

		const splat = sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '';
		code += `
fn load(texel: vec2i, layer: u32) -> vec4f {
	return textureLoad(ds_source, texel${ array ? ', i32(layer)' : '' }, 0)${ splat };
}
fn ds_size() -> vec2u { return textureDimensions(ds_source); }
`;

	} else {

		code += `fn ds_size() -> vec2u { return ${ size === undefined ? 'ds_params.size' : 'size()' }; }\n`;

	}

	// Combine two accumulations, prepare a loaded texel (weighted channels carry
	// w·v), and normalise an accumulation for emission.
	let prepare = ( v: string ) => v;
	let emitted = false;
	if ( resolved.custom !== undefined ) {

		code += `${ ( reduction as { wgsl: string } ).wgsl }\nfn ds_combine(a: ${ T }, b: ${ T }) -> ${ T } { return reduce(a, b); }\n`;

	} else {

		const ops = resolved.ops;
		const weight = ops.indexOf( 'weight' );
		const terms = ops.map( ( op, c ) => op === 'min' || op === 'max' ? `${ op }(${ ch( 'a', c ) }, ${ ch( 'b', c ) })` : `${ ch( 'a', c ) } + ${ ch( 'b', c ) }` );
		// Payload folds (argmin/argmax with payload channels) need a comparison first.
		code += payloadCombine( ops, lane, c => terms[ c ] ) ?? `fn ds_combine(a: ${ T }, b: ${ T }) -> ${ T } { return ${ build( terms ) }; }\n`;

		if ( ops.includes( 'weighted' ) ) {

			code += `fn ds_prepare(v: ${ T }) -> ${ T } { return ${ build( ops.map( ( op, c ) => op === 'weighted' ? `${ ch( 'v', c ) } * ${ ch( 'v', weight ) }` : ch( 'v', c ) ) ) }; }\n`;
			prepare = v => `ds_prepare(${ v })`;

		}

		if ( ops.includes( 'average' ) || ops.includes( 'weighted' ) ) {

			emitted = true;
			code += `fn ds_emit(v: ${ T }, n: f32) -> ${ T } {
	return ${ build( ops.map( ( op, c ) => {

				if ( op === 'average' ) return `${ ch( 'v', c ) } / n`;
				if ( op === 'weighted' ) return `select(0.0, ${ ch( 'v', c ) } / ${ ch( 'v', weight ) }, ${ ch( 'v', weight ) } != 0.0)`;
				return ch( 'v', c );

			} ) ) };
}
`;

		}

	}

	const averages = resolved.ops?.includes( 'average' ) ?? false;
	// Level sizes: ⌈n / 2^l⌉ ('zero-weight') or ⌊n / 2^l⌋, at least 1 ('clamp').
	const levelSize = ( level: number ) => zeroWeight ? `((ds_size() + vec2u(${ ( 1 << level ) - 1 }u)) >> vec2u(${ level }u))` : `max(ds_size() >> vec2u(${ level }u), vec2u(1u))`;
	// Texels behind an output texel: the in-bounds part of its block, or the whole clamped block.
	const texelCount = ( level: number, pos: string ) => ! averages ? '1.0' : zeroWeight ? `ds_count(${ level }u, ${ pos })` : `${ 4 ** level }.0`;
	if ( averages && zeroWeight ) code += `
fn ds_count(level: u32, texel: vec2u) -> f32 {
	let lo = texel << vec2u(level);
	let n = min(ds_size(), lo + vec2u(1u << level)) - lo;
	return f32(n.x * n.y);
}
`;

	// 'zero-weight': b, c and d count only when inside the level they come from (a
	// always is, for any output inside its own level). 'clamp': all four count.
	// (select() takes vectors, so DsWide selects per half.)
	if ( channels === 8 ) code += zeroWeight ? `
fn ds_reduce4(a: DsWide, b: DsWide, c: DsWide, d: DsWide, inside: vec2<bool>) -> DsWide {
	let ab = ds_combine(a, b);
	let cd = ds_combine(c, d);
	let top = DsWide(select(a.lo, ab.lo, inside.x), select(a.hi, ab.hi, inside.x));
	let bottom = DsWide(select(c.lo, cd.lo, inside.x), select(c.hi, cd.hi, inside.x));
	let both = ds_combine(top, bottom);
	return DsWide(select(top.lo, both.lo, inside.y), select(top.hi, both.hi, inside.y));
}
` : `
fn ds_reduce4(a: DsWide, b: DsWide, c: DsWide, d: DsWide) -> DsWide {
	return ds_combine(ds_combine(a, b), ds_combine(c, d));
}
`;
	else code += zeroWeight ? `
fn ds_reduce4(a: vec4f, b: vec4f, c: vec4f, d: vec4f, inside: vec2<bool>) -> vec4f {
	let top = select(a, ds_combine(a, b), inside.x);
	let bottom = select(c, ds_combine(c, d), inside.x);
	return select(top, ds_combine(top, bottom), inside.y);
}
` : `
fn ds_reduce4(a: vec4f, b: vec4f, c: vec4f, d: vec4f) -> vec4f {
	return ds_combine(ds_combine(a, b), ds_combine(c, d));
}
`;

	// Level 1 from a 2×2 block of the source. Coordinates are clamped, so the load
	// hook only sees in-bounds texels; under 'zero-weight' the clamped duplicates
	// are then left out. 'sample' takes one bilinear sample at the block's shared
	// corner (weights ½, ½): clamp-to-edge addressing repeats the edge texel past
	// the image exactly as clamped loads do. The tile holds sums, hence × 4.
	if ( access === 'sample' ) code += `
fn ds_first(p: vec2u, layer: u32) -> vec4f {
	let uv = vec2f(p * 2u + vec2u(1u)) / vec2f(ds_size());
	return textureSampleLevel(ds_source, ds_sampler, uv${ array ? ', i32(layer)' : '' }, 0.0)${ sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '' } * 4.0;
}
`;
	else code += `
fn ds_first(p: vec2u, layer: u32) -> ${ T } {
	let size = ds_size();
	let last = vec2i(size) - vec2i(1);
	let lo = min(vec2i(p * 2u), last);
	let hi = min(vec2i(p * 2u + vec2u(1u)), last);
	let a = ${ prepare( 'load(lo, layer)' ) };
	let b = ${ prepare( 'load(vec2i(hi.x, lo.y), layer)' ) };
	let c = ${ prepare( 'load(vec2i(lo.x, hi.y), layer)' ) };
	let d = ${ prepare( 'load(hi, layer)' ) };
	return ds_reduce4(a, b, c, d${ zeroWeight ? ', p * 2u + vec2u(1u) < size' : '' });
}
`;

	const emit = ( level: number, pos: string ) => {

		const value = emitted ? `ds_emit(value, ${ texelCount( level, pos ) })` : 'value';
		if ( buffer ) return `ds_put(${ levels.indexOf( level ) }u, vec2u(${ pos }), group.z, ${ value });`;
		if ( packed !== null && store === undefined ) {

			const [ pack, unpack ] = packed === 'rgb9e5ufloat' ? [ 'ds_pack9e5', 'ds_unpack9e5' ] : [ 'ds_pack11_11_10', 'ds_unpack11_11_10' ];
			return `textureStore(ds_out${ level }, vec2i(${ pos })${ array ? ', i32(group.z)' : '' }, vec4f(${ unpack }(${ pack }((${ value }).xyz)), 1.0));`;

		}

		// float16 outputs are rounded before the store (f16Rounding), as Downsampler does:
		// textureStore truncates f32 → f16 on Apple and D3D12 (results/probes/f16-store-rounding.mjs).
		if ( half ) return `textureStore(ds_out${ level }, vec2i(${ pos })${ array ? ', i32(group.z)' : '' }, ds_round_f16(${ value }));`;

		return store !== undefined ? `store(${ level }u, vec2i(${ pos }), group.z, ${ value });` : `textureStore(ds_out${ level }, vec2i(${ pos })${ array ? ', i32(group.z)' : '' }, ${ value });`;

	};

	// Partial updates offset the workgroup grid and the layer, and store only the region's texels.
	const entry = region ? `@builtin(workgroup_id) ds_group: vec3u) {
	let group = ds_group + ds_region.origin.xyz;` : '@builtin(workgroup_id) group: vec3u) {';
	const inside = ( level: number ) => region ? ` && ds_inside(${ levels.indexOf( level ) }u, pos)` : '';
	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, ${ entry }
	var value: ${ T };
`;
	if ( levels.includes( 1 ) ) code += `	let size1 = ${ levelSize( 1 ) };\n`;

	for ( let i = 0; i < samples * samples; i ++ ) {

		code += `
	{
		let p = local.xy + vec2u(${ ( i % samples ) * side }u, ${ Math.floor( i / samples ) * side }u);
		let pos = group.xy * ${ tile }u + p;
		value = ds_first(pos, group.z);
${ count > 1 ? `		ds_tile[p.y * ${ tile }u + p.x] = value;\n` : '' }${ levels.includes( 1 ) ? `		if (all(pos < size1)${ inside( 1 ) }) { ${ emit( 1, 'pos' ) } }\n` : '' }	}
`;

	}

	for ( let level = 2; level <= count; level ++ ) {

		const extent = tile >> ( level - 1 );
		const at = ( dx: number, dy: number ) => `ds_tile[(lo.y + ${ dy }u) * ${ tile }u + lo.x + ${ dx }u]`;
		code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let lo = local.xy * 2u;
		value = ds_reduce4(${ at( 0, 0 ) }, ${ at( 1, 0 ) }, ${ at( 0, 1 ) }, ${ at( 1, 1 ) }${ zeroWeight ? `, group.xy * ${ extent * 2 }u + lo + vec2u(1u) < ${ levelSize( level - 1 ) }` : '' });
	}
`;
		if ( level < count ) code += `	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) { ds_tile[local.y * ${ tile }u + local.x] = value; }
`;
		if ( levels.includes( level ) ) code += `	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let pos = group.xy * ${ extent }u + local.xy;
		if (all(pos < ${ levelSize( level ) })${ inside( level ) }) { ${ emit( level, 'pos' ) } }
	}
`;

	}

	return code + '}\n';

}
