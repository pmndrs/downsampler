/**
 * `downsampler/mipmap`: in-place `'average'` mip chains, in one self-contained module.
 *
 * This file imports nothing, so an engine can copy its build output (`dist/mipmap.js`,
 * plain ES2022) into its own tree, as three.js does for `WebGPUTextureMipmapUtils`. It is
 * the subset of the library a renderer's mipmap generator needs:
 * - {@link createMipmapKernel}: the WGSL of one batch, the 'exact' kernel or the tail kernel,
 *   reading level n and writing levels n + 1 … n + count of the same texture.
 * - {@link planMipmapBatches}: how a chain is split into those batches.
 * - {@link kernelDefaults}: levels per dispatch and tile sizes for a device, the rules
 *   `Downsampler` itself uses.
 * - {@link mipmapFormats} and {@link mipmapAccess}: the formats it writes, and how it reads them.
 *
 * Everything here is a pruned copy of the full generator (`createKernel` in
 * `src/webgpu/kernel.ts`, `planBatches` in `src/core/plan.ts`), and must stay byte for byte
 * equal to it: `test/core/mipmap.test.js` compares every option set, and
 * `test/core/golden.test.js` pins both to the WGSL in `test/golden/mipmap/`. A vendored copy
 * is checked against those goldens with `node scripts/check-vendored.mjs <file>`
 * (SPEC.md §10).
 *
 * Results match render-pass mipmaps ('exact' semantics, SPEC.md §2.1): a batch's first level
 * is one bilinear sample at each texel centre in normalized coordinates, so odd sizes give
 * what a render pass gives. Further levels are exact 2:1 reductions in workgroup memory, so a
 * batch ends where a non-unit dimension is odd; once a level fits one workgroup, a tail kernel
 * finishes the chain by the same rule, odd sizes included. Every level is quantized as
 * stored: normalized formats in integer units with ties to even, float16 rounded to nearest
 * even in the shader.
 */

/** How values are rounded between levels, per {@link mipmapFormats} entry. */
export type MipmapQuantization = 'unorm8' | 'snorm8' | 'unorm16' | 'snorm16' | 'float16' | 'float32';

export interface MipmapFormatInfo {
	quantization: MipmapQuantization;
	/** Whether a filtering sampler reads it without optional features (else it loads, see {@link mipmapAccess}). */
	filterable: boolean;
	/** Device feature its storage needs, if any. */
	feature?: 'bgra8unorm-storage' | 'texture-formats-tier1';
}

/**
 * The formats this module writes: storage-capable formats with a float view (a subset of
 * `formats` in `src/core/formats.ts`). sRGB, packed, rgb10a2unorm and integer formats need
 * the full library (`Downsampler`).
 */
export const mipmapFormats = {
	rgba8unorm: { quantization: 'unorm8', filterable: true },
	rgba16float: { quantization: 'float16', filterable: true },
	rgba8snorm: { quantization: 'snorm8', filterable: true },
	bgra8unorm: { quantization: 'unorm8', filterable: true, feature: 'bgra8unorm-storage' },
	r8unorm: { quantization: 'unorm8', filterable: true, feature: 'texture-formats-tier1' },
	rg8unorm: { quantization: 'unorm8', filterable: true, feature: 'texture-formats-tier1' },
	r16float: { quantization: 'float16', filterable: true, feature: 'texture-formats-tier1' },
	rg16float: { quantization: 'float16', filterable: true, feature: 'texture-formats-tier1' },
	r8snorm: { quantization: 'snorm8', filterable: true, feature: 'texture-formats-tier1' },
	rg8snorm: { quantization: 'snorm8', filterable: true, feature: 'texture-formats-tier1' },
	r16unorm: { quantization: 'unorm16', filterable: false, feature: 'texture-formats-tier1' },
	rg16unorm: { quantization: 'unorm16', filterable: false, feature: 'texture-formats-tier1' },
	rgba16unorm: { quantization: 'unorm16', filterable: false, feature: 'texture-formats-tier1' },
	r16snorm: { quantization: 'snorm16', filterable: false, feature: 'texture-formats-tier1' },
	rg16snorm: { quantization: 'snorm16', filterable: false, feature: 'texture-formats-tier1' },
	rgba16snorm: { quantization: 'snorm16', filterable: false, feature: 'texture-formats-tier1' },
	r32float: { quantization: 'float32', filterable: false },
	// No storage in compatibility mode (it needs 'core-features-and-limits').
	rg32float: { quantization: 'float32', filterable: false },
	rgba32float: { quantization: 'float32', filterable: false }
} as const satisfies Record<string, MipmapFormatInfo>;

export type MipmapFormat = keyof typeof mipmapFormats;

/**
 * How a batch reads its source level: 'sample' (one hardware bilinear sample) where the
 * format is filterable, float32 included when the device has `float32-filterable`; else
 * 'load' (`textureLoad` with the bilinear weights computed in the shader). Both give the
 * 'exact' rule. Pass `features` as `device.features` (anything with `has`).
 */
export function mipmapAccess( format: MipmapFormat, features?: { has( feature: string ): boolean } ): 'sample' | 'load' {

	const info: MipmapFormatInfo = mipmapFormats[ format ];
	return info.filterable || ( info.quantization === 'float32' && features?.has( 'float32-filterable' ) === true ) ? 'sample' : 'load';

}

/** Normalized formats are reduced in integer units, so 2:1 averages are exact and ties round to even. */
const integerDomains = {
	unorm8: { scale: 255, min: 0, max: 255 },
	snorm8: { scale: 127, min: - 127, max: 127 },
	unorm16: { scale: 65535, min: 0, max: 65535 },
	snorm16: { scale: 32767, min: - 32767, max: 32767 }
} as const;

/** Largest first output of a tail batch: one 32×32 workgroup tile. */
const tailTile = 32;

const mipSize = ( size: number, level: number ) => Math.max( 1, Math.floor( size / 2 ** level ) );

/** First-output tile of a batch of `count` levels: 8×8 up to four levels, 16×16 for five, 32×32 for six. */
const tileSize = ( count: number ) => Math.max( 8, 1 << ( count - 1 ) );

/** Invocations per workgroup edge; each writes ( tile / side )² first-output texels. */
const workgroupSide = ( tile: number ) => Math.min( tile, 16 );

/** Tail tile: the first output's larger side, rounded up to 8, 16 or 32. */
const tailTileFor = ( width: number, height: number ) => Math.min( tailTile, Math.max( 8, 2 ** Math.ceil( Math.log2( Math.max( width, height ) ) ) ) );

/** What {@link kernelDefaults} resolves for a device. */
export interface KernelDefaults {
	/** Most levels one dispatch writes to storage textures: one binding per level, so min(6, maxStorageTexturesPerShaderStage); 4 below 256 invocations per workgroup. */
	levelsPerDispatch: number;
	/** The same where levels aren't storage textures (the library's buffer output path): 6, or 4 below 256 invocations. */
	bufferLevelsPerDispatch: number;
	/** Smallest first-output tile of 'exact' batches (`exactMinTile` of {@link planMipmapBatches}). */
	exactTile: 8 | 16 | 32;
	/** Largest tail tile: 32, or 8 below 256 invocations per workgroup. */
	tailMaxTile: 8 | 32;
	/** Whether chains finish in a tail kernel. Always true: the tail was 0.81–1.02× of a batch per odd size on every GPU measured (SPEC.md §2.1). */
	tail: true;
}

/**
 * The library's defaults for a device (`Downsampler` resolves its options from these):
 * - Five and six levels per dispatch take 16 × 16 invocations, so they need
 *   `maxComputeInvocationsPerWorkgroup` ≥ 256 (compatibility mode defaults to 128).
 * - 'exact' tiles: 32×32 (16 × 16 invocations, 2 × 2 texels each) was fastest on NVIDIA and
 *   Intel, 16×16 on Apple GPUs; 8×8 below 256 invocations and on fallback (software)
 *   adapters, which take 30 s or more to compile a 32×32-tile pipeline.
 *
 * Pass `device.limits` and `device.adapterInfo` (or the adapter's). Pure: no WebGPU globals.
 */
export function kernelDefaults( limits: { readonly maxComputeInvocationsPerWorkgroup: number; readonly maxStorageTexturesPerShaderStage: number }, adapterInfo?: { readonly vendor?: string; readonly isFallbackAdapter?: boolean } | null ): KernelDefaults {

	const wide = limits.maxComputeInvocationsPerWorkgroup >= 256;
	const invocations = wide ? 6 : 4;
	const software = adapterInfo?.isFallbackAdapter === true;
	return {
		levelsPerDispatch: Math.min( invocations, limits.maxStorageTexturesPerShaderStage ),
		bufferLevelsPerDispatch: invocations,
		exactTile: wide && ! software ? ( adapterInfo?.vendor === 'apple' ? 16 : 32 ) : 8,
		tailMaxTile: wide ? tailTile : 8,
		tail: true
	};

}

/** Options of {@link planMipmapBatches}. */
export interface MipmapPlanOptions {
	/** Levels of the texture, level 0 included. Default the full chain. */
	mipLevelCount?: number;
	/** Most levels one dispatch writes (1–6), e.g. {@link KernelDefaults.levelsPerDispatch}. */
	levelsPerDispatch: number;
	/** Finish the chain in one workgroup once a first output fits `tailMaxTile`². Default true. */
	tail?: boolean;
	/** Largest tail tile: 8, 16 or 32. Default 32. */
	tailMaxTile?: number;
	/** Smallest tile of 'exact' batches: 8, 16 or 32. Default 8. */
	exactMinTile?: number;
	/** The same as `exactMinTile`, under its {@link KernelDefaults} name, so `planMipmapBatches( w, h, kernelDefaults( … ) )` works. */
	exactTile?: number;
}

/** One dispatch of an in-place chain (the library's `Batch`). */
export interface MipmapBatch {
	/** 'exact': first output sampled, later levels 2:1 in workgroup memory. 'tail': one workgroup, every level by the 'exact' rule. */
	kernel: 'exact' | 'tail';
	/** Level read. */
	sourceMipLevel: number;
	/** First level written (`sourceMipLevel + 1`). */
	firstMipLevel: number;
	/** Levels written: `firstMipLevel` … `firstMipLevel + count - 1`. */
	count: number;
	/** Size of the first level written. */
	outputWidth: number;
	outputHeight: number;
	/** First-output texels per workgroup edge: the kernel's `tile`. */
	tile: number;
	/** Workgroups to dispatch: `groupsX` × `groupsY` × array layers. */
	groupsX: number;
	groupsY: number;
}

/**
 * Splits the mip chain of a `width` × `height` level 0 into dispatches. A batch reads one
 * level and writes up to `levelsPerDispatch` levels; it ends early where a non-unit
 * dimension of its last level is odd. With `tail` (default), once a first output fits
 * `tailMaxTile`², one tail batch finishes the chain. Equal to `planBatches` in core with
 * `baseMipLevel` 0 and `oddSizes` 'exact'.
 */
export function planMipmapBatches( width: number, height: number, options: MipmapPlanOptions ): MipmapBatch[] {

	const { levelsPerDispatch, tail = true, tailMaxTile = tailTile } = options;
	const mipLevelCount = options.mipLevelCount ?? Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;
	if ( options.exactMinTile !== undefined && options.exactTile !== undefined && options.exactMinTile !== options.exactTile ) throw new Error( 'downsampler: exactMinTile and exactTile name the same option; pass one.' );
	const exactMinTile = options.exactMinTile ?? options.exactTile ?? 8;
	if ( exactMinTile !== 8 && exactMinTile !== 16 && exactMinTile !== 32 ) throw new Error( 'downsampler: exactMinTile must be 8, 16 or 32.' );
	if ( ! Number.isInteger( levelsPerDispatch ) || levelsPerDispatch < 1 || levelsPerDispatch > 6 ) throw new Error( 'downsampler: levelsPerDispatch must be an integer from 1 to 6.' );

	const batches: MipmapBatch[] = [];
	const lastLevel = mipLevelCount - 1;

	for ( let level = 1; level <= lastLevel; ) {

		const outputWidth = mipSize( width, level ), outputHeight = mipSize( height, level );
		let count = 1, cw = outputWidth, ch = outputHeight;

		// Continue in workgroup memory only while each non-unit dimension halves exactly.
		while ( count < levelsPerDispatch && level + count <= lastLevel && ( cw === 1 || cw % 2 === 0 ) && ( ch === 1 || ch % 2 === 0 ) ) {

			cw = Math.max( 1, cw / 2 );
			ch = Math.max( 1, ch / 2 );
			count ++;

		}

		// The tail: one workgroup finishes what would otherwise take more batches.
		const tailCount = Math.min( levelsPerDispatch, lastLevel - level + 1 );
		if ( tail && count < tailCount && outputWidth <= tailMaxTile && outputHeight <= tailMaxTile ) {

			batches.push( { kernel: 'tail', sourceMipLevel: level - 1, firstMipLevel: level, count: tailCount, outputWidth, outputHeight, tile: tailTileFor( outputWidth, outputHeight ), groupsX: 1, groupsY: 1 } );
			level += tailCount;
			continue;

		}

		const tile = Math.max( exactMinTile, tileSize( count ) );
		batches.push( { kernel: 'exact', sourceMipLevel: level - 1, firstMipLevel: level, count, outputWidth, outputHeight, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ) } );
		level += count;

	}

	return batches;

}

/** Options of {@link createMipmapKernel}: a {@link MipmapBatch}'s `count`, `tile` and kind, plus the texture's. */
export interface MipmapKernelOptions {
	/** Format of the texture (source and outputs). */
	format: MipmapFormat;
	/** '2d', or '2d-array' (every layer; cube maps as 6 layers; workgroup z is the layer). Default '2d'. */
	dimension?: '2d' | '2d-array';
	/** Levels the dispatch writes, 1–6. */
	count: number;
	/** How the first output reads the source ({@link mipmapAccess}). Default 'sample'. */
	access?: 'sample' | 'load';
	/** A tail kernel (`kernel: 'tail'`): one workgroup writes all `count` levels. Default false. */
	tail?: boolean;
	/** First-output tile edge, 8, 16 or 32. Default what `count` needs (8 up to four levels), 32 for a tail. */
	tile?: number;
	/**
	 * float16: 'nearest' (default) rounds to nearest even in the shader, the same on every GPU;
	 * 'device' uses `pack2x16float`, whose rounding varies (D3D12 truncates).
	 */
	f16Rounding?: 'nearest' | 'device';
}

/** WGSL of `ds_round_f16(v: vec4f) -> vec4f`: each channel rounded to a half, as f32 (`float16Wgsl( rounding, false )` in the full generator). */
function float16RoundWgsl( rounding: 'nearest' | 'device' ): string {

	// In the half normal range the f32 bits are rounded to 10 mantissa bits with integer math (a
	// carry into the exponent is correct); below it, round( x · 2^24 ) · 2^-24. Not ( x + c ) - c,
	// which fast math folds to x on Metal.
	return rounding === 'device' ? `fn ds_round_f16(v: vec4f) -> vec4f { return vec4f(unpack2x16float(pack2x16float(v.xy)), unpack2x16float(pack2x16float(v.zw))); }
` : `fn ds_round_f16(v: vec4f) -> vec4f {
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
 * The WGSL of one in-place batch. Bindings (group 0): 0 a filtering sampler ('sample' only),
 * 1 the source level (`texture_2d<f32>`, or `texture_2d_array`; 'float' sample type for
 * 'sample', 'unfilterable-float' for 'load'), 2 … count + 1 one write-only storage texture
 * of `format` per output level. Dispatch `groupsX` × `groupsY` × layers; entry point `main`.
 *
 * Each workgroup writes a tile×tile block of the first output. Adjacent invocations read and
 * store adjacent texels (a 2×2 block per invocation measured 30–45 % slower). Later levels are
 * 2:1 reductions of the tile in workgroup memory, with clamped reads for unit dimensions and
 * partial tiles. A tail kernel is one workgroup: the first output as above, then each further
 * level from the previous one by the same rule at any parity, with shader bilinear weights.
 *
 * Equal to `createKernel( { ...options, f16Pack: false } )` of `downsampler/webgpu`.
 */
export function createMipmapKernel( options: MipmapKernelOptions ): string {

	const { format, dimension = '2d', count, tail = false, f16Rounding = 'nearest' } = options;
	const info: MipmapFormatInfo | undefined = Object.prototype.hasOwnProperty.call( mipmapFormats, format ) ? mipmapFormats[ format ] : undefined;
	if ( info === undefined ) throw new Error( `downsampler: format ${ JSON.stringify( format ) } is not a mipmap format.` );
	if ( dimension !== '2d' && dimension !== '2d-array' ) throw new Error( `downsampler: view dimension ${ JSON.stringify( dimension ) } is not supported.` );
	if ( ! Number.isInteger( count ) || count < 1 || count > 6 ) throw new Error( 'downsampler: count must be an integer from 1 to 6.' );
	const access = options.access ?? 'sample';
	if ( access !== 'sample' && access !== 'load' ) throw new Error( `downsampler: unknown access ${ JSON.stringify( access ) }.` );
	if ( f16Rounding !== 'nearest' && f16Rounding !== 'device' ) throw new Error( `downsampler: f16Rounding must be 'nearest' or 'device', got ${ JSON.stringify( f16Rounding ) }.` );
	const tile = options.tile ?? ( tail ? tailTile : tileSize( count ) );
	if ( tile !== 8 && tile !== 16 && tile !== 32 ) throw new Error( `downsampler: a tile is 8, 16 or 32, got ${ tile }.` );
	if ( ! tail && tile < tileSize( count ) ) throw new Error( `downsampler: ${ count } levels need a tile of at least ${ tileSize( count ) }, got ${ tile }.` );

	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const quantization = info.quantization;
	const domain = quantization === 'float16' || quantization === 'float32' ? undefined : integerDomains[ quantization ];
	// Single-channel formats read as ( v, 0, 0, 1 ): splat the channel, as for separate sources.
	const splat = /^r\d/.test( format ) ? '.xxxx' : '';
	const side = workgroupSide( tile );
	const samples = tile / side;

	let code = access === 'sample' ? '@group(0) @binding(0) var imageSampler: sampler;\n' : '';
	code += `@group(0) @binding(1) var source: texture_2d${ suffix }<f32>;
var<workgroup> tile: array<vec4f, ${ tile * tile }>;
`;
	if ( quantization === 'float16' ) code += float16RoundWgsl( f16Rounding );
	for ( let i = 0; i < count; i ++ ) code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;

	// Intermediate levels are quantized exactly as they are stored. Normalized values stay in
	// integer units, so 2:1 averages are exact and ties round to even (with k/255 floats, float
	// error turns ties into round-ups that drift along 1D tails). The parity test uses i32:
	// converting a negative float to u32 is undefined (snorm).
	if ( domain !== undefined ) {

		code += `
const SCALE = ${ domain.scale }.0;
fn quantize(value: vec4f) -> vec4f {
	let scaled = clamp(value, vec4f(${ domain.min }.0), vec4f(${ domain.max }.0));
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

	} else {

		code += `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return value; }
`;

	}

	if ( access === 'load' ) {

		// Normalized texels are rounded back to exact integer units, so later sums are exact.
		code += `
fn load(p: vec2u, layer: u32) -> vec4f {
	let v = textureLoad(source, vec2i(p)${ array ? ', i32(layer)' : '' }, 0)${ splat } * SCALE;
	return ${ domain !== undefined ? 'round(v)' : 'v' };
}
`;

	}

	code += `
fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
`;

	if ( access === 'sample' ) {

		code += `	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv${ array ? ', i32(layer)' : '' }, 0.0)${ splat } * SCALE);
`;

	} else {

		// Bilinear weights at the output texel centre, as the sampler applies them but without its
		// fixed-point weight rounding. The sample position (x + 0.5) * S / n - 0.5 is
		// ((2x + 1) * S - n) / 2n: integer math keeps the base texel exact (GPU float division
		// isn't correctly rounded), and gives weights of exactly 0 or 0.5 at even sizes. A
		// same-size level reads its texel alone (never in place; the full generator's copies).
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

	}

	code += '}\n';

	// Later sizes are derived from the first (floored halves, at least 1), not queried:
	// textureDimensions on a storage texture is a resource query per call.
	const dims = ( i: number ) => i === 0 ? 'textureDimensions(mip0)' : `max(size0 >> vec2u(${ i }u), vec2u(1u))`;
	const store = ( i: number, pos: string ) => `textureStore(mip${ i }, vec2i(${ pos })${ layer }, value / SCALE);`;

	if ( tail ) {

		// The same rule from a level held in the tile (scaled, quantized units).
		code += `
fn tileAt(p: vec2u) -> vec4f { return tile[p.y * ${ tile }u + p.x]; }
fn tailStep(pos: vec2u, sourceSize: vec2u, size: vec2u) -> vec4f {
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(tileAt(lo), tileAt(vec2u(hi.x, lo.y)), f.x);
	let bottom = mix(tileAt(vec2u(lo.x, hi.y)), tileAt(hi), f.x);
	return quantize(mix(top, bottom, f.y));
}
`;

	}

	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
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
		if (all(pos0 < size0)) { ${ store( 0, 'pos0' ) } }
	}
`;

	}

	for ( let i = 1; i < count; i ++ ) {

		if ( tail ) {

			// The first output fits the tile, so every later level fits half of it, at most the
			// workgroup's side: one texel per invocation.
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
		${ store( i, 'local.xy' ) }
	}
`;

		} else {

			const extent = tile >> i;
			const at = ( x: string, y: string ) => `tile[${ y }.y * ${ tile }u + ${ x }.x]`;
			code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = ${ dims( i - 1 ) } - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = quantize((${ at( 'lo', 'lo' ) } + ${ at( 'hi', 'lo' ) } + ${ at( 'lo', 'hi' ) } + ${ at( 'hi', 'hi' ) }) * 0.25);
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		tile[local.y * ${ tile }u + local.x] = value;
		let pos = group.xy * ${ extent }u + local.xy;
		if (all(pos < ${ dims( i ) })) { ${ store( i, 'pos' ) } }
	}
`;

		}

	}

	return code + '}\n';

}
