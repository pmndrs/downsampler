import { bufferWritable, formats, isIntFormat, isSupportedFormat, sampleKind, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { outputBufferLayout, planBatches, type Batch, type BatchKernel, type OddSizePolicy } from '../core/plan.js';
import { needsTileEdges, reductionKey, type Reduction } from '../core/reduction.js';
import { layersProblem, planRegions, rectProblem, type BatchRegion, type Rect } from '../core/region.js';
import { isCompatibilityTexture } from './compat.js';
import { createKernel, kernelLayoutEntries, outputBufferBinding, outputLayoutBinding, outputLayoutBytes, regionBinding, regionBytes, srgbTable, srgbTableBinding, usesSrgbTable, type F16Rounding, type KernelOptions, type OutputPath, type SourceAccess, type ViewDimension } from './kernel.js';
import { requiredLimits } from './limits.js';
import { asTextureLevel, bindingViewDimension, levelSizeOf, sourceShapeReason, sourceTextureReason, sourceView, type TextureLevel } from './source.js';
import { retireBuffers } from './tool.js';
import { kernelDefaults } from '../mipmap.js';

/** Options of `new Downsampler( device, options )`. Every option is validated by the constructor. See docs/guide.md. */
export interface DownsamplerOptions {
	/**
	 * Output levels per dispatch. Defaults to the most the path allows: on the
	 * storage path min(6, maxStorageTexturesPerShaderStage) (six needs a device
	 * created with a raised limit — see {@link requiredLimits}); on
	 * the buffer path 6 at any storage texture limit. 5 and 6 levels also need
	 * 256 invocations per workgroup (compatibility mode defaults to 128). Above
	 * the storage limit only with output 'buffer'. Given, it sets both paths, so
	 * below 6 storage textures `{ levelsPerDispatch: other.levelsPerDispatch }`
	 * plans sRGB and the other buffer-path formats unlike `other` (whose
	 * {@link Downsampler.bufferLevelsPerDispatch} is 6), and their chains differ
	 * in the last bit where one stores a level the other keeps in the tile.
	 */
	levelsPerDispatch?: number;
	/**
	 * How levels are written (SPEC.md §6.1 Buffer output path):
	 * - 'storage': write-only storage textures, one binding per level.
	 * - 'buffer': packed texels into one storage buffer, then `copyBufferToTexture`
	 *   per level. Needs COPY_DST usage and a format with 4-byte-multiple texels;
	 *   writes formats that can't be storage textures, and isn't limited by
	 *   maxStorageTexturesPerShaderStage. Can't record into a caller's compute pass.
	 * - 'auto' (default): storage whenever the device can write the format as a
	 *   storage texture (the texture then needs STORAGE_BINDING); the buffer path
	 *   only for formats it can't: rgb9e5ufloat, rg11b10ufloat without
	 *   texture-formats-tier1, rg32float in compatibility mode.
	 */
	output?: 'auto' | OutputPath;
	/**
	 * How 'average' reads the source level of each batch: one hardware bilinear
	 * sample, or four `textureLoad`s with weights computed in the shader. Default
	 * 'sample'. Unfilterable formats (float32, depth) and fold reductions always load.
	 * Samplers round bilinear weights to 8 bits in vendor-specific ways (NVIDIA per 2D
	 * tap, Intel by truncation), so 'sample' results differ slightly between GPUs;
	 * 'load' is identical on every GPU and costs 2–31 % (results/windows/timings.md).
	 */
	averageAccess?: SourceAccess;
	/**
	 * The compatibility-mode path (SPEC.md §4 Compatibility mode): compatibility
	 * mode binds a texture through views of one mip range per dispatch, so each
	 * batch writes per-level scratch textures, and a copy dispatch per level moves
	 * them into the destination. 'auto' (default) takes it for textures of a
	 * device without 'core-features-and-limits' (their `textureBindingViewDimension`
	 * is defined only there); true always; false never. It applies to the storage
	 * path only: the buffer path already reads one level per dispatch and writes
	 * through copies, so it needs no scratch textures.
	 */
	compatibility?: boolean | 'auto';
	/**
	 * 'exact' chains: once a level fits one workgroup (32×32, or 8×8 below 256
	 * invocations per workgroup, as in compatibility mode), finish the chain in
	 * one dispatch (the tail kernel) instead of a batch per odd size. Results
	 * follow the same 'exact' rule, with shader bilinear weights in the tail levels
	 * (within one rgba8 step of the sampler's). Works on both output paths and in
	 * compatibility mode. Default true: 0.82–0.90× of 'exact' on an RTX 3060, 0.81–1.02×
	 * on an M5 Pro, neutral on Intel UHD (SPEC.md §2.1).
	 */
	tail?: boolean;
	/**
	 * float16 formats: how each level is rounded to a half, before it is stored and
	 * before the next level reads it. 'nearest' (default): round to nearest even in the
	 * shader, the same on every GPU. 'device': the device's own conversion
	 * (`pack2x16float`), which truncates on D3D12. See {@link F16Rounding}.
	 */
	f16Rounding?: F16Rounding;
	/**
	 * 'exact' batches: the smallest output tile per workgroup, 8, 16 or 32. Default 32
	 * (16 × 16 invocations, each writing 2 × 2 first-level texels) where a workgroup can
	 * have 256 invocations, 16 on Apple GPUs, else 8; also 8 on fallback (software)
	 * adapters, which compile 32×32-tile pipelines slowly. Results are the same; 8 × 8
	 * workgroups were up to 1.6× slower on Intel UHD and 1.3× on an RTX 3060 at render
	 * sizes (results/windows/timings.md), while 32 was 1.04–1.28× slower than 8 or 16 on
	 * an M5 Pro (results/render/m5-pro-render-tiles-*.json).
	 */
	exactTile?: number;
	/**
	 * Largest source ÷ destination size ratio per axis that `source.resample: 'area'`
	 * accepts. Default {@link maxResampleRatio} (8: at most 9 × 9 taps per texel of
	 * the first level). Any number ≥ 1; the taps grow with its square, and only the
	 * validation reads it (the kernel loops over each texel's span at run time).
	 */
	maxResampleRatio?: number;
}

/**
 * A separate source for the first destination level (`EncodeOptions.source`),
 * e.g. a depth buffer for Hi-Z: a {@link TextureLevel} (any float-sampleable 2D
 * format, including depth, read through its depth aspect; for an integer
 * destination, an integer format of the same signedness, values clamped to the
 * destination's range; with TEXTURE_BINDING usage and the destination's array
 * layer count) and how to resample it.
 */
export interface SourceOptions extends TextureLevel {
	/**
	 * 'area': the destination's first level may be any size (e.g. a power-of-two
	 * Hi-Z, 1024×512 or 2048×1024 from 1920×1080), up to `maxResampleRatio`
	 * ({@link DownsamplerOptions.maxResampleRatio}, default 8) times smaller per axis.
	 * Destination texel x covers [x·S/n, (x+1)·S/n) of the
	 * source (S source texels, n destination texels per axis): 'average' weights
	 * every source texel by its overlap, and folds combine every texel it overlaps,
	 * so min/max stay conservative at any ratio, up or down. Later levels follow
	 * `oddSizes`. Without it, the destination level must be the source's size or
	 * half of it. SPEC.md §4 Separate source.
	 */
	resample?: 'area';
}

/** Default of {@link DownsamplerOptions.maxResampleRatio}: the largest source ÷ destination ratio per axis with `resample: 'area'` (at most 9 taps per axis). */
export const maxResampleRatio = 8;

/** A separate source with its defaults filled in. */
interface ResolvedSource {
	texture: GPUTexture;
	mipLevel: number;
	resample: 'area' | undefined;
}

const resolveSource = ( source: GPUTexture | SourceOptions ): ResolvedSource => ( { ...asTextureLevel( source ), resample: 'texture' in source ? source.resample : undefined } );

/** Options of one chain: `encode`, `encodePass`, `generate` and `prepare`. Plans are cached per texture and option set. */
export interface EncodeOptions {
	/**
	 * In place: the source level, with outputs from `baseMipLevel + 1`. With
	 * `source`: the first level written. Default 0.
	 */
	baseMipLevel?: number;
	/**
	 * Destination levels involved. In place this includes the source level, so
	 * SPD's `mips = N` is `mipLevelCount: N + 1`. With `source`: the levels written.
	 * Default: the rest of the chain.
	 */
	mipLevelCount?: number;
	/**
	 * A separate source, e.g. a depth buffer for a Hi-Z pyramid (depth formats
	 * cannot be storage textures): a texture (its level 0), or `{ texture, mipLevel,
	 * resample }`. Destination level `baseMipLevel` must be the same size as the
	 * source level (a converting copy) or half of it (one reduction step), or any
	 * size with `resample: 'area'`; every destination level in range is written.
	 * Single-channel sources are splatted, so 'minmax' of depth gives (d, d).
	 */
	source?: GPUTexture | SourceOptions;
	/**
	 * 'exact' (default): odd, non-unit sizes end a batch and the next level is
	 * re-sampled with normalized coordinates, matching render-pass/WebGL mipmaps.
	 * 'spd': AMD SPD semantics, 'average' only. Each dispatch reduces up to
	 * `levelsPerDispatch` levels in one tile with clamped loads and ⌊n/2⌋ sizes
	 * (the last odd row or column is dropped), without rounding between its
	 * levels; the next dispatch reads the last stored level. Any size takes
	 * ⌈levels / levelsPerDispatch⌉ dispatches. Lower quality at odd sizes.
	 * Folds are rejected: under these semantics min/max are not a conservative
	 * Hi-Z. Works on both output paths and in compatibility mode.
	 * 'box': area-weighted odd sizes (nvpro_pyramid's weights): each source texel
	 * contributes in proportion to its overlap, so a 31×31 texture reduces to its
	 * true mean. Same batches as 'exact'; batches that read an odd level (and tail
	 * batches) load, with weights (n − x, n, x + 1) / (2n + 1) along an odd size
	 * 2n + 1. Folds are the same as under 'exact' (the same 2x … 2x+2 footprint).
	 * Works with the tail kernel, both output paths and compatibility mode.
	 * See SPEC.md §2.1.
	 */
	oddSizes?: OddSizePolicy;
	/**
	 * 'average' (default) matches render-pass mipmaps. 'min', 'max' and 'minmax'
	 * (even channels min, odd channels max) combine every texel an output covers,
	 * including the extra row/column of odd sizes, as Hi-Z needs. A custom
	 * `{ wgsl }` defines `fn reduce(a: vec4f, b: vec4f) -> vec4f`, which must be
	 * associative, commutative and idempotent. Sums, per-channel ops and custom
	 * sums have no 'exact' meaning and are rejected: use a `Reducer`.
	 * Integer formats (SPEC.md §6.1): 'average' is the exact mean rounded to nearest,
	 * ties to even; a custom fold takes and returns `vec4u` (uint formats) or `vec4i`
	 * (sint formats), e.g. a bitwise or for occupancy pyramids.
	 */
	reduction?: Reduction;
	/**
	 * Reduce only these array layers, `[ first, count ]` (2d-array destinations, cube
	 * faces as layers); the other layers are untouched. With `source`, the same layers
	 * of the source are read. Default: every layer. See SPEC.md §5.
	 */
	layers?: readonly [ number, number ];
	/**
	 * A partial update (SPD's rectInfo): the texels of level `baseMipLevel` (in place)
	 * or of the source level (with `source`) that changed since the chain was last
	 * generated. Each later level rewrites only the texels whose footprint touches the
	 * previous level's rewritten ones, so the result is bit-identical to a full
	 * recompute when the old chain was valid and nothing changed outside the rect;
	 * every other texel is untouched. Works with every odd-size policy, reduction,
	 * output path, the tail kernel, compatibility mode and separate sources.
	 * An empty rect (width or height 0) records nothing, as WebGPU's zero-size copies
	 * do; it must still lie inside the level (x ≤ width, y ≤ height).
	 * Each distinct rect and layer range costs a small uniform buffer and bind groups
	 * (the 16 most recent are cached per texture; older ones are destroyed once the
	 * queue has finished the work submitted so far). See SPEC.md §5.
	 */
	rect?: Rect;
	/**
	 * Buffer output path (sRGB, rgb9e5ufloat, …; SPEC.md §6.1): write through this buffer
	 * instead of one the Downsampler allocates and keeps per texture and option set. Share
	 * one across textures to bound memory (each chain's copies are recorded before the
	 * next dispatch writes it, and commands run in order; don't interleave batches of
	 * different chains when dispatching `prepare()` yourself). Needs STORAGE and COPY_SRC
	 * usage and at least {@link Downsampler.scratchSize} bytes. Never destroyed by the
	 * Downsampler. Ignored on the storage path.
	 */
	scratch?: GPUBuffer;
}

/** A `copyBufferToTexture` of one level, after a buffer-path batch. */
export interface BufferCopy {
	mipLevel: number;
	/** Byte offset in the batch's buffer: of the copied region's first texel (256-aligned for whole levels). */
	offset: number;
	bytesPerRow: number;
	rowsPerImage: number;
	/** Texture origin of the copy in texels (0, 0 for whole levels). */
	x: number;
	y: number;
	/** Copy extent in texels. */
	width: number;
	height: number;
	/** Array layers copied, `[ first, count ]` (every layer for whole levels). */
	layers: readonly [ number, number ];
}

/**
 * One dispatch of a chain, as `prepare()` returns it: dispatch it with
 * `pipeline`, `bindGroup` at group 0 and `dispatchWorkgroups( groupsX, groupsY,
 * layers )`, then (buffer path) record `copies` from `buffer` into the texture.
 */
export interface PreparedBatch extends Batch {
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
	/** How the batch writes its levels. */
	output: OutputPath;
	/** Buffer path: the buffer the dispatch writes, and the copies that follow it. */
	buffer?: GPUBuffer;
	copies?: readonly BufferCopy[];
	/**
	 * Compatibility-mode storage path only: this dispatch copies level
	 * `firstMipLevel` from its scratch texture into the destination (no reduction).
	 */
	copy?: boolean;
	/** Array layers the dispatch covers, `[ first, count ]`: workgroup z is the layer, `count` the z dispatch size. */
	layers: readonly [ number, number ];
	/**
	 * Partial updates (`rect` or `layers`): per level of the batch, the texels it
	 * writes, or null for none. `groupsX`/`groupsY` then count the workgroups that
	 * cover them, from the first workgroup `groupX`/`groupY` (in the batch's tiles).
	 */
	regions?: readonly ( Rect | null )[];
	groupX?: number;
	groupY?: number;
}

interface CachedPlan {
	source: GPUTexture | undefined;
	/** The caller's scratch buffer the plan writes (EncodeOptions.scratch), if any. */
	scratch: GPUBuffer | undefined;
	batches: PreparedBatch[];
	/** Buffer path: the scratch buffer and layout uniforms, destroyed on release. */
	buffers: GPUBuffer[];
}

const mipSize = ( size: number, level: number ) => Math.max( 1, size >> level );

interface BatchKernelKey {
	format: SupportedFormat;
	dimension: ViewDimension;
	count: number;
	access: SourceAccess;
	reduction: Reduction;
	sourceFormat: string;
	output: OutputPath;
	kind: BatchKernel;
	tile: number;
	/** Area-weighted first output: see {@link KernelOptions.area}. */
	area: boolean;
	/** `oddSizes: 'box'`: area-weighted first output and tail steps ({@link KernelOptions.box}). */
	box: boolean;
	/** A partial update: the region uniform offsets the grid and restricts stores. */
	region: boolean;
}

/** Recent partial-update plans kept per texture (each holds a uniform buffer and bind groups). */
const regionPlanLimit = 16;
/** Uniform slot stride: the default minUniformBufferOffsetAlignment. */
const slotBytes = 256;

interface PlanParameters {
	source: ResolvedSource | undefined;
	baseMipLevel: number;
	mipLevelCount: number;
	oddSizes: OddSizePolicy;
	reduction: Reduction;
	sourceMipLevel: number;
	scratch: GPUBuffer | undefined;
}

/**
 * Generates mip chains with compute shaders, up to six levels per dispatch:
 * averages (render-pass semantics by default), min/max/minmax and custom fold
 * pyramids (conservative at odd sizes, so a valid Hi-Z), in place or from a
 * separate source such as a depth buffer.
 *
 * ```ts
 * const downsampler = new Downsampler( device );
 * downsampler.encode( encoder, texture );
 * downsampler.encode( encoder, hiZ, { source: depth, reduction: 'minmax' } );
 * ```
 *
 * Dependent batches are separated by dispatch boundaries. There is deliberately
 * no single-dispatch (atomic last-workgroup) mode: WGSL barriers are
 * workgroup-scoped and atomics are relaxed, so that handoff is not guaranteed,
 * and the tested prototype was also slower. See SPEC.md.
 */
export class Downsampler {

	readonly device: GPUDevice;
	/** Levels per dispatch on the storage path (and on every path with output 'buffer'). */
	readonly levelsPerDispatch: number;
	/** Levels per dispatch on the buffer path: `levelsPerDispatch` if given, else 6 (4 below 256 invocations per workgroup). */
	readonly bufferLevelsPerDispatch: number;
	/** The `output` option: 'auto', 'storage' or 'buffer'. {@link Downsampler.outputPath} resolves it per texture. */
	readonly output: 'auto' | OutputPath;
	/** How 'average' reads filterable sources (resolved `averageAccess` option, default 'sample'). */
	readonly averageAccess: SourceAccess;
	/** The `compatibility` option. {@link Downsampler.usesCompatibility} resolves it per texture. */
	readonly compatibility: boolean | 'auto';
	/** Whether 'exact' chains finish in a tail kernel (resolved `tail` option). */
	readonly tail: boolean;
	/** How float16 levels are rounded (resolved `f16Rounding` option). */
	readonly f16Rounding: F16Rounding;
	/** Smallest first-output tile of 'exact' batches (resolved `exactTile` option: 8, 16 or 32). */
	readonly exactTile: number;
	/** Largest source ÷ destination ratio per axis with `resample: 'area'` (resolved `maxResampleRatio` option, default 8). */
	readonly maxResampleRatio: number;
	/** Largest tail tile: 32, or 8 below 256 invocations per workgroup (tail tiles of 16 and 32 take 16 × 16 invocations). */
	private readonly tailMaxTile: number;
	private readonly scratch = new WeakMap<GPUTexture, Map<number, GPUTexture>>();
	private readonly pipelines = new Map<string, GPUComputePipeline>();
	private readonly layouts = new Map<string, GPUBindGroupLayout>();
	private readonly plans = new WeakMap<GPUTexture, Map<string, CachedPlan>>();
	/** Partial-update plans per texture, least recently used first. */
	private readonly regionPlans = new WeakMap<GPUTexture, Map<string, CachedPlan>>();
	/** Buffer path, partial updates: one scratch buffer per texture, shared by its region plans. */
	private readonly regionScratch = new WeakMap<GPUTexture, GPUBuffer>();
	private readonly sampler: GPUSampler;

	/**
	 * Limits to request when creating a device: the standalone
	 * {@link requiredLimits} (exported from `downsampler/webgpu`), which also
	 * covers 5- and 6-level `Reducer`s and `ScalarReduction`.
	 */
	static requiredLimits( adapter: GPUAdapter ): Record<string, number> {

		return requiredLimits( adapter );

	}

	constructor( device: GPUDevice, options: DownsamplerOptions = {} ) {

		const output = options.output ?? 'auto';
		if ( output !== 'auto' && output !== 'storage' && output !== 'buffer' ) throw new Error( 'downsampler: output must be "auto", "storage" or "buffer".' );
		// The device's defaults (`kernelDefaults`, shared with `downsampler/mipmap`). 5 and 6 levels
		// take 16 × 16 invocations; compatibility mode defaults to 128. Software adapters
		// (SwiftShader) keep 8×8 'exact' tiles: they take 30 s or more to compile a 32×32-tile pipeline.
		const defaults = kernelDefaults( device.limits, ( device as GPUDevice & { adapterInfo?: GPUAdapterInfo } ).adapterInfo );
		const invocations = defaults.bufferLevelsPerDispatch;
		const limit = output === 'buffer' ? invocations : defaults.levelsPerDispatch;
		const levels = options.levelsPerDispatch ?? limit;
		if ( ! Number.isInteger( levels ) || levels < 1 || levels > limit ) throw new Error( `downsampler: levelsPerDispatch must be an integer from 1 to ${ limit } on this device${ output === 'buffer' || limit === invocations ? '' : ` (up to ${ invocations } with output 'buffer')` }.` );
		this.device = device;
		this.levelsPerDispatch = levels;
		this.bufferLevelsPerDispatch = options.levelsPerDispatch ?? invocations;
		this.output = output;
		if ( options.averageAccess !== undefined && options.averageAccess !== 'sample' && options.averageAccess !== 'load' ) throw new Error( 'downsampler: averageAccess must be "sample" or "load".' );
		this.averageAccess = options.averageAccess ?? 'sample';
		const compatibility = options.compatibility ?? 'auto';
		if ( compatibility !== 'auto' && typeof compatibility !== 'boolean' ) throw new Error( 'downsampler: compatibility must be true, false or "auto".' );
		this.compatibility = compatibility;
		this.tail = options.tail ?? defaults.tail;
		const f16Rounding = options.f16Rounding ?? 'nearest';
		if ( f16Rounding !== 'nearest' && f16Rounding !== 'device' ) throw new Error( 'downsampler: f16Rounding must be "nearest" or "device".' );
		this.f16Rounding = f16Rounding;
		const exactTile = options.exactTile ?? defaults.exactTile;
		if ( exactTile !== 8 && ( exactTile !== 16 && exactTile !== 32 || invocations < 6 ) ) throw new Error( 'downsampler: exactTile must be 8, or 16 or 32 on devices with 256 invocations per workgroup.' );
		this.exactTile = exactTile;
		const ratio = options.maxResampleRatio ?? maxResampleRatio;
		if ( typeof ratio !== 'number' || ! Number.isFinite( ratio ) || ratio < 1 ) throw new Error( 'downsampler: maxResampleRatio must be a finite number ≥ 1.' );
		this.maxResampleRatio = ratio;
		this.tailMaxTile = defaults.tailMaxTile;
		this.sampler = device.createSampler( { label: 'downsampler', minFilter: 'linear', magFilter: 'linear' } );

	}

	/** Why a texture cannot be a destination, or null if it can. */
	unsupportedReason( texture: GPUTexture ): string | null {

		if ( ! isSupportedFormat( texture.format ) ) return `format ${ texture.format } is not supported`;
		if ( texture.dimension !== '2d' ) return `dimension ${ texture.dimension } is not supported`;
		if ( texture.sampleCount !== 1 ) return 'multisampled textures are not supported';
		if ( ( texture.usage & GPUTextureUsage.TEXTURE_BINDING ) === 0 ) return 'texture needs TEXTURE_BINDING usage';
		const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: string } ).textureBindingViewDimension;
		if ( viewDimension !== undefined && viewDimension !== '2d' && viewDimension !== '2d-array' ) return `textureBindingViewDimension ${ viewDimension } is not supported`;
		if ( this.outputPath( texture ) !== null ) return null;

		// Why no path applies.
		const format = texture.format;
		const info: FormatInfo = formats[ format ];
		const formatReason = this.storageFormatReason( texture );
		const storageReason = formatReason ?? 'texture needs STORAGE_BINDING usage';
		const bufferReason = ! bufferWritable( format ) ? `format ${ format } has ${ info.bytesPerTexel }-byte texels, and the buffer output path writes 4-byte words` : 'texture needs COPY_DST usage for the buffer output path';
		if ( this.output === 'storage' ) return storageReason;
		if ( this.output === 'buffer' ) return bufferReason;
		// 'auto' takes the buffer path only for formats the device can't store.
		if ( formatReason === null ) return bufferWritable( format ) ? `${ storageReason } (or COPY_DST usage with output 'buffer')` : storageReason;
		return bufferWritable( format ) ? `${ storageReason }, so it takes the buffer output path, and the ${ bufferReason.replace( 'texture needs ', 'texture needs ' ) }` : storageReason;

	}

	/**
	 * How levels of `texture` are written: 'storage' (storage textures) or
	 * 'buffer' (a storage buffer and copies), or null when neither applies (see
	 * {@link Downsampler.unsupportedReason}). Follows the `output` option; 'auto'
	 * takes the buffer path only for formats the device can't write as storage
	 * textures, never for a texture that merely lacks STORAGE_BINDING usage.
	 */
	outputPath( texture: GPUTexture ): OutputPath | null {

		if ( ! isSupportedFormat( texture.format ) ) return null;
		const storageFormat = this.storageFormatReason( texture ) === null;
		const storage = storageFormat && ( texture.usage & GPUTextureUsage.STORAGE_BINDING ) !== 0;
		const buffer = bufferWritable( texture.format ) && ( texture.usage & GPUTextureUsage.COPY_DST ) !== 0;
		if ( this.output === 'storage' ) return storage ? 'storage' : null;
		if ( this.output === 'buffer' ) return buffer ? 'buffer' : null;
		return storageFormat ? ( storage ? 'storage' : null ) : buffer ? 'buffer' : null;

	}

	/** Why this device can't write `texture`'s format as a storage texture (here: compatibility mode counts), or null. */
	private storageFormatReason( texture: GPUTexture ): string | null {

		const format = texture.format as SupportedFormat;
		const info: FormatInfo = formats[ format ];
		if ( info.storage === false ) return `format ${ format } can't be a storage texture`;
		if ( info.feature !== undefined && ! this.device.features.has( info.feature ) ) return `format ${ format } needs the "${ info.feature }" device feature`;
		if ( info.coreStorage && this.usesCompatibility( texture ) ) return `format ${ format } has no storage in compatibility mode (it needs "core-features-and-limits")`;
		return null;

	}

	/** Whether `texture` can be a destination: {@link Downsampler.unsupportedReason} is null. */
	supports( texture: GPUTexture ): boolean {

		return this.unsupportedReason( texture ) === null;

	}

	/**
	 * Why `source` (a texture, or `{ texture, mipLevel, resample }`) cannot feed
	 * `destination` from level `baseMipLevel`, or null if it can.
	 */
	sourceUnsupportedReason( destination: GPUTexture, source: GPUTexture | SourceOptions, baseMipLevel = 0 ): string | null {

		const shape = sourceShapeReason( source );
		if ( shape !== null ) return shape;
		const resolved = resolveSource( source );
		const { texture, resample } = resolved;
		if ( texture === destination ) return 'source is the destination: use baseMipLevel to reduce in place';
		if ( resample !== undefined && resample !== 'area' ) return `source resample ${ JSON.stringify( resample ) } is unknown (only 'area')`;
		// Float destinations read float-sampleable sources; integer ones an integer source of the same signedness.
		const kind = isSupportedFormat( destination.format ) ? sampleKind( destination.format ) : 'float';
		const reason = sourceTextureReason( resolved, 'source', kind === 'float' ? {} : { kind, mismatch: format => `destination ${ destination.format } is an integer format, so its source must be a ${ kind } integer format too, not ${ format }` } );
		if ( reason !== null ) return reason;
		if ( texture.depthOrArrayLayers !== destination.depthOrArrayLayers ) return `source has ${ texture.depthOrArrayLayers } layers, destination ${ destination.depthOrArrayLayers }`;
		const [ sw, sh ] = levelSizeOf( resolved );
		const dw = mipSize( destination.width, baseMipLevel ), dh = mipSize( destination.height, baseMipLevel );
		if ( resample === 'area' ) {

			const ratio = this.maxResampleRatio;
			if ( sw > ratio * dw || sh > ratio * dh ) return `destination level ${ baseMipLevel } (${ dw }×${ dh }) is more than ${ ratio }× smaller than the source level (${ sw }×${ sh }) along an axis; reduce into a larger level, start from a smaller source level, or raise maxResampleRatio`;
			return null;

		}

		const same = dw === sw && dh === sh, half = dw === Math.max( 1, sw >> 1 ) && dh === Math.max( 1, sh >> 1 );
		if ( ! same && ! half ) return `destination level ${ baseMipLevel } (${ dw }×${ dh }) must be the source level's size (${ sw }×${ sh }) or half of it; for another size (e.g. a power-of-two Hi-Z) pass source.resample 'area'`;
		return null;

	}

	/**
	 * Bytes an {@link EncodeOptions.scratch} buffer needs for `texture` with these options
	 * (`baseMipLevel`, `mipLevelCount`, `source`): every written level and layer in the
	 * buffer path's layout, which holds any batch of the chain (about a third of the base
	 * level in place, 256-byte rows). 0 when `texture` takes the storage path.
	 */
	scratchSize( texture: GPUTexture, options: EncodeOptions = {} ): number {

		if ( this.outputPath( texture ) !== 'buffer' ) return 0;
		const baseMipLevel = options.baseMipLevel ?? 0;
		const end = options.mipLevelCount === undefined ? texture.mipLevelCount : baseMipLevel + options.mipLevelCount;
		const first = options.source === undefined ? baseMipLevel + 1 : baseMipLevel;
		const levels = Array.from( { length: Math.max( 0, end - first ) }, ( _, i ) => [ mipSize( texture.width, first + i ), mipSize( texture.height, first + i ) ] as const );
		return levels.length === 0 ? 0 : outputBufferLayout( levels, texture.depthOrArrayLayers, formats[ texture.format as SupportedFormat ].bytesPerTexel ).size;

	}

	/** Record the mip chain into an encoder. Throws for unsupported textures. */
	encode( encoder: GPUCommandEncoder, texture: GPUTexture, options: EncodeOptions = {} ): void {

		for ( const batch of this.prepare( texture, options ) ) {

			const pass = encoder.beginComputePass( { label: 'downsampler' } );
			pass.setPipeline( batch.pipeline );
			pass.setBindGroup( 0, batch.bindGroup );
			pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, batch.layers[ 1 ] );
			pass.end();
			// Buffer path: each level into its texture, before the next batch reads it.
			for ( const copy of batch.copies ?? [] ) {

				encoder.copyBufferToTexture( { buffer: batch.buffer!, offset: copy.offset, bytesPerRow: copy.bytesPerRow, rowsPerImage: copy.rowsPerImage }, { texture, mipLevel: copy.mipLevel, origin: { x: copy.x, y: copy.y, z: copy.layers[ 0 ] } }, [ copy.width, copy.height, copy.layers[ 1 ] ] );

			}

		}

	}

	/**
	 * Record the mip chain into a compute pass the caller owns, one dispatch per
	 * batch. Dispatches in a compute pass are separate usage scopes, so each batch
	 * reads what the previous one wrote. Use it to share one pass (and one pair of
	 * timestamp writes) with other dispatches. Throws for unsupported textures,
	 * and on the buffer output path, whose copies can't be recorded in a pass.
	 */
	encodePass( pass: GPUComputePassEncoder, texture: GPUTexture, options: EncodeOptions = {} ): void {

		const batches = this.prepare( texture, options );
		if ( batches.some( batch => batch.output === 'buffer' ) ) throw new Error( `downsampler: ${ texture.format } levels are written through a buffer and copied into the texture between dispatches, which a compute pass can't record; use encode().` );
		for ( const batch of batches ) {

			pass.setPipeline( batch.pipeline );
			pass.setBindGroup( 0, batch.bindGroup );
			pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, batch.layers[ 1 ] );

		}

	}

	/** Encode and submit in one call. */
	generate( texture: GPUTexture, options: EncodeOptions = {} ): void {

		const encoder = this.device.createCommandEncoder( { label: 'downsampler' } );
		this.encode( encoder, texture, options );
		this.device.queue.submit( [ encoder.finish() ] );

	}

	/**
	 * What `encode` records for a chain, for tools that dispatch it themselves
	 * (and to inspect the plan, or compile its pipelines ahead of time): one
	 * {@link PreparedBatch} per dispatch, in order, with its pipeline, group-0
	 * bind group and workgroup counts, and on the buffer path the copies that
	 * follow it. Validates like `encode` and throws with the reason. Cached per
	 * destination and option set (see {@link Downsampler.release}); with a
	 * separate source only the most recent source texture's plan is kept per
	 * option set, so recreating a depth buffer on resize does not grow the cache.
	 * Partial updates (`rect`, `layers`) keep the 16 most recent regions per
	 * texture; batches that write nothing are left out, so an empty rect gives none.
	 * Plans dropped from a cache (an evicted region, a replaced source texture) have
	 * their buffers destroyed once the queue has finished the work submitted so
	 * far: submit what you record before awaiting anything.
	 */
	prepare( texture: GPUTexture, options: EncodeOptions = {} ): readonly PreparedBatch[] {

		const reason = this.unsupportedReason( texture );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		const baseMipLevel = options.baseMipLevel ?? 0;
		const mipLevelCount = options.mipLevelCount ?? texture.mipLevelCount - baseMipLevel;
		if ( baseMipLevel < 0 || mipLevelCount < 1 || baseMipLevel + mipLevelCount > texture.mipLevelCount ) throw new Error( 'downsampler: mip range is outside the texture.' );
		if ( options.source !== undefined ) {

			const sourceReason = this.sourceUnsupportedReason( texture, options.source, baseMipLevel );
			if ( sourceReason !== null ) throw new Error( `downsampler: ${ sourceReason }.` );

		}

		const source = options.source === undefined ? undefined : resolveSource( options.source );
		const oddSizes = options.oddSizes ?? 'exact';
		if ( oddSizes !== 'exact' && oddSizes !== 'spd' && oddSizes !== 'box' ) throw new Error( `downsampler: odd size policy ${ JSON.stringify( oddSizes ) } is not implemented ('exact', 'box' or 'spd').` );
		const reduction = options.reduction ?? 'average';
		if ( needsTileEdges( reduction ) ) throw new Error( 'downsampler: sums, per-channel reductions and custom sums have no \'exact\' mip-chain meaning; use a Reducer (edges \'zero-weight\' or \'clamp\').' );
		if ( oddSizes === 'spd' && isIntFormat( texture.format as SupportedFormat ) ) throw new Error( `downsampler: oddSizes 'spd' isn't available for integer formats (${ texture.format }): its batches carry unrounded float sums between levels, which can't hold 32-bit integers exactly. Use oddSizes 'exact' (the default) or 'box', which reduce integers exactly.` );
		if ( oddSizes === 'spd' && reduction !== 'average' ) throw new Error( 'downsampler: oddSizes \'spd\' is for \'average\' only. It drops the last odd row or column, so min/max/minmax and custom folds would not be a conservative Hi-Z; use oddSizes \'exact\' (the default), or a Reducer with edges \'zero-weight\' (⌈n/2⌉ sizes, every texel counted).' );
		const sourceMipLevel = source?.mipLevel ?? 0;
		const parameters: PlanParameters = { source, baseMipLevel, mipLevelCount, oddSizes, reduction, sourceMipLevel, scratch: options.scratch };
		const key = `${ baseMipLevel }:${ mipLevelCount }:${ oddSizes }:${ reductionKey( reduction ) }:${ source === undefined ? 'self' : `source:${ sourceMipLevel }${ source.resample === 'area' ? ':area' : '' }` }`;

		if ( options.rect === undefined && options.layers === undefined ) {

			let cache = this.plans.get( texture );
			if ( cache === undefined ) this.plans.set( texture, cache = new Map() );
			const cached = cache.get( key );
			if ( cached !== undefined && cached.source === source?.texture && cached.scratch === options.scratch ) return cached.batches;
			// A new source texture (e.g. a depth buffer recreated on resize) or scratch buffer replaces the plan.
			if ( cached !== undefined ) retireBuffers( this.device, cached.buffers );
			const plan = this.build( texture, parameters, null );
			cache.set( key, plan );
			return plan.batches;

		}

		// A partial update: the dirty rect of the first source level, and a layer range.
		const layerCount = texture.depthOrArrayLayers;
		const layersReason = options.layers === undefined ? null : layersProblem( options.layers, layerCount );
		if ( layersReason !== null ) throw new Error( `downsampler: ${ layersReason }.` );
		const layers: [ number, number ] = options.layers === undefined ? [ 0, layerCount ] : [ options.layers[ 0 ], options.layers[ 1 ] ];
		const [ rw, rh ] = source === undefined ? [ mipSize( texture.width, baseMipLevel ), mipSize( texture.height, baseMipLevel ) ] : levelSizeOf( source );
		const rectReason = options.rect === undefined ? null : rectProblem( options.rect, rw, rh );
		if ( rectReason !== null ) throw new Error( `downsampler: ${ rectReason }${ source === undefined ? ` (level ${ baseMipLevel })` : ' (the source level)' }.` );
		const { x, y, width, height } = options.rect ?? { x: 0, y: 0, width: rw, height: rh };
		// An empty rect: nothing depends on it, so nothing is recorded (and nothing cached).
		if ( width === 0 || height === 0 ) return [];
		const rect = { x, y, width, height };
		const regionKey = `${ key }|${ layers.join() }|${ x },${ y },${ width },${ height }`;
		let cache = this.regionPlans.get( texture );
		if ( cache === undefined ) this.regionPlans.set( texture, cache = new Map() );
		const cached = cache.get( regionKey );
		cache.delete( regionKey );
		if ( cached !== undefined && cached.source === source?.texture && cached.scratch === options.scratch ) {

			cache.set( regionKey, cached );
			return cached.batches;

		}

		if ( cached !== undefined ) retireBuffers( this.device, cached.buffers );
		const plan = this.build( texture, parameters, { rect, layers } );
		cache.set( regionKey, plan );
		// Evicted plans may still be recorded in an encoder not yet submitted: their
		// region uniforms are destroyed once the queue has finished the work submitted so far.
		while ( cache.size > regionPlanLimit ) {

			const oldest = cache.keys().next().value!;
			retireBuffers( this.device, cache.get( oldest )!.buffers );
			cache.delete( oldest );

		}

		return plan.batches;

	}

	/** Prepares the batches of one plan; `region` makes it a partial update. */
	private build( texture: GPUTexture, { source, baseMipLevel, mipLevelCount, oddSizes, reduction, sourceMipLevel, scratch: scratchBuffer }: PlanParameters, region: { rect: Rect; layers: readonly [ number, number ] } | null ): CachedPlan {

		const format = texture.format as SupportedFormat;
		const dimension = bindingViewDimension( texture );
		const view = ( level: number ) => texture.createView( { label: `downsampler mip ${ level }`, dimension, baseMipLevel: level, mipLevelCount: 1 } );

		const output = this.outputPath( texture )!;
		const levelsPerDispatch = output === 'buffer' && this.output === 'auto' ? this.bufferLevelsPerDispatch : this.levelsPerDispatch;
		const sourceWidth = source === undefined ? undefined : mipSize( source.texture.width, sourceMipLevel ), sourceHeight = source === undefined ? undefined : mipSize( source.texture.height, sourceMipLevel );
		const copySource = source !== undefined && sourceWidth === mipSize( texture.width, baseMipLevel ) && sourceHeight === mipSize( texture.height, baseMipLevel );
		const { tailMaxTile } = this;
		// Under 'spd', a resampled source is its own 1-level batch, like a copy.
		const resample = source?.resample === 'area';
		const planned = planBatches( texture.width, texture.height, { baseMipLevel, mipLevelCount, levelsPerDispatch, oddSizes, fromSource: source !== undefined, copySource: copySource || resample, tail: oddSizes !== 'spd' && this.tail, tailMaxTile, exactMinTile: this.exactTile } );
		const layerCount = texture.depthOrArrayLayers;
		const layers: readonly [ number, number ] = region?.layers ?? [ 0, layerCount ];
		const regions: BatchRegion[] | null = region === null ? null : planRegions( planned, region.rect, texture.width, texture.height, sourceWidth, sourceHeight );
		// A partial update leaves out batches that write nothing (possible under 'spd' only).
		const included = planned.map( ( batch, index ) => ( { batch, index } ) ).filter( ( { index } ) => regions === null || regions[ index ].groupsX * regions[ index ].groupsY > 0 );
		const sourceSize = ( batch: Batch ): [ number, number ] => batch.sourceMipLevel === null ? [ sourceWidth!, sourceHeight! ] : [ mipSize( texture.width, batch.sourceMipLevel ), mipSize( texture.height, batch.sourceMipLevel ) ];
		const odd = ( size: number ) => size > 1 && size % 2 === 1;

		// Buffer path: one scratch buffer, sized for the largest batch, serves every
		// batch in turn (each batch's copies are recorded before the next dispatch).
		// Its layout holds every layer and whole levels, so partial updates write it
		// at the same offsets and copy their regions out.
		const bytesPerTexel = formats[ format ].bytesPerTexel;
		const layouts = output === 'buffer' ? planned.map( batch => outputBufferLayout( Array.from( { length: batch.count }, ( _, i ) => [ mipSize( texture.width, batch.firstMipLevel + i ), mipSize( texture.height, batch.firstMipLevel + i ) ] as const ), layerCount, bytesPerTexel ) ) : [];
		const buffers: GPUBuffer[] = [];
		let scratch: GPUBuffer | undefined;
		if ( output === 'buffer' && included.length > 0 ) {

			const size = Math.max( ...included.map( ( { index } ) => layouts[ index ].size ) );
			const limit = Math.min( this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize );
			if ( size > limit ) throw new Error( `downsampler: the buffer output path needs a ${ size }-byte buffer for this texture, above this device's maxStorageBufferBindingSize / maxBufferSize (${ limit }); request higher limits.` );
			// A caller's scratch buffer (EncodeOptions.scratch): checked, never destroyed here.
			if ( scratchBuffer !== undefined ) {

				const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
				if ( ( scratchBuffer.usage & usage ) !== usage ) throw new Error( 'downsampler: the scratch buffer needs STORAGE and COPY_SRC usage.' );
				if ( scratchBuffer.size < size ) throw new Error( `downsampler: the scratch buffer has ${ scratchBuffer.size } bytes; this chain needs ${ size } (scratchSize( texture, options ) gives a size that fits).` );
				scratch = scratchBuffer;

			} else if ( region === null ) buffers.push( scratch = this.device.createBuffer( { label: `downsampler output ${ format }`, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } ) );
			else scratch = this.sharedScratch( texture, size );

		}

		// Compatibility mode, storage path: batches write scratch textures and read written
		// levels from them (SPEC §4). The buffer path needs neither: it reads one level per
		// dispatch and writes through copies. A partial update writes only part of each
		// scratch level, so it reads the destination, which the copies have brought up to date.
		const compat = output === 'storage' && this.usesCompatibility( texture );
		const firstWritten = source === undefined ? baseMipLevel + 1 : baseMipLevel;
		const target = compat ? ( level: number ) => this.scratchView( texture, level, dimension ) : view;
		const input = ( level: number ) => compat && region === null && level >= firstWritten ? this.scratchView( texture, level, dimension ) : view( level );

		// Partial updates: one uniform buffer, a 256-byte slot per region uniform (and per
		// layout uniform on the buffer path), written once: plans are cached per region.
		const slots: Uint32Array[] = [];
		const slot = ( words: ArrayLike<number> ) => {

			const data = new Uint32Array( slotBytes / 4 );
			data.set( words );
			slots.push( data );
			return ( slots.length - 1 ) * slotBytes;

		};
		const regionWords = ( groupX: number, groupY: number, levels: readonly ( Rect | null )[] ) => {

			const words = [ groupX, groupY, layers[ 0 ], 0 ];
			for ( const level of levels ) words.push( ...( level === null ? [ 0, 0, 0, 0 ] : [ level.x, level.y, level.x + level.width, level.y + level.height ] ) );
			return words;

		};
		// Bind group entries that point into the uniform buffer, filled in once it exists.
		const pending: { entry: GPUBindGroupEntry; offset: number; size: number }[] = [];
		const uniformEntry = ( binding: number, offset: number, size: number ) => {

			const entry: GPUBindGroupEntry = { binding, resource: { buffer: undefined as unknown as GPUBuffer, offset, size } };
			pending.push( { entry, offset, size } );
			return entry;

		};
		// Bind groups are created after the uniform buffer.
		const deferred: ( () => void )[] = [];

		let batches = included.map( ( { batch, index } ): PreparedBatch => {

			const external = batch.sourceMipLevel === null;
			const sourceFormat = external ? source!.texture.format : format;
			// A same-size converting copy loads: a sampler at texel centres isn't exact on every
			// GPU (Intel truncates the coordinate a hair below the centre and blends in 1/256 of a
			// neighbour; results/windows/correctness.md).
			// Area-weighted batches load: a resampled source (any reduction), and under 'box' the
			// averages that read an odd level, or reduce odd levels in a tail. The others are
			// the 'exact' kernels: an even step is the same 2×2 box, and folds are the same.
			const [ sw, sh ] = sourceSize( batch );
			const box = batch.kernel !== 'clamp' && oddSizes === 'box' && reduction === 'average' && ( batch.kernel === 'tail' || odd( sw ) || odd( sh ) );
			const area = box || ( batch.kernel !== 'clamp' && external && resample );
			const access = area || ( external && copySource && batch.firstMipLevel === baseMipLevel ) ? 'load' : this.access( reduction, sourceFormat );
			const kernel: BatchKernelKey = { format, dimension, count: batch.count, access, reduction, sourceFormat, output, kind: batch.kernel, tile: batch.tile, area, box, region: region !== null };
			const layout = this.layout( kernel );
			const pipeline = this.pipeline( kernel );
			const entries: GPUBindGroupEntry[] = access === 'sample' ? [ { binding: 0, resource: this.sampler } ] : [];
			entries.push( { binding: 1, resource: external ? sourceView( source!, dimension, 'downsampler source' ) : input( batch.sourceMipLevel! ) } );
			if ( usesSrgbTable( this.kernelOptions( kernel ) ) ) entries.push( { binding: srgbTableBinding, resource: { buffer: srgbTable( this.device ) } } );
			const update = regions?.[ index ];
			const prepared: PreparedBatch = { ...batch, pipeline, bindGroup: undefined as unknown as GPUBindGroup, output, layers };
			if ( update !== undefined ) {

				Object.assign( prepared, { groupX: update.groupX, groupY: update.groupY, groupsX: update.groupsX, groupsY: update.groupsY, regions: update.levels } );
				entries.push( uniformEntry( regionBinding, slot( regionWords( update.groupX, update.groupY, update.levels ) ), regionBytes ) );

			}

			if ( output === 'storage' ) {

				for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: target( batch.firstMipLevel + i ) } );

			} else {

				// The layout uniform: first-output size, then per level its offset and row stride in words and its rows.
				const levels = layouts[ index ].levels;
				const words = new Uint32Array( outputLayoutBytes / 4 );
				words.set( [ batch.outputWidth, batch.outputHeight ] );
				levels.forEach( ( level, i ) => { words[ 4 + i ] = level.offset / 4; words[ 12 + i ] = level.bytesPerRow / 4; words[ 20 + i ] = level.rowsPerImage; } );
				if ( update === undefined ) {

					const uniform = this.device.createBuffer( { label: 'downsampler output layout', size: outputLayoutBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
					this.device.queue.writeBuffer( uniform, 0, words );
					buffers.push( uniform );
					entries.push( { binding: outputLayoutBinding, resource: { buffer: uniform } } );

				} else entries.push( uniformEntry( outputLayoutBinding, slot( words ), outputLayoutBytes ) );
				entries.push( { binding: outputBufferBinding, resource: { buffer: scratch! } } );
				// Whole levels, or each level's region of the layers in range.
				prepared.buffer = scratch;
				prepared.copies = levels.flatMap( ( { offset, bytesPerRow, rowsPerImage, width, height }, i ): BufferCopy[] => {

					const mipLevel = batch.firstMipLevel + i;
					if ( update === undefined ) return [ { mipLevel, offset, bytesPerRow, rowsPerImage, x: 0, y: 0, width, height, layers: [ 0, layerCount ] } ];
					const r = update.levels[ i ];
					if ( r === null ) return [];
					return [ { mipLevel, offset: offset + ( layers[ 0 ] * rowsPerImage + r.y ) * bytesPerRow + r.x * bytesPerTexel, bytesPerRow, rowsPerImage, x: r.x, y: r.y, width: r.width, height: r.height, layers } ];

				} );

			}

			deferred.push( () => { prepared.bindGroup = this.device.createBindGroup( { label: 'downsampler', layout, entries } ); } );
			return prepared;

		} );
		if ( compat ) batches = batches.flatMap( batch => [ batch, ...this.compatCopies( texture, batch, dimension, view, region === null ? null : { slot, regionWords, uniformEntry, deferred } ) ] );

		if ( slots.length > 0 ) {

			const uniform = this.device.createBuffer( { label: 'downsampler region', size: slots.length * slotBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			const data = new Uint32Array( slots.length * slotBytes / 4 );
			slots.forEach( ( words, i ) => data.set( words, i * slotBytes / 4 ) );
			this.device.queue.writeBuffer( uniform, 0, data );
			buffers.push( uniform );
			for ( const { entry, offset, size } of pending ) entry.resource = { buffer: uniform, offset, size };

		}

		for ( const create of deferred ) create();
		return { source: source?.texture, scratch: scratchBuffer, batches, buffers };

	}

	/**
	 * Drop cached bind groups for a destination texture (pipelines stay cached per
	 * device), and destroy its buffer-path and partial-update buffers: call it after
	 * submitting work that uses them.
	 */
	release( texture: GPUTexture ): void {

		for ( const plans of [ this.plans, this.regionPlans ] ) {

			for ( const plan of plans.get( texture )?.values() ?? [] ) for ( const buffer of plan.buffers ) buffer.destroy();
			plans.delete( texture );

		}

		this.regionScratch.get( texture )?.destroy();
		this.regionScratch.delete( texture );
		for ( const scratch of this.scratch.get( texture )?.values() ?? [] ) scratch.destroy();
		this.scratch.delete( texture );

	}

	/** Whether `texture` takes the compatibility-mode path (see {@link DownsamplerOptions.compatibility}); on the buffer path it changes nothing. */
	usesCompatibility( texture: GPUTexture ): boolean {

		return this.compatibility === 'auto' ? isCompatibilityTexture( this.device, texture ) : this.compatibility;

	}

	/**
	 * The buffer path's scratch buffer for partial updates of `texture`, at least `size`
	 * bytes. Region plans share it (commands run in order); a larger one replaces it, and
	 * the old one is left to plans that still hold it.
	 */
	private sharedScratch( texture: GPUTexture, size: number ): GPUBuffer {

		let scratch = this.regionScratch.get( texture );
		if ( scratch === undefined || scratch.size < size ) this.regionScratch.set( texture, scratch = this.device.createBuffer( { label: `downsampler output ${ texture.format } (partial updates)`, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } ) );
		return scratch;

	}

	/**
	 * Compatibility mode allows one view (mip range and aspect) of a texture per
	 * dispatch. So a batch writes each level to a scratch texture of that
	 * level's size, and one dispatch per level copies it into the destination:
	 * the 1-level fold kernel on a same-size source, which reads one texel per
	 * output (an exact copy; an average could turn an Inf neighbour into NaN).
	 * A partial update copies only each level's region of the layers in range.
	 */
	private compatCopies( texture: GPUTexture, batch: PreparedBatch, dimension: ViewDimension, view: ( level: number ) => GPUTextureView, partial: {
		slot: ( words: ArrayLike<number> ) => number;
		regionWords: ( groupX: number, groupY: number, levels: readonly ( Rect | null )[] ) => number[];
		uniformEntry: ( binding: number, offset: number, size: number ) => GPUBindGroupEntry;
		deferred: ( () => void )[];
	} | null ): PreparedBatch[] {

		const format = texture.format as SupportedFormat;
		const kernel: BatchKernelKey = { format, dimension, count: 1, access: 'load', reduction: 'max', sourceFormat: format, output: 'storage', kind: 'exact', tile: 8, area: false, box: false, region: partial !== null };
		const layout = this.layout( kernel );
		const pipeline = this.pipeline( kernel );
		const tile = 8;
		return Array.from( { length: batch.count }, ( _, i ) => i ).flatMap( ( i ): PreparedBatch[] => {

			const level = batch.firstMipLevel + i;
			const outputWidth = mipSize( texture.width, level ), outputHeight = mipSize( texture.height, level );
			const entries: GPUBindGroupEntry[] = [ { binding: 1, resource: this.scratchView( texture, level, dimension ) }, { binding: 2, resource: view( level ) } ];
			if ( usesSrgbTable( this.kernelOptions( kernel ) ) ) entries.push( { binding: srgbTableBinding, resource: { buffer: srgbTable( this.device ) } } );
			const copy: PreparedBatch = {
				kernel: 'exact', sourceMipLevel: level, firstMipLevel: level, count: 1, outputWidth, outputHeight, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ), copy: true, output: 'storage',
				layers: batch.layers, pipeline, bindGroup: undefined as unknown as GPUBindGroup
			};
			if ( partial === null ) {

				copy.bindGroup = this.device.createBindGroup( { label: 'downsampler compat copy', layout, entries } );
				return [ copy ];

			}

			const r = batch.regions![ i ];
			if ( r === null ) return [];
			const groupX = Math.floor( r.x / tile ), groupY = Math.floor( r.y / tile );
			Object.assign( copy, { groupX, groupY, groupsX: Math.ceil( ( r.x + r.width ) / tile ) - groupX, groupsY: Math.ceil( ( r.y + r.height ) / tile ) - groupY, regions: [ r ] } );
			entries.push( partial.uniformEntry( regionBinding, partial.slot( partial.regionWords( groupX, groupY, [ r ] ) ), regionBytes ) );
			partial.deferred.push( () => { copy.bindGroup = this.device.createBindGroup( { label: 'downsampler compat copy', layout, entries } ); } );
			return [ copy ];

		} );

	}

	/** A view of the scratch texture for `level` of `texture` (compatibility mode), created on first use. */
	private scratchView( texture: GPUTexture, level: number, dimension: ViewDimension ): GPUTextureView {

		let levels = this.scratch.get( texture );
		if ( levels === undefined ) this.scratch.set( texture, levels = new Map() );
		let scratch = levels.get( level );
		if ( scratch === undefined ) levels.set( level, scratch = this.device.createTexture( {
			label: `downsampler compat scratch, level ${ level }`, format: texture.format,
			size: [ mipSize( texture.width, level ), mipSize( texture.height, level ), texture.depthOrArrayLayers ],
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING, textureBindingViewDimension: dimension
		} as GPUTextureDescriptor ) );
		return scratch.createView( { label: `downsampler compat scratch, level ${ level }`, dimension } );

	}

	/** How a batch reads a texture of `format`: folds and unfilterable formats load. */
	private access( reduction: Reduction, format: string ): SourceAccess {

		if ( reduction !== 'average' ) return 'load';
		const float32 = format.endsWith( '32float' ) && this.device.features.has( 'float32-filterable' );
		const known = isSupportedFormat( format ) ? ( formats[ format ] as FormatInfo ).filterable : ! format.startsWith( 'depth' ) && ! format.endsWith( '32float' ) && ! /16[us]norm$/.test( format );
		const filterable = known || float32;
		return filterable ? this.averageAccess : 'load';

	}

	/**
	 * Kernel options for a batch. 'clamp' batches (`oddSizes: 'spd'`) are the
	 * Reducer's 'clamp' tile kernel emitting every level.
	 */
	private kernelOptions( { format, dimension, count, access, reduction, sourceFormat, output, kind, tile, area, box, region }: BatchKernelKey ): KernelOptions {

		const { f16Rounding } = this;
		const partial = region ? { region: true } : {};
		if ( kind !== 'clamp' ) return { format, dimension, count, reduction, access, sourceFormat, output, f16Rounding, tile, ...( kind === 'tail' ? { tail: true } : {} ), ...( box ? { box: true } : area ? { area: true } : {} ), ...partial };
		return { format, dimension, edges: 'clamp', count, reduction, access, sourceFormat, output, f16Rounding, ...partial };

	}

	private layout( key: BatchKernelKey ): GPUBindGroupLayout {

		const { format, dimension, count, access, output, kind, region } = key;
		// Exact and tail kernels share layouts; 'clamp' ones differ (no sampler under 'load').
		// An sRGB source or destination adds the table binding.
		const srgb = usesSrgbTable( this.kernelOptions( key ) );
		const id = `${ format }:${ dimension }:${ count }:${ access }:${ output }${ kind === 'clamp' ? ':clamp' : '' }${ region ? ':region' : '' }${ srgb ? ':srgb' : '' }`;
		let layout = this.layouts.get( id );
		if ( layout === undefined ) this.layouts.set( id, layout = this.device.createBindGroupLayout( { label: `downsampler ${ id }`, entries: kernelLayoutEntries( this.kernelOptions( key ) ) } ) );
		return layout;

	}

	private pipeline( key: BatchKernelKey ): GPUComputePipeline {

		const { format, dimension, count, access, reduction, sourceFormat, output, kind, tile, area, box, region } = key;
		const id = `${ kind }${ kind === 'clamp' ? '' : tile }${ box ? ':box' : area ? ':area' : '' }:${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ reductionKey( reduction ) }:${ output }${ region ? ':region' : '' }`;
		let pipeline = this.pipelines.get( id );
		if ( pipeline !== undefined ) return pipeline;
		const label = `downsampler ${ kind }${ box ? ' box' : area ? ' area' : '' }:${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ typeof reduction === 'string' ? reduction : 'custom' }:${ output }${ region ? ':region' : '' }`;
		const module = this.device.createShaderModule( { label, code: createKernel( this.kernelOptions( key ) ) } );
		const layout = this.device.createPipelineLayout( { label, bindGroupLayouts: [ this.layout( key ) ] } );
		pipeline = this.device.createComputePipeline( { label, layout, compute: { module, entryPoint: 'main' } } );
		this.pipelines.set( id, pipeline );
		return pipeline;

	}

}
