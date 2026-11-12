import { bufferWritable, formats, isSupportedFormat, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { outputBufferLayout, planBatches, tailTile, type Batch, type BatchKernel, type OddSizePolicy } from '../core/plan.js';
import { needsTileEdges, reductionKey, type Reduction } from '../core/reduction.js';
import { isCompatibilityTexture } from './compat.js';
import { createKernel, kernelLayoutEntries, outputBufferBinding, outputLayoutBinding, outputLayoutBytes, type F16Rounding, type KernelOptions, type OutputPath, type SourceAccess, type ViewDimension } from './kernel.js';

export interface DownsamplerOptions {
	/**
	 * Output levels per dispatch. Defaults to the most the path allows: on the
	 * storage path min(6, maxStorageTexturesPerShaderStage) (six needs a device
	 * created with a raised limit — see {@link Downsampler.requiredLimits}); on
	 * the buffer path 6 at any storage texture limit. 5 and 6 levels also need
	 * 256 invocations per workgroup (compatibility mode defaults to 128). Above
	 * the storage limit only with output 'buffer'.
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
	 * follow the same 'exact' rule. Works on both output paths and in
	 * compatibility mode. Default false (SPEC.md §2.1).
	 */
	tail?: boolean;
	/**
	 * float16 formats: how each level is rounded to a half, before it is stored and
	 * before the next level reads it. 'nearest' (default): round to nearest even in the
	 * shader, the same on every GPU. 'device': the device's own conversion
	 * (`pack2x16float`), which truncates on D3D12. See {@link F16Rounding}.
	 */
	f16Rounding?: F16Rounding;
}

export interface SourceOptions {
	/**
	 * Texture to read the first destination level from: any float-sampleable 2D
	 * format, including depth (read through its depth aspect). Needs
	 * TEXTURE_BINDING usage and the destination's array layer count.
	 */
	texture: GPUTexture;
	/** Level to read. Default 0. */
	mipLevel?: number;
}

export interface EncodeOptions {
	/**
	 * In place: the source level, with outputs from `baseMipLevel + 1`. With
	 * `source`: the first level written. Default 0.
	 */
	baseMipLevel?: number;
	/**
	 * Destination levels involved. In place this includes the source level.
	 * Default: the rest of the chain.
	 */
	mipLevelCount?: number;
	/**
	 * A separate source, e.g. a depth buffer for a Hi-Z pyramid (depth formats
	 * cannot be storage textures). Destination level `baseMipLevel` must be the
	 * same size as the source level (a converting copy) or half of it (one
	 * reduction step); every destination level in range is written.
	 * Single-channel sources are splatted, so 'minmax' of depth gives (d, d).
	 */
	source?: SourceOptions;
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
	 * 'box' (planned): area-weighted odd sizes. See SPEC.md §2.1.
	 */
	oddSizes?: OddSizePolicy;
	/**
	 * 'average' (default) matches render-pass mipmaps. 'min', 'max' and 'minmax'
	 * (even channels min, odd channels max) combine every texel an output covers,
	 * including the extra row/column of odd sizes, as Hi-Z needs. A custom
	 * `{ wgsl }` defines `fn reduce(a: vec4f, b: vec4f) -> vec4f`, which must be
	 * associative, commutative and idempotent. Sums, per-channel ops and custom
	 * sums have no 'exact' meaning and are rejected: use a `Reducer`.
	 */
	reduction?: Reduction;
}

/** A `copyBufferToTexture` of one level, after a buffer-path batch. */
export interface BufferCopy {
	mipLevel: number;
	/** Byte offset in the batch's buffer, a multiple of 256. */
	offset: number;
	bytesPerRow: number;
	rowsPerImage: number;
	width: number;
	height: number;
}

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
}

interface CachedPlan {
	source: GPUTexture | undefined;
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
}

/**
 * Generates mip chains with compute shaders, up to six levels per dispatch.
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
	readonly output: 'auto' | OutputPath;
	readonly averageAccess: SourceAccess | undefined;
	readonly compatibility: boolean | 'auto';
	readonly tail: boolean;
	readonly f16Rounding: F16Rounding;
	private readonly scratch = new WeakMap<GPUTexture, Map<number, GPUTexture>>();
	private readonly pipelines = new Map<string, GPUComputePipeline>();
	private readonly layouts = new Map<string, GPUBindGroupLayout>();
	private readonly plans = new WeakMap<GPUTexture, Map<string, CachedPlan>>();
	private readonly sampler: GPUSampler;

	/**
	 * Limits to request when creating a device, so six levels fit in one
	 * dispatch: storage textures, and on compatibility-mode adapters (default
	 * 128) the 256 invocations of the 5- and 6-level kernels and of
	 * `ScalarReduction` (whose combine workgroup is 256 wide).
	 */
	static requiredLimits( adapter: GPUAdapter ): Record<string, number> {

		const limits: Record<string, number> = { maxStorageTexturesPerShaderStage: Math.max( 4, Math.min( 6, adapter.limits.maxStorageTexturesPerShaderStage ) ) };
		if ( adapter.limits.maxComputeInvocationsPerWorkgroup >= 256 ) limits.maxComputeInvocationsPerWorkgroup = 256;
		if ( adapter.limits.maxComputeWorkgroupSizeX >= 256 ) limits.maxComputeWorkgroupSizeX = 256;
		return limits;

	}

	constructor( device: GPUDevice, options: DownsamplerOptions = {} ) {

		const output = options.output ?? 'auto';
		if ( output !== 'auto' && output !== 'storage' && output !== 'buffer' ) throw new Error( 'downsampler: output must be "auto", "storage" or "buffer".' );
		// 5 and 6 levels take 16 × 16 invocations; compatibility mode defaults to 128.
		const invocations = device.limits.maxComputeInvocationsPerWorkgroup >= 256 ? 6 : 4;
		const limit = output === 'buffer' ? invocations : Math.min( invocations, device.limits.maxStorageTexturesPerShaderStage );
		const levels = options.levelsPerDispatch ?? limit;
		if ( ! Number.isInteger( levels ) || levels < 1 || levels > limit ) throw new Error( `downsampler: levelsPerDispatch must be an integer from 1 to ${ limit } on this device${ output === 'buffer' || limit === invocations ? '' : ` (up to ${ invocations } with output 'buffer')` }.` );
		this.device = device;
		this.levelsPerDispatch = levels;
		this.bufferLevelsPerDispatch = options.levelsPerDispatch ?? invocations;
		this.output = output;
		if ( options.averageAccess !== undefined && options.averageAccess !== 'sample' && options.averageAccess !== 'load' ) throw new Error( 'downsampler: averageAccess must be "sample" or "load".' );
		this.averageAccess = options.averageAccess;
		const compatibility = options.compatibility ?? 'auto';
		if ( compatibility !== 'auto' && typeof compatibility !== 'boolean' ) throw new Error( 'downsampler: compatibility must be true, false or "auto".' );
		this.compatibility = compatibility;
		this.tail = options.tail ?? false;
		const f16Rounding = options.f16Rounding ?? 'nearest';
		if ( f16Rounding !== 'nearest' && f16Rounding !== 'device' ) throw new Error( 'downsampler: f16Rounding must be "nearest" or "device".' );
		this.f16Rounding = f16Rounding;
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

	supports( texture: GPUTexture ): boolean {

		return this.unsupportedReason( texture ) === null;

	}

	/** Why `source` cannot feed `destination` with these options, or null if it can. */
	sourceUnsupportedReason( destination: GPUTexture, source: SourceOptions, baseMipLevel = 0 ): string | null {

		const { texture } = source, mipLevel = source.mipLevel ?? 0;
		if ( texture === destination ) return 'source is the destination: use baseMipLevel to reduce in place';
		if ( texture.dimension !== '2d' ) return `source dimension ${ texture.dimension } is not supported`;
		if ( texture.sampleCount !== 1 ) return 'multisampled sources are not supported (resolve first)';
		if ( ( texture.usage & GPUTextureUsage.TEXTURE_BINDING ) === 0 ) return 'source needs TEXTURE_BINDING usage';
		if ( /int$/.test( texture.format ) || texture.format === 'stencil8' ) return `source format ${ texture.format } is not float-sampleable`;
		if ( ! Number.isInteger( mipLevel ) || mipLevel < 0 || mipLevel >= texture.mipLevelCount ) return 'source mipLevel is outside the source texture';
		if ( texture.depthOrArrayLayers !== destination.depthOrArrayLayers ) return `source has ${ texture.depthOrArrayLayers } layers, destination ${ destination.depthOrArrayLayers }`;
		const sw = mipSize( texture.width, mipLevel ), sh = mipSize( texture.height, mipLevel );
		const dw = mipSize( destination.width, baseMipLevel ), dh = mipSize( destination.height, baseMipLevel );
		const same = dw === sw && dh === sh, half = dw === Math.max( 1, sw >> 1 ) && dh === Math.max( 1, sh >> 1 );
		if ( ! same && ! half ) return `destination level ${ baseMipLevel } (${ dw }×${ dh }) must be the source level's size (${ sw }×${ sh }) or half of it`;
		return null;

	}

	/** Record the mip chain into an encoder. Throws for unsupported textures. */
	encode( encoder: GPUCommandEncoder, texture: GPUTexture, options: EncodeOptions = {} ): void {

		for ( const batch of this.prepare( texture, options ) ) {

			const pass = encoder.beginComputePass( { label: 'downsampler' } );
			pass.setPipeline( batch.pipeline );
			pass.setBindGroup( 0, batch.bindGroup );
			pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, texture.depthOrArrayLayers );
			pass.end();
			// Buffer path: each level into its texture, before the next batch reads it.
			for ( const copy of batch.copies ?? [] ) {

				encoder.copyBufferToTexture( { buffer: batch.buffer!, offset: copy.offset, bytesPerRow: copy.bytesPerRow, rowsPerImage: copy.rowsPerImage }, { texture, mipLevel: copy.mipLevel }, [ copy.width, copy.height, texture.depthOrArrayLayers ] );

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
			pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, texture.depthOrArrayLayers );

		}

	}

	/** Encode and submit in one call. */
	generate( texture: GPUTexture, options: EncodeOptions = {} ): void {

		const encoder = this.device.createCommandEncoder( { label: 'downsampler' } );
		this.encode( encoder, texture, options );
		this.device.queue.submit( [ encoder.finish() ] );

	}

	/**
	 * The dispatches `encode` will record, prepared and cached per destination.
	 * With a separate source, only the plan for the most recent source texture is
	 * kept per option set, so recreating a depth buffer on resize does not grow
	 * the cache.
	 */
	prepare( texture: GPUTexture, options: EncodeOptions = {} ): readonly PreparedBatch[] {

		const reason = this.unsupportedReason( texture );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		const source = options.source;
		const baseMipLevel = options.baseMipLevel ?? 0;
		const mipLevelCount = options.mipLevelCount ?? texture.mipLevelCount - baseMipLevel;
		if ( baseMipLevel < 0 || mipLevelCount < 1 || baseMipLevel + mipLevelCount > texture.mipLevelCount ) throw new Error( 'downsampler: mip range is outside the texture.' );
		if ( source !== undefined ) {

			const sourceReason = this.sourceUnsupportedReason( texture, source, baseMipLevel );
			if ( sourceReason !== null ) throw new Error( `downsampler: ${ sourceReason }.` );

		}

		const oddSizes = options.oddSizes ?? 'exact';
		if ( oddSizes !== 'exact' && oddSizes !== 'spd' ) throw new Error( `downsampler: odd size policy ${ JSON.stringify( oddSizes ) } is not implemented ('exact' or 'spd').` );
		const reduction = options.reduction ?? 'average';
		if ( needsTileEdges( reduction ) ) throw new Error( 'downsampler: sums, per-channel reductions and custom sums have no \'exact\' mip-chain meaning; use a Reducer (edges \'zero-weight\' or \'clamp\').' );
		if ( oddSizes === 'spd' && reduction !== 'average' ) throw new Error( 'downsampler: oddSizes \'spd\' is for \'average\' only. It drops the last odd row or column, so min/max/minmax and custom folds would not be a conservative Hi-Z; use oddSizes \'exact\' (the default), or a Reducer with edges \'zero-weight\' (⌈n/2⌉ sizes, every texel counted).' );
		const sourceMipLevel = source?.mipLevel ?? 0;
		const key = `${ baseMipLevel }:${ mipLevelCount }:${ oddSizes }:${ reductionKey( reduction ) }:${ source === undefined ? 'self' : `source:${ sourceMipLevel }` }`;
		let cache = this.plans.get( texture );
		if ( cache === undefined ) this.plans.set( texture, cache = new Map() );
		const cached = cache.get( key );
		if ( cached !== undefined && cached.source === source?.texture ) return cached.batches;

		const format = texture.format as SupportedFormat;
		const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: ViewDimension } ).textureBindingViewDimension;
		const dimension: ViewDimension = viewDimension ?? ( texture.depthOrArrayLayers > 1 ? '2d-array' : '2d' );
		const view = ( level: number ) => texture.createView( { label: `downsampler mip ${ level }`, dimension, baseMipLevel: level, mipLevelCount: 1 } );
		const sourceView = () => {

			const aspect = source!.texture.format.includes( 'stencil' ) ? 'depth-only' : 'all';
			return source!.texture.createView( { label: 'downsampler source', dimension, baseMipLevel: sourceMipLevel, mipLevelCount: 1, aspect } );

		};

		const output = this.outputPath( texture )!;
		const levelsPerDispatch = output === 'buffer' && this.output === 'auto' ? this.bufferLevelsPerDispatch : this.levelsPerDispatch;
		const copySource = source !== undefined && mipSize( source.texture.width, sourceMipLevel ) === mipSize( texture.width, baseMipLevel ) && mipSize( source.texture.height, sourceMipLevel ) === mipSize( texture.height, baseMipLevel );
		// Tail tiles of 16 and 32 take 16 × 16 invocations; below 256 (compatibility mode) only 8.
		const tailMaxTile = this.device.limits.maxComputeInvocationsPerWorkgroup >= 256 ? tailTile : 8;
		const planned = planBatches( texture.width, texture.height, { baseMipLevel, mipLevelCount, levelsPerDispatch, oddSizes, fromSource: source !== undefined, copySource, tail: oddSizes === 'exact' && this.tail, tailMaxTile } );

		// Buffer path: one scratch buffer, sized for the largest batch, serves every
		// batch in turn (each batch's copies are recorded before the next dispatch).
		const layers = texture.depthOrArrayLayers;
		const layouts = output === 'buffer' ? planned.map( batch => outputBufferLayout( Array.from( { length: batch.count }, ( _, i ) => [ mipSize( texture.width, batch.firstMipLevel + i ), mipSize( texture.height, batch.firstMipLevel + i ) ] as const ), layers, formats[ format ].bytesPerTexel ) ) : [];
		const buffers: GPUBuffer[] = [];
		let scratch: GPUBuffer | undefined;
		if ( output === 'buffer' && planned.length > 0 ) {

			const size = Math.max( ...layouts.map( layout => layout.size ) );
			const limit = Math.min( this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize );
			if ( size > limit ) throw new Error( `downsampler: the buffer output path needs a ${ size }-byte buffer for this texture, above this device's maxStorageBufferBindingSize / maxBufferSize (${ limit }); request higher limits.` );
			buffers.push( scratch = this.device.createBuffer( { label: `downsampler output ${ format }`, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } ) );

		}

		// Compatibility mode, storage path: batches write scratch textures and read written
		// levels from them (SPEC §4). The buffer path needs neither: it reads one level per
		// dispatch and writes through copies.
		const compat = output === 'storage' && this.usesCompatibility( texture );
		const firstWritten = source === undefined ? baseMipLevel + 1 : baseMipLevel;
		const target = compat ? ( level: number ) => this.scratchView( texture, level, dimension ) : view;
		const input = ( level: number ) => compat && level >= firstWritten ? this.scratchView( texture, level, dimension ) : view( level );
		let batches = planned.map( ( batch, index ): PreparedBatch => {

			const external = batch.sourceMipLevel === null;
			const sourceFormat = external ? source!.texture.format : format;
			const access = this.access( reduction, sourceFormat );
			const kernel: BatchKernelKey = { format, dimension, count: batch.count, access, reduction, sourceFormat, output, kind: batch.kernel, tile: batch.tile };
			const layout = this.layout( kernel );
			const pipeline = this.pipeline( kernel );
			const entries: GPUBindGroupEntry[] = access === 'sample' ? [ { binding: 0, resource: this.sampler } ] : [];
			entries.push( { binding: 1, resource: external ? sourceView() : input( batch.sourceMipLevel! ) } );
			if ( output === 'storage' ) {

				for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: target( batch.firstMipLevel + i ) } );
				return { ...batch, pipeline, bindGroup: this.device.createBindGroup( { label: 'downsampler', layout, entries } ), output };

			}

			// The layout uniform: first-output size, then per level its offset and row stride in words and its rows.
			const levels = layouts[ index ].levels;
			const words = new Uint32Array( outputLayoutBytes / 4 );
			words.set( [ batch.outputWidth, batch.outputHeight ] );
			levels.forEach( ( level, i ) => { words[ 4 + i ] = level.offset / 4; words[ 12 + i ] = level.bytesPerRow / 4; words[ 20 + i ] = level.rowsPerImage; } );
			const uniform = this.device.createBuffer( { label: 'downsampler output layout', size: outputLayoutBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			this.device.queue.writeBuffer( uniform, 0, words );
			buffers.push( uniform );
			entries.push( { binding: outputLayoutBinding, resource: { buffer: uniform } }, { binding: outputBufferBinding, resource: { buffer: scratch! } } );
			const copies = levels.map( ( { offset, bytesPerRow, rowsPerImage, width, height }, i ) => ( { mipLevel: batch.firstMipLevel + i, offset, bytesPerRow, rowsPerImage, width, height } ) );
			return { ...batch, pipeline, bindGroup: this.device.createBindGroup( { label: 'downsampler', layout, entries } ), output, buffer: scratch, copies };

		} );
		if ( compat ) batches = batches.flatMap( batch => [ batch, ...this.compatCopies( texture, batch, dimension, view ) ] );

		cache.set( key, { source: source?.texture, batches, buffers } );
		return batches;

	}

	/**
	 * Drop cached bind groups for a destination texture (pipelines stay cached per
	 * device), and destroy its buffer-path buffers: call it after submitting work
	 * that uses them.
	 */
	release( texture: GPUTexture ): void {

		for ( const plan of this.plans.get( texture )?.values() ?? [] ) for ( const buffer of plan.buffers ) buffer.destroy();
		this.plans.delete( texture );
		for ( const scratch of this.scratch.get( texture )?.values() ?? [] ) scratch.destroy();
		this.scratch.delete( texture );

	}

	/** Whether `texture` takes the compatibility-mode path (see {@link DownsamplerOptions.compatibility}); on the buffer path it changes nothing. */
	usesCompatibility( texture: GPUTexture ): boolean {

		return this.compatibility === 'auto' ? isCompatibilityTexture( this.device, texture ) : this.compatibility;

	}

	/**
	 * Compatibility mode allows one view (mip range and aspect) of a texture per
	 * dispatch. So a batch writes each level to a scratch texture of that
	 * level's size, and one dispatch per level copies it into the destination:
	 * the 1-level fold kernel on a same-size source, which reads one texel per
	 * output (an exact copy; an average could turn an Inf neighbour into NaN).
	 */
	private compatCopies( texture: GPUTexture, batch: Batch, dimension: ViewDimension, view: ( level: number ) => GPUTextureView ): PreparedBatch[] {

		const format = texture.format as SupportedFormat;
		const kernel: BatchKernelKey = { format, dimension, count: 1, access: 'load', reduction: 'max', sourceFormat: format, output: 'storage', kind: 'exact', tile: 8 };
		const layout = this.layout( kernel );
		const pipeline = this.pipeline( kernel );
		return Array.from( { length: batch.count }, ( _, i ) => {

			const level = batch.firstMipLevel + i;
			const outputWidth = mipSize( texture.width, level ), outputHeight = mipSize( texture.height, level ), tile = 8;
			const entries: GPUBindGroupEntry[] = [ { binding: 1, resource: this.scratchView( texture, level, dimension ) }, { binding: 2, resource: view( level ) } ];
			return {
				kernel: 'exact', sourceMipLevel: level, firstMipLevel: level, count: 1, outputWidth, outputHeight, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ), copy: true, output: 'storage',
				pipeline, bindGroup: this.device.createBindGroup( { label: 'downsampler compat copy', layout, entries } )
			};

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
		return filterable ? this.averageAccess ?? 'sample' : 'load';

	}

	/**
	 * Kernel options for a batch. 'clamp' batches (`oddSizes: 'spd'`) are the
	 * Reducer's 'clamp' tile kernel emitting every level.
	 */
	private kernelOptions( { format, dimension, count, access, reduction, sourceFormat, output, kind, tile }: BatchKernelKey ): KernelOptions {

		const { f16Rounding } = this;
		if ( kind !== 'clamp' ) return { format, dimension, count, reduction, access, sourceFormat, output, f16Rounding, ...( kind === 'tail' ? { tail: true, tile } : {} ) };
		return { format, dimension, edges: 'clamp', count, reduction, access, sourceFormat, output, f16Rounding };

	}

	private layout( key: BatchKernelKey ): GPUBindGroupLayout {

		const { format, dimension, count, access, output, kind } = key;
		// Exact and tail kernels share layouts; 'clamp' ones differ (no sampler under 'load').
		const id = `${ format }:${ dimension }:${ count }:${ access }:${ output }${ kind === 'clamp' ? ':clamp' : '' }`;
		let layout = this.layouts.get( id );
		if ( layout === undefined ) this.layouts.set( id, layout = this.device.createBindGroupLayout( { label: `downsampler ${ id }`, entries: kernelLayoutEntries( this.kernelOptions( key ) ) } ) );
		return layout;

	}

	private pipeline( key: BatchKernelKey ): GPUComputePipeline {

		const { format, dimension, count, access, reduction, sourceFormat, output, kind, tile } = key;
		const id = `${ kind }${ kind === 'tail' ? tile : '' }:${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ reductionKey( reduction ) }:${ output }`;
		let pipeline = this.pipelines.get( id );
		if ( pipeline !== undefined ) return pipeline;
		const label = `downsampler ${ kind }:${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ typeof reduction === 'string' ? reduction : 'custom' }:${ output }`;
		const module = this.device.createShaderModule( { label, code: createKernel( this.kernelOptions( key ) ) } );
		const layout = this.device.createPipelineLayout( { label, bindGroupLayouts: [ this.layout( key ) ] } );
		pipeline = this.device.createComputePipeline( { label, layout, compute: { module, entryPoint: 'main' } } );
		this.pipelines.set( id, pipeline );
		return pipeline;

	}

}
