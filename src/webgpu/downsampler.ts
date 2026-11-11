import { formats, isSupportedFormat, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { planBatches, type Batch, type OddSizePolicy } from '../core/plan.js';
import { needsTileEdges, reductionKey, type Reduction } from '../core/reduction.js';
import { isCompatibilityTexture } from './compat.js';
import { createKernel, kernelLayoutEntries, type SourceAccess, type ViewDimension } from './kernel.js';

export interface DownsamplerOptions {
	/**
	 * Output levels per dispatch. Defaults to the most the device allows:
	 * min(6, maxStorageTexturesPerShaderStage). Six needs a device created with a
	 * raised limit — see {@link Downsampler.requiredLimits}.
	 */
	levelsPerDispatch?: number;
	/**
	 * How 'average' reads the source level of each batch: one hardware bilinear
	 * sample, or four `textureLoad`s with weights computed in the shader. Default
	 * 'sample'. Unfilterable formats (float32, depth) and fold reductions always load.
	 */
	averageAccess?: SourceAccess;
	/**
	 * The compatibility-mode path (SPEC.md §4 Compatibility mode): compatibility
	 * mode binds a texture through views of one mip range per dispatch, so each
	 * batch writes per-level scratch textures, and a copy dispatch per level moves
	 * them into the destination. 'auto' (default) takes it for textures of a
	 * device without 'core-features-and-limits' (their `textureBindingViewDimension`
	 * is defined only there); true always; false never.
	 */
	compatibility?: boolean | 'auto';
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
	 * 'box' (planned): area-weighted odd sizes. SPD-style 'fast' was dropped
	 * because it loses the last row/column. See SPEC.md §2.1.
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

export interface PreparedBatch extends Batch {
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
	/**
	 * Compatibility-mode path only: this dispatch copies level `firstMipLevel`
	 * from its scratch texture into the destination (no reduction).
	 */
	copy?: boolean;
}

interface CachedPlan {
	source: GPUTexture | undefined;
	batches: PreparedBatch[];
}

const mipSize = ( size: number, level: number ) => Math.max( 1, size >> level );

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
	readonly levelsPerDispatch: number;
	readonly averageAccess: SourceAccess | undefined;
	readonly compatibility: boolean | 'auto';
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

		// 5 and 6 levels take 16 × 16 invocations; compatibility mode defaults to 128.
		const limit = Math.min( device.limits.maxComputeInvocationsPerWorkgroup >= 256 ? 6 : 4, device.limits.maxStorageTexturesPerShaderStage );
		const levels = options.levelsPerDispatch ?? limit;
		if ( ! Number.isInteger( levels ) || levels < 1 || levels > limit ) throw new Error( `downsampler: levelsPerDispatch must be an integer from 1 to ${ limit } on this device.` );
		this.device = device;
		this.levelsPerDispatch = levels;
		if ( options.averageAccess !== undefined && options.averageAccess !== 'sample' && options.averageAccess !== 'load' ) throw new Error( 'downsampler: averageAccess must be "sample" or "load".' );
		this.averageAccess = options.averageAccess;
		const compatibility = options.compatibility ?? 'auto';
		if ( compatibility !== 'auto' && typeof compatibility !== 'boolean' ) throw new Error( 'downsampler: compatibility must be true, false or "auto".' );
		this.compatibility = compatibility;
		this.sampler = device.createSampler( { label: 'downsampler', minFilter: 'linear', magFilter: 'linear' } );

	}

	/** Why a texture cannot be a destination, or null if it can. */
	unsupportedReason( texture: GPUTexture ): string | null {

		if ( ! isSupportedFormat( texture.format ) ) return `format ${ texture.format } is not supported`;
		const feature: string | undefined = ( formats[ texture.format ] as FormatInfo ).feature;
		if ( feature !== undefined && ! this.device.features.has( feature ) ) return `format ${ texture.format } needs the "${ feature }" device feature`;
		if ( ( formats[ texture.format ] as FormatInfo ).coreStorage && this.usesCompatibility( texture ) ) return `format ${ texture.format } has no storage in compatibility mode (it needs "core-features-and-limits")`;
		if ( texture.dimension !== '2d' ) return `dimension ${ texture.dimension } is not supported`;
		if ( texture.sampleCount !== 1 ) return 'multisampled textures are not supported';
		if ( ( texture.usage & GPUTextureUsage.STORAGE_BINDING ) === 0 ) return 'texture needs STORAGE_BINDING usage';
		if ( ( texture.usage & GPUTextureUsage.TEXTURE_BINDING ) === 0 ) return 'texture needs TEXTURE_BINDING usage';
		const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: string } ).textureBindingViewDimension;
		if ( viewDimension !== undefined && viewDimension !== '2d' && viewDimension !== '2d-array' ) return `textureBindingViewDimension ${ viewDimension } is not supported`;
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

		}

	}

	/**
	 * Record the mip chain into a compute pass the caller owns, one dispatch per
	 * batch. Dispatches in a compute pass are separate usage scopes, so each batch
	 * reads what the previous one wrote. Use it to share one pass (and one pair of
	 * timestamp writes) with other dispatches. Throws for unsupported textures.
	 */
	encodePass( pass: GPUComputePassEncoder, texture: GPUTexture, options: EncodeOptions = {} ): void {

		for ( const batch of this.prepare( texture, options ) ) {

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
		const reduction = options.reduction ?? 'average';
		if ( needsTileEdges( reduction ) ) throw new Error( 'downsampler: sums, per-channel reductions and custom sums have no \'exact\' mip-chain meaning; use a Reducer (edges \'zero-weight\' or \'clamp\').' );
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

		// Compatibility mode: batches write scratch textures and read written levels from them (SPEC §4).
		const compat = this.usesCompatibility( texture );
		const firstWritten = source === undefined ? baseMipLevel + 1 : baseMipLevel;
		const output = compat ? ( level: number ) => this.scratchView( texture, level, dimension ) : view;
		const input = ( level: number ) => compat && level >= firstWritten ? this.scratchView( texture, level, dimension ) : view( level );
		let batches: PreparedBatch[] = planBatches( texture.width, texture.height, { baseMipLevel, mipLevelCount, levelsPerDispatch: this.levelsPerDispatch, oddSizes, fromSource: source !== undefined } ).map( batch => {

			const external = batch.sourceMipLevel === null;
			const sourceFormat = external ? source!.texture.format : format;
			const access = this.access( reduction, sourceFormat );
			const layout = this.layout( format, dimension, batch.count, access );
			const pipeline = this.pipeline( format, dimension, batch.count, access, reduction, sourceFormat );
			const entries: GPUBindGroupEntry[] = access === 'sample' ? [ { binding: 0, resource: this.sampler } ] : [];
			entries.push( { binding: 1, resource: external ? sourceView() : input( batch.sourceMipLevel! ) } );
			for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: output( batch.firstMipLevel + i ) } );
			return { ...batch, pipeline, bindGroup: this.device.createBindGroup( { label: 'downsampler', layout, entries } ) };

		} );
		if ( compat ) batches = batches.flatMap( batch => [ batch, ...this.compatCopies( texture, batch, dimension, view ) ] );

		cache.set( key, { source: source?.texture, batches } );
		return batches;

	}

	/**
	 * Drop cached bind groups for a destination texture (pipelines stay cached
	 * per device), and destroy its compatibility-mode scratch textures.
	 */
	release( texture: GPUTexture ): void {

		this.plans.delete( texture );
		for ( const scratch of this.scratch.get( texture )?.values() ?? [] ) scratch.destroy();
		this.scratch.delete( texture );

	}

	/** Whether `texture` takes the compatibility-mode path (see {@link DownsamplerOptions.compatibility}). */
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
		const layout = this.layout( format, dimension, 1, 'load' );
		const pipeline = this.pipeline( format, dimension, 1, 'load', 'max', format );
		return Array.from( { length: batch.count }, ( _, i ) => {

			const level = batch.firstMipLevel + i;
			const outputWidth = mipSize( texture.width, level ), outputHeight = mipSize( texture.height, level ), tile = 8;
			const entries: GPUBindGroupEntry[] = [ { binding: 1, resource: this.scratchView( texture, level, dimension ) }, { binding: 2, resource: view( level ) } ];
			return {
				sourceMipLevel: level, firstMipLevel: level, count: 1, outputWidth, outputHeight, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ), copy: true,
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

	private layout( format: SupportedFormat, dimension: ViewDimension, count: number, access: SourceAccess ): GPUBindGroupLayout {

		const key = `${ format }:${ dimension }:${ count }:${ access }`;
		let layout = this.layouts.get( key );
		if ( layout === undefined ) this.layouts.set( key, layout = this.device.createBindGroupLayout( { label: `downsampler ${ key }`, entries: kernelLayoutEntries( { format, dimension, count, access } ) } ) );
		return layout;

	}

	private pipeline( format: SupportedFormat, dimension: ViewDimension, count: number, access: SourceAccess, reduction: Reduction, sourceFormat: string ): GPUComputePipeline {

		const key = `${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ reductionKey( reduction ) }`;
		let pipeline = this.pipelines.get( key );
		if ( pipeline !== undefined ) return pipeline;
		const label = `downsampler ${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ typeof reduction === 'string' ? reduction : 'custom' }`;
		const module = this.device.createShaderModule( { label, code: createKernel( { format, dimension, count, reduction, access, sourceFormat } ) } );
		const layout = this.device.createPipelineLayout( { label, bindGroupLayouts: [ this.layout( format, dimension, count, access ) ] } );
		pipeline = this.device.createComputePipeline( { label, layout, compute: { module, entryPoint: 'main' } } );
		this.pipelines.set( key, pipeline );
		return pipeline;

	}

}
