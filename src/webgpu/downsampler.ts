import { bufferWritable, formats, isSupportedFormat, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { outputBufferLayout, planBatches, type Batch, type OddSizePolicy } from '../core/plan.js';
import { needsTileEdges, reductionKey, type Reduction } from '../core/reduction.js';
import { createKernel, kernelLayoutEntries, outputBufferBinding, outputLayoutBinding, outputLayoutBytes, type OutputPath, type SourceAccess, type ViewDimension } from './kernel.js';

export interface DownsamplerOptions {
	/**
	 * Output levels per dispatch. Defaults to the most the path allows: on the
	 * storage path min(6, maxStorageTexturesPerShaderStage) (six needs a device
	 * created with a raised limit — see {@link Downsampler.requiredLimits}); on
	 * the buffer path 6 at any limit. Above the storage limit only with output 'buffer'.
	 */
	levelsPerDispatch?: number;
	/**
	 * How levels are written (SPEC.md §6.1 Buffer output path):
	 * - 'storage': write-only storage textures, one binding per level.
	 * - 'buffer': packed texels into one storage buffer, then `copyBufferToTexture`
	 *   per level. Needs COPY_DST usage and a format with 4-byte-multiple texels;
	 *   writes formats that can't be storage textures (rgb9e5ufloat, and
	 *   rg11b10ufloat without texture-formats-tier1), and isn't limited by
	 *   maxStorageTexturesPerShaderStage. Can't record into a caller's compute pass.
	 * - 'auto' (default): storage when the texture has STORAGE_BINDING usage and
	 *   the device can write its format there, else buffer.
	 */
	output?: 'auto' | OutputPath;
	/**
	 * How 'average' reads the source level of each batch: one hardware bilinear
	 * sample, or four `textureLoad`s with weights computed in the shader. Default
	 * 'sample'. Unfilterable formats (float32, depth) and fold reductions always load.
	 */
	averageAccess?: SourceAccess;
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
}

interface CachedPlan {
	source: GPUTexture | undefined;
	batches: PreparedBatch[];
	/** Buffer path: the scratch buffer and layout uniforms, destroyed on release. */
	buffers: GPUBuffer[];
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
	/** Levels per dispatch on the storage path (and on every path with output 'buffer'). */
	readonly levelsPerDispatch: number;
	/** Levels per dispatch on the buffer path: `levelsPerDispatch` if given, else 6. */
	readonly bufferLevelsPerDispatch: number;
	readonly output: 'auto' | OutputPath;
	readonly averageAccess: SourceAccess | undefined;
	private readonly pipelines = new Map<string, GPUComputePipeline>();
	private readonly layouts = new Map<string, GPUBindGroupLayout>();
	private readonly plans = new WeakMap<GPUTexture, Map<string, CachedPlan>>();
	private readonly sampler: GPUSampler;

	/** Limits to request when creating a device, so six levels fit in one dispatch. */
	static requiredLimits( adapter: GPUAdapter ): Record<string, number> {

		return { maxStorageTexturesPerShaderStage: Math.max( 4, Math.min( 6, adapter.limits.maxStorageTexturesPerShaderStage ) ) };

	}

	constructor( device: GPUDevice, options: DownsamplerOptions = {} ) {

		const output = options.output ?? 'auto';
		if ( output !== 'auto' && output !== 'storage' && output !== 'buffer' ) throw new Error( 'downsampler: output must be "auto", "storage" or "buffer".' );
		const limit = output === 'buffer' ? 6 : Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
		const levels = options.levelsPerDispatch ?? limit;
		if ( ! Number.isInteger( levels ) || levels < 1 || levels > limit ) throw new Error( `downsampler: levelsPerDispatch must be an integer from 1 to ${ limit } on this device${ output === 'buffer' ? '' : ' (up to 6 with output \'buffer\')' }.` );
		this.device = device;
		this.levelsPerDispatch = levels;
		this.bufferLevelsPerDispatch = options.levelsPerDispatch ?? 6;
		this.output = output;
		if ( options.averageAccess !== undefined && options.averageAccess !== 'sample' && options.averageAccess !== 'load' ) throw new Error( 'downsampler: averageAccess must be "sample" or "load".' );
		this.averageAccess = options.averageAccess;
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

		// Why neither path applies.
		const format = texture.format;
		const info: FormatInfo = formats[ format ];
		const storageReason = info.storage === false ? `format ${ format } can't be a storage texture` :
			info.feature !== undefined && ! this.device.features.has( info.feature ) ? `format ${ format } needs the "${ info.feature }" device feature` : 'texture needs STORAGE_BINDING usage';
		const bufferReason = ! bufferWritable( format ) ? `format ${ format } has ${ info.bytesPerTexel }-byte texels, and the buffer output path writes 4-byte words` : 'texture needs COPY_DST usage for the buffer output path';
		if ( this.output === 'storage' ) return storageReason;
		if ( this.output === 'buffer' ) return bufferReason;
		return bufferWritable( format ) ? `${ storageReason }, or ${ bufferReason.replace( 'texture needs ', '' ) }` : storageReason;

	}

	/**
	 * How levels of `texture` are written: 'storage' (storage textures) or
	 * 'buffer' (a storage buffer and copies), or null when neither applies (see
	 * {@link Downsampler.unsupportedReason}). Follows the `output` option; 'auto'
	 * prefers storage textures.
	 */
	outputPath( texture: GPUTexture ): OutputPath | null {

		if ( ! isSupportedFormat( texture.format ) ) return null;
		const info: FormatInfo = formats[ texture.format ];
		const storage = info.storage !== false && ( info.feature === undefined || this.device.features.has( info.feature ) ) && ( texture.usage & GPUTextureUsage.STORAGE_BINDING ) !== 0;
		const buffer = bufferWritable( texture.format ) && ( texture.usage & GPUTextureUsage.COPY_DST ) !== 0;
		if ( this.output === 'storage' ) return storage ? 'storage' : null;
		if ( this.output === 'buffer' ) return buffer ? 'buffer' : null;
		return storage ? 'storage' : buffer ? 'buffer' : null;

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

		const output = this.outputPath( texture )!;
		const levelsPerDispatch = output === 'buffer' && this.output === 'auto' ? this.bufferLevelsPerDispatch : this.levelsPerDispatch;
		const planned = planBatches( texture.width, texture.height, { baseMipLevel, mipLevelCount, levelsPerDispatch, oddSizes, fromSource: source !== undefined } );

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

		const batches = planned.map( ( batch, index ): PreparedBatch => {

			const external = batch.sourceMipLevel === null;
			const sourceFormat = external ? source!.texture.format : format;
			const access = this.access( reduction, sourceFormat );
			const layout = this.layout( format, dimension, batch.count, access, output );
			const pipeline = this.pipeline( format, dimension, batch.count, access, reduction, sourceFormat, output );
			const entries: GPUBindGroupEntry[] = access === 'sample' ? [ { binding: 0, resource: this.sampler } ] : [];
			entries.push( { binding: 1, resource: external ? sourceView() : view( batch.sourceMipLevel! ) } );
			if ( output === 'storage' ) {

				for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: view( batch.firstMipLevel + i ) } );
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

	}

	/** How a batch reads a texture of `format`: folds and unfilterable formats load. */
	private access( reduction: Reduction, format: string ): SourceAccess {

		if ( reduction !== 'average' ) return 'load';
		const float32 = format.endsWith( '32float' ) && this.device.features.has( 'float32-filterable' );
		const known = isSupportedFormat( format ) ? ( formats[ format ] as FormatInfo ).filterable : ! format.startsWith( 'depth' ) && ! format.endsWith( '32float' ) && ! /16[us]norm$/.test( format );
		const filterable = known || float32;
		return filterable ? this.averageAccess ?? 'sample' : 'load';

	}

	private layout( format: SupportedFormat, dimension: ViewDimension, count: number, access: SourceAccess, output: OutputPath ): GPUBindGroupLayout {

		const key = `${ format }:${ dimension }:${ count }:${ access }:${ output }`;
		let layout = this.layouts.get( key );
		if ( layout === undefined ) this.layouts.set( key, layout = this.device.createBindGroupLayout( { label: `downsampler ${ key }`, entries: kernelLayoutEntries( { format, dimension, count, access, output } ) } ) );
		return layout;

	}

	private pipeline( format: SupportedFormat, dimension: ViewDimension, count: number, access: SourceAccess, reduction: Reduction, sourceFormat: string, output: OutputPath ): GPUComputePipeline {

		const key = `${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ reductionKey( reduction ) }:${ output }`;
		let pipeline = this.pipelines.get( key );
		if ( pipeline !== undefined ) return pipeline;
		const label = `downsampler ${ format }:${ dimension }:${ count }:${ access }:${ sourceFormat }:${ typeof reduction === 'string' ? reduction : 'custom' }:${ output }`;
		const module = this.device.createShaderModule( { label, code: createKernel( { format, dimension, count, reduction, access, sourceFormat, output } ) } );
		const layout = this.device.createPipelineLayout( { label, bindGroupLayouts: [ this.layout( format, dimension, count, access, output ) ] } );
		pipeline = this.device.createComputePipeline( { label, layout, compute: { module, entryPoint: 'main' } } );
		this.pipelines.set( key, pipeline );
		return pipeline;

	}

}
