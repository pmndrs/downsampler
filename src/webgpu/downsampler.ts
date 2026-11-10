import { isSupportedFormat, type SupportedFormat } from '../core/formats.js';
import { planBatches, type Batch, type OddSizePolicy } from '../core/plan.js';
import { createKernel, type ViewDimension } from './kernel.js';

export interface DownsamplerOptions {
	/**
	 * Output levels per dispatch. Defaults to the most the device allows:
	 * min(6, maxStorageTexturesPerShaderStage). Six needs a device created with a
	 * raised limit — see {@link Downsampler.requiredLimits}.
	 */
	levelsPerDispatch?: number;
}

export interface EncodeOptions {
	/** Source level. Outputs start at `baseMipLevel + 1`. Default 0. */
	baseMipLevel?: number;
	/** Levels involved, including the source. Default: the rest of the chain. */
	mipLevelCount?: number;
	/**
	 * 'exact' (default): odd, non-unit sizes end a batch and the next level is
	 * re-sampled with normalized coordinates, matching render-pass/WebGL mipmaps.
	 * 'fast' (planned): SPD-style clamped reduction through odd sizes, fewer
	 * dispatches but different edge weights. See SPEC.md.
	 */
	oddSizes?: OddSizePolicy;
}

interface PreparedBatch extends Batch {
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
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
	readonly levelsPerDispatch: number;
	private readonly pipelines = new Map<string, GPUComputePipeline>();
	private readonly plans = new WeakMap<GPUTexture, Map<string, PreparedBatch[]>>();
	private readonly sampler: GPUSampler;

	/** Limits to request when creating a device, so six levels fit in one dispatch. */
	static requiredLimits( adapter: GPUAdapter ): Record<string, number> {

		return { maxStorageTexturesPerShaderStage: Math.max( 4, Math.min( 6, adapter.limits.maxStorageTexturesPerShaderStage ) ) };

	}

	constructor( device: GPUDevice, options: DownsamplerOptions = {} ) {

		const limit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
		const levels = options.levelsPerDispatch ?? limit;
		if ( ! Number.isInteger( levels ) || levels < 1 || levels > limit ) throw new Error( `downsampler: levelsPerDispatch must be an integer from 1 to ${ limit } on this device.` );
		this.device = device;
		this.levelsPerDispatch = levels;
		this.sampler = device.createSampler( { label: 'downsampler', minFilter: 'linear', magFilter: 'linear' } );

	}

	/** Why a texture cannot be processed, or null if it can. */
	unsupportedReason( texture: GPUTexture ): string | null {

		if ( ! isSupportedFormat( texture.format ) ) return `format ${ texture.format } is not supported`;
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

	/** Encode and submit in one call. */
	generate( texture: GPUTexture, options: EncodeOptions = {} ): void {

		const encoder = this.device.createCommandEncoder( { label: 'downsampler' } );
		this.encode( encoder, texture, options );
		this.device.queue.submit( [ encoder.finish() ] );

	}

	/** The dispatches `encode` will record, prepared and cached per texture. */
	prepare( texture: GPUTexture, options: EncodeOptions = {} ): readonly PreparedBatch[] {

		const reason = this.unsupportedReason( texture );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		const baseMipLevel = options.baseMipLevel ?? 0;
		const mipLevelCount = options.mipLevelCount ?? texture.mipLevelCount - baseMipLevel;
		if ( baseMipLevel < 0 || mipLevelCount < 1 || baseMipLevel + mipLevelCount > texture.mipLevelCount ) throw new Error( 'downsampler: mip range is outside the texture.' );
		const oddSizes = options.oddSizes ?? 'exact';
		const key = `${ baseMipLevel }:${ mipLevelCount }:${ oddSizes }`;
		let cache = this.plans.get( texture );
		if ( cache === undefined ) this.plans.set( texture, cache = new Map() );
		let prepared = cache.get( key );
		if ( prepared !== undefined ) return prepared;

		const format = texture.format as SupportedFormat;
		const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: ViewDimension } ).textureBindingViewDimension;
		const dimension: ViewDimension = viewDimension ?? ( texture.depthOrArrayLayers > 1 ? '2d-array' : '2d' );
		const view = ( level: number ) => texture.createView( { label: `downsampler mip ${ level }`, dimension, baseMipLevel: level, mipLevelCount: 1 } );

		prepared = planBatches( texture.width, texture.height, { baseMipLevel, mipLevelCount, levelsPerDispatch: this.levelsPerDispatch, oddSizes } ).map( batch => {

			const pipeline = this.pipeline( format, dimension, batch.count );
			const entries: GPUBindGroupEntry[] = [ { binding: 0, resource: this.sampler }, { binding: 1, resource: view( batch.sourceMipLevel ) } ];
			for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: view( batch.sourceMipLevel + 1 + i ) } );
			return { ...batch, pipeline, bindGroup: this.device.createBindGroup( { label: 'downsampler', layout: pipeline.getBindGroupLayout( 0 ), entries } ) };

		} );

		cache.set( key, prepared );
		return prepared;

	}

	/** Drop cached bind groups for a texture (pipelines stay cached per device). */
	release( texture: GPUTexture ): void {

		this.plans.delete( texture );

	}

	private pipeline( format: SupportedFormat, dimension: ViewDimension, count: number ): GPUComputePipeline {

		const key = `${ format }:${ dimension }:${ count }`;
		let pipeline = this.pipelines.get( key );
		if ( pipeline !== undefined ) return pipeline;
		const module = this.device.createShaderModule( { label: `downsampler ${ key }`, code: createKernel( { format, dimension, count } ) } );
		pipeline = this.device.createComputePipeline( { label: `downsampler ${ key }`, layout: 'auto', compute: { module, entryPoint: 'main' } } );
		this.pipelines.set( key, pipeline );
		return pipeline;

	}

}
