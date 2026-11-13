// `--compare=srgb` (M2.3.5, SPEC §6.1 sRGB): how to write sRGB levels, which WebGPU can't
// store from a shader. Both candidates read the sRGB view (linear), reduce in linear and
// encode in the shader with the same kernel; they differ only in the output path:
// - `buffer`: the library's path, packed texels into a storage buffer, then
//   `copyBufferToTexture` per level (6 levels per dispatch at any storage limit).
// - `scratch`: storage writes to an rgba8unorm scratch texture (the linear alias), then
//   `copyTextureToTexture` per level. A measurement prototype: the kernel's storage output
//   of sRGB formats, dispatched here with the library's plan.
// - `bufferLoad`: the buffer path with `averageAccess: 'load'` (loads snap the hardware's
//   decode to the exact table, so results don't depend on the GPU).
// - `render`: one render pass per level (three.js today).
// Ratios are paired per round against `buffer`.
import { Downsampler, createKernel, kernelLayoutEntries, srgbScratchFormat } from '/dist/webgpu/index.js';
import { createRenderBaseline } from './render-baseline.js';

export const srgbFormats = [ 'rgba8unorm-srgb' ];

export function srgbStrategies( device ) {

	const buffer = new Downsampler( device ), load = new Downsampler( device, { averageAccess: 'load' } );
	const run = ( downsampler, options = {} ) => ( { passes: texture => downsampler.prepare( texture, options ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture, options ) } );
	const strategies = {
		render: { ...createRenderBaseline( device ), parity: true },
		buffer: { ...run( buffer ), parity: true },
		bufferLoad: { ...run( load ), parity: true },
		scratch: { ...scratchPrototype( device, buffer ), parity: true }
	};
	return { strategies, reference: () => 'buffer' };

}

/** The scratch-texture path, with the plan, kernels and layouts of the library's buffer path. */
export function scratchPrototype( device, downsampler ) {

	const sampler = device.createSampler( { minFilter: 'linear', magFilter: 'linear' } );
	const pipelines = new Map(), plans = new WeakMap();
	const pipeline = options => {

		const key = JSON.stringify( options );
		if ( ! pipelines.has( key ) ) {

			const layout = device.createBindGroupLayout( { entries: kernelLayoutEntries( options ) } );
			const module = device.createShaderModule( { code: createKernel( options ) } );
			pipelines.set( key, { layout, pipeline: device.createComputePipeline( { layout: device.createPipelineLayout( { bindGroupLayouts: [ layout ] } ), compute: { module, entryPoint: 'main' } } ) } );

		}

		return pipelines.get( key );

	};
	const plan = texture => {

		if ( plans.has( texture ) ) return plans.get( texture );
		const dimension = texture.depthOrArrayLayers > 1 ? '2d-array' : '2d';
		const scratch = device.createTexture( { label: 'srgb scratch', format: srgbScratchFormat( texture.format ), size: [ texture.width, texture.height, texture.depthOrArrayLayers ], mipLevelCount: texture.mipLevelCount, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC } );
		const view = ( target, level ) => target.createView( { dimension, baseMipLevel: level, mipLevelCount: 1 } );
		const batches = downsampler.prepare( texture ).map( batch => {

			const options = { format: texture.format, dimension, count: batch.count, access: 'sample', sourceFormat: texture.format, output: 'storage', tile: batch.tile, ...( batch.kernel === 'tail' ? { tail: true } : {} ) };
			const { layout, pipeline: compute } = pipeline( options );
			const entries = [ { binding: 0, resource: sampler }, { binding: 1, resource: view( texture, batch.sourceMipLevel ) } ];
			for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: view( scratch, batch.firstMipLevel + i ) } );
			return { batch, pipeline: compute, bindGroup: device.createBindGroup( { layout, entries } ) };

		} );
		const result = { scratch, batches };
		plans.set( texture, result );
		return result;

	};

	return {
		passes: texture => plan( texture ).batches.length,
		encode( encoder, texture ) {

			const { scratch, batches } = plan( texture );
			for ( const { batch, pipeline: compute, bindGroup } of batches ) {

				const pass = encoder.beginComputePass();
				pass.setPipeline( compute );
				pass.setBindGroup( 0, bindGroup );
				pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, texture.depthOrArrayLayers );
				pass.end();
				for ( let i = 0; i < batch.count; i ++ ) {

					const level = batch.firstMipLevel + i;
					const size = [ Math.max( 1, texture.width >> level ), Math.max( 1, texture.height >> level ), texture.depthOrArrayLayers ];
					encoder.copyTextureToTexture( { texture: scratch, mipLevel: level }, { texture, mipLevel: level }, size );

				}

			}

		}
	};

}
