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
import { Downsampler, createKernel, kernelLayoutEntries, srgbScratchFormat, srgbTable, srgbTableBinding } from '/dist/webgpu/index.js';
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

			// One storage texture per level: a plan of the buffer path's 6 levels needs 6.
			if ( batch.count > device.limits.maxStorageTexturesPerShaderStage ) throw new Error( `srgb scratch: a ${ batch.count }-level batch needs ${ batch.count } storage textures, above this device's ${ device.limits.maxStorageTexturesPerShaderStage }; plan with levelsPerDispatch ≤ the limit` );
			const options = { format: texture.format, dimension, count: batch.count, access: 'sample', sourceFormat: texture.format, output: 'storage', tile: batch.tile, ...( batch.kernel === 'tail' ? { tail: true } : {} ) };
			const { layout, pipeline: compute } = pipeline( options );
			const entries = [ { binding: 0, resource: sampler }, { binding: 1, resource: view( texture, batch.sourceMipLevel ) }, { binding: srgbTableBinding, resource: { buffer: srgbTable( device ) } } ];
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

// `--compare=srgb-cost` (DIVERGES #16): where an sRGB chain's time goes. Timing only: every
// variant but `buffer` writes wrong values. Each is the library's buffer path on the same
// rgba8unorm-srgb texture with one part of the shader's sRGB work edited out of the WGSL
// (through a device whose createShaderModule edits the code), so the differences split the cost:
// - `buffer`: the library.
// - `noCorrect`: ds_srgb_code without its table correction step (the ±1 code fix-up).
// - `noPow`: ds_srgb_code with sqrt in place of pow(c, 1/2.4) (correction kept).
// - `noSnap`: in-tile values not snapped through encode/decode (`quantize` keeps linear colour);
//   stores still encode.
// - `fastStore` (a candidate, same bits expected): stores encode without the correction step.
//   Every stored value has been through `quantize`, so it is exactly decode(k) for some code k,
//   and the formula alone rounds it back to k (largest miss on the CPU in f32: 0.000016 of a step).
// - `noEncode`: no snap, and stores write round(v · 255): no sRGB maths in the shader at all.
// - `linearBuffer`: an rgba8unorm texture of the same size on the buffer path (`output: 'buffer'`):
//   the same dispatches and copies with the linear format's arithmetic.
// - `linearStorage`: that texture on the storage path, no copies.
// - `render`: one render pass per level.
const srgbEdits = {
	noCorrect: code => edit( code, /\n\tif \(k < 255u && c >= ds_srgb_table[^\n]+/, '' ),
	noPow: code => edit( code, '1.055 * pow(c, 1.0 / 2.4) - 0.055', 'sqrt(c)' ),
	noSnap: code => edit( code, 'return vec4f(ds_srgb_quantize(value.xyz), round(', 'return vec4f(value.xyz, round(' ),
	fastStore: code => edit( code.replace( /^.*<< 24u.*$/m, line => line.replaceAll( 'ds_srgb_code(', 'ds_srgb_stored_code(' ) ), 'fn ds_srgb_quantize(', `fn ds_srgb_stored_code(v: f32) -> u32 {
	let c = clamp(v, 0.0, 1.0);
	return u32(round(select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308) * 255.0));
}
fn ds_srgb_quantize(` ),
	noEncode: code => edit( srgbEdits.noSnap( code ), /fn ds_srgb_code\(v: f32\) -> u32 \{[\s\S]*?\n\}/, 'fn ds_srgb_code(v: f32) -> u32 { return u32(clamp(round(v * 255.0), 0.0, 255.0)); }' )
};

function edit( code, from, to ) {

	if ( ! code.includes( 'ds_srgb_code' ) ) return code;
	const out = code.replace( from, to );
	if ( out === code ) throw new Error( `srgb-cost: edit ${ from } didn't apply` );
	return out;

}

/** `device`, with every shader module's WGSL passed through `transform`. */
function editingDevice( device, transform ) {

	return new Proxy( device, { get( target, key ) {

		if ( key === 'createShaderModule' ) return descriptor => target.createShaderModule( { ...descriptor, code: transform( descriptor.code ) } );
		const value = Reflect.get( target, key );
		return typeof value === 'function' ? value.bind( target ) : value;

	} } );

}

export function srgbCostStrategies( device ) {

	const run = downsampler => ( { passes: texture => downsampler.prepare( texture ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture ) } );
	const linear = new WeakMap();
	const linearTexture = texture => {

		if ( ! linear.has( texture ) ) linear.set( texture, device.createTexture( { label: 'srgb-cost linear', format: 'rgba8unorm', size: [ texture.width, texture.height, texture.depthOrArrayLayers ], mipLevelCount: texture.mipLevelCount, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } ) );
		return linear.get( texture );

	};
	const onLinear = downsampler => ( { passes: texture => downsampler.prepare( linearTexture( texture ) ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, linearTexture( texture ) ) } );
	const strategies = {
		render: { ...createRenderBaseline( device ), parity: false },
		buffer: { ...run( new Downsampler( device ) ), parity: true }
	};
	for ( const [ name, transform ] of Object.entries( srgbEdits ) ) strategies[ name ] = { ...run( new Downsampler( editingDevice( device, transform ) ) ), parity: name === 'fastStore' };
	strategies.linearBuffer = { ...onLinear( new Downsampler( device, { output: 'buffer' } ) ), parity: false };
	strategies.linearStorage = { ...onLinear( new Downsampler( device ) ), parity: false };
	return { strategies, reference: () => 'buffer' };

}
