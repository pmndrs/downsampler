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
// - `alias`: an rgba8unorm texture with an sRGB view format, read through the sRGB view and
//   written as storage, no copies (aliasPrototype; parity checked byte for byte by its `check`).
// - `colorSpace`: the same, through the library (`colorSpace: 'srgb'`, aliasLibrary).
// - `renderPath`: the library's render path (`output: 'render'`, what 'auto' picks for sRGB
//   textures that can render): `render`'s passes with the exact kernel's rule and rounding in
//   the fragment shader. `renderPathLoad`: the same with `averageAccess: 'load'`.
// - `renderDevice`: the render path with `srgbRounding: 'device'` (the attachment encodes, as
//   `render` does); its `check` counts the bytes that differ from `renderPath`'s.
// Ratios are paired per round against `buffer`.
import { Downsampler, createKernel, kernelLayoutEntries, srgbScratchFormat, srgbTable, srgbTableBinding } from '/dist/webgpu/index.js';
import { createRenderBaseline } from './render-baseline.js';

export const srgbFormats = [ 'rgba8unorm-srgb' ];

export function srgbStrategies( device ) {

	// Bench textures can render, so 'auto' would take the render path: the buffer strategies ask for theirs.
	const buffer = new Downsampler( device, { output: 'buffer' } ), load = new Downsampler( device, { averageAccess: 'load', output: 'buffer' } );
	const run = ( downsampler, options = {} ) => ( { passes: texture => downsampler.prepare( texture, options ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture, options ) } );
	// The library's render path: a pass per level and layer.
	const rendered = downsampler => ( { passes: texture => ( texture.mipLevelCount - 1 ) * texture.depthOrArrayLayers, encode: ( encoder, texture ) => downsampler.encode( encoder, texture ) } );
	const strategies = {
		render: { ...createRenderBaseline( device ), parity: true },
		buffer: { ...run( buffer ), parity: true },
		bufferLoad: { ...run( load ), parity: true },
		scratch: { ...scratchPrototype( device, buffer ), parity: true },
		alias: { ...aliasPrototype( device, buffer ), parity: false },
		colorSpace: { ...aliasLibrary( device, new Downsampler( device ) ), parity: false },
		renderPath: { ...rendered( new Downsampler( device, { output: 'render' } ) ), parity: true },
		renderPathLoad: { ...rendered( new Downsampler( device, { output: 'render', averageAccess: 'load' } ) ), parity: true },
		renderDevice: { ...rendered( new Downsampler( device, { output: 'render', srgbRounding: 'device' } ) ), check: againstRenderPath( device ), parity: true }
	};
	return { strategies, reference: () => 'buffer' };

}

/**
 * `check` of `renderDevice`: every level's bytes against the library's exact render path
 * (`renderPath`), reported, not failed: how many differ, by how much, and from which level.
 */
function againstRenderPath( device ) {

	return async ( textures, name ) => {

		if ( ! textures.renderPath ) return { compared: false };
		const [ ours, theirs ] = [ await rawLevels( device, textures[ name ] ), await rawLevels( device, textures.renderPath ) ];
		let differing = 0, worst = 0, firstLevel = null, bytes = 0;
		ours.forEach( ( level, i ) => level.forEach( ( byte, j ) => { bytes ++; if ( byte !== theirs[ i ][ j ] ) { differing ++; worst = Math.max( worst, Math.abs( byte - theirs[ i ][ j ] ) ); firstLevel ??= i; } } ) );
		return { compared: true, levels: ours.length, bytes, differing, worst, firstLevel };

	};

}

/**
 * The view trick (DIVERGES #16): the texture is rgba8unorm with `viewFormats: ['rgba8unorm-srgb']`.
 * The kernel reads the level above through an rgba8unorm-srgb view (the hardware decodes, as
 * for any sRGB texture) and writes sRGB codes straight into the rgba8unorm levels: no buffer,
 * no copies. Materials would sample the sRGB view, so filtering still decodes first.
 * Each bench texture gets an aliased twin with the same level 0; `check` compares every level's
 * bytes with the buffer path's.
 */
export function aliasPrototype( device, downsampler ) {

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
		const linear = srgbScratchFormat( texture.format );
		const dimension = texture.depthOrArrayLayers > 1 ? '2d-array' : '2d';
		const alias = device.createTexture( { label: 'srgb alias', format: linear, viewFormats: [ texture.format ], size: [ texture.width, texture.height, texture.depthOrArrayLayers ], mipLevelCount: texture.mipLevelCount, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		const copy = device.createCommandEncoder();
		copy.copyTextureToTexture( { texture }, { texture: alias }, [ texture.width, texture.height, texture.depthOrArrayLayers ] );
		device.queue.submit( [ copy.finish() ] );
		// The sRGB view can't carry STORAGE_BINDING (sRGB isn't a storage format), so it names its usage.
		const read = level => alias.createView( { format: texture.format, usage: GPUTextureUsage.TEXTURE_BINDING, dimension, baseMipLevel: level, mipLevelCount: 1 } );
		const write = level => alias.createView( { format: linear, usage: GPUTextureUsage.STORAGE_BINDING, dimension, baseMipLevel: level, mipLevelCount: 1 } );
		const batches = downsampler.prepare( texture ).map( batch => {

			if ( batch.count > device.limits.maxStorageTexturesPerShaderStage ) throw new Error( `srgb alias: a ${ batch.count }-level batch needs ${ batch.count } storage textures, above this device's ${ device.limits.maxStorageTexturesPerShaderStage }` );
			const options = { format: texture.format, dimension, count: batch.count, access: 'sample', sourceFormat: texture.format, output: 'storage', tile: batch.tile, ...( batch.kernel === 'tail' ? { tail: true } : {} ) };
			const { layout, pipeline: compute } = pipeline( options );
			const entries = [ { binding: 0, resource: sampler }, { binding: 1, resource: read( batch.sourceMipLevel ) }, { binding: srgbTableBinding, resource: { buffer: srgbTable( device ) } } ];
			for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: write( batch.firstMipLevel + i ) } );
			return { batch, pipeline: compute, bindGroup: device.createBindGroup( { layout, entries } ) };

		} );
		const result = { alias, batches };
		plans.set( texture, result );
		return result;

	};

	return {
		passes: texture => plan( texture ).batches.length,
		encode( encoder, texture ) {

			const { batches } = plan( texture );
			for ( const { batch, pipeline: compute, bindGroup } of batches ) {

				const pass = encoder.beginComputePass();
				pass.setPipeline( compute );
				pass.setBindGroup( 0, bindGroup );
				pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, texture.depthOrArrayLayers );
				pass.end();

			}

		},
		/** Every level's bytes, against the buffer path's. */
		async check( textures, name ) {

			if ( ! textures.buffer ) return { compared: false };
			const [ ours, theirs ] = [ await rawLevels( device, plan( textures[ name ] ).alias ), await rawLevels( device, textures.buffer ) ];
			let differing = 0;
			ours.forEach( ( level, i ) => level.forEach( ( byte, j ) => { if ( byte !== theirs[ i ][ j ] ) differing ++; } ) );
			if ( differing ) throw new Error( `srgb alias ${ textures[ name ].width }x${ textures[ name ].height }: ${ differing } bytes differ from the buffer path` );
			return { compared: true, levels: ours.length, differing };

		}
	};

}

/** The library's `colorSpace: 'srgb'` on an aliased twin of each bench texture (as `alias`, built in). */
export function aliasLibrary( device, downsampler ) {

	const twins = new WeakMap();
	const twin = texture => {

		if ( twins.has( texture ) ) return twins.get( texture );
		const alias = device.createTexture( { label: 'srgb colorSpace', format: srgbScratchFormat( texture.format ), viewFormats: [ texture.format ], size: [ texture.width, texture.height, texture.depthOrArrayLayers ], mipLevelCount: texture.mipLevelCount, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		const copy = device.createCommandEncoder();
		copy.copyTextureToTexture( { texture }, { texture: alias }, [ texture.width, texture.height, texture.depthOrArrayLayers ] );
		device.queue.submit( [ copy.finish() ] );
		twins.set( texture, alias );
		return alias;

	};

	return {
		passes: texture => downsampler.prepare( twin( texture ), { colorSpace: 'srgb' } ).length,
		encode: ( encoder, texture ) => downsampler.encode( encoder, twin( texture ), { colorSpace: 'srgb' } ),
		async check( textures, name ) {

			if ( ! textures.buffer ) return { compared: false };
			const [ ours, theirs ] = [ await rawLevels( device, twin( textures[ name ] ) ), await rawLevels( device, textures.buffer ) ];
			let differing = 0;
			ours.forEach( ( level, i ) => level.forEach( ( byte, j ) => { if ( byte !== theirs[ i ][ j ] ) differing ++; } ) );
			if ( differing ) throw new Error( `srgb colorSpace ${ textures[ name ].width }x${ textures[ name ].height }: ${ differing } bytes differ from the buffer path` );
			return { compared: true, levels: ours.length, differing };

		}
	};

}

/** Every level of a 4-byte-per-texel texture, as bytes. */
async function rawLevels( device, texture ) {

	const layers = texture.depthOrArrayLayers, levels = [];
	let size = 0;
	for ( let level = 0; level < texture.mipLevelCount; level ++ ) {

		const width = Math.max( 1, texture.width >> level ), height = Math.max( 1, texture.height >> level );
		const bytesPerRow = Math.ceil( width * 4 / 256 ) * 256;
		levels.push( { level, width, height, bytesPerRow, offset: size } );
		size += bytesPerRow * height * layers;

	}

	const buffer = device.createBuffer( { size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	for ( const { level, width, height, bytesPerRow, offset } of levels ) encoder.copyTextureToBuffer( { texture, mipLevel: level }, { buffer, offset, bytesPerRow, rowsPerImage: height }, [ width, height, layers ] );
	device.queue.submit( [ encoder.finish() ] );
	await buffer.mapAsync( GPUMapMode.READ );
	const mapped = buffer.getMappedRange();
	const out = levels.map( ( { width, height, bytesPerRow, offset } ) => {

		const data = new Uint8Array( width * height * layers * 4 );
		for ( let row = 0; row < height * layers; row ++ ) data.set( new Uint8Array( mapped, offset + row * bytesPerRow, width * 4 ), row * width * 4 );
		return data;

	} );
	buffer.unmap();
	buffer.destroy();
	return out;

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
// - `buffer`: the library. Its stores encode without the correction step (`ds_srgb_stored_code`):
//   every stored value has been through `quantize`, so it is exactly decode(k) for some code k,
//   and the formula alone rounds it back to k (largest miss on the CPU in f32: 0.000016 of a step).
// - `slowStore` (same bits): stores with the correction step, as the library did before
//   2026-11-15 (results/windows/round2/srgb-cost.md measured that version as `buffer`, and the
//   current one as `fastStore`).
// - `noCorrect`: ds_srgb_code without its table correction step (the ±1 code fix-up).
// - `noPow`: sqrt in place of pow(c, 1/2.4) everywhere (correction kept).
// - `noSnap`: in-tile values not snapped through encode/decode (`quantize` keeps linear color);
//   stores still encode.
// - `noEncode`: no snap, and stores write round(v · 255): no sRGB maths in the shader at all.
// - `linearBuffer`: an rgba8unorm texture of the same size on the buffer path (`output: 'buffer'`):
//   the same dispatches and copies with the linear format's arithmetic.
// - `linearStorage`: that texture on the storage path, no copies.
// - `render`: one render pass per level.
const srgbEdits = {
	noCorrect: code => edit( code, /\n\tif \(k < 255u && c >= ds_srgb_table[^\n]+/, '' ),
	noPow: code => edit( code, /1\.055 \* pow\(c, 1\.0 \/ 2\.4\) - 0\.055/g, 'sqrt(c)' ),
	noSnap: code => edit( code, 'return vec4f(ds_srgb_quantize(value.xyz), round(', 'return vec4f(value.xyz, round(' ),
	slowStore: code => edit( code, /^.*<< 24u.*$/m, line => line.replaceAll( 'ds_srgb_stored_code(', 'ds_srgb_code(' ) ),
	noEncode: code => edit( edit( srgbEdits.noSnap( code ), /fn ds_srgb_code\(v: f32\) -> u32 \{[\s\S]*?\n\}/, 'fn ds_srgb_code(v: f32) -> u32 { return u32(clamp(round(v * 255.0), 0.0, 255.0)); }' ), /fn ds_srgb_stored_code\(v: f32\) -> u32 \{[\s\S]*?\n\}/, 'fn ds_srgb_stored_code(v: f32) -> u32 { return u32(clamp(round(v * 255.0), 0.0, 255.0)); }' )
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
		buffer: { ...run( new Downsampler( device, { output: 'buffer' } ) ), parity: true }
	};
	for ( const [ name, transform ] of Object.entries( srgbEdits ) ) strategies[ name ] = { ...run( new Downsampler( editingDevice( device, transform ), { output: 'buffer' } ) ), parity: name === 'slowStore' };
	strategies.linearBuffer = { ...onLinear( new Downsampler( device, { output: 'buffer' } ) ), parity: false };
	strategies.linearStorage = { ...onLinear( new Downsampler( device ) ), parity: false };
	return { strategies, reference: () => 'buffer' };

}
