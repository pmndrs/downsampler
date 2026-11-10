// The common render-pass approach: one pass per level and layer, sampling the
// previous level with a linear sampler. Used as the comparison baseline.
const pipelines = new Map();

export function createRenderBaseline( device ) {

	const sampler = device.createSampler( { minFilter: 'linear', magFilter: 'linear' } );
	const module = device.createShaderModule( { label: 'render baseline', code: `
struct Varyings { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@vertex fn vs(@builtin(vertex_index) index: u32) -> Varyings {
	let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3))[index];
	return Varyings(vec4f(p, 0, 1), p * vec2f(0.5, -0.5) + 0.5);
}
@fragment fn fs(in: Varyings) -> @location(0) vec4f {
	return textureSampleLevel(source, imageSampler, in.uv, 0.0);
}` } );
	const plans = new WeakMap();

	const pipeline = format => {

		if ( ! pipelines.has( format ) ) pipelines.set( format, device.createRenderPipeline( {
			label: 'render baseline', layout: 'auto',
			vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [ { format } ] }
		} ) );
		return pipelines.get( format );

	};

	return {
		passes: texture => ( texture.mipLevelCount - 1 ) * texture.depthOrArrayLayers,
		encode( encoder, texture ) {

			let plan = plans.get( texture );
			if ( ! plan ) {

				plan = [];
				const p = pipeline( texture.format );
				for ( let layer = 0; layer < texture.depthOrArrayLayers; layer ++ ) for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

					const view = ( mip ) => texture.createView( { dimension: '2d', baseMipLevel: mip, mipLevelCount: 1, baseArrayLayer: layer, arrayLayerCount: 1 } );
					plan.push( { target: view( level ), bindGroup: device.createBindGroup( { layout: p.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: view( level - 1 ) }, { binding: 1, resource: sampler } ] } ), pipeline: p } );

				}

				plans.set( texture, plan );

			}

			for ( const { target, bindGroup, pipeline } of plan ) {

				const pass = encoder.beginRenderPass( { colorAttachments: [ { view: target, loadOp: 'clear', storeOp: 'store' } ] } );
				pass.setPipeline( pipeline );
				pass.setBindGroup( 0, bindGroup );
				pass.draw( 3 );
				pass.end();

			}

		}
	};

}
