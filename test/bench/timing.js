// GPU timing for arbitrary command sequences.
//
// Timestamps are written by two one-pixel render "marker" passes that bracket a
// batch of repeats. The markers have real resource dependencies on the texture
// under test (the first writes level 1, the last samples the final level), so an
// implementation cannot overlap them with the measured work. Marker cost is
// measured separately (an empty batch) and subtracted by the caller. Empty or
// independent compute markers produced invalid timestamps on Apple/Metal during
// development, which is why render markers are used.
const markers = new WeakMap();

function getMarkers( device, texture ) {

	let marker = markers.get( texture );
	if ( marker ) return marker;
	const module = device.createShaderModule( { label: 'timing marker', code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	return vec4f(array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3))[index], 0, 1);
}
@fragment fn fs() -> @location(0) vec4f { return textureLoad(image, vec2i(0), 0); }` } );
	const pipeline = format => device.createRenderPipeline( { layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [ { format } ] } } );
	const before = pipeline( texture.format ), after = pipeline( 'rgba8unorm' );
	const last = texture.createView( { dimension: '2d', baseMipLevel: texture.mipLevelCount - 1, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 1 } );
	const small = device.createTexture( { size: [ 1, 1 ], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT } );
	marker = {
		before, after,
		beforeTarget: texture.createView( { dimension: '2d', baseMipLevel: Math.min( 1, texture.mipLevelCount - 1 ), mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 1 } ),
		afterTarget: small.createView(),
		beforeBindings: device.createBindGroup( { layout: before.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: last } ] } ),
		afterBindings: device.createBindGroup( { layout: after.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: last } ] } )
	};
	markers.set( texture, marker );
	return marker;

}

/** Median-ready sample: GPU milliseconds per repeat of `encode`, including marker overhead. */
export async function timeGpu( device, texture, encode, repeats ) {

	const marker = getMarkers( device, texture );
	const querySet = device.createQuerySet( { type: 'timestamp', count: 2 } );
	const resolve = device.createBuffer( { size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC } );
	const readback = device.createBuffer( { size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST } );
	const encoder = device.createCommandEncoder();
	const pass = ( view, pipeline, bindings, timestampWrites, loadOp ) => {

		const p = encoder.beginRenderPass( { colorAttachments: [ { view, loadOp, storeOp: 'store' } ], timestampWrites } );
		p.setPipeline( pipeline ); p.setBindGroup( 0, bindings ); p.setViewport( 0, 0, 1, 1, 0, 1 ); p.setScissorRect( 0, 0, 1, 1 ); p.draw( 3 ); p.end();

	};

	pass( marker.beforeTarget, marker.before, marker.beforeBindings, { querySet, beginningOfPassWriteIndex: 0 }, 'load' );
	for ( let i = 0; i < repeats; i ++ ) encode( encoder );
	pass( marker.afterTarget, marker.after, marker.afterBindings, { querySet, endOfPassWriteIndex: 1 }, 'clear' );
	encoder.resolveQuerySet( querySet, 0, 2, resolve, 0 );
	encoder.copyBufferToBuffer( resolve, 0, readback, 0, 16 );
	device.queue.submit( [ encoder.finish() ] );
	await readback.mapAsync( GPUMapMode.READ );
	const [ start, end ] = new BigUint64Array( readback.getMappedRange() );
	readback.unmap(); readback.destroy(); resolve.destroy(); querySet.destroy();
	if ( start === 0n || end <= start ) throw new Error( `invalid GPU timestamps: ${ start }, ${ end }` );
	return Number( end - start ) / 1e6 / repeats;

}
