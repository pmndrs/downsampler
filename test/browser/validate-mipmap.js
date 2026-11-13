// `downsampler/mipmap` (SPEC.md §10 Vendoring): an engine's in-place chain built from the
// standalone entry alone (kernelDefaults, planMipmapBatches, createMipmapKernel, the bindings
// its documentation names), as three.js vendors it, must give a Downsampler's chain bit for
// bit. Its kernels equal the Downsampler's but for float16, where they define ds_round_f16
// alone (`f16Pack: false`); this runs them on the device.
import { createMipmapKernel, kernelDefaults, mipmapAccess, mipmapFormats, planMipmapBatches } from '/dist/mipmap.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, poison, readMips, upload } from './gpu.js';
import { compatibility } from './validate.js';

const shapes = [ [ 1920, 1080 ], [ 255, 257 ], [ 31, 31 ], [ 1, 300 ], [ 300, 1 ], [ 1, 1 ], [ 33, 17, 3 ], [ 64, 64, 6 ], [ 96, 40, 1, '2d-array' ] ];
const quickShapes = [ [ 255, 257 ], [ 31, 31 ], [ 1, 300 ], [ 33, 17, 3 ] ];
const chainFormats = [ 'rgba8unorm', 'bgra8unorm', 'rgba8snorm', 'r8unorm', 'rgba16unorm', 'r16snorm', 'rgba16float', 'r16float', 'rg16float', 'rgba32float', 'r32float' ];
const quickFormats = [ 'rgba8unorm', 'rgba16float', 'r16float', 'rgba32float' ];

/** An in-place chain recorder from `downsampler/mipmap` alone. */
function mipmapChain( device ) {

	const defaults = kernelDefaults( device.limits, device.adapterInfo );
	const sampler = device.createSampler( { minFilter: 'linear', magFilter: 'linear' } );
	const pipelines = new Map();
	const pipeline = ( format, dimension, access, batch ) => {

		const options = { format, dimension, count: batch.count, access, tail: batch.kernel === 'tail', tile: batch.tile };
		const key = JSON.stringify( options );
		if ( ! pipelines.has( key ) ) {

			const visibility = GPUShaderStage.COMPUTE;
			const entries = access === 'sample' ? [ { binding: 0, visibility, sampler: { type: 'filtering' } } ] : [];
			entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
			for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format, viewDimension: dimension } } );
			const layout = device.createBindGroupLayout( { entries } );
			const module = device.createShaderModule( { code: createMipmapKernel( options ) } );
			pipelines.set( key, { layout, pipeline: device.createComputePipeline( { layout: device.createPipelineLayout( { bindGroupLayouts: [ layout ] } ), compute: { module, entryPoint: 'main' } } ) } );

		}

		return pipelines.get( key );

	};

	return {
		defaults,
		plan: texture => planMipmapBatches( texture.width, texture.height, { mipLevelCount: texture.mipLevelCount, ...defaults } ),
		encode( encoder, texture, dimension ) {

			const { format } = texture;
			const access = mipmapAccess( format, device.features );
			const view = level => texture.createView( { dimension, baseMipLevel: level, mipLevelCount: 1 } );
			const pass = encoder.beginComputePass();
			for ( const batch of this.plan( texture ) ) {

				const { layout, pipeline: compute } = pipeline( format, dimension, access, batch );
				const entries = access === 'sample' ? [ { binding: 0, resource: sampler } ] : [];
				entries.push( { binding: 1, resource: view( batch.sourceMipLevel ) } );
				for ( let i = 0; i < batch.count; i ++ ) entries.push( { binding: i + 2, resource: view( batch.firstMipLevel + i ) } );
				pass.setPipeline( compute );
				pass.setBindGroup( 0, device.createBindGroup( { layout, entries } ) );
				pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, texture.depthOrArrayLayers );

			}

			pass.end();

		}
	};

}

export async function validateMipmap( device, { quick = false } = {} ) {

	// Compatibility mode binds one view of a texture per dispatch: no in-place chain.
	if ( compatibility( device ) ) return [ { option: 'mipmap: skipped in compatibility mode (in-place chains need core)' } ];

	const chain = mipmapChain( device );
	const downsampler = new Downsampler( device );
	const { defaults } = chain;
	for ( const key of [ 'levelsPerDispatch', 'exactTile', 'tail' ] ) if ( defaults[ key ] !== downsampler[ key ] ) throw new Error( `mipmap: kernelDefaults ${ key } ${ defaults[ key ] } but the Downsampler resolved ${ downsampler[ key ] }` );

	let chains = 0;
	const skipped = [];
	for ( const format of quick ? quickFormats : chainFormats ) {

		const feature = mipmapFormats[ format ].feature;
		if ( feature !== undefined && ! device.features.has( feature ) ) {

			skipped.push( format );
			continue;

		}

		for ( const [ width, height, layers = 1, view ] of quick ? quickShapes : shapes ) {

			const dimension = view ?? ( layers > 1 ? '2d-array' : '2d' );
			const label = `mipmap ${ format } ${ width }×${ height }×${ layers } ${ dimension }`;
			const ours = createTexture( device, { width, height, layers, format, dimension } );
			const theirs = createTexture( device, { width, height, layers, format, dimension } );
			try {

				// The plan is the Downsampler's.
				const shape = batches => batches.map( ( { kernel, sourceMipLevel, firstMipLevel, count, tile, groupsX, groupsY } ) => `${ kernel }:${ sourceMipLevel }:${ firstMipLevel }:${ count }:${ tile }:${ groupsX }×${ groupsY }` ).join( ' ' );
				if ( shape( chain.plan( ours ) ) !== shape( downsampler.prepare( theirs ) ) ) throw new Error( `${ label }: planMipmapBatches ${ shape( chain.plan( ours ) ) }, Downsampler ${ shape( downsampler.prepare( theirs ) ) }` );
				const data = makeData( width, height, layers, format, 'random', width * 7 + height );
				for ( const texture of [ ours, theirs ] ) {

					upload( device, texture, data );
					poison( device, texture, 1 );

				}

				const encoder = device.createCommandEncoder();
				chain.encode( encoder, ours, dimension );
				downsampler.encode( encoder, theirs );
				device.queue.submit( [ encoder.finish() ] );
				const [ a, b ] = await Promise.all( [ readMips( device, ours, 1 ), readMips( device, theirs, 1 ) ] );
				a.forEach( ( level, l ) => {

					const i = level.findIndex( ( value, k ) => ! Object.is( value, b[ l ][ k ] ) );
					if ( i >= 0 ) throw new Error( `${ label }: level ${ l + 1 } differs from the Downsampler's at value ${ i } (${ level[ i ] } vs ${ b[ l ][ i ] })` );

				} );
				chains ++;

			} finally {

				ours.destroy();
				theirs.destroy();

			}

		}

	}

	return [ { option: `mipmap: ${ chains } chains from downsampler/mipmap alone bit-identical to a Downsampler's (kernelDefaults ${ defaults.levelsPerDispatch } levels, exact tile ${ defaults.exactTile }, tail tile ${ defaults.tailMaxTile })${ skipped.length > 0 ? `; ${ skipped.join( ', ' ) } skipped (feature missing)` : '' }` } ];

}
