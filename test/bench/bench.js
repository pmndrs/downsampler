// Benchmark: render-pass baseline vs the downsampler at each kernel width, plus
// 'average' read with textureLoad instead of the sampler, and a 'max' fold.
// Results are medians over rotating-order rounds; marker cost is subtracted.
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, median, readMip, upload } from '../browser/gpu.js';
import { createRenderBaseline } from './render-baseline.js';
import { timeGpu } from './timing.js';

export const defaultSizes = [ [ 256, 256 ], [ 512, 512 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ], [ 1920, 1080 ], [ 512, 512, 6 ] ];

// The render baseline filters linearly, so it needs filterable formats.
export const defaultFormats = [ 'rgba8unorm', 'rgba16float' ];

export async function benchmark( device, { sizes = defaultSizes, formats: benchFormats = defaultFormats, rounds = 11, repeats = 16, onProgress = () => {} } = {} ) {

	if ( ! device.features.has( 'timestamp-query' ) ) return { timestamps: false, results: [] };
	const limit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	const strategies = { render: createRenderBaseline( device ) };
	for ( const levels of [ 4, limit ].filter( ( value, index, all ) => all.indexOf( value ) === index ) ) {

		const downsampler = new Downsampler( device, { levelsPerDispatch: levels } );
		strategies[ `compute${ levels }` ] = { passes: texture => downsampler.prepare( texture ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture ) };

	}

	const load = new Downsampler( device, { levelsPerDispatch: limit, averageAccess: 'load' } );
	strategies[ `load${ limit }` ] = { passes: texture => load.prepare( texture ).length, encode: ( encoder, texture ) => load.encode( encoder, texture ) };
	const max = new Downsampler( device, { levelsPerDispatch: limit } );
	strategies[ `max${ limit }` ] = { parity: false, passes: texture => max.prepare( texture, { reduction: 'max' } ).length, encode: ( encoder, texture ) => max.encode( encoder, texture, { reduction: 'max' } ) };

	const names = Object.keys( strategies );
	const results = [];
	device.pushErrorScope( 'validation' );

	try {

		for ( const format of benchFormats ) for ( const [ width, height, layers = 1 ] of sizes ) {

			const textures = Object.fromEntries( names.map( name => [ name, createTexture( device, { width, height, layers, format, render: true } ) ] ) );

			try {

				const data = makeData( width, height, layers, format, 'random', 1337 );
				for ( const name of names ) upload( device, textures[ name ], data );
				const run = name => {

					const encoder = device.createCommandEncoder();
					strategies[ name ].encode( encoder, textures[ name ] );
					device.queue.submit( [ encoder.finish() ] );

				};

				for ( let i = 0; i < 8; i ++ ) for ( const name of names ) run( name );

				// Parity with the baseline on the timed input (a sanity check; the
				// validation suite is the correctness oracle).
				const last = textures.render.mipLevelCount - 1;
				const baseline = await readMip( device, textures.render, last );
				for ( const name of names.slice( 1 ).filter( name => strategies[ name ].parity !== false ) ) {

					const actual = await readMip( device, textures[ name ], last );
					const error = Math.max( ...actual.map( ( value, i ) => Math.abs( value - baseline[ i ] ) ) );
					if ( ! ( error < 0.05 ) ) throw new Error( `${ name } ${ width }x${ height } differs from baseline by ${ error }` );

				}

				await device.queue.onSubmittedWorkDone();
				const samples = Object.fromEntries( [ 'marker', ...names ].map( name => [ name, [] ] ) );
				const order = [ 'marker', ...names ];

				for ( let round = 0; round < rounds; round ++ ) {

					for ( const name of [ ...order.slice( round % order.length ), ...order.slice( 0, round % order.length ) ] ) {

						const texture = textures[ name === 'marker' ? 'render' : name ];
						const encode = name === 'marker' ? () => {} : encoder => strategies[ name ].encode( encoder, texture );
						samples[ name ].push( await timeGpu( device, texture, encode, repeats ) );

					}

				}

				const marker = median( samples.marker );
				const gpuMs = Object.fromEntries( names.map( name => [ name, median( samples[ name ] ) - marker ] ) );
				const passes = Object.fromEntries( names.map( name => [ name, strategies[ name ].passes( textures[ name ] ) ] ) );
				const result = { width, height, layers, format, passes, gpuMs, markerMs: marker, samples };
				results.push( result );
				onProgress( result );

			} finally {

				for ( const texture of Object.values( textures ) ) texture.destroy();

			}

		}

	} finally {

		const error = await device.popErrorScope();
		if ( error ) throw new Error( `GPU validation error: ${ error.message }` );

	}

	return { timestamps: true, rounds, repeats, strategies: names, results };

}
