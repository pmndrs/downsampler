// Async warm-up must preserve every texel on the actual encode paths. Scheduling,
// cache identity, rejection and creation counts are covered by core/warmup.test.js.
import { boxReference, formats } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, readMips, toFloats, upload } from './gpu.js';
import { check } from './validate.js';

export async function validateWarmup( device ) {

	const results = [];
	for ( const [ name, config, format ] of [
		[ 'storage', {}, 'rgba8unorm' ],
		[ 'buffer', { output: 'buffer' }, 'rgba8unorm' ],
		[ 'compat-copy', { compatibility: true }, 'rgba8unorm' ],
		[ 'render', { output: 'render' }, 'rgba8unorm' ],
		[ 'srgb-render', { output: 'render' }, 'rgba8unorm-srgb' ]
	] ) {

		const width = 65, height = 33, layers = 2;
		const texture = createTexture( device, { width, height, layers, format, render: config.output === 'render' } );
		const ds = new Downsampler( device, { ...config, averageAccess: 'load', tail: false } );
		const reason = ds.unsupportedReason( texture );
		if ( reason !== null ) { texture.destroy(); results.push( { warmup: name, skip: reason } ); continue; }
		device.pushErrorScope( 'validation' );
		try {

			const data = makeData( width, height, layers, format, 'random', 17 );
			upload( device, texture, data );
			const options = { oddSizes: 'box' };
			await Promise.all( [ ds.warmup( texture, options ), ds.warmup( texture, options ) ] );
			ds.generate( texture, options );
			const expected = await readMips( device, texture, 1 );
			let previous = toFloats( data, format ), w = width, h = height;
			for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

				const oracle = boxReference( previous, w, h, layers, formats[ format ].quantization, 'average', { weights: 'shader' } );
				check( expected[ level - 1 ], oracle, formats[ format ].quantization, `warmup ${ name } mip ${ level }` );
				previous = expected[ level - 1 ]; w = oracle.width; h = oracle.height;

			}
			// Release execution resources, retain pipelines, then warm and encode moving
			// rects/layer ranges. Immutable regions and warmed pipelines leave pixels intact.
			ds.release( texture );
			for ( const x of [ 1, 9, 1 ] ) {

				const partial = { ...options, rect: { x, y: 3, width: 11, height: 9 }, layers: [ 1, 1 ] };
				await ds.warmup( texture, partial );
				ds.generate( texture, partial );

			}
			const actual = await readMips( device, texture, 1 );
			if ( actual.some( ( level, i ) => level.some( ( value, j ) => value !== expected[ i ][ j ] ) ) ) throw new Error( `warmup ${ name }: partial revisit changed mip contents` );
			results.push( { warmup: name, levels: actual.length, partialUpdates: 3 } );

		} finally {

			await device.queue.onSubmittedWorkDone();
			const error = await device.popErrorScope();
			ds.release( texture );
			texture.destroy();
			if ( error !== null ) throw new Error( `warmup ${ name }: ${ error.message }` );

		}

	}
	return results;

}
