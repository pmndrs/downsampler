// Async warm-up must preserve every texel on the actual encode paths. Scheduling,
// cache identity, rejection and creation counts are covered by core/warmup.test.js.
import { boxReference, formats } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, readMips, toFloats, upload } from './gpu.js';
import { check } from './validate.js';

/** Native sync creation returns invalid pipeline objects and reports validation later.
 * Nested scopes consume the intentional errors without leaking into the suite's scope.
 */
async function validateDeferredCompute( device, ds, texture, data ) {

	const invalid = { reduction: { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { return ds_intentionally_undefined_value; }' } };
	let prepared, syncError;
	device.pushErrorScope( 'validation' );
	try {

		prepared = ds.prepare( texture, invalid );

	} finally {

		syncError = await device.popErrorScope();

	}
	if ( prepared.length === 0 || prepared.some( batch => batch.pipeline === undefined ) ) throw new Error( 'warmup deferred compute: synchronous creation did not return a prepared plan' );
	if ( syncError === null ) throw new Error( 'warmup deferred compute: invalid WGSL produced no deferred validation error' );
	for ( let retry = 0; retry < 2; retry ++ ) {

		let rejection;
		device.pushErrorScope( 'validation' );
		try {

			await ds.warmup( texture, invalid );

		} catch ( error ) {

			rejection = error;

		} finally {

			// createShaderModule can report the syntax/type error through this scope;
			// createComputePipelineAsync reports its own failure through its promise.
			await device.popErrorScope();

		}
		if ( rejection?.name !== 'GPUPipelineError' || rejection.reason !== 'validation' ) throw new Error( `warmup deferred compute retry ${ retry }: expected native validation rejection, got ${ rejection?.name ?? 'success' }` );

	}
	const valid = { oddSizes: 'box', reduction: { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { return max(a, b); }' } };
	await ds.warmup( texture, valid );
	ds.generate( texture, valid );
	const mips = await readMips( device, texture, 1 );
	let previous = toFloats( data, texture.format ), width = texture.width, height = texture.height;
	for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

		const oracle = boxReference( previous, width, height, texture.depthOrArrayLayers, formats[ texture.format ].quantization, 'max', { weights: 'shader' } );
		check( mips[ level - 1 ], oracle, formats[ texture.format ].quantization, `warmup deferred compute recovery mip ${ level }`, 'max' );
		previous = mips[ level - 1 ]; width = oracle.width; height = oracle.height;

	}

}

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
			if ( name === 'storage' ) await validateDeferredCompute( device, ds, texture, data );
			// Render plans obtain their bind-group layout from the pipeline. Exercise
			// sync-first bindings, then their replacement by a native async-validated plan.
			let syncMips;
			if ( config.output === 'render' ) {

				ds.generate( texture, options );
				syncMips = await readMips( device, texture, 1 );

			}
			await Promise.all( [ ds.warmup( texture, options ), ds.warmup( texture, options ) ] );
			ds.generate( texture, options );
			const expected = await readMips( device, texture, 1 );
			if ( syncMips !== undefined && expected.some( ( level, i ) => level.some( ( value, j ) => value !== syncMips[ i ][ j ] ) ) ) throw new Error( `warmup ${ name }: sync-first render differs after async validation` );
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
			results.push( { warmup: name, levels: actual.length, partialUpdates: 3, ...( name === 'storage' ? { deferredValidationRetries: 2, customRecovery: 'max' } : {} ), ...( syncMips === undefined ? {} : { syncFirstRender: true } ) } );

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
