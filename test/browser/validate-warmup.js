// Async warm-up must preserve every texel on the actual encode paths. Scheduling,
// cache identity, rejection and creation counts are covered by core/warmup.test.js.
import { boxReference, formats } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, mipLevels, readMips, toFloats, upload } from './gpu.js';
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

/** Exercise both auto fallbacks with a small tunable boundary: full chain renders,
 * a 63-texel update computes, and a 64-texel update renders again. In particular,
 * an alias's dynamic storage template discovers full-size topology without changing
 * the compute path selected from the small update rectangle.
 */
async function validateAutoThresholdWarmup( device ) {

	for ( const mode of [ 'srgb', 'alias', 'render-only-alias' ] ) {

		const alias = mode !== 'srgb', renderOnly = mode === 'render-only-alias';

		const width = 65, height = 33, layers = 2;
		const format = alias ? 'rgba8unorm' : 'rgba8unorm-srgb';
		const texture = device.createTexture( {
			size: [ width, height, layers ], format, mipLevelCount: mipLevels( width, height ), textureBindingViewDimension: '2d-array',
			...( alias ? { viewFormats: [ 'rgba8unorm-srgb' ] } : {} ),
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | ( alias && ! renderOnly ? GPUTextureUsage.STORAGE_BINDING : 0 )
		} );
		const counts = { createComputePipeline: 0, createRenderPipeline: 0, createComputePipelineAsync: 0, createRenderPipelineAsync: 0 };
		const measured = new Proxy( device, { get( target, key ) {

			const value = Reflect.get( target, key, target );
			if ( Object.hasOwn( counts, key ) ) return descriptor => { counts[ key ] ++; return value.call( target, descriptor ); };
			return typeof value === 'function' ? value.bind( target ) : value;

		} } );
		const ds = new Downsampler( measured, { renderPathMinTexels: renderOnly ? Infinity : 64, averageAccess: 'load', tail: false } );
		const base = { oddSizes: 'box', ...( alias ? { colorSpace: 'srgb' } : {} ) };
		const data = makeData( width, height, layers, 'rgba8unorm-srgb', 'random', 29 );
		const decode = level => alias ? toFloats( Uint8Array.from( level, value => Math.round( value * 255 ) ), 'rgba8unorm-srgb' ) : level;
		const checkChain = async label => {

			const mips = ( await readMips( device, texture, 0 ) ).map( decode );
			let w = width, h = height;
			for ( let level = 1; level < mips.length; level ++ ) {

				const oracle = boxReference( mips[ level - 1 ], w, h, layers, 'srgb8', 'average', { weights: 'shader' } );
				check( mips[ level ], oracle, 'srgb8', `warmup auto ${ mode } ${ label } mip ${ level }` );
				w = oracle.width; h = oracle.height;

			}

		};
		try {

			upload( device, texture, data );
			if ( renderOnly && ( ds.outputPath( texture ) !== null || ds.unsupportedReason( texture ) === null || ds.scratchSize( texture, base ) !== 0 ) ) throw new Error( 'warmup render-only alias: preserve option-blind queries and report zero scratch for the alias' );
			await ds.warmup( texture, base );
			if ( counts.createRenderPipelineAsync === 0 || counts.createComputePipelineAsync !== 0 ) throw new Error( 'warmup auto threshold: full chain must warm render only' );
			ds.generate( texture, base );
			await checkChain( 'full' );
			const small = { ...base, rect: { x: 56, y: 24, width: 7, height: 9 }, layers: [ 1, 1 ] };
			await ds.warmup( texture, small );
			if ( renderOnly ? counts.createComputePipelineAsync !== 0 : counts.createComputePipelineAsync === 0 ) throw new Error( 'warmup auto threshold: small update chose the wrong compute/render fallback' );
			if ( renderOnly && ds.scratchSize( texture, small ) !== 0 ) throw new Error( 'warmup render-only alias: small update needs no scratch' );
			ds.generate( texture, small );
			await checkChain( 'small' );
			const compiled = counts.createComputePipelineAsync + counts.createRenderPipelineAsync;
			for ( const [ label, rect ] of [
				[ 'clipped edge', { x: 60, y: 26, width: 5, height: 7 } ],
				[ 'boundary', { x: 57, y: 25, width: 8, height: 8 } ],
				[ 'small revisit', small.rect ]
			] ) {

				const options = { ...base, rect, layers: [ 1, 1 ] };
				await ds.warmup( texture, options );
				ds.generate( texture, options );
				await checkChain( label );

			}
			if ( counts.createComputePipeline !== 0 || counts.createRenderPipeline !== 0 ) throw new Error( 'warmup auto threshold: encode missed a warmed pipeline (check dynamic full-size topology path)' );
			if ( counts.createComputePipelineAsync + counts.createRenderPipelineAsync !== compiled ) throw new Error( 'warmup auto threshold: moving/clipped/boundary updates compiled new variants' );

		} finally {

			await device.queue.onSubmittedWorkDone();
			ds.release( texture );
			texture.destroy();

		}

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
			if ( name === 'storage' ) {

				await validateDeferredCompute( device, ds, texture, data );
				if ( device.features.has( 'core-features-and-limits' ) ) await validateAutoThresholdWarmup( device );

			}
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
			results.push( { warmup: name, levels: actual.length, partialUpdates: 3, ...( name === 'storage' ? { deferredValidationRetries: 2, customRecovery: 'max', ...( device.features.has( 'core-features-and-limits' ) ? { autoThresholdChains: 3 } : {} ) } : {} ), ...( syncMips === undefined ? {} : { syncFirstRender: true } ) } );

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
