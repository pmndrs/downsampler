// Benchmark: render-pass baseline vs the downsampler at each kernel width, plus
// 'average' read with textureLoad instead of the sampler, and a 'max' fold.
// `compare: 'spd'` (M2T.4) instead times the odd-size strategies against each
// other: 'exact', the exact tail kernel, `oddSizes: 'spd'` and the single-dispatch
// atomic probe (measurement only, test/bench/atomic-spd.js), for average and max.
// Results are medians over rotating-order rounds; marker cost is subtracted.
import { spdReference, tolerance } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, median, readMip, readMips, upload } from '../browser/gpu.js';
import { AtomicSpd } from './atomic-spd.js';
import { createRenderBaseline } from './render-baseline.js';
import { timeGpu } from './timing.js';
import { compareSpdSizes, firstUnitStep, webgpuSpdVariants } from './webgpu-spd.js';

export const defaultSizes = [ [ 256, 256 ], [ 512, 512 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ], [ 1920, 1080 ], [ 512, 512, 6 ] ];

// The render baseline filters linearly, so it needs filterable formats.
export const defaultFormats = [ 'rgba8unorm', 'rgba16float' ];

export const compareSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 3840, 2160 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ] ];

/** M2T.4 strategies: exact and tail for average and max, spd and the atomic probe for average, and the render baseline. */
function compareStrategies( device ) {

	const exact = new Downsampler( device ), tail = new Downsampler( device, { tail: true } );
	const strategies = { render: createRenderBaseline( device ) };
	for ( const [ suffix, reduction ] of [ [ '', 'average' ], [ 'Max', 'max' ] ] ) {

		const run = ( downsampler, options ) => ( { passes: texture => downsampler.prepare( texture, options ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture, options ) } );
		strategies[ `exact${ suffix }` ] = { ...run( exact, { reduction } ), parity: reduction === 'average' };
		strategies[ `tail${ suffix }` ] = { ...run( tail, { reduction } ), parity: reduction === 'average' };
		// 'spd' (and the probe, which reproduces it) is 'average' only: Downsampler
		// rejects folds under 'spd'. The 2026-11-11 runs in results/spd/ predate that
		// and also timed spdMax and atomicMax.
		if ( reduction !== 'average' ) continue;
		strategies.spd = { ...run( exact, { reduction, oddSizes: 'spd' } ), parity: false };
		// Windows decisions (2026-11-12, results/windows/correctness.md): shader weights instead of
		// the sampler ('load', identical on every GPU), and the device's own f16 conversion
		// ('device', the cost of rounding to nearest even; the same as exact for non-f16 formats).
		const load = new Downsampler( device, { averageAccess: 'load' } ), tailLoad = new Downsampler( device, { averageAccess: 'load', tail: true } );
		strategies.exactLoad = { ...run( load, { reduction } ), parity: true };
		strategies.tailLoad = { ...run( tailLoad, { reduction } ), parity: true };
		strategies.exactDevice = { ...run( new Downsampler( device, { f16Rounding: 'device' } ), { reduction } ), parity: true };
		{

			// All batches in one compute pass (encodePass), as a tool records them:
			// separates per-pass overhead from the dispatch structure itself.
			const inPass = ( downsampler, options ) => ( { passes: texture => downsampler.prepare( texture, options ).length, encode: ( encoder, texture ) => {

				const pass = encoder.beginComputePass();
				downsampler.encodePass( pass, texture, options );
				pass.end();

			} } );
			strategies.exactPass = { ...inPass( exact, {} ), parity: true };
			strategies.tailPass = { ...inPass( tail, {} ), parity: true };
			strategies.spdPass = { ...inPass( exact, { oddSizes: 'spd' } ), parity: false };

		}

		const atomic = new AtomicSpd( device, { reduction } );
		strategies.atomic = { passes: texture => atomic.passes( texture ), encode: ( encoder, texture ) => atomic.encode( encoder, texture ), parity: false, atomic, reduction };
		// Kernel only: packed tail levels stay in the buffer (not texture-ready).
		strategies.atomicKernel = { passes: () => 1, encode: ( encoder, texture ) => atomic.encode( encoder, texture, { materialize: false } ), parity: false, kernelOnly: atomic };

	}

	return { strategies, exact };

}

/**
 * Before timing, the atomic probe must reproduce `oddSizes: 'spd'` on four
 * changing inputs (stale payloads would show), with its counter reset: levels
 * 1–6 bit for bit (same tile kernel), the tail within the format tolerance
 * (payloads are packed with pack4x8unorm / pack2x16float, stored levels by the
 * texture store). 'spd' itself is checked against `spdReference` here too.
 */
async function checkAtomic( device, { atomic, reduction }, exact, width, height, format ) {

	const a = createTexture( device, { width, height, format } ), b = createTexture( device, { width, height, format } );
	const quantization = format === 'rgba8unorm' ? 'unorm8' : 'float16';
	let tailError = 0, tailOutside = 0;
	// 'spd' against its oracle is reported, not fatal: samplers that round their weights
	// differently (Intel truncates, results/windows/correctness.md) shouldn't stop a timing run.
	const spd = { outside: 0, values: 0, first: null };

	try {

		for ( const seed of [ 3, 4, 5, 6 ] ) {

			const data = makeData( width, height, 1, format, seed % 2 ? 'random' : 'tiles', seed );
			upload( device, a, data ); upload( device, b, data );
			const encoder = device.createCommandEncoder();
			atomic.encode( encoder, a );
			exact.encode( encoder, b, { reduction, oddSizes: 'spd' } );
			device.queue.submit( [ encoder.finish() ] );
			const [ ma, mb ] = [ await readMips( device, a, 1 ), await readMips( device, b, 1 ) ];
			const counter = await atomic.counter( a );
			if ( counter !== 0 ) throw new Error( `atomic probe ${ width }x${ height } ${ format }: counter ${ counter } after the chain` );
			const label = `atomic probe ${ width }x${ height } ${ format } ${ reduction } seed ${ seed }`;
			for ( let level = 1; level <= ma.length; level ++ ) {

				const x = ma[ level - 1 ], y = mb[ level - 1 ];
				for ( let i = 0; i < x.length; i ++ ) {

					const error = Math.abs( x[ i ] - y[ i ] );
					// Levels 1–6 are the same tile kernel: bit for bit. Levels 7+ read mip 6 by load (the
					// probe's payload) where 'spd' samples it, so a truncating sampler (Intel) differs there.
					if ( level <= 6 && ! Object.is( x[ i ], y[ i ] ) ) throw new Error( `${ label }: level ${ level } value ${ i }: ${ x[ i ] } vs spd ${ y[ i ] }` );
					if ( level > 6 ) {

						tailError = Math.max( tailError, error );
						if ( error > tolerance( y[ i ], quantization ) ) tailOutside ++;

					}

				}

			}

			// 'spd' against its oracle (first input only; the GPU matrix covers the rest).
			if ( seed === 3 ) for ( const batch of exact.prepare( b, { reduction, oddSizes: 'spd' } ) ) {

				const source = batch.sourceMipLevel === 0 ? ( await readMips( device, b, 0, 1 ) )[ 0 ] : mb[ batch.sourceMipLevel - 1 ];
				spdReference( source, Math.max( 1, width >> batch.sourceMipLevel ), Math.max( 1, height >> batch.sourceMipLevel ), 1, quantization, reduction, batch.count ).forEach( ( expected, i ) => {

					const actual = mb[ batch.firstMipLevel + i - 1 ];
					for ( let j = 0; j < actual.length; j ++ ) {

						spd.values ++;
						if ( Math.abs( actual[ j ] - expected.values[ j ] ) <= tolerance( expected.values[ j ], quantization, expected.filterTolerance[ j ], reduction ) ) continue;
						spd.outside ++;
						spd.first ??= `mip ${ batch.firstMipLevel + i } value ${ j }: ${ actual[ j ] } vs ${ expected.values[ j ] }`;

					}

				} );

			}

		}

	} finally {

		a.destroy(); b.destroy();

	}

	if ( spd.outside ) console.warn( `spd ${ width }x${ height } ${ format } ${ reduction }: ${ spd.outside } of ${ spd.values } values outside the oracle's tolerance (first: ${ spd.first })` );
	if ( tailOutside ) console.warn( `atomic probe ${ width }x${ height } ${ format } ${ reduction }: ${ tailOutside } tail values outside the format tolerance of 'spd' (max error ${ tailError })` );
	return { tailError, tailOutside, spdOutside: spd.outside, spdValues: spd.values };

}

/**
 * webgpu-spd (results/compare/webgpu-spd.md) against 'exact', the tail kernel,
 * `oddSizes: 'spd'` and render passes. Averages for rgba8unorm/rgba16float,
 * max (Hi-Z) for every format, r32float max only. Before timing, each
 * webgpu-spd variant is compared with our 'spd' (average; format tolerance)
 * or with wspd6 (max; exact) on every level above its first unit step (where
 * WGSL leaves its out-of-bounds reads implementation-defined): levels 1–6 must
 * match; mismatches from level 7 on, which depend on its atomic handoff, are
 * recorded in `checks` (they occur on this device: results/compare/webgpu-spd.md).
 */
function webgpuSpdStrategies( device, format ) {

	const exact = new Downsampler( device ), tail = new Downsampler( device, { tail: true } );
	const variants = webgpuSpdVariants( device );
	const run = ( downsampler, options ) => ( { passes: texture => downsampler.prepare( texture, options ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture, options ) } );
	const spd = ( variant, config, against ) => ( {
		passes: texture => variant.pass( texture, config ).numPasses,
		encode: ( encoder, texture ) => variant.encode( encoder, texture, config ),
		parity: false,
		check: async ( textures, name ) => {

			const texture = textures[ name ], reference = textures[ against ];
			const unit = firstUnitStep( texture.width, texture.height, texture.mipLevelCount - 1 );
			const [ a, b ] = [ await readMips( device, texture, 1, Math.min( unit, texture.mipLevelCount ) ), await readMips( device, reference, 1, Math.min( unit, texture.mipLevelCount ) ) ];
			const quantization = { rgba8unorm: 'unorm8', rgba16float: 'float16', r32float: 'float32' }[ format ];
			// Levels 1–6 come from one workgroup's own tile and must match. Levels from 7 on
			// depend on the handoff (or on a level that did): mismatches there are recorded,
			// not fatal, since the timing is still that of the same work.
			const mismatches = {};
			a.forEach( ( level, l ) => level.forEach( ( value, i ) => {

				const limit = config.filter ? 0 : tolerance( b[ l ][ i ], quantization );
				if ( Math.abs( value - b[ l ][ i ] ) <= limit ) return;
				// Max is exact on every GPU; averages depend on the sampler's weight rounding (Intel
				// truncates, results/windows/correctness.md), so they're recorded like the handoff's.
				if ( l + 1 <= 6 && config.filter ) throw new Error( `${ name } ${ texture.width }x${ texture.height } ${ format } level ${ l + 1 } value ${ i }: ${ value } vs ${ against } ${ b[ l ][ i ] }` );
				mismatches[ l + 1 ] = ( mismatches[ l + 1 ] ?? 0 ) + 1;

			} ) );
			return { against, levelsCompared: a.length, mismatches };

		}
	} );
	const strategies = {};
	if ( format !== 'r32float' ) {

		strategies.exact = { ...run( exact, {} ), parity: true };
		strategies.render = createRenderBaseline( device );
		strategies.tail = { ...run( tail, {} ), parity: true };
		strategies.spd = { ...run( exact, { oddSizes: 'spd' } ), parity: false };
		for ( const [ name, variant ] of Object.entries( variants ) ) strategies[ name ] = spd( variant, {}, 'spd' );

	}

	strategies.exactMax = { ...run( exact, { reduction: 'max' } ), parity: false };
	// wspd6 has no handoff; its own correctness is in the check run (results/compare/webgpu-spd-check.json).
	strategies.wspd6Max = { ...spd( variants.wspd6, { filter: 'max' }, null ), check: undefined };
	strategies.wspdMax = spd( variants.wspd, { filter: 'max' }, 'wspd6Max' );
	return { strategies, reference: name => name.endsWith( 'Max' ) ? 'exactMax' : 'exact' };

}

export async function benchmark( device, { sizes, formats: benchFormats = defaultFormats, rounds = 11, repeats = 16, compare, onProgress = () => {} } = {} ) {

	if ( ! device.features.has( 'timestamp-query' ) ) return { timestamps: false, results: [] };
	sizes ??= compare === 'spd' ? compareSizes : compare === 'webgpu-spd' ? compareSpdSizes : defaultSizes;
	if ( compare === 'webgpu-spd' ) {

		// Strategies differ per format (r32float has no render baseline or average), so measure per format.
		const all = { timestamps: true, rounds, repeats, strategies: [], results: [] };
		for ( const format of benchFormats ) {

			const part = await measure( device, webgpuSpdStrategies( device, format ), { sizes, benchFormats: [ format ], rounds, repeats, onProgress } );
			all.strategies = [ ...new Set( [ ...all.strategies, ...part.strategies ] ) ];
			all.results.push( ...part.results );

		}

		return all;

	}

	const limit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	if ( compare === 'spd' ) return measure( device, compareStrategies( device ), { sizes, benchFormats, rounds, repeats, onProgress } );
	const strategies = { render: createRenderBaseline( device ) };
	for ( const levels of [ 4, limit ].filter( ( value, index, all ) => all.indexOf( value ) === index ) ) {

		const downsampler = new Downsampler( device, { levelsPerDispatch: levels } );
		strategies[ `compute${ levels }` ] = { passes: texture => downsampler.prepare( texture ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture ) };

	}

	const load = new Downsampler( device, { levelsPerDispatch: limit, averageAccess: 'load' } );
	strategies[ `load${ limit }` ] = { passes: texture => load.prepare( texture ).length, encode: ( encoder, texture ) => load.encode( encoder, texture ) };
	const max = new Downsampler( device, { levelsPerDispatch: limit } );
	strategies[ `max${ limit }` ] = { parity: false, passes: texture => max.prepare( texture, { reduction: 'max' } ).length, encode: ( encoder, texture ) => max.encode( encoder, texture, { reduction: 'max' } ) };
	// The buffer output path (SPEC §6.1): packed texels into a buffer, then a copy per level.
	// Six levels per dispatch at any storage texture limit.
	const buffer = new Downsampler( device, { output: 'buffer' } );
	strategies.buffer6 = { passes: texture => buffer.prepare( texture ).length, encode: ( encoder, texture ) => buffer.encode( encoder, texture ) };
	return measure( device, { strategies }, { sizes, benchFormats, rounds, repeats, onProgress } );

}

async function measure( device, { strategies, exact, reference }, { sizes, benchFormats, rounds, repeats, onProgress } ) {

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

				// The atomic probe reproduces 'spd' (and 'spd' its oracle) before it is timed.
				const checks = {};
				for ( const name of names.filter( name => strategies[ name ].atomic ) ) checks[ name ] = { ...await checkAtomic( device, strategies[ name ], exact, width, height, format ), tail: strategies[ name ].atomic.prepare( textures[ name ] ).tail, packedLevels: strategies[ name ].atomic.prepare( textures[ name ] ).packed.length };

				// Parity with the baseline on the timed input (a sanity check; the
				// validation suite is the correctness oracle).
				const baselineName = 'render' in textures ? 'render' : names[ 0 ];
				const last = textures[ baselineName ].mipLevelCount - 1;
				const baseline = await readMip( device, textures[ baselineName ], last );
				for ( const name of names.filter( name => strategies[ name ].check ) ) checks[ name ] = await strategies[ name ].check( textures, name );
				for ( const name of names.filter( name => name !== baselineName && strategies[ name ].parity !== false ) ) {

					const actual = await readMip( device, textures[ name ], last );
					const error = Math.max( ...actual.map( ( value, i ) => Math.abs( value - baseline[ i ] ) ) );
					// Packed floats have 6- or 9-bit mantissas over HDR fixtures, and render passes
					// round each level their own way (Apple truncates), so allow 10 % of the range.
					const limit = format === 'rg11b10ufloat' || format === 'rgb9e5ufloat' ? 0.1 * Math.max( 1, ...baseline.map( Math.abs ) ) : 0.05;
					if ( ! ( error < limit ) ) throw new Error( `${ name } ${ width }x${ height } differs from baseline by ${ error }` );

				}

				await device.queue.onSubmittedWorkDone();
				const samples = Object.fromEntries( [ 'marker', ...names ].map( name => [ name, [] ] ) );
				const order = [ 'marker', ...names ];

				for ( let round = 0; round < rounds; round ++ ) {

					for ( const name of [ ...order.slice( round % order.length ), ...order.slice( 0, round % order.length ) ] ) {

						const texture = textures[ name === 'marker' ? baselineName : name ];
						const encode = name === 'marker' ? () => {} : encoder => strategies[ name ].encode( encoder, texture );
						samples[ name ].push( await timeGpu( device, texture, encode, repeats ) );

					}

				}

				const marker = median( samples.marker );
				const gpuMs = Object.fromEntries( names.map( name => [ name, median( samples[ name ] ) - marker ] ) );
				const passes = Object.fromEntries( names.map( name => [ name, strategies[ name ].passes( textures[ name ] ) ] ) );
				// Paired per-round ratios to the reference strategy (rounds rotate order, so
				// both samples of a pair come from the same round), median over rounds.
				const ratios = reference ? Object.fromEntries( names.map( name => [ name, median( samples[ name ].map( ( value, round ) => ( value - marker ) / ( samples[ reference( name ) ][ round ] - marker ) ) ) ] ) ) : undefined;
				const result = { width, height, layers, format, passes, gpuMs, ratios, markerMs: marker, samples, checks };
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
