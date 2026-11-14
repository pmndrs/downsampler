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
import { measureUnits, orderAgreement, throughputOptions } from './throughput.js';
import { compareSpdSizes, firstUnitStep, webgpuSpdVariants } from './webgpu-spd.js';
import { storageUsage } from './storage-usage.js';
import { kernelVariants } from './kernel-variants.js';
import { srgbFormats, srgbStrategies } from './srgb.js';

export const defaultSizes = [ [ 256, 256 ], [ 512, 512 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ], [ 1920, 1080 ], [ 512, 512, 6 ] ];

// The render baseline filters linearly, so it needs filterable formats.
export const defaultFormats = [ 'rgba8unorm', 'rgba16float' ];

export const compareSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 3840, 2160 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ] ];

/** M2T.4 strategies: exact and tail for average and max, spd and the atomic probe for average, and the render baseline. */
function compareStrategies( device ) {

	const exact = new Downsampler( device, { tail: false } ), tail = new Downsampler( device, { tail: true } );
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
		// M2.5: area-weighted odd sizes, the 'exact' batches (odd-level batches load), with and without the tail.
		strategies.box = { ...run( new Downsampler( device, { tail: false } ), { reduction, oddSizes: 'box' } ), parity: false };
		strategies.tailBox = { ...run( tail, { reduction, oddSizes: 'box' } ), parity: false };
		strategies.haloBox = { ...run( new Downsampler( device, { tail: true, halo: true } ), { reduction, oddSizes: 'box' } ), parity: false };
		// Windows decisions (2026-11-12, results/windows/correctness.md): shader weights instead of
		// the sampler ('load', identical on every GPU), and the device's own f16 conversion
		// ('device', the cost of rounding to nearest even; the same as exact for non-f16 formats).
		const load = new Downsampler( device, { averageAccess: 'load', tail: false } ), tailLoad = new Downsampler( device, { averageAccess: 'load', tail: true } );
		strategies.exactLoad = { ...run( load, { reduction } ), parity: true };
		strategies.tailLoad = { ...run( tailLoad, { reduction } ), parity: true };
		strategies.exactDevice = { ...run( new Downsampler( device, { f16Rounding: 'device', tail: false } ), { reduction } ), parity: true };
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

/** The native harnesses' sizes (results/windows/native-tables.md). */
export const renderCompareSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 1024, 1024 ], [ 4096, 4096 ] ];

/**
 * `compare: 'render'` (DIVERGES #17): where render passes beat 'exact' compute.
 * `hybrid<K>` renders levels 1..K and finishes with 'exact' from level K: the same
 * results as both (render passes sample texel centres with normalized coordinates,
 * as 'exact' does). `hybridSpd1` finishes with 'spd' instead (different semantics).
 * `renderL1` / `exactL1` write only level 1, isolating the largest level; `exactBuffer`
 * writes through the buffer output path, isolating storage-texture writes.
 * Ratios are paired per round against render passes.
 */
function renderStrategies( device ) {

	// 'exact' strategies pin tail: false (the library's default is the tail kernel since 2026-11-13); 'tail' is its own column.
	const exact = new Downsampler( device, { tail: false } ), buffer = new Downsampler( device, { output: 'buffer', tail: false } );
	const render = createRenderBaseline( device );
	const run = ( downsampler, options ) => ( { passes: texture => downsampler.prepare( texture, options ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture, options ) } );
	const strategies = { render, exact: { ...run( exact, {} ), parity: true }, spd: { ...run( exact, { oddSizes: 'spd' } ), parity: false }, exactBuffer: { ...run( buffer, {} ), parity: true } };
	const hybrid = ( levels, oddSizes ) => ( {
		parity: oddSizes === 'exact',
		passes: texture => Math.min( levels, texture.mipLevelCount - 1 ) + ( levels < texture.mipLevelCount - 1 ? exact.prepare( texture, { baseMipLevel: levels, oddSizes } ).length : 0 ),
		encode: ( encoder, texture ) => {

			render.encodeLevels( encoder, texture, 1, levels );
			if ( levels < texture.mipLevelCount - 1 ) exact.encode( encoder, texture, { baseMipLevel: levels, oddSizes } );

		}
	} );
	for ( const levels of [ 1, 2, 3 ] ) strategies[ `hybrid${ levels }` ] = hybrid( levels, 'exact' );
	strategies.hybridSpd1 = hybrid( 1, 'spd' );
	strategies.renderL1 = { parity: false, passes: () => 1, encode: ( encoder, texture ) => render.encodeLevels( encoder, texture, 1, 1 ) };
	strategies.exactL1 = { ...run( exact, { mipLevelCount: 2 } ), parity: false };
	// Workgroup shape: 'exact' batches default to 32 × 32 tiles (16 × 16 invocations);
	// exact8 is the previous default for batches of up to four levels.
	strategies.exact8 = { ...run( new Downsampler( device, { exactTile: 8, tail: false } ), {} ), parity: true };
	strategies.tail = { ...run( new Downsampler( device, { tail: true } ), {} ), parity: true };
	// Tile sweep (Apple: 32 measured slower than 8 at render sizes, 2026-11-13).
	strategies.exact16 = { ...run( new Downsampler( device, { exactTile: 16, tail: false } ), {} ), parity: true };
	strategies.tail8 = { ...run( new Downsampler( device, { tail: true, exactTile: 8 } ), {} ), parity: true };
	strategies.tail16 = { ...run( new Downsampler( device, { tail: true, exactTile: 16 } ), {} ), parity: true };
	// Halo batches (SPEC §13 item 3): the default chain (tail on) with odd steps carried inside batches.
	strategies.halo = { ...run( new Downsampler( device, { tail: true, halo: true } ), {} ), parity: true };
	return { strategies, reference: () => 'render' };

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

	const exact = new Downsampler( device, { tail: false } ), tail = new Downsampler( device, { tail: true } );
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

/**
 * `strategies` (`--strategies=a,b,c`): time only these, plus the ones a mode's ratios
 * divide by (`keep`). Unknown names throw with the mode's list.
 */
export function pickStrategies( set, names, keep = [] ) {

	if ( ! names?.length ) return set;
	const all = Object.keys( set.strategies );
	const unknown = names.filter( name => ! all.includes( name ) );
	if ( unknown.length ) throw new Error( `unknown strategies ${ unknown.join( ', ' ) }; this mode has ${ all.join( ', ' ) }` );
	const wanted = new Set( [ ...names, ...keep.filter( name => all.includes( name ) ) ] );
	return { ...set, strategies: Object.fromEntries( Object.entries( set.strategies ).filter( ( [ name ] ) => wanted.has( name ) ) ) };

}

/** `compare: 'throughput'`: the strategies timed per frame (and, as a check, by timestamps). */
export const throughputStrategies = [ 'render', 'exact', 'spd', 'tail' ];

export async function benchmark( device, { sizes, formats: benchFormats, rounds, repeats = 16, compare, strategies: only, throughput, noTimestamps = false, windowMs, pacing, inFlight, batchMs, maxBatch, textures, onProgress = () => {} } = {} ) {

	const timestamps = device.features.has( 'timestamp-query' );
	const tp = { windowMs, pacing, inFlight, batchMs, maxBatch, textures };
	// Modes with their own harness (and their own default sizes and formats). Without
	// timestamp-query they measure throughput only.
	if ( compare === 'storage-usage' ) return storageUsage( device, { sizes, formats: benchFormats, rounds: rounds ?? 11, repeats, shaders: only, throughput: throughput || ! timestamps ? tp : null, onProgress } );
	if ( compare === 'subgroups' || compare === 'f16' ) return kernelVariants( device, { probe: compare, sizes, formats: benchFormats, rounds: rounds ?? 11, repeats, variants: only, throughput: throughput || ! timestamps ? tp : null, onProgress } );
	// sRGB output paths (test/bench/srgb.js): ratios paired against the buffer path.
	if ( compare === 'srgb' ) return timestamps ? measure( device, pickStrategies( srgbStrategies( device ), only, [ 'buffer' ] ), { sizes: sizes ?? renderCompareSizes, benchFormats: benchFormats ?? srgbFormats, rounds: rounds ?? 11, repeats, onProgress } ) : { timestamps: false, results: [] };
	benchFormats ??= defaultFormats;
	if ( compare === 'throughput' ) {

		sizes ??= renderCompareSizes;
		const set = pickStrategies( pickStrategies( renderStrategies( device ), throughputStrategies ), only, [ 'render' ] );
		return measureThroughput( device, set, { sizes, benchFormats, rounds: rounds ?? 5, repeats, timestamps: timestamps && ! noTimestamps, ...tp, onProgress } );

	}

	rounds ??= 11;
	if ( ! timestamps ) return { timestamps: false, results: [] };
	sizes ??= compare === 'spd' ? compareSizes : compare === 'webgpu-spd' ? compareSpdSizes : compare === 'render' ? renderCompareSizes : defaultSizes;
	if ( compare === 'render' ) return measure( device, pickStrategies( renderStrategies( device ), only, [ 'render' ] ), { sizes, benchFormats, rounds, repeats, onProgress } );
	if ( compare === 'webgpu-spd' ) {

		// Strategies differ per format (r32float has no render baseline or average), so measure per format.
		const all = { timestamps: true, rounds, repeats, strategies: [], results: [] };
		for ( const format of benchFormats ) {

			const set = webgpuSpdStrategies( device, format );
			const part = await measure( device, only ? pickStrategies( set, only.filter( name => name in set.strategies ), [ 'exact', 'exactMax' ] ) : set, { sizes, benchFormats: [ format ], rounds, repeats, onProgress } );
			all.strategies = [ ...new Set( [ ...all.strategies, ...part.strategies ] ) ];
			all.results.push( ...part.results );

		}

		return all;

	}

	const limit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	if ( compare === 'spd' ) {

		// run.js prints ratios to 'exact' (averages) and 'exactMax' (max).
		const set = compareStrategies( device );
		return measure( device, pickStrategies( set, only, [ 'exact', ...( only?.some( name => name.endsWith( 'Max' ) ) ? [ 'exactMax' ] : [] ) ] ), { sizes, benchFormats, rounds, repeats, onProgress } );

	}

	if ( compare ) throw new Error( `unknown --compare=${ compare }; expected spd, render, webgpu-spd, throughput, storage-usage, subgroups, f16 or srgb` );
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
	return measure( device, pickStrategies( { strategies }, only, [ 'render' ] ), { sizes, benchFormats, rounds, repeats, onProgress } );

}

/**
 * Parity with the baseline (render passes when timed, else the first strategy) on the
 * timed input: a sanity check that every strategy did its work; the validation suite is
 * the correctness oracle. Returns the baseline's name.
 */
async function checkParity( device, strategies, textures, format ) {

	const names = Object.keys( strategies );
	const baselineName = 'render' in textures ? 'render' : names[ 0 ];
	const last = textures[ baselineName ].mipLevelCount - 1;
	const baseline = await readMip( device, textures[ baselineName ], last );
	for ( const name of names.filter( name => name !== baselineName && strategies[ name ].parity !== false ) ) {

		const actual = await readMip( device, textures[ name ], last );
		const error = Math.max( ...actual.map( ( value, i ) => Math.abs( value - baseline[ i ] ) ) );
		// Packed floats have 6- or 9-bit mantissas over HDR fixtures, and render passes
		// round each level their own way (Apple truncates), so allow 10 % of the range.
		const limit = format === 'rg11b10ufloat' || format === 'rgb9e5ufloat' ? 0.1 * Math.max( 1, ...baseline.map( Math.abs ) ) : 0.05;
		if ( ! ( error < limit ) ) throw new Error( `${ name } ${ textures[ name ].width }x${ textures[ name ].height } differs from baseline by ${ error }` );

	}

	return baselineName;

}

/**
 * `compare: 'throughput'`: wall ms per chain with `n` chains per submit and submits kept in
 * flight for a fixed window (test/bench/throughput.js), rotated order, median over rounds;
 * no timestamp-query needed. `textures` > 1 cycles through that many identical textures per
 * strategy (independent chains, as when an app generates mips for several textures a frame).
 * Where timestamps exist (and unless `timestamps` is false), the same strategies are then
 * timed with timestamps, and `agreement` says whether both order them the same way.
 */
async function measureThroughput( device, set, { sizes, benchFormats, rounds, repeats, timestamps, onProgress, ...options } ) {

	const { strategies } = set;
	const names = Object.keys( strategies );
	const tp = throughputOptions( options );
	const reference = names.includes( 'render' ) ? () => 'render' : undefined;
	const results = [];
	device.pushErrorScope( 'validation' );

	try {

		for ( const format of benchFormats ) for ( const [ width, height, layers = 1 ] of sizes ) {

			const pools = Object.fromEntries( names.map( name => [ name, Array.from( { length: tp.textures }, () => createTexture( device, { width, height, layers, format, render: true } ) ) ] ) );

			try {

				const data = makeData( width, height, layers, format, 'random', 1337 );
				for ( const pool of Object.values( pools ) ) for ( const texture of pool ) upload( device, texture, data );
				for ( const name of names ) for ( const texture of pools[ name ] ) {

					const encoder = device.createCommandEncoder();
					strategies[ name ].encode( encoder, texture );
					device.queue.submit( [ encoder.finish() ] );

				}

				await checkParity( device, strategies, Object.fromEntries( names.map( name => [ name, pools[ name ][ 0 ] ] ) ), format );
				const units = Object.fromEntries( names.map( name => {

					let next = 0;
					const pool = pools[ name ];
					return [ name, encoder => strategies[ name ].encode( encoder, pool[ next ++ % pool.length ] ) ];

				} ) );
				const measured = await measureUnits( device, units, { rounds, reference, ...tp } );
				const passes = Object.fromEntries( names.map( name => [ name, strategies[ name ].passes( pools[ name ][ 0 ] ) ] ) );
				const msPerChain = Object.fromEntries( names.map( name => [ name, measured.units[ name ].msPerUnit ] ) );
				const result = { width, height, layers, format, passes, msPerChain, throughput: measured.units };
				results.push( result );
				onProgress( result );

			} finally {

				for ( const pool of Object.values( pools ) ) for ( const texture of pool ) texture.destroy();

			}

		}

	} finally {

		const error = await device.popErrorScope();
		if ( error ) throw new Error( `GPU validation error: ${ error.message }` );

	}

	const out = { mode: 'throughput', timestamps, throughputOptions: tp, rounds, strategies: names, results };
	if ( ! timestamps ) return out;
	// The check: the same strategies, sizes and formats by timestamps (marker-subtracted GPU ms).
	const timed = await measure( device, { ...set, reference }, { sizes, benchFormats, rounds: Math.max( rounds, 5 ), repeats, onProgress } );
	timed.results.forEach( ( row, i ) => {

		results[ i ].gpuMs = row.gpuMs;
		results[ i ].gpuRatios = row.ratios;
		results[ i ].agreement = orderAgreement( row.gpuMs, results[ i ].msPerChain );

	} );
	const sum = key => results.reduce( ( total, row ) => total + row.agreement[ key ], 0 );
	out.agreement = { agree: sum( 'agree' ), counted: sum( 'counted' ), meanTau: sum( 'tau' ) / results.length, tie: 0.05 };
	return out;

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

				for ( const name of names.filter( name => strategies[ name ].check ) ) checks[ name ] = await strategies[ name ].check( textures, name );
				const baselineName = await checkParity( device, strategies, textures, format );
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
