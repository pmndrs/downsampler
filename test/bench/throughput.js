// Throughput: work per frame instead of timestamps (greggman's method, gpuweb#5253).
//
// No timestamp-query: a unit of work (one mip chain, one sampling pass) is encoded N
// times per submit, and submits are kept in flight for a fixed wall-clock window. The
// result is wall milliseconds per unit (and units per second), which is what an app
// sees: GPU time plus whatever per-pass, per-submit and encoding cost doesn't overlap.
//
// - `pacing: 'queue'` (default): up to `inFlight` submits outstanding, each awaited
//   with `queue.onSubmittedWorkDone()`, so the GPU never idles waiting for the CPU
//   unless encoding is the bottleneck.
// - `pacing: 'raf'`: one submit per requestAnimationFrame, as an app records a frame.
//   N is calibrated so the work exceeds a frame (`frameMs`), otherwise the rate would be
//   the display's, not the GPU's.
//
// N is calibrated per unit so one submit takes about `batchMs` of wall time, capped at
// `maxBatch`. `cpuMs` is the time spent in JS encoding and submitting; when it is most
// of the wall time (`cpuBound`), the number measures the CPU side, not the GPU. The GPU
// process (Dawn) also works between JS and the GPU, and JS can't see it: compare with
// the timestamp bench where timestamps exist (`orderAgreement`).
import { median } from '../browser/gpu.js';

const now = () => performance.now();
const frame = () => new Promise( resolve => requestAnimationFrame( resolve ) );

/** Wall ms of one submit of `n` units, from an idle queue. */
async function batchWall( device, encode, n ) {

	await device.queue.onSubmittedWorkDone();
	const start = now();
	const encoder = device.createCommandEncoder();
	for ( let i = 0; i < n; i ++ ) encode( encoder );
	device.queue.submit( [ encoder.finish() ] );
	await device.queue.onSubmittedWorkDone();
	return now() - start;

}

/** Units per submit so one submit takes about `targetMs` (doubling from 1). */
export async function calibrate( device, encode, { targetMs = 8, maxBatch = 1024 } = {} ) {

	let n = 1;
	while ( n < maxBatch && await batchWall( device, encode, n ) < targetMs ) n *= 2;
	return Math.min( n, maxBatch );

}

/**
 * One window of throughput for `encode( encoder )`, `n` units per submit.
 * Returns wall ms per unit, units per second, the CPU (JS) share and the submit count.
 */
export async function throughputWindow( device, encode, n, { windowMs = 1000, inFlight = 2, pacing = 'queue' } = {} ) {

	await device.queue.onSubmittedWorkDone();
	const pending = [];
	let units = 0, submits = 0, cpu = 0;
	const start = now();
	while ( now() - start < windowMs ) {

		if ( pacing === 'raf' ) await frame();
		const t0 = now();
		const encoder = device.createCommandEncoder();
		for ( let i = 0; i < n; i ++ ) encode( encoder );
		device.queue.submit( [ encoder.finish() ] );
		cpu += now() - t0;
		units += n; submits ++;
		pending.push( device.queue.onSubmittedWorkDone() );
		if ( pending.length >= inFlight ) await pending.shift();

	}

	await Promise.all( pending );
	const wall = now() - start;
	return { msPerUnit: wall / units, unitsPerSecond: units / wall * 1000, cpuMsPerUnit: cpu / units, cpuShare: cpu / wall, units, submits, wallMs: wall };

}

/** Throughput options from the bench options (defaults documented in test/bench/README.md). */
export function throughputOptions( { windowMs = 1000, pacing = 'queue', inFlight = 2, batchMs, maxBatch = 1024, textures = 1 } = {} ) {

	if ( pacing !== 'queue' && pacing !== 'raf' ) throw new Error( `throughput: pacing must be 'queue' or 'raf', got ${ JSON.stringify( pacing ) }` );
	// rAF pacing: the work of one frame must exceed a 60 Hz frame, or the rate is the display's.
	return { windowMs, pacing, inFlight, batchMs: batchMs ?? ( pacing === 'raf' ? 25 : 8 ), maxBatch, textures };

}

/**
 * Throughput of named units for one case (rotated order per round, median per unit).
 * `units`: { name: encode( encoder ) }. Returns per unit the median ms per unit, units/s,
 * CPU ms per unit, the batch size, whether it was CPU-bound, the raw windows, and
 * paired per-round ratios to `reference` when given.
 */
export async function measureUnits( device, units, { rounds = 5, reference, ...options } = {} ) {

	const opts = throughputOptions( options );
	const names = Object.keys( units );
	const batch = {};
	for ( const name of names ) batch[ name ] = await calibrate( device, units[ name ], { targetMs: opts.batchMs, maxBatch: opts.maxBatch } );
	const windows = Object.fromEntries( names.map( name => [ name, [] ] ) );
	for ( let round = 0; round < rounds; round ++ ) {

		for ( const name of [ ...names.slice( round % names.length ), ...names.slice( 0, round % names.length ) ] ) windows[ name ].push( await throughputWindow( device, units[ name ], batch[ name ], opts ) );

	}

	const out = {};
	for ( const name of names ) {

		const w = windows[ name ];
		const ms = median( w.map( x => x.msPerUnit ) ), cpu = median( w.map( x => x.cpuMsPerUnit ) ), share = median( w.map( x => x.cpuShare ) );
		out[ name ] = { msPerUnit: ms, unitsPerSecond: 1000 / ms, cpuMsPerUnit: cpu, cpuShare: share, cpuBound: share > 0.8, batch: batch[ name ], windows: w.map( x => x.msPerUnit ) };
		if ( reference ) out[ name ].ratio = median( w.map( ( x, round ) => x.msPerUnit / windows[ reference( name ) ][ round ].msPerUnit ) );

	}

	return { options: { ...opts, rounds }, units: out };

}

/**
 * Do two timings order the same strategies the same way? Pairs whose timestamp times
 * differ by less than `tie` (relative) are ties and not counted. Returns agreeing and
 * counted pairs, Kendall's tau-a over all pairs (ties count as neither), and the disagreeing pairs.
 */
export function orderAgreement( timestamps, throughput, { tie = 0.05 } = {} ) {

	const names = Object.keys( timestamps ).filter( name => name in throughput );
	let agree = 0, counted = 0, concordant = 0, discordant = 0;
	const disagree = [];
	for ( let i = 0; i < names.length; i ++ ) for ( let j = i + 1; j < names.length; j ++ ) {

		const [ a, b ] = [ names[ i ], names[ j ] ];
		const ts = Math.sign( timestamps[ a ] - timestamps[ b ] ), tp = Math.sign( throughput[ a ] - throughput[ b ] );
		if ( ts * tp > 0 ) concordant ++; else if ( ts * tp < 0 ) discordant ++;
		if ( Math.abs( timestamps[ a ] - timestamps[ b ] ) < tie * Math.max( timestamps[ a ], timestamps[ b ] ) ) continue;
		counted ++;
		if ( ts === tp ) agree ++; else disagree.push( `${ a } vs ${ b }: timestamps ${ timestamps[ a ].toFixed( 4 ) } / ${ timestamps[ b ].toFixed( 4 ) }, throughput ${ throughput[ a ].toFixed( 4 ) } / ${ throughput[ b ].toFixed( 4 ) }` );

	}

	const pairs = names.length * ( names.length - 1 ) / 2;
	return { agree, counted, tau: pairs ? ( concordant - discordant ) / pairs : 1, disagree };

}
