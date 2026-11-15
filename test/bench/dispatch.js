// `--compare=dispatch`: what one dependent compute dispatch costs on this device.
// The same chain (same levels, same arithmetic) planned with `levelsPerDispatch` 1 … 6 and the
// tail kernel off, so only the number of dependent dispatches changes: `lpdN` is N levels per
// dispatch. Two dispatches with nothing between them still get a barrier (each dispatch is its own
// usage scope), so a dispatch's cost here is its launch plus the wait for the one before it.
// `pass` records the 1-level plan in one compute pass (encodePass), to separate the pass boundaries
// from the dispatch boundaries. The summary fits time = a + b · dispatches per size and format:
// b is the cost of one dependent dispatch, a the work.
import { Downsampler } from '/dist/webgpu/index.js';

export const dispatchSizes = [ [ 64, 64 ], [ 256, 256 ], [ 1024, 1024 ], [ 4096, 4096 ], [ 1920, 1080 ] ];
export const dispatchFormats = [ 'rgba8unorm' ];

export function dispatchStrategies( device ) {

	const strategies = {};
	const max = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	for ( let n = max; n >= 1; n -- ) {

		const downsampler = new Downsampler( device, { levelsPerDispatch: n, tail: false } );
		strategies[ `lpd${ n }` ] = { passes: texture => downsampler.prepare( texture ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture ), parity: true };
		if ( n === 1 ) strategies.pass = { passes: texture => downsampler.prepare( texture ).length, encode: ( encoder, texture ) => {

			const pass = encoder.beginComputePass();
			downsampler.encodePass( pass, texture );
			pass.end();

		}, parity: true };

	}

	return { strategies, reference: () => `lpd${ max }` };

}

/** Least-squares fit of gpuMs = a + b · passes over the lpdN strategies of one result row. */
export function dispatchFit( row ) {

	const points = Object.keys( row.gpuMs ).filter( name => name.startsWith( 'lpd' ) ).map( name => [ row.passes[ name ], row.gpuMs[ name ] ] );
	const n = points.length, mx = points.reduce( ( s, [ x ] ) => s + x, 0 ) / n, my = points.reduce( ( s, [ , y ] ) => s + y, 0 ) / n;
	const sxx = points.reduce( ( s, [ x ] ) => s + ( x - mx ) ** 2, 0 ), sxy = points.reduce( ( s, [ x, y ] ) => s + ( x - mx ) * ( y - my ), 0 );
	const b = sxx > 0 ? sxy / sxx : 0;
	return { perDispatchUs: b * 1000, workUs: ( my - b * mx ) * 1000, dispatches: points.map( ( [ x ] ) => x ) };

}
