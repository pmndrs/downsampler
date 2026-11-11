/**
 * CPU reference for `oddSizes: 'spd'` mip chains (SPEC.md §2.1): AMD SPD
 * semantics, built on the 'clamp' tile reduction oracle (`reducePyramid`).
 *
 * One batch reads its source level and writes levels 1…count (source-relative).
 * Level l texel x reduces the source block [x·2^l, (x+1)·2^l) per axis with
 * coordinates clamped to the source; sizes are ⌊n/2^l⌋, at least 1, so the
 * last odd row or column is dropped. Nothing is rounded between the levels of a
 * batch; each is quantized once, as stored. The next batch reads the last
 * stored (quantized) level, as SPD's last workgroup reads mip 6.
 *
 * Min/max under these semantics are NOT conservative: a dropped row or column
 * is in no texel's block, so a max pyramid can miss it (Hi-Z false culling).
 */
import { quantize, type Quantization } from './formats.js';
import { pyramidTolerance, reducePyramid } from './pyramid.js';
import type { PyramidReferenceReduction, ReferenceReduction } from './reduction.js';
import type { ReferenceLevel } from './reference.js';

/** A Downsampler reduction as the tile oracle takes it: a custom fold is a fold. */
function tileReduction( reduction: ReferenceReduction ): PyramidReferenceReduction {

	return typeof reduction === 'function' ? { reference: reduction, kind: 'fold' } : reduction;

}

/**
 * Reference for one 'spd' batch: levels 1…`count` of `source` (RGBA,
 * layer-major then row-major), each quantized to the storage format.
 * `filterTolerance` carries the float allowance of the kernel's f32 sums
 * (`pyramidTolerance` of the terms' magnitude), so `tolerance()` applies as for
 * other levels; folds select values and stay exact.
 */
export function spdReference( source: ArrayLike<number>, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction = 'average', count = 1 ): ReferenceLevel[] {

	const levels = Array.from( { length: count }, ( _, i ) => i + 1 );
	return reducePyramid( source, width, height, layers, { reduction: tileReduction( reduction ), levels, edges: 'clamp' } ).map( level => ( {
		width: level.width, height: level.height, layers,
		values: Float32Array.from( level.values, value => quantize( value, quantization ) ),
		filterTolerance: Float32Array.from( level.magnitude, pyramidTolerance )
	} ) );

}

/**
 * Reference for a whole 'spd' chain of `levels` output levels at
 * `levelsPerDispatch` levels per batch, each batch reading the previous
 * batch's last quantized level. Level i of the result is mip i + 1.
 */
export function spdChainReference( source: ArrayLike<number>, width: number, height: number, layers: number, quantization: Quantization, reduction: ReferenceReduction, levels: number, levelsPerDispatch: number ): ReferenceLevel[] {

	const chain: ReferenceLevel[] = [];
	let input: ArrayLike<number> = source, w = width, h = height;

	while ( chain.length < levels ) {

		const batch = spdReference( input, w, h, layers, quantization, reduction, Math.min( levelsPerDispatch, levels - chain.length ) );
		chain.push( ...batch );
		const last = batch[ batch.length - 1 ];
		input = last.values; w = last.width; h = last.height;

	}

	return chain;

}

/**
 * Hi-Z conservativeness of one max level against the base level: the number of
 * level texels (channel 0, all layers) smaller than some base texel inside
 * their normalized area [x, x+1) · width / w. Zero for a conservative pyramid.
 * `base` and `level` are RGBA, layer-major then row-major.
 */
export function maxViolations( base: ArrayLike<number>, width: number, height: number, level: ArrayLike<number>, w: number, h: number, layers = 1 ): number {

	let violations = 0;
	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		const x0 = Math.floor( x * width / w ), x1 = Math.ceil( ( x + 1 ) * width / w ) - 1;
		const y0 = Math.floor( y * height / h ), y1 = Math.ceil( ( y + 1 ) * height / h ) - 1;
		const value = level[ ( ( layer * h + y ) * w + x ) * 4 ];
		let bounded = true;
		for ( let sy = y0; sy <= y1 && bounded; sy ++ ) for ( let sx = x0; sx <= x1 && bounded; sx ++ ) bounded = value >= base[ ( ( layer * height + sy ) * width + sx ) * 4 ];
		if ( ! bounded ) violations ++;

	}

	return violations;

}
