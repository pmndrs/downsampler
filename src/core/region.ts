/**
 * Backend-neutral region math for partial updates (SPEC.md §5 `rect`, SPD's
 * rectInfo): given a dirty rectangle of a chain's first source level, which
 * texels of each later level depend on it.
 *
 * A texel is recomputed exactly when its footprint touches the previous level's
 * region, so a partial update writes the same bits a full recompute would, as
 * long as the source changed only inside the rect and the old chain was valid.
 * Every other texel is left untouched.
 *
 * Footprints, per axis, from a source level of size S:
 * - 'exact' and 'tail' batches (`oddSizes: 'exact'`): texels 2x and 2x + 1, plus
 *   2x + 2 along an odd S, clamped (`footprint()`; the bilinear sample of an
 *   odd-size average lies within it, at any sampler weight rounding). A same-size
 *   level (a separate source copied into the first level) reads texel x alone.
 *   `oddSizes: 'box'` reads the same footprint, and a resampled separate source
 *   (`resample: 'area'`) the area span of [x·S/n, (x+1)·S/n) (`areaDependents`).
 *   Each level follows from the previous one.
 * - 'clamp' batches (`oddSizes: 'spd'`): level l of the batch reduces the source
 *   block [x·2^l, (x+1)·2^l) with coordinates clamped to S, straight from the
 *   batch source (a dropped row or column belongs to no level-1 texel but can
 *   reach a deeper one through the clamp, so levels aren't chained inside a batch).
 */
import { mipSize, type Batch } from './plan.js';

/** A texel rectangle: x ≤ column < x + width, y ≤ row < y + height. */
export interface Rect {
	x: number;
	y: number;
	width: number;
	height: number;
}

/** What one batch writes in a partial update. */
export interface BatchRegion {
	/** Per output level i of the batch (`firstMipLevel + i`): the texels written, or null for none. */
	levels: ( Rect | null )[];
	/** First workgroup of the dispatch, in the batch's tiles (a tail batch: 0, 0). */
	groupX: number;
	groupY: number;
	/** Workgroups to dispatch; 0 when the batch writes nothing. */
	groupsX: number;
	groupsY: number;
}

type Range = readonly [ number, number ] | null;

/**
 * Texels of a level of `size` that read texels `range` (inclusive) of a source
 * level of `sourceSize` through one 'exact' step: the inverse of `footprint()`,
 * or the identity for a same-size copy.
 */
export function exactDependents( range: Range, sourceSize: number, size: number ): Range {

	if ( range === null ) return null;
	const [ a, b ] = range;
	if ( size === sourceSize ) return [ a, b ];
	// Any other ratio: a resampled separate source (`resample: 'area'`).
	if ( size !== Math.max( 1, sourceSize >> 1 ) ) return areaDependents( range, sourceSize, size );
	// Output x covers 2x … 2x + 1 (even S, or S = 1), or 2x … 2x + 2 (odd S > 1).
	const lo = sourceSize > 1 && sourceSize % 2 === 1 ? Math.max( 0, ( a - 1 ) >> 1 ) : a >> 1;
	return [ lo, Math.min( size - 1, b >> 1 ) ];

}

/**
 * Texels of a level of `size` whose area span (`areaSpan` in core: the source texels
 * of [x·S/n, (x+1)·S/n)) meets texels `range` (inclusive) of a level of `sourceSize`:
 * a resampled source (`resample: 'area'`) at any ratio. At 2:1 the span is the 'exact'
 * footprint, so this agrees with {@link exactDependents} there, and the identity for a copy.
 */
export function areaDependents( range: Range, sourceSize: number, size: number ): Range {

	if ( range === null ) return null;
	const [ a, b ] = range;
	// The last texel of x's span is ⌊((x + 1)·S − 1) / n⌋ ≥ a, and its first ⌊x·S / n⌋ ≤ b.
	const lo = Math.max( 0, Math.ceil( ( a * size + 1 ) / sourceSize ) - 1 );
	const hi = Math.min( size - 1, Math.ceil( ( b + 1 ) * size / sourceSize ) - 1 );
	return lo <= hi ? [ lo, hi ] : null;

}

/**
 * Texels of level `level` (1 = half) of a 'clamp' batch, `size` texels long, that
 * read texels `range` (inclusive) of the batch source: those whose clamped block
 * [x·2^l, (x+1)·2^l − 1] meets the range. None when only dropped texels changed.
 */
export function clampDependents( range: Range, level: number, size: number ): Range {

	if ( range === null ) return null;
	const lo = range[ 0 ] >> level, hi = Math.min( size - 1, range[ 1 ] >> level );
	return lo <= hi ? [ lo, hi ] : null;

}

const rangesOf = ( rect: Rect | null ): [ Range, Range ] => rect === null || rect.width === 0 || rect.height === 0 ? [ null, null ] : [ [ rect.x, rect.x + rect.width - 1 ], [ rect.y, rect.y + rect.height - 1 ] ];
const rectOf = ( x: Range, y: Range ): Rect | null => x === null || y === null ? null : { x: x[ 0 ], y: y[ 0 ], width: x[ 1 ] - x[ 0 ] + 1, height: y[ 1 ] - y[ 0 ] + 1 };

/**
 * Why `rect` is not a texel rectangle inside a `width` × `height` level, or null.
 * An empty rect (width or height 0) is valid, as WebGPU's zero-size copies are:
 * nothing depends on it. It must still lie inside the level (x ≤ width, y ≤ height).
 */
export function rectProblem( rect: Rect, width: number, height: number ): string | null {

	if ( rect === null || typeof rect !== 'object' ) return 'rect must be { x, y, width, height }';
	const { x, y, width: w, height: h } = rect;
	if ( ! [ x, y, w, h ].every( Number.isInteger ) ) return 'rect x, y, width and height must be integers';
	if ( w < 0 || h < 0 ) return `rect width and height must not be negative, got ${ w }×${ h }`;
	if ( x < 0 || y < 0 || x + w > width || y + h > height ) return `rect ${ w }×${ h } at (${ x }, ${ y }) is outside the ${ width }×${ height } level it marks dirty`;
	return null;

}

/**
 * Per batch, the texels a partial update writes and the workgroups that cover
 * them. `rect` marks the dirty texels of the first batch's source: level
 * `sourceMipLevel` of the destination (in place), or the separate source of
 * `sourceWidth` × `sourceHeight` (`Batch.sourceMipLevel === null`). `width` and
 * `height` are the destination's level-0 size; batches come from `planBatches`.
 *
 * A batch's workgroups span every written texel of every level it writes; they
 * may compute more texels than that, but store only the written ones. An empty
 * `rect` writes nothing: every level is null and every batch has 0 workgroups.
 */
export function planRegions( batches: readonly Batch[], rect: Rect, width: number, height: number, sourceWidth?: number, sourceHeight?: number ): BatchRegion[] {

	if ( batches.length === 0 ) return [];
	const first = batches[ 0 ];
	const external = first.sourceMipLevel === null;
	if ( external && ( sourceWidth === undefined || sourceHeight === undefined ) ) throw new Error( 'downsampler: a batch reads a separate source; pass its size.' );
	const levelWidth = ( level: number | null ) => level === null ? sourceWidth! : mipSize( width, level );
	const levelHeight = ( level: number | null ) => level === null ? sourceHeight! : mipSize( height, level );
	const problem = rectProblem( rect, levelWidth( first.sourceMipLevel ), levelHeight( first.sourceMipLevel ) );
	if ( problem !== null ) throw new Error( `downsampler: ${ problem }.` );

	// Destination level → the texels written (or dirty, for the first source level).
	const written = new Map<number | null, Rect | null>( [ [ first.sourceMipLevel, rect ] ] );

	return batches.map( batch => {

		if ( ! written.has( batch.sourceMipLevel ) ) throw new Error( `downsampler: batch reads level ${ batch.sourceMipLevel }, which no earlier batch writes.` );
		const source = written.get( batch.sourceMipLevel )!;
		const sw = levelWidth( batch.sourceMipLevel ), sh = levelHeight( batch.sourceMipLevel );
		const levels: ( Rect | null )[] = [];
		let [ rx, ry ] = rangesOf( source ), pw = sw, ph = sh;

		for ( let i = 0; i < batch.count; i ++ ) {

			const level = batch.firstMipLevel + i;
			const w = mipSize( width, level ), h = mipSize( height, level );
			if ( batch.kernel === 'clamp' ) {

				const [ sx, sy ] = rangesOf( source );
				levels.push( rectOf( clampDependents( sx, i + 1, w ), clampDependents( sy, i + 1, h ) ) );

			} else {

				rx = exactDependents( rx, pw, w );
				ry = exactDependents( ry, ph, h );
				levels.push( rectOf( rx, ry ) );
				pw = w; ph = h;

			}

			written.set( level, levels[ i ] );

		}

		const any = levels.some( region => region !== null );
		if ( batch.kernel === 'tail' ) return { levels, groupX: 0, groupY: 0, groupsX: any ? 1 : 0, groupsY: any ? 1 : 0 };

		// Workgroup g writes texels [g·e, (g + 1)·e) of level i, with e = tile >> i.
		let x0 = Infinity, x1 = - Infinity, y0 = Infinity, y1 = - Infinity;
		levels.forEach( ( region, i ) => {

			if ( region === null ) return;
			const extent = batch.tile >> i;
			x0 = Math.min( x0, Math.floor( region.x / extent ) );
			x1 = Math.max( x1, Math.floor( ( region.x + region.width - 1 ) / extent ) );
			y0 = Math.min( y0, Math.floor( region.y / extent ) );
			y1 = Math.max( y1, Math.floor( ( region.y + region.height - 1 ) / extent ) );

		} );
		return any ? { levels, groupX: x0, groupY: y0, groupsX: x1 - x0 + 1, groupsY: y1 - y0 + 1 } : { levels, groupX: 0, groupY: 0, groupsX: 0, groupsY: 0 };

	} );

}

/** Why `layers` is not a `[ first, count ]` range of a texture with `layerCount` layers, or null. */
export function layersProblem( layers: readonly [ number, number ], layerCount: number ): string | null {

	if ( ! Array.isArray( layers ) || layers.length !== 2 ) return 'layers must be [ first, count ]';
	const [ first, count ] = layers;
	if ( ! Number.isInteger( first ) || ! Number.isInteger( count ) ) return 'layers first and count must be integers';
	if ( count < 1 ) return `layers count must be at least 1, got ${ count }`;
	if ( first < 0 || first + count > layerCount ) return `layers [ ${ first }, ${ count } ] are outside the texture's ${ layerCount } layer${ layerCount === 1 ? '' : 's' }`;
	return null;

}
