/**
 * Reductions: how a footprint of source texels becomes one output texel.
 *
 * - 'average' is a normalized-coordinate bilinear sample at the output texel
 *   centre (an exact 2×2 box for even sizes), matching render-pass mipmaps.
 * - Every other reduction is a fold over the texels the output covers, with a
 *   binary `reduce` that must be associative, commutative and idempotent (a
 *   semilattice, like min, max or bitwise or). Idempotence makes duplicated reads
 *   (clamped unit dimensions) and overlapping odd footprints harmless.
 */

export type BuiltinReduction = 'average' | 'min' | 'max' | 'minmax';

/** A custom fold reduction. */
export interface CustomReduction {
	/**
	 * WGSL defining `fn reduce(a: vec4f, b: vec4f) -> vec4f`, associative,
	 * commutative and idempotent. Values are in the texture's normalized units.
	 */
	wgsl: string;
}

export type Reduction = BuiltinReduction | CustomReduction;

/** A reduction as the CPU reference understands it: a builtin, or a JS fold. */
export type ReferenceReduction = BuiltinReduction | ( ( a: number, b: number, channel: number ) => number );

export function isBuiltinReduction( value: unknown ): value is BuiltinReduction {

	return value === 'average' || value === 'min' || value === 'max' || value === 'minmax';

}

/** Validates a reduction and returns a stable cache key for it. */
export function reductionKey( reduction: Reduction ): string {

	if ( isBuiltinReduction( reduction ) ) return reduction;
	if ( typeof reduction !== 'object' || reduction === null || typeof reduction.wgsl !== 'string' ) throw new Error( `downsampler: unknown reduction ${ JSON.stringify( reduction ) }.` );
	if ( ! /\bfn\s+reduce\s*\(/.test( reduction.wgsl ) ) throw new Error( 'downsampler: a custom reduction must define `fn reduce(a: vec4f, b: vec4f) -> vec4f`.' );
	return `custom:${ reduction.wgsl }`;

}

/** The builtin fold for one channel: minmax takes the min of even channels and the max of odd ones. */
export function foldChannel( reduction: Exclude<BuiltinReduction, 'average'>, a: number, b: number, channel: number ): number {

	if ( reduction === 'min' || ( reduction === 'minmax' && channel % 2 === 0 ) ) return Math.min( a, b );
	return Math.max( a, b );

}

/**
 * Source texels covered by output texel `x` along one axis, inclusive: `2x` and
 * `2x + 1`, plus `2x + 2` when the source size is odd (the output then spans
 * 2 + 1/n source texels), clamped to the source. This is the conservative rule
 * min/max (Hi-Z) needs.
 */
export function footprint( x: number, size: number ): [ number, number ] {

	const first = Math.min( 2 * x, size - 1 );
	return [ first, Math.min( 2 * x + 1 + ( size & 1 ), size - 1 ) ];

}
