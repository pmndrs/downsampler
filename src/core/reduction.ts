/**
 * Reductions: how a footprint of source texels becomes one output texel.
 *
 * Under the 'exact' edge policy (mip chains, `Downsampler`):
 * - 'average' is a normalized-coordinate bilinear sample at the output texel
 *   centre (an exact 2×2 box for even sizes), matching render-pass mipmaps.
 * - Every other reduction is a fold over the texels the output covers, with a
 *   binary `reduce` that must be associative, commutative and idempotent (a
 *   semilattice, like min, max or bitwise or). Idempotence makes duplicated reads
 *   (clamped unit dimensions) and overlapping odd footprints harmless.
 *
 * Under the tile edge policies ('zero-weight', 'clamp'; `Reducer`), output
 * texel x of level l reduces the 2^l × 2^l block of source texels it covers,
 * channel by channel (see `ChannelOp`). Sums stay unnormalised until a level is
 * emitted. See SPEC.md §14.1.
 */

export type BuiltinReduction = 'average' | 'min' | 'max' | 'minmax';

/**
 * What one channel of a tile reduction computes over a block of source texels:
 * - 'sum': Σ v
 * - 'average': Σ v / n, with n the texels the edge policy counts
 * - 'min', 'max'
 * - 'weight': Σ w. This channel weighs the 'weighted' channels.
 * - 'weighted': Σ w·v / Σ w (0 where Σ w = 0). Needs exactly one 'weight' channel.
 * - 'argmin', 'argmax': the key of a payload fold. The texel with the smallest
 *   (largest) key wins, and the 'payload' channels that follow the key carry
 *   that texel's values. Ties go to the smaller payload, compared
 *   lexicographically in channel order, so the result doesn't depend on the
 *   order of combination (see {@link payloadGroups}).
 * - 'payload': carried with the key before it.
 */
export type ChannelOp = 'sum' | 'average' | 'min' | 'max' | 'weight' | 'weighted' | 'argmin' | 'argmax' | 'payload';

/** A payload fold: the key channel, its kind, and the payload channels that follow it. */
export interface PayloadGroup {
	key: number;
	kind: 'argmin' | 'argmax';
	payload: number[];
}

/**
 * The payload folds in a list of channel ops: each 'argmin' or 'argmax' key
 * with the run of 'payload' channels right after it. Throws for a 'payload'
 * that doesn't follow a key or another payload.
 *
 * The fold is the lexicographic min (argmin) of (key, payload₁, payload₂, …),
 * or for argmax the max of the key and then the min of the payload: a total
 * order, so it is associative, commutative and idempotent, and the same at any
 * tree shape, edge policy or kernel width. Ties on the key go to the smaller
 * payload; a tool that wants ties broken by position puts a texel index in the
 * first payload channel. Keys and payloads must not be NaN (comparisons with
 * NaN are false, which makes the winner order-dependent); −0 and +0 compare
 * equal, so which of them is returned depends on the order.
 */
export function payloadGroups( ops: readonly string[] ): PayloadGroup[] {

	const groups: PayloadGroup[] = [];
	ops.forEach( ( op, c ) => {

		if ( op === 'argmin' || op === 'argmax' ) groups.push( { key: c, kind: op, payload: [] } );
		else if ( op === 'payload' ) {

			const group = groups[ groups.length - 1 ];
			const previous = group === undefined ? - 1 : group.payload.length > 0 ? group.payload[ group.payload.length - 1 ] : group.key;
			if ( group === undefined || previous !== c - 1 ) throw new Error( `downsampler: a 'payload' channel (${ c }) must follow an 'argmin' or 'argmax' key, or another payload of it; got ${ JSON.stringify( ops ) }.` );
			group.payload.push( c );

		}

	} );
	return groups;

}

/**
 * Does vector `a` win over `b` in a payload fold (`at( vector, channel )` reads
 * a value)? The CPU form of the kernels' `ds_winsN`.
 */
export function payloadWins( group: PayloadGroup, at: ( vector: 'a' | 'b', channel: number ) => number ): boolean {

	const ka = at( 'a', group.key ), kb = at( 'b', group.key );
	if ( ka !== kb ) return group.kind === 'argmin' ? ka < kb : ka > kb;
	for ( const c of group.payload ) {

		const pa = at( 'a', c ), pb = at( 'b', c );
		if ( pa !== pb ) return pa < pb;

	}

	return true;

}

/**
 * Channel ops, one per accumulator channel: four (a `vec4f` per texel), or
 * eight (two `vec4f`s, the `DsWide` struct of SPEC.md §14.1).
 */
export type ChannelOps = readonly [ ChannelOp, ChannelOp, ChannelOp, ChannelOp ] | readonly [ ChannelOp, ChannelOp, ChannelOp, ChannelOp, ChannelOp, ChannelOp, ChannelOp, ChannelOp ];

/** Accumulator channels per texel: one `vec4f`, or two (`DsWide`). */
export type Channels = 4 | 8;

/** A custom reduction. */
export interface CustomReduction {
	/**
	 * WGSL defining `fn reduce(a: vec4f, b: vec4f) -> vec4f` (with 8 channels,
	 * `fn reduce(a: DsWide, b: DsWide) -> DsWide`). Associative and
	 * commutative; for kind 'fold' also idempotent. Under 'exact', values are in
	 * the texture's normalized units.
	 */
	wgsl: string;
	/**
	 * 'fold' (default): idempotent, so duplicate reads don't matter. 'sum': not
	 * idempotent, so every texel the edge policy counts is combined exactly once,
	 * and the result is emitted as is. 'sum' needs a tile edge policy
	 * ('zero-weight' or 'clamp').
	 */
	kind?: 'fold' | 'sum';
}

/** 'sum' is shorthand for four 'sum' channels (tile edge policies only). */
export type Reduction = BuiltinReduction | 'sum' | CustomReduction | ChannelOps;

/**
 * How odd sizes and image edges are handled (SPEC.md §2.1, §14.1):
 * - 'exact': render-pass parity; odd sizes end a batch (mip chains, `Downsampler`).
 * - 'zero-weight': texels outside the image contribute nothing; level sizes are
 *   ⌈n/2⌉; averages divide by the in-bounds count. Every texel counts once.
 * - 'clamp': SPD/FSR-compatible. Source coordinates are clamped to the edge;
 *   level sizes are ⌊n/2⌋ (at least 1), dropping the last odd row or column.
 * Both tile policies keep every level of a dispatch in one tile. Area weights
 * are `Downsampler`'s `oddSizes: 'box'` (core `areaReference`), not a tile policy.
 */
export type EdgePolicy = 'exact' | 'zero-weight' | 'clamp';

/** The edge policies a tile kernel (`Reducer`) implements. */
export type TileEdgePolicy = Exclude<EdgePolicy, 'exact'>;

/** A reduction as the CPU reference understands it: a builtin, or a JS fold. */
export type ReferenceReduction = BuiltinReduction | ( ( a: number, b: number, channel: number ) => number );

/**
 * A tile reduction for the CPU reference: a builtin, channel ops, or a JS custom
 * combine (called per channel) with its kind.
 */
export type PyramidReferenceReduction = BuiltinReduction | 'sum' | ChannelOps | {
	reference: ( a: number, b: number, channel: number ) => number;
	kind?: 'fold' | 'sum';
};

/**
 * A tile reduction resolved: channel ops, or a custom combine of a kind, and
 * the channel count. Builtins expand to channel ops ('minmax' to min, max, min,
 * max, …).
 */
export type ResolvedReduction = { ops: readonly ChannelOp[], custom?: undefined, channels: Channels } | { ops?: undefined, custom: 'fold' | 'sum', channels: Channels };

const channelOps = [ 'sum', 'average', 'min', 'max', 'weight', 'weighted', 'argmin', 'argmax', 'payload' ];

export function isBuiltinReduction( value: unknown ): value is BuiltinReduction {

	return value === 'average' || value === 'min' || value === 'max' || value === 'minmax';

}

export function isChannelOps( value: unknown ): value is ChannelOps {

	return Array.isArray( value );

}

/** Sums, channel ops and custom sums are defined only under the tile edge policies. */
export function needsTileEdges( reduction: Reduction ): boolean {

	return reduction === 'sum' || isChannelOps( reduction ) || ( typeof reduction === 'object' && reduction !== null && reduction.kind === 'sum' );

}

function checkKind( kind: unknown ): 'fold' | 'sum' {

	if ( kind === undefined ) return 'fold';
	if ( kind !== 'fold' && kind !== 'sum' ) throw new Error( `downsampler: custom reduction kind must be 'fold' or 'sum', got ${ JSON.stringify( kind ) }.` );
	return kind;

}

/** Validates a channel count (4 or 8). */
export function checkChannels( channels: unknown ): Channels {

	if ( channels !== 4 && channels !== 8 ) throw new Error( `downsampler: channels must be 4 (one vec4f per texel) or 8 (two, DsWide), got ${ JSON.stringify( channels ) }.` );
	return channels;

}

/**
 * Validates a tile reduction and expands builtins to channel ops. `channels`
 * (4 or 8) sizes builtins and custom reductions; channel ops set it by their
 * length, and must agree with it when it is given.
 */
export function resolveReduction( reduction: Reduction | PyramidReferenceReduction, channels?: number ): ResolvedReduction {

	if ( isChannelOps( reduction ) ) {

		if ( ( reduction.length !== 4 && reduction.length !== 8 ) || ! reduction.every( op => channelOps.includes( op ) ) ) throw new Error( `downsampler: per-channel reductions are four of ${ channelOps.join( ', ' ) } (or eight, for 8-channel accumulators); got ${ JSON.stringify( reduction ) }.` );
		if ( channels !== undefined && channels !== reduction.length ) throw new Error( `downsampler: ${ reduction.length } channel ops, but channels is ${ channels }.` );
		const weights = reduction.filter( op => op === 'weight' ).length;
		if ( weights > 1 ) throw new Error( 'downsampler: at most one channel can be the \'weight\'.' );
		if ( weights === 0 && reduction.includes( 'weighted' ) ) throw new Error( 'downsampler: \'weighted\' channels need a \'weight\' channel.' );
		payloadGroups( reduction );
		return { ops: reduction, channels: reduction.length as Channels };

	}

	const count = checkChannels( channels ?? 4 );
	if ( reduction === 'minmax' ) return { ops: Array.from( { length: count }, ( _, c ) => c % 2 === 0 ? 'min' : 'max' ), channels: count };
	if ( isBuiltinReduction( reduction ) || reduction === 'sum' ) return { ops: Array<ChannelOp>( count ).fill( reduction ), channels: count };
	if ( typeof reduction !== 'object' || reduction === null ) throw new Error( `downsampler: unknown reduction ${ JSON.stringify( reduction ) }.` );
	if ( 'wgsl' in reduction ) reductionKey( reduction );
	else if ( typeof ( reduction as { reference?: unknown } ).reference !== 'function' ) throw new Error( 'downsampler: a reference custom reduction needs a `reference` function.' );
	return { custom: checkKind( reduction.kind ), channels: count };

}

/** Validates a reduction and returns a stable cache key for it. */
export function reductionKey( reduction: Reduction ): string {

	if ( isBuiltinReduction( reduction ) || reduction === 'sum' ) return reduction;
	if ( isChannelOps( reduction ) ) {

		resolveReduction( reduction );
		return `ops:${ reduction.join( ',' ) }`;

	}

	if ( typeof reduction !== 'object' || reduction === null || typeof reduction.wgsl !== 'string' ) throw new Error( `downsampler: unknown reduction ${ JSON.stringify( reduction ) }.` );
	if ( ! /\bfn\s+reduce\s*\(/.test( reduction.wgsl ) ) throw new Error( 'downsampler: a custom reduction must define `fn reduce(a: vec4f, b: vec4f) -> vec4f` (`DsWide` with 8 channels).' );
	return checkKind( reduction.kind ) === 'fold' ? `custom:${ reduction.wgsl }` : `custom-sum:${ reduction.wgsl }`;

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
