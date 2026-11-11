/**
 * Tile reductions (`Reducer`, SPEC.md §14.1): level sizes, dispatch geometry and
 * the CPU reference.
 *
 * Output texel x of level l (1 = half size) reduces the source block
 * [x·2^l, (x+1)·2^l) on each axis:
 * - 'zero-weight': only the block's texels inside the image count. Level sizes
 *   are ⌈n / 2^l⌉, so every source texel is counted exactly once per level.
 * - 'clamp': every texel of the block counts, with its coordinates clamped to
 *   the image (SPD/FSR). Level sizes are ⌊n / 2^l⌋, at least 1, so the last odd
 *   row or column is dropped, and blocks that reach past the image (unit or
 *   partial dimensions) count the edge texel more than once.
 *
 * The reference computes each level straight from the source in f64, without the
 * kernel's 2×2 tree or intermediate rounding, so it is independent of the
 * kernel's structure.
 */
import { tileSize, workgroupSide } from './plan.js';
import { resolveReduction, type ChannelOp, type PyramidReferenceReduction, type TileEdgePolicy } from './reduction.js';

/** Most levels one tile dispatch reduces (a 64×64 source tile, as in SPD). */
export const maxTileLevels = 6;

/** Size of level `level` (1 = half) along one axis of a `size`-texel source. */
export function levelSize( size: number, level: number, edges: TileEdgePolicy ): number {

	if ( edges === 'zero-weight' ) return Math.ceil( size / 2 ** level );
	if ( edges === 'clamp' ) return Math.max( 1, Math.floor( size / 2 ** level ) );
	throw new Error( `downsampler: unknown tile edge policy ${ JSON.stringify( edges ) }.` );

}

export interface TileDispatch {
	/** Levels reduced in the tile: the deepest emitted level. */
	count: number;
	/** Level-1 texels per workgroup edge. */
	tile: number;
	/** Invocations per workgroup edge. */
	side: number;
	/** Size of level 1. */
	width: number;
	height: number;
	groupsX: number;
	groupsY: number;
}

/** Workgroup grid of one tile dispatch over a `width` × `height` source. */
export function tileDispatch( width: number, height: number, count: number, edges: TileEdgePolicy ): TileDispatch {

	if ( ! Number.isInteger( width ) || ! Number.isInteger( height ) || width < 1 || height < 1 ) throw new Error( `downsampler: source size must be positive integers, got ${ width }×${ height }.` );
	const tile = tileSize( count ), side = workgroupSide( tile );
	const w = levelSize( width, 1, edges ), h = levelSize( height, 1, edges );
	return { count, tile, side, width: w, height: h, groupsX: Math.ceil( w / tile ), groupsY: Math.ceil( h / tile ) };

}

/** Validates emitted levels: strictly increasing integers from 1 to 6. */
export function checkLevels( levels: readonly number[] ): void {

	if ( ! Array.isArray( levels ) || levels.length === 0 ) throw new Error( 'downsampler: levels must list at least one level to emit.' );
	levels.forEach( ( level, i ) => {

		if ( ! Number.isInteger( level ) || level < 1 || level > maxTileLevels ) throw new Error( `downsampler: emitted levels are integers from 1 (half size) to ${ maxTileLevels }; got ${ level }.` );
		if ( i > 0 && level <= levels[ i - 1 ] ) throw new Error( `downsampler: levels must be strictly increasing; got ${ JSON.stringify( levels ) }.` );

	} );

}

export interface PyramidReferenceOptions {
	reduction: PyramidReferenceReduction;
	levels: readonly number[];
	edges: TileEdgePolicy;
}

export interface PyramidLevel {
	level: number;
	width: number;
	height: number;
	layers: number;
	/** RGBA values, layer-major then row-major, as emitted (normalised). */
	values: Float64Array;
	/**
	 * Per value, the magnitude of the terms behind it, normalised like the value
	 * (Σ|v| for a sum, Σ|v|/n for an average, Σ|w·v|/Σw for a weighted mean, 0
	 * for min/max and custom folds). Float error scales with it.
	 */
	magnitude: Float64Array;
}

/**
 * Reference for a tile reduction of `source` (RGBA, layer-major then row-major)
 * at each level of `levels`.
 */
export function reducePyramid( source: ArrayLike<number>, width: number, height: number, layers: number, { reduction, levels, edges }: PyramidReferenceOptions ): PyramidLevel[] {

	checkLevels( levels );
	const resolved = resolveReduction( reduction );
	const ops: readonly ( ChannelOp | 'fold' | 'custom-sum' )[] = resolved.ops ?? Array( 4 ).fill( resolved.custom === 'sum' ? 'custom-sum' : 'fold' );
	const custom = typeof reduction === 'object' && ! Array.isArray( reduction ) ? ( reduction as { reference: ( a: number, b: number, channel: number ) => number } ).reference : null;
	const weightChannel = ops.indexOf( 'weight' );

	return levels.map( level => {

		const w = levelSize( width, level, edges ), h = levelSize( height, level, edges ), block = 2 ** level;
		const values = new Float64Array( w * h * layers * 4 ), magnitude = new Float64Array( values.length );
		// Source coordinates of a block along one axis, per the edge policy.
		const span = ( x: number, size: number ) => {

			const out: number[] = [];
			for ( let s = x * block; s < ( x + 1 ) * block; s ++ ) {

				if ( edges === 'clamp' ) out.push( Math.min( s, size - 1 ) );
				else if ( s < size ) out.push( s );

			}

			return out;

		};

		const spansX = Array.from( { length: w }, ( _, x ) => span( x, width ) );
		const spansY = Array.from( { length: h }, ( _, y ) => span( y, height ) );

		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const index = ( ( layer * h + y ) * w + x ) * 4;
			const n = spansX[ x ].length * spansY[ y ].length;
			let weight = 0;
			if ( weightChannel >= 0 ) for ( const sy of spansY[ y ] ) for ( const sx of spansX[ x ] ) weight += source[ ( ( layer * height + sy ) * width + sx ) * 4 + weightChannel ];

			for ( let c = 0; c < 4; c ++ ) {

				const op = ops[ c ];
				let value = NaN, total = 0, first = true;

				for ( const sy of spansY[ y ] ) for ( const sx of spansX[ x ] ) {

					const base = ( ( layer * height + sy ) * width + sx ) * 4;
					const v = op === 'weighted' ? source[ base + c ] * source[ base + weightChannel ] : source[ base + c ];
					total += Math.abs( v );
					if ( first ) { value = v; first = false; continue; }
					if ( op === 'min' ) value = Math.min( value, v );
					else if ( op === 'max' ) value = Math.max( value, v );
					else if ( op === 'fold' || op === 'custom-sum' ) value = custom!( value, v, c );
					else value += v;

				}

				let scale = 0;
				if ( op === 'average' ) { value /= n; scale = total / n; }
				else if ( op === 'weighted' ) { value = weight === 0 ? 0 : value / weight; scale = weight === 0 ? 0 : total / Math.abs( weight ); }
				else if ( op === 'sum' || op === 'weight' ) scale = total;
				else if ( op === 'custom-sum' ) scale = total + n;
				values[ index + c ] = value;
				magnitude[ index + c ] = scale;

			}

		}

		return { level, width: w, height: h, layers, values, magnitude };

	} );

}

/**
 * Declared comparison tolerance for a tile reduction computed in f32: float
 * error relative to the magnitude of the terms (a 4096-texel tree of f32 sums
 * stays well inside it), so min/max and custom folds of loaded values are exact.
 * A test tolerance for this repository's fixtures, not a formal error bound.
 */
export function pyramidTolerance( magnitude: number ): number {

	return magnitude * 4e-6;

}
