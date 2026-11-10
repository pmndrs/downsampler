/**
 * Backend-neutral planning: how a mip chain is split into dispatches.
 *
 * A batch reads one source level and writes `count` consecutive output levels.
 * Inside a batch the first output is produced by sampling the source with
 * normalized coordinates; every further level is an exact 2:1 reduction of the
 * previous one in workgroup memory. A batch therefore ends where the next 2:1
 * reduction would not be exact (an odd, non-unit dimension) — the "exact"
 * odd-size policy — or when it reaches the backend's levels-per-dispatch limit.
 */

/** How odd, non-unit dimensions are handled. */
export type OddSizePolicy = 'exact';
// 'fast' (SPD-style clamped 2x2 reduction through odd sizes) is specified in
// SPEC.md and not implemented yet.

export interface PlanOptions {
	/** Source level. Outputs start at `baseMipLevel + 1`. */
	baseMipLevel?: number;
	/** Number of levels involved, including the source level. */
	mipLevelCount: number;
	/** Maximum output levels per dispatch (bounded by storage texture bindings). */
	levelsPerDispatch: number;
	oddSizes?: OddSizePolicy;
}

export interface Batch {
	/** Level sampled by the first output of this batch. */
	sourceMipLevel: number;
	/** Number of output levels written: sourceMipLevel + 1 … sourceMipLevel + count. */
	count: number;
	/** Size of the first output level. */
	outputWidth: number;
	outputHeight: number;
	/** First-output texels per workgroup edge. */
	tile: number;
	groupsX: number;
	groupsY: number;
}

/** Size of a mip level, following the WebGPU/Vulkan/D3D floor convention. */
export function mipSize( size: number, level: number ): number {

	return Math.max( 1, Math.floor( size / 2 ** level ) );

}

/** Full mip chain length for a size. */
export function mipLevelCountFor( width: number, height: number ): number {

	return Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;

}

/**
 * Tile size for a batch of `count` levels: 8×8 for up to four levels, 16×16 for
 * five and 32×32 (a 64×64 source tile, as in SPD) for six.
 */
export function tileSize( count: number ): number {

	return Math.max( 8, 1 << ( count - 1 ) );

}

/** Invocations per workgroup edge for a tile; each produces (tile / side)² first-level texels. */
export function workgroupSide( tile: number ): number {

	return Math.min( tile, 16 );

}

export function planBatches( width: number, height: number, options: PlanOptions ): Batch[] {

	const { baseMipLevel = 0, mipLevelCount, levelsPerDispatch, oddSizes = 'exact' } = options;
	if ( oddSizes !== 'exact' ) throw new Error( `downsampler: odd size policy "${ oddSizes }" is not implemented.` );
	if ( ! Number.isInteger( levelsPerDispatch ) || levelsPerDispatch < 1 || levelsPerDispatch > 6 ) throw new Error( 'downsampler: levelsPerDispatch must be an integer from 1 to 6.' );

	const lastLevel = baseMipLevel + mipLevelCount - 1;
	const batches: Batch[] = [];
	let w = mipSize( width, baseMipLevel ), h = mipSize( height, baseMipLevel );

	for ( let level = baseMipLevel; level < lastLevel; ) {

		const outputWidth = Math.max( 1, Math.floor( w / 2 ) );
		const outputHeight = Math.max( 1, Math.floor( h / 2 ) );
		let count = 1, cw = outputWidth, ch = outputHeight;

		// Continue in workgroup memory only while each non-unit dimension halves exactly.
		while ( count < levelsPerDispatch && level + count < lastLevel &&
			( cw === 1 || cw % 2 === 0 ) && ( ch === 1 || ch % 2 === 0 ) ) {

			cw = Math.max( 1, cw / 2 );
			ch = Math.max( 1, ch / 2 );
			count ++;

		}

		const tile = tileSize( count );
		batches.push( {
			sourceMipLevel: level, count, outputWidth, outputHeight, tile,
			groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile )
		} );

		w = cw;
		h = ch;
		level += count;

	}

	return batches;

}
