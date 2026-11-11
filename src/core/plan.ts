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
// 'box' (area-weighted odd sizes) is planned; SPD-style 'fast' was dropped.
// See SPEC.md §2.1.

export interface PlanOptions {
	/**
	 * In place: the source level, with outputs from `baseMipLevel + 1`. With
	 * `fromSource`: the first level written, read from the separate source.
	 */
	baseMipLevel?: number;
	/** Destination levels involved: in place this includes the source level. */
	mipLevelCount: number;
	/** Maximum output levels per dispatch (bounded by storage texture bindings). */
	levelsPerDispatch: number;
	oddSizes?: OddSizePolicy;
	/** The first batch reads a separate source texture instead of a level of this one. */
	fromSource?: boolean;
}

export interface Batch {
	/** Level sampled by the first output, or null for the separate source. */
	sourceMipLevel: number | null;
	/** First level written. */
	firstMipLevel: number;
	/** Number of output levels written: firstMipLevel … firstMipLevel + count - 1. */
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

	const { baseMipLevel = 0, mipLevelCount, levelsPerDispatch, oddSizes = 'exact', fromSource = false } = options;
	if ( oddSizes !== 'exact' ) throw new Error( `downsampler: odd size policy "${ oddSizes }" is not implemented.` );
	if ( ! Number.isInteger( levelsPerDispatch ) || levelsPerDispatch < 1 || levelsPerDispatch > 6 ) throw new Error( 'downsampler: levelsPerDispatch must be an integer from 1 to 6.' );

	const firstLevel = fromSource ? baseMipLevel : baseMipLevel + 1;
	const lastLevel = baseMipLevel + mipLevelCount - 1;
	const batches: Batch[] = [];

	for ( let level = firstLevel; level <= lastLevel; ) {

		const outputWidth = mipSize( width, level ), outputHeight = mipSize( height, level );
		let count = 1, cw = outputWidth, ch = outputHeight;

		// Continue in workgroup memory only while each non-unit dimension halves exactly.
		while ( count < levelsPerDispatch && level + count <= lastLevel &&
			( cw === 1 || cw % 2 === 0 ) && ( ch === 1 || ch % 2 === 0 ) ) {

			cw = Math.max( 1, cw / 2 );
			ch = Math.max( 1, ch / 2 );
			count ++;

		}

		const tile = tileSize( count );
		batches.push( {
			sourceMipLevel: fromSource && level === firstLevel ? null : level - 1, firstMipLevel: level, count, outputWidth, outputHeight, tile,
			groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile )
		} );

		level += count;

	}

	return batches;

}

/** One level's region of a buffer-output batch (SPEC.md §6.1 Buffer output path). */
export interface BufferLevel {
	width: number;
	height: number;
	/** Byte offset of the level's first row, a multiple of 256. */
	offset: number;
	/** Row pitch, a multiple of 256 (`copyBufferToTexture` needs it). */
	bytesPerRow: number;
	/** Rows per array layer: the level's height. */
	rowsPerImage: number;
}

/**
 * Layout of the buffer a batch writes on the buffer output path: each level at
 * its own 256-byte-aligned offset, each layer `rowsPerImage` rows after the
 * previous one, rows padded to 256 bytes. `size` is the buffer size in bytes.
 */
export function outputBufferLayout( sizes: readonly ( readonly [ number, number ] )[], layers: number, bytesPerTexel: number ): { levels: BufferLevel[]; size: number } {

	let size = 0;
	const levels = sizes.map( ( [ width, height ] ) => {

		const bytesPerRow = Math.ceil( width * bytesPerTexel / 256 ) * 256;
		const level = { width, height, offset: size, bytesPerRow, rowsPerImage: height };
		size += bytesPerRow * height * layers;
		return level;

	} );
	return { levels, size };

}
