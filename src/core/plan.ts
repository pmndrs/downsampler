/**
 * Backend-neutral planning: how a mip chain is split into dispatches.
 *
 * A batch reads one source level and writes `count` consecutive output levels.
 *
 * Under 'exact' (default), the first output of a batch is produced by sampling
 * the source with normalized coordinates; every further level is an exact 2:1
 * reduction of the previous one in workgroup memory. A batch therefore ends
 * where the next 2:1 reduction would not be exact (an odd, non-unit dimension),
 * or when it reaches the backend's levels-per-dispatch limit. With `tail`, once
 * a first output fits one workgroup (32×32), one 'tail' batch finishes the
 * chain in that workgroup with the same per-level 'exact' rule, odd sizes
 * included.
 *
 * 'box' plans exactly as 'exact': its odd steps read the three taps 2x … 2x+2
 * that the 'exact' folds read, which a tile of the previous level doesn't hold
 * at its far edge without a halo, so an odd size ends the batch the same way.
 *
 * Under 'spd', every batch is a 'clamp' tile reduction (SPD semantics): clamped
 * loads, ⌊n/2⌋ sizes, the last odd row or column dropped, so batches never end
 * at odd sizes and a chain takes ⌈levels / levelsPerDispatch⌉ dispatches.
 */

/**
 * How odd, non-unit dimensions are handled (SPEC.md §2.1):
 * - 'exact' (default): render-pass parity; an odd size ends a batch.
 * - 'spd': AMD SPD semantics. Each batch reduces up to `levelsPerDispatch`
 *   levels in one tile with clamped loads and ⌊n/2⌋ sizes, dropping the last
 *   odd row or column; the next batch reads the last stored level. Lower
 *   quality at odd sizes, and min/max are NOT conservative (not a valid Hi-Z).
 * - 'box': area-weighted odd sizes (nvpro_pyramid's weights, `areaReference` in
 *   core). Each source texel contributes in proportion to its overlap; for an
 *   odd size 2n+1, three taps weighted (n − x, n, x + 1) / (2n + 1). Same
 *   batches as 'exact'. Folds are the same as under 'exact'.
 */
export type OddSizePolicy = 'exact' | 'spd' | 'box';

/**
 * Kernel family of a batch:
 * - 'exact': first output sampled from the source level, later levels exact 2:1.
 * - 'tail': one workgroup; every level from its previous one by the 'exact' rule.
 * - 'clamp': a 'clamp' tile reduction (`oddSizes: 'spd'`).
 */
export type BatchKernel = 'exact' | 'tail' | 'clamp';

/** Largest first output a 'tail' batch takes: one 32×32 workgroup tile (16 KiB of vec4f). */
export const tailTile = 32;

/** Tile of a tail batch: the first output's larger side rounded up to 8, 16 or 32. */
export function tailTileFor( width: number, height: number ): number {

	return Math.min( tailTile, Math.max( 8, 2 ** Math.ceil( Math.log2( Math.max( width, height ) ) ) ) );

}

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
	/**
	 * With `fromSource`: the source level is the size of the first destination
	 * level (a converting copy), or is resampled into it (`resample: 'area'`).
	 * Under 'spd' that step is its own 1-level 'exact' batch; under 'exact' and
	 * 'box' the kernel handles it at run time.
	 */
	copySource?: boolean;
	/**
	 * 'exact' and 'box' only: finish the chain in one workgroup ('tail' batch) once a first
	 * output fits 32×32 and the regular plan would need more batches for the
	 * remaining levels. Default false.
	 */
	tail?: boolean;
	/** Largest tail tile (8, 16 or 32): 8 where a workgroup has fewer than 256 invocations. Default 32. */
	tailMaxTile?: number;
	/** 'exact' batches: the smallest tile (8, 16 or 32), above what the level count needs. Default 8. */
	exactMinTile?: number;
}

export interface Batch {
	/** Kernel family. */
	kernel: BatchKernel;
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

/**
 * The dispatches of a mip chain for a `width` × `height` level 0: what
 * `Downsampler.prepare()` records, without any GPU objects.
 */
export function planBatches( width: number, height: number, options: PlanOptions ): Batch[] {

	const { baseMipLevel = 0, mipLevelCount, levelsPerDispatch, oddSizes = 'exact', fromSource = false, copySource = false, tail = false, tailMaxTile = tailTile, exactMinTile = 8 } = options;
	if ( exactMinTile !== 8 && exactMinTile !== 16 && exactMinTile !== 32 ) throw new Error( 'downsampler: exactMinTile must be 8, 16 or 32.' );
	if ( oddSizes !== 'exact' && oddSizes !== 'spd' && oddSizes !== 'box' ) throw new Error( `downsampler: odd size policy "${ oddSizes }" is not implemented.` );
	if ( ! Number.isInteger( levelsPerDispatch ) || levelsPerDispatch < 1 || levelsPerDispatch > 6 ) throw new Error( 'downsampler: levelsPerDispatch must be an integer from 1 to 6.' );
	if ( oddSizes === 'spd' && tail ) throw new Error( 'downsampler: the tail kernel is an \'exact\' or \'box\' plan; \'spd\' chains never end a batch at odd sizes.' );

	const firstLevel = fromSource ? baseMipLevel : baseMipLevel + 1;
	const lastLevel = baseMipLevel + mipLevelCount - 1;
	const batches: Batch[] = [];
	const sourceOf = ( level: number ) => fromSource && level === firstLevel ? null : level - 1;

	if ( oddSizes === 'spd' ) {

		let level = firstLevel;
		if ( fromSource && copySource && level <= lastLevel ) {

			// A tile reduction can't copy: the converting copy is a 1-level exact batch.
			const outputWidth = mipSize( width, level ), outputHeight = mipSize( height, level ), tile = tileSize( 1 );
			batches.push( { kernel: 'exact', sourceMipLevel: null, firstMipLevel: level, count: 1, outputWidth, outputHeight, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ) } );
			level ++;

		}

		for ( ; level <= lastLevel; ) {

			const count = Math.min( levelsPerDispatch, lastLevel - level + 1 );
			const outputWidth = mipSize( width, level ), outputHeight = mipSize( height, level ), tile = tileSize( count );
			batches.push( { kernel: 'clamp', sourceMipLevel: sourceOf( level ), firstMipLevel: level, count, outputWidth, outputHeight, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ) } );
			level += count;

		}

		return batches;

	}

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

		// The tail: one workgroup finishes what would otherwise take more batches.
		const tailCount = Math.min( levelsPerDispatch, lastLevel - level + 1 );
		if ( tail && count < tailCount && outputWidth <= tailMaxTile && outputHeight <= tailMaxTile ) {

			batches.push( { kernel: 'tail', sourceMipLevel: sourceOf( level ), firstMipLevel: level, count: tailCount, outputWidth, outputHeight, tile: tailTileFor( outputWidth, outputHeight ), groupsX: 1, groupsY: 1 } );
			level += tailCount;
			continue;

		}

		const tile = Math.max( exactMinTile, tileSize( count ) );
		batches.push( {
			kernel: 'exact', sourceMipLevel: sourceOf( level ), firstMipLevel: level, count, outputWidth, outputHeight, tile,
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
