import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planBatches, mipLevelCountFor, tileSize } from '../../dist/index.js';

const counts = ( width, height, levelsPerDispatch, extra = {} ) => planBatches( width, height, { mipLevelCount: mipLevelCountFor( width, height ), levelsPerDispatch, ...extra } ).map( batch => batch.count );

test( 'power-of-two chains use full batches', () => {

	assert.deepEqual( counts( 4096, 4096, 6 ), [ 6, 6 ] );
	assert.deepEqual( counts( 4096, 4096, 4 ), [ 4, 4, 4 ] );
	assert.deepEqual( counts( 64, 64, 6 ), [ 6 ] );
	assert.deepEqual( counts( 128, 128, 6 ), [ 6, 1 ] );

} );

test( 'workgroup grid covers the first output with SPD-sized tiles', () => {

	const [ first, second ] = planBatches( 4096, 4096, { mipLevelCount: 13, levelsPerDispatch: 6 } );
	assert.equal( first.tile, 32 );
	assert.deepEqual( [ first.groupsX, first.groupsY, first.outputWidth ], [ 64, 64, 2048 ] );
	assert.deepEqual( [ second.sourceMipLevel, second.groupsX, second.groupsY ], [ 6, 1, 1 ] );
	assert.deepEqual( [ 1, 2, 3, 4, 5, 6 ].map( tileSize ), [ 8, 8, 8, 8, 16, 32 ] );

} );

test( 'odd sizes end a batch (exact policy)', () => {

	// 1920x1080: 960x540, 480x270, 240x135 (odd) -> re-sample, ...
	assert.equal( counts( 1920, 1080, 6 ).length, 7 );
	assert.deepEqual( counts( 3, 7, 6 ), [ 1, 1 ] );
	// 1D tails reduce in workgroup memory: unit dimensions never end a batch.
	assert.deepEqual( counts( 1, 128, 6 ), [ 6, 1 ] );
	assert.deepEqual( counts( 128, 8, 6 ), [ 6, 1 ] );

} );

test( 'trivial chains and sub-ranges', () => {

	assert.deepEqual( counts( 1, 1, 6 ), [] );
	assert.deepEqual( counts( 2, 2, 6 ), [ 1 ] );
	const range = planBatches( 1024, 1024, { baseMipLevel: 2, mipLevelCount: 5, levelsPerDispatch: 6 } );
	assert.deepEqual( range.map( batch => [ batch.sourceMipLevel, batch.count, batch.outputWidth ] ), [ [ 2, 4, 128 ] ] );

} );

test( 'invalid options throw', () => {

	assert.throws( () => planBatches( 8, 8, { mipLevelCount: 4, levelsPerDispatch: 7 } ) );
	assert.throws( () => planBatches( 8, 8, { mipLevelCount: 4, levelsPerDispatch: 6, oddSizes: 'fast' } ) );

} );

test( 'a separate source writes every destination level, starting at baseMipLevel', () => {

	const batches = planBatches( 1024, 1024, { mipLevelCount: 11, levelsPerDispatch: 6, fromSource: true } );
	assert.deepEqual( batches.map( batch => [ batch.sourceMipLevel, batch.firstMipLevel, batch.count, batch.outputWidth ] ), [ [ null, 0, 6, 1024 ], [ 5, 6, 5, 16 ] ] );
	// In place, the same range writes one level fewer.
	assert.equal( planBatches( 1024, 1024, { mipLevelCount: 11, levelsPerDispatch: 6 } ).reduce( ( sum, batch ) => sum + batch.count, 0 ), 10 );
	// A single destination level: one copy or reduction from the source.
	assert.deepEqual( planBatches( 7, 3, { baseMipLevel: 1, mipLevelCount: 1, levelsPerDispatch: 6, fromSource: true } ).map( batch => [ batch.sourceMipLevel, batch.firstMipLevel, batch.count ] ), [ [ null, 1, 1 ] ] );

} );
