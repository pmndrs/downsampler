import { test } from 'node:test';
import assert from 'node:assert/strict';
import { batchHalo, mipLevelCountFor, mipSize, planBatches } from '../../dist/index.js';

const plan = ( width, height, extra = {} ) => planBatches( width, height, { mipLevelCount: mipLevelCountFor( width, height ), levelsPerDispatch: 6, tail: true, exactMinTile: 32, ...extra } );
const renderSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 3840, 2160 ] ];

test( 'batchHalo: e_i = 2·e_(i+1) + 1 at each odd step, per axis, the last level 0', () => {

	// 1920×1080 from level 1: 960×540 → 480×270 → 240×135 → 120×67 (135 is odd).
	assert.deepEqual( batchHalo( 1920, 1080, 1, 4 ), { x: [ 0, 0, 0, 0 ], y: [ 4, 2, 1, 0 ] } );
	// 1477×831 from level 1: 738×415 (415 odd) → 369×207 (both odd) → 184×103.
	assert.deepEqual( batchHalo( 1477, 831, 1, 3 ), { x: [ 2, 1, 0 ], y: [ 3, 1, 0 ] } );
	// Unit dimensions are never odd steps.
	assert.deepEqual( batchHalo( 1, 7, 0, 3 ).x, [ 0, 0, 0 ] );

} );

test( 'halo plans: same levels in order, fewer or equal dispatches, tiles with halo within the budget', () => {

	for ( const [ width, height ] of [ ...renderSizes, [ 255, 257 ], [ 191, 107 ], [ 31, 33 ], [ 1, 300 ], [ 4096, 4096 ] ] ) {

		const plain = plan( width, height ), halo = plan( width, height, { halo: true } );
		assert.ok( halo.length <= plain.length, `${ width }×${ height }: ${ halo.length } > ${ plain.length }` );
		let level = 1;
		for ( const batch of halo ) {

			assert.equal( batch.firstMipLevel, level );
			level += batch.count;
			if ( batch.halo === undefined ) continue;
			assert.equal( batch.kernel, 'exact' );
			assert.ok( batch.tile >= 16 );
			assert.deepEqual( batch.halo, batchHalo( width, height, batch.firstMipLevel, batch.count ) );
			const texels = ( batch.tile + batch.halo.x[ 0 ] ) * ( batch.tile + batch.halo.y[ 0 ] );
			assert.ok( texels <= 1024 && texels / batch.tile ** 2 - 1 <= 0.25, `${ width }×${ height } level ${ batch.firstMipLevel }: ${ texels } texels` );
			assert.ok( batch.halo.x[ 0 ] + batch.halo.y[ 0 ] > 0, 'a halo batch has an odd step' );
			assert.equal( batch.outputWidth, mipSize( width, batch.firstMipLevel ) );

		}

		assert.equal( level, mipLevelCountFor( width, height ) );

	}

} );

test( 'halo plans save dispatches at render sizes', () => {

	const saved = renderSizes.map( ( [ w, h ] ) => plan( w, h ).length - plan( w, h, { halo: true } ).length );
	assert.deepEqual( saved, [ 1, 2, 0, 3, 1, 1, 0, 1 ] );

} );

test( 'halo: powers of two and plans without odd steps are unchanged; limits respected', () => {

	assert.deepEqual( plan( 4096, 4096, { halo: true } ), plan( 4096, 4096 ) );
	assert.deepEqual( plan( 1024, 512, { halo: true } ), plan( 1024, 512 ) );
	// No room for a halo (a 16-texel tile alone fills it): the plain plan.
	assert.deepEqual( plan( 1477, 831, { halo: true, haloMaxTexels: 256 } ), plan( 1477, 831 ) );
	assert.deepEqual( plan( 1477, 831, { halo: true, haloMaxExtra: 0 } ), plan( 1477, 831 ) );
	assert.throws( () => plan( 1477, 831, { halo: true, oddSizes: 'spd', tail: false } ), /halo batches are an 'exact' or 'box' plan/ );

} );

test( 'halo: a separate source\'s first batch is not extended', () => {

	const batches = planBatches( 1129, 635, { mipLevelCount: mipLevelCountFor( 1129, 635 ), levelsPerDispatch: 6, fromSource: true, halo: true } );
	assert.equal( batches[ 0 ].halo, undefined );

} );
