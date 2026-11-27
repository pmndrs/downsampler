import { test } from 'node:test';
import assert from 'node:assert/strict';
import { floorFootprint, intReference, isOrderedFold, planBatches, reduceReference, reductionKey, resolveReduction } from '../../dist/index.js';
import { createKernel } from '../../dist/webgpu/index.js';

// Floor footprints and ordered folds (`oddSizes: 'floor'`, `kind: 'ordered'`; pmndrs/downsampler#6):
// GI-1's probe-mask mips keep the first valid of each texel's 2×2 children in raster order.

const INVALID = 0xffffffff;
const firstValid = ( a, b ) => a !== INVALID ? a : b;
const firstValidWgsl = { wgsl: 'fn reduce(a: vec4u, b: vec4u) -> vec4u { return select(b, a, a.x != 0xffffffffu); }', kind: 'ordered' };

// Capsaicin's FilterProbeMask (gi1.comp:248-284) as gi ports it: children past the level's edge
// are invalid; the first valid of (2x, 2y), (2x + 1, 2y), (2x, 2y + 1), (2x + 1, 2y + 1) wins.
function filterProbeMask( level, width, height ) {

	const w = Math.max( 1, width >> 1 ), h = Math.max( 1, height >> 1 );
	const out = new Array( w * h ).fill( INVALID );
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		let mask = INVALID;
		for ( let dy = 0; dy < 2 && mask === INVALID; dy ++ ) for ( let dx = 0; dx < 2 && mask === INVALID; dx ++ ) {

			const px = 2 * x + dx, py = 2 * y + dy;
			mask = px < width && py < height ? level[ py * width + px ] : INVALID;

		}

		out[ y * w + x ] = mask;

	}

	return { values: out, width: w, height: h };

}

let state = 11;
const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 );

test( 'floorFootprint: 2x and 2x + 1, clamped, never the last texel of an odd size', () => {

	assert.deepEqual( floorFootprint( 0, 1 ), [ 0, 0 ] );
	assert.deepEqual( floorFootprint( 0, 2 ), [ 0, 1 ] );
	assert.deepEqual( floorFootprint( 0, 3 ), [ 0, 1 ] );
	assert.deepEqual( floorFootprint( 2, 5 ), [ 4, 4 ] ); // past the 5 → 2 level, only reached clamped
	assert.deepEqual( floorFootprint( 1, 5 ), [ 2, 3 ] );

} );

test( 'intReference with floor and a first-valid fold is Capsaicin\'s FilterProbeMask, level by level', () => {

	const sizes = [ [ 240, 135 ], [ 31, 17 ], [ 33, 1 ], [ 1, 9 ], [ 7, 5 ], [ 2, 3 ], [ 160, 90 ], [ 1, 1 ] ];
	for ( const [ width, height ] of sizes ) for ( const density of [ 0.02, 0.3, 0.9 ] ) {

		let w = width, h = height;
		let level = Array.from( { length: w * h }, () => ( next() / 2 ** 32 ) < density ? next() % 0xfffffff0 : INVALID );
		while ( w > 1 || h > 1 ) {

			const expected = filterProbeMask( level, w, h );
			const rgba = level.flatMap( v => [ v, 0, 0, 0 ] );
			const got = intReference( rgba, w, h, 1, 'uint32', ( a, b, c ) => c === 0 ? firstValid( a, b ) : a, { floor: true } );
			assert.equal( got.width, expected.width );
			assert.equal( got.height, expected.height );
			const values = Array.from( { length: got.width * got.height }, ( _, i ) => got.values[ i * 4 ] );
			assert.deepEqual( values, expected.values, `${ w }×${ h } at density ${ density }` );
			level = values;
			w = got.width;
			h = got.height;

		}

	}

} );

test( 'floor footprints drop the odd last row and column; exact ones keep them', () => {

	// 3×3, the max in the last column and row only.
	const values = [ 0, 0, 5, 0, 0, 5, 5, 5, 9 ].flatMap( v => [ v, v, v, v ] );
	const max = ( a, b ) => Math.max( a, b );
	assert.equal( intReference( values, 3, 3, 1, 'uint8', max, { floor: true } ).values[ 0 ], 0 );
	assert.equal( intReference( values, 3, 3, 1, 'uint8', max ).values[ 0 ], 9 );
	const floats = new Float32Array( values.map( v => v / 255 ) );
	assert.equal( reduceReference( floats, 3, 3, 1, 'float32', 'max', { floor: true } ).values[ 0 ], 0 );
	assert.equal( reduceReference( floats, 3, 3, 1, 'float32', 'max' ).values[ 0 ], Math.fround( 9 / 255 ) );
	assert.throws( () => intReference( values, 3, 3, 1, 'uint8', 'average', { floor: true } ), /folds/ );
	assert.throws( () => reduceReference( floats, 3, 3, 1, 'float32', 'average', { floor: true } ), /folds/ );

} );

test( 'ordered folds: their own cache key, Downsampler-only', () => {

	assert.ok( isOrderedFold( firstValidWgsl ) );
	assert.ok( ! isOrderedFold( { wgsl: firstValidWgsl.wgsl } ) );
	assert.ok( ! isOrderedFold( 'min' ) );
	assert.notEqual( reductionKey( firstValidWgsl ), reductionKey( { wgsl: firstValidWgsl.wgsl } ) );
	assert.throws( () => resolveReduction( firstValidWgsl ), /Downsampler reduction with oddSizes 'floor'/ );
	assert.throws( () => reductionKey( { wgsl: firstValidWgsl.wgsl, kind: 'scan' } ), /'fold', 'sum' or 'ordered'/ );

} );

test( 'floor plans as exact, without halo batches', () => {

	for ( const [ width, height ] of [ [ 240, 135 ], [ 1920, 1080 ], [ 1129, 635 ], [ 1024, 1024 ] ] ) for ( const tail of [ false, true ] ) {

		const options = { mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1, levelsPerDispatch: 6, tail };
		assert.deepEqual( planBatches( width, height, { ...options, oddSizes: 'floor' } ), planBatches( width, height, { ...options, oddSizes: 'exact' } ) );

	}

	assert.throws( () => planBatches( 240, 135, { mipLevelCount: 8, levelsPerDispatch: 6, oddSizes: 'floor', halo: true } ), /halo/ );

} );

test( 'floor kernels: the 2×2 fold only, for folds only; ordered folds need it', () => {

	for ( const format of [ 'r32uint', 'r32float', 'rgba8unorm' ] ) for ( const tail of [ false, true ] ) {

		const reduction = format === 'r32uint' ? firstValidWgsl : 'max';
		const options = { format, count: 3, access: 'load', reduction, tail };
		const floor = createKernel( { ...options, floor: true } );
		assert.ok( ! floor.includes( 'odd.x' ), `${ format } tail ${ tail }` );
		assert.ok( floor.includes( 'fold(fold(' ) );
		if ( reduction === 'max' ) assert.ok( createKernel( options ).includes( 'odd.x' ) );

	}

	assert.throws( () => createKernel( { format: 'r32uint', count: 2, access: 'load', reduction: firstValidWgsl } ), /needs floor footprints/ );
	assert.throws( () => createKernel( { format: 'r32float', count: 2, access: 'load', reduction: firstValidWgsl } ), /needs floor footprints/ );
	assert.throws( () => createKernel( { format: 'rgba8unorm', count: 2, reduction: 'average', floor: true } ), /for folds/ );
	assert.throws( () => createKernel( { format: 'r32uint', count: 2, access: 'load', reduction: 'max', floor: true, box: true } ), /2:1 steps/ );
	assert.throws( () => createKernel( { format: 'r32float', count: 2, reduction: 'max', floor: true, edges: 'clamp' } ), /'exact' kernel option/ );

} );
