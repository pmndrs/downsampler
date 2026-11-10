import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reduceReference } from '../../dist/index.js';

const rgba = values => Float32Array.from( values.flatMap( value => [ value, value, value, 1 ] ) );

test( 'even sizes are an exact 2x2 box', () => {

	const level = reduceReference( rgba( [ 0, 1 / 255, 2 / 255, 5 / 255 ] ), 2, 2, 1, 'unorm8' );
	assert.equal( level.width, 1 );
	assert.equal( Math.round( level.values[ 0 ] * 255 ), 2 ); // 8 / 4
	assert.equal( level.filterTolerance[ 0 ], 0 );

} );

test( 'odd sizes use normalized-coordinate bilinear weights', () => {

	// 3 -> 1: sample at x = 1.0, i.e. exactly the centre texel.
	const level = reduceReference( rgba( [ 0, 0.5, 1 ] ), 3, 1, 1, 'float16' );
	assert.equal( level.values[ 0 ], 0.5 );

} );

test( 'layers are independent', () => {

	const level = reduceReference( rgba( [ 0, 0, 0, 0, 1, 1, 1, 1 ] ), 2, 2, 2, 'float16' );
	assert.deepEqual( [ level.values[ 0 ], level.values[ 4 ] ], [ 0, 1 ] );

} );
