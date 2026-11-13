import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reduceReference } from '../../dist/index.js';

const rgba = values => Float32Array.from( values.flatMap( value => [ value, value, value, 1 ] ) );

test( 'even sizes are an exact 2x2 box', () => {

	const level = reduceReference( rgba( [ 0, 1 / 255, 2 / 255, 5 / 255 ] ), 2, 2, 1, 'unorm8' );
	assert.equal( level.width, 1 );
	assert.equal( Math.round( level.values[ 0 ] * 255 ), 2 ); // 8 / 4
	// Shader weights of exactly 1/2 need no allowance; a hardware sampler may be a weight step
	// off even there (Intel truncates), 1.5 steps per axis times the contrast (gx 3, gy 4 units).
	assert.equal( reduceReference( rgba( [ 0, 1 / 255, 2 / 255, 5 / 255 ] ), 2, 2, 1, 'unorm8', 'average', { weights: 'shader' } ).filterTolerance[ 0 ], 0 );
	assert.equal( level.filterTolerance[ 0 ], Math.fround( 1.5 * ( 3 + 4 ) / 256 / 255 ) );

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

test( 'normalized averages round exact ties to even, despite float readback', () => {

	// 8/255 and 9/255 as f32 average to a hair off 8.5/255; in integer units it's an exact tie -> 8.
	const level = reduceReference( Float32Array.of( 8 / 255, 0, 0, 1, 9 / 255, 0, 0, 1 ).map( Math.fround ), 1, 2, 1, 'unorm8' );
	assert.equal( Math.round( level.values[ 0 ] * 255 ), 8 );
	// A separate source of another encoding is not snapped to integer units first.
	// 8.4 and 9.4 average to 8.9 -> 9; snapped to 8 and 9 they would tie -> 8.
	const raw = reduceReference( Float32Array.of( 8.4 / 255, 0, 0, 1, 9.4 / 255, 0, 0, 1 ), 1, 2, 1, 'unorm8', 'average', { quantizedInput: false } );
	assert.equal( Math.round( raw.values[ 0 ] * 255 ), 9 );

} );
