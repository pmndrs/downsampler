import { test } from 'node:test';
import assert from 'node:assert/strict';
import { footprint, mipLevelCountFor, reduceReference, reductionKey } from '../../dist/index.js';

const rgba = values => Float32Array.from( values.flatMap( value => [ value, value, value, value ] ) );

test( 'footprint covers 2x..2x+1, plus 2x+2 along odd sizes', () => {

	assert.deepEqual( [ 0, 1 ].map( x => footprint( x, 4 ) ), [ [ 0, 1 ], [ 2, 3 ] ] );
	assert.deepEqual( [ 0, 1 ].map( x => footprint( x, 5 ) ), [ [ 0, 2 ], [ 2, 4 ] ] );
	assert.deepEqual( footprint( 0, 3 ), [ 0, 2 ] );
	assert.deepEqual( footprint( 0, 1 ), [ 0, 0 ] );
	assert.deepEqual( footprint( 0, 2 ), [ 0, 1 ] );

} );

test( 'min/max include the odd last column', () => {

	// 3 -> 1: average samples only the centre texel; max must see all three.
	const source = rgba( [ 0, 0.25, 1 ] );
	assert.equal( reduceReference( source, 3, 1, 1, 'float16', 'average' ).values[ 0 ], 0.25 );
	assert.equal( reduceReference( source, 3, 1, 1, 'float16', 'max' ).values[ 0 ], 1 );
	assert.equal( reduceReference( source, 3, 1, 1, 'float16', 'min' ).values[ 0 ], 0 );

} );

test( 'minmax: even channels take the min, odd channels the max', () => {

	const level = reduceReference( Float32Array.of( 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9 ), 2, 1, 1, 'float16', 'minmax' );
	assert.deepEqual( Array.from( level.values, v => Math.round( v * 10 ) / 10 ), [ 0.1, 0.9, 0.1, 0.9 ] );

} );

test( 'max chains are conservative: each texel bounds every base texel its area touches', () => {

	let state = 7;
	const random = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;

	for ( const [ width, height ] of [ [ 7, 5 ], [ 17, 9 ], [ 31, 33 ], [ 3, 64 ], [ 255, 1 ], [ 120, 67 ] ] ) {

		const base = Float32Array.from( { length: width * height * 4 }, () => Math.round( random() * 255 ) / 255 );
		let level = { width, height, values: base };

		for ( let mip = 1; mip < mipLevelCountFor( width, height ); mip ++ ) {

			level = reduceReference( level.values, level.width, level.height, 1, 'unorm8', 'max' );
			const { width: w, height: h, values } = level;

			for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

				// Base texels overlapping [x, x+1) * width / w (open at the far edge).
				const x0 = Math.floor( x * width / w ), x1 = Math.ceil( ( x + 1 ) * width / w ) - 1;
				const y0 = Math.floor( y * height / h ), y1 = Math.ceil( ( y + 1 ) * height / h ) - 1;
				for ( let sy = y0; sy <= y1; sy ++ ) for ( let sx = x0; sx <= x1; sx ++ ) {

					assert.ok( values[ ( y * w + x ) * 4 ] >= base[ ( sy * width + sx ) * 4 ], `${ width }x${ height } mip ${ mip } texel ${ x },${ y } misses base ${ sx },${ sy }` );

				}

			}

		}

	}

} );

test( 'reductions are validated', () => {

	assert.equal( reductionKey( 'minmax' ), 'minmax' );
	assert.throws( () => reductionKey( 'median' ) );
	assert.throws( () => reductionKey( { wgsl: 'fn combine(a: vec4f, b: vec4f) -> vec4f { return a; }' } ) );
	assert.match( reductionKey( { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { return max(a, b); }' } ), /^custom:/ );

} );
