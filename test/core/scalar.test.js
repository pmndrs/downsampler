import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scalarDepth, scalarOps, scalarReference, scalarWorkgroups } from '../../dist/index.js';

let state = 3;
const random = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
const image = ( width, height, layers, value ) => {

	const texels = new Float32Array( width * height * layers * 4 );
	for ( let i = 0; i < texels.length; i ++ ) texels[ i ] = value( i % 4, Math.floor( i / 4 ) );
	return texels;

};

test( 'ops: one op or four, at most one weight, weighted needs a weight', () => {

	assert.deepEqual( scalarOps(), [ 'average', 'average', 'average', 'average' ] );
	assert.deepEqual( scalarOps( [ 'sum', 'min', 'max', 'average' ] ), [ 'sum', 'min', 'max', 'average' ] );
	assert.throws( () => scalarOps( [ 'sum', 'sum' ] ), /four/ );
	assert.throws( () => scalarOps( 'median' ), /unknown/ );
	assert.throws( () => scalarOps( [ 'weight', 'weight', 'sum', 'sum' ] ), /one channel/ );
	assert.throws( () => scalarOps( [ 'weighted', 'sum', 'sum', 'sum' ] ), /weight/ );

} );

test( 'workgroups and depth cover the image with 64×64 tiles', () => {

	assert.deepEqual( scalarWorkgroups( 1, 1 ), [ 1, 1 ] );
	assert.deepEqual( scalarWorkgroups( 1920, 1080 ), [ 30, 17 ] );
	assert.deepEqual( scalarWorkgroups( 3840, 2160 ), [ 60, 34 ] );
	assert.deepEqual( scalarWorkgroups( 65, 64, 2 ), [ 3, 2 ] );
	// 4 (in-invocation) + 8 (workgroup) + sequential partials + 8 (combine).
	assert.equal( scalarDepth( 64, 64 ), 20 );
	assert.equal( scalarDepth( 3840, 2160 ), 20 + Math.ceil( 60 * 34 / 256 ) - 1 );

} );

test( 'every texel is counted once at any size: sums of 1 equal the texel count exactly', () => {

	for ( const [ width, height, layers ] of [ [ 1, 1, 1 ], [ 3, 7, 1 ], [ 63, 65, 1 ], [ 65, 129, 2 ], [ 17, 11, 3 ], [ 1, 300, 1 ], [ 1477, 831, 1 ] ] ) {

		const texels = image( width, height, layers, () => 1 );
		for ( const result of scalarReference( texels, width, height, layers, [ 'sum', 'average', 'min', 'max' ] ) ) {

			assert.deepEqual( result.value, [ width * height, 1, 1, 1 ] );
			assert.equal( result.weight, width * height );

		}

	}

} );

test( 'min and max see every texel, including the last row and column', () => {

	const [ width, height ] = [ 67, 33 ];
	const last = width * height - 1;
	const texels = image( width, height, 1, ( c, i ) => c % 2 === 0 ? i : last - i );
	const [ result ] = scalarReference( texels, width, height, 1, [ 'max', 'max', 'min', 'min' ] );
	assert.deepEqual( result.value, [ last, last, 0, 0 ] );

} );

test( 'layers reduce separately', () => {

	const texels = image( 9, 5, 3, ( c, i ) => Math.floor( i / 45 ) + 1 );
	assert.deepEqual( scalarReference( texels, 9, 5, 3, 'average' ).map( result => result.value[ 0 ] ), [ 1, 2, 3 ] );

} );

test( 'weighted: Σ w·v / Σ w, with the weight channel emitted as Σ w', () => {

	// Two texels: (v = 1, w = 3) and (v = 5, w = 1) → (3 + 5) / 4 = 2.
	const texels = Float32Array.of( 1, 10, 0, 3, 5, 20, 0, 1 );
	const [ result ] = scalarReference( texels, 2, 1, 1, [ 'weighted', 'weighted', 'sum', 'weight' ] );
	assert.deepEqual( result.value, [ 2, 12.5, 0, 4 ] );
	assert.equal( result.weight, 4 );
	const [ empty ] = scalarReference( Float32Array.of( 7, 0, 0, 0 ), 1, 1, 1, [ 'weighted', 'sum', 'sum', 'weight' ] );
	assert.equal( empty.value[ 0 ], 0 );

} );

test( 'pairing: the oracle combines in the GPU order (stride-halving trees)', () => {

	// One 16×16 tile with block 1: the 256 slots are combined as
	// slot[i] += slot[i + stride], stride 128 … 1. Here with 1 in slot 0 and
	// 2^-24 elsewhere: a serial f32 sum loses every small term, the tree keeps most.
	const small = 2 ** - 24;
	const texels = image( 16, 16, 1, ( c, i ) => i === 0 ? 1 : small );
	const [ result ] = scalarReference( texels, 16, 16, 1, 'sum', { block: 1 } );
	const slots = new Float32Array( 256 ).fill( small );
	slots[ 0 ] = 1;
	for ( let stride = 128; stride > 0; stride >>= 1 ) for ( let i = 0; i < stride; i ++ ) slots[ i ] += slots[ i + stride ];
	let serial = Math.fround( 1 );
	for ( let i = 1; i < 256; i ++ ) serial = Math.fround( serial + small );
	assert.equal( result.value[ 0 ], slots[ 0 ] );
	assert.equal( serial, 1 );
	assert.ok( slots[ 0 ] > 1 + 200 * small );

} );

test( 'error bound holds for HDR data at render sizes, and naive serial summation breaks it', () => {

	for ( const [ width, height ] of [ [ 1920, 1080 ], [ 1477, 831 ], [ 255, 257 ] ] ) {

		const texels = image( width, height, 1, c => c === 3 ? random() : ( random() * 16 - 4 ) * ( c + 1 ) );
		const [ result ] = scalarReference( texels, width, height, 1, [ 'sum', 'average', 'weighted', 'weight' ] );
		for ( let c = 0; c < 4; c ++ ) assert.ok( Math.abs( result.value[ c ] - result.exact[ c ] ) <= result.bound[ c ], `${ width }x${ height } channel ${ c }: ${ result.value[ c ] } vs ${ result.exact[ c ] }, bound ${ result.bound[ c ] }` );

		if ( width === 1920 ) {

			let serial = 0;
			for ( let i = 0; i < width * height; i ++ ) serial = Math.fround( serial + texels[ i * 4 ] );
			assert.ok( Math.abs( serial - result.exact[ 0 ] ) > result.bound[ 0 ], 'a serial f32 sum should exceed the tree bound at 1080p' );

		}

	}

} );

test( 'block size only changes the pairing, not the result beyond the bound', () => {

	const texels = image( 300, 200, 1, () => random() );
	for ( const block of [ 1, 2, 4, 8 ] ) {

		const [ result ] = scalarReference( texels, 300, 200, 1, 'average', { block } );
		assert.ok( Math.abs( result.value[ 0 ] - result.exact[ 0 ] ) <= result.bound[ 0 ] );

	}

} );
