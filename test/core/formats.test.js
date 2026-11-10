import { test } from 'node:test';
import assert from 'node:assert/strict';
import { integerDomains, roundHalfEven, toHalf, fromHalf, quantize } from '../../dist/index.js';

test( 'roundHalfEven breaks exact ties to even', () => {

	assert.deepEqual( [ 0.5, 1.5, 2.5, 179.5, 158.5, 158.50001 ].map( roundHalfEven ), [ 0, 2, 2, 180, 158, 159 ] );

} );

test( 'every finite half value round-trips', () => {

	for ( let bits = 0; bits < 0x10000; bits ++ ) {

		if ( ( bits & 0x7c00 ) === 0x7c00 ) continue;
		const value = fromHalf( bits );
		const back = toHalf( value );
		assert.ok( back === bits || ( value === 0 && ( back & 0x7fff ) === 0 ), `bits ${ bits } -> ${ value } -> ${ back }` );

	}

} );

test( 'half rounding is nearest-even, with overflow to infinity', () => {

	assert.equal( fromHalf( toHalf( 1 / 3 ) ), 0.333251953125 );
	assert.equal( toHalf( 1 + 2 ** - 11 ), 0x3c00 ); // tie -> even
	assert.equal( toHalf( 1 + 3 * 2 ** - 11 ), 0x3c02 ); // tie -> even (up)
	assert.equal( fromHalf( toHalf( 70000 ) ), Infinity );

} );

test( 'unorm8 quantization clamps and rounds', () => {

	assert.equal( quantize( 1.2, 'unorm8' ), 1 );
	assert.equal( quantize( - 1, 'unorm8' ), 0 );
	assert.equal( quantize( 179.5 / 255, 'unorm8' ) * 255, 180 );

} );

test( 'normalized quantization: every stored integer round-trips, ties go to even, snorm never yields -128', () => {

	for ( const [ quantization, { scale, min, max } ] of Object.entries( integerDomains ) ) {

		for ( let k = min; k <= max; k ++ ) assert.equal( quantize( k / scale, quantization ) * scale, k, `${ quantization } ${ k }` );
		assert.equal( quantize( 2, quantization ), 1 );
		assert.equal( quantize( - 2, quantization ), min / scale );
		assert.equal( quantize( 2.5 / scale, quantization ) * scale, 2 );
		assert.equal( quantize( 3.5 / scale, quantization ) * scale, 4 );

	}

	assert.equal( quantize( - 2.5 / 127, 'snorm8' ) * 127, - 2 );
	assert.equal( quantize( - 3.5 / 127, 'snorm8' ) * 127, - 4 );

} );
