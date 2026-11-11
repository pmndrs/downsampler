// Packed float formats: rgb9e5ufloat and rg11b10ufloat (SPEC §6.1). Exhaustive
// over the encodings (decode → encode for every code point), every rounding
// boundary of the unsigned floats, and independent oracles for random values.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	copyReference, fromUfloat, packedStep, packRg11b10, packRgb9e5, quantizeTexel, reduceReference, rg11b10Max, rgb9e5Max,
	tolerance, toUfloat, unpackRg11b10, unpackRgb9e5
} from '../../dist/index.js';

const f32 = Math.fround;

/** Deterministic float32 values spanning many binades, both signs. */
function randomFloats( count, seed, { minExponent = - 30, maxExponent = 18 } = {} ) {

	let state = seed;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 );
	return Array.from( { length: count }, () => {

		const exponent = minExponent + next() % ( maxExponent - minExponent + 1 );
		return f32( ( next() / 2 ** 32 + 1 ) * 2 ** exponent );

	} );

}

/** The next float32 above (direction 1) or below (−1) a positive float32. */
function nextFloat( value, direction ) {

	const bits = new Uint32Array( new Float32Array( [ value ] ).buffer )[ 0 ];
	return new Float32Array( new Uint32Array( [ bits + direction ] ).buffer )[ 0 ];

}

// ---------------------------------------------------------------------------
// Unsigned 11- and 10-bit floats.

for ( const bits of [ 6, 5 ] ) {

	const codes = 1 << ( bits + 5 );
	const inf = 0x1f << bits;
	const name = `${ bits + 5 }-bit ufloat`;

	test( `${ name }: every code point decodes and re-encodes to itself (NaN to the canonical NaN)`, () => {

		for ( let code = 0; code < codes; code ++ ) {

			const value = fromUfloat( code, bits );
			if ( code > inf ) {

				assert.ok( Number.isNaN( value ), `code ${ code }` );
				assert.equal( toUfloat( value, bits ), inf | 1 );
				continue;

			}

			assert.equal( f32( value ), value, `code ${ code } is a float32` );
			assert.equal( toUfloat( value, bits ), code, `code ${ code } -> ${ value }` );

		}

		assert.equal( fromUfloat( inf, bits ), Infinity );
		assert.equal( fromUfloat( inf - 1, bits ), bits === 6 ? rg11b10Max : 64512 );
		assert.equal( fromUfloat( 1, bits ), 2 ** ( - 14 - bits ) ); // smallest subnormal

	} );

	test( `${ name }: every rounding boundary rounds to nearest, ties to even`, () => {

		for ( let code = 0; code < inf - 1; code ++ ) {

			const lo = fromUfloat( code, bits ), hi = fromUfloat( code + 1, bits );
			const mid = ( lo + hi ) / 2;
			assert.equal( f32( mid ), mid, 'midpoints are float32' );
			assert.equal( toUfloat( mid, bits ), code % 2 === 0 ? code : code + 1, `tie between ${ code } and ${ code + 1 }` );
			assert.equal( toUfloat( nextFloat( mid, - 1 ), bits ), code, `just below the midpoint of ${ code }` );
			assert.equal( toUfloat( nextFloat( mid, 1 ), bits ), code + 1, `just above the midpoint of ${ code }` );

		}

	} );

	test( `${ name }: negatives, overflow, infinities and NaN`, () => {

		const max = fromUfloat( inf - 1, bits );
		assert.equal( toUfloat( - 1, bits ), 0 );
		assert.equal( toUfloat( - 0, bits ), 0 );
		assert.equal( toUfloat( - Infinity, bits ), 0 );
		assert.equal( toUfloat( - ( 2 ** - 30 ), bits ), 0 );
		assert.equal( toUfloat( 2 ** - 140, bits ), 0 ); // float32 subnormal
		// Finite overflow goes to the largest finite value (Vulkan allows it or Inf).
		for ( const value of [ max, nextFloat( max, 1 ), ( max + 2 ** ( 15 - bits ) ) , 1e6, 3.4e38 ] ) assert.equal( toUfloat( value, bits ), inf - 1, `${ value }` );
		assert.equal( toUfloat( Infinity, bits ), inf );
		assert.ok( Number.isNaN( fromUfloat( toUfloat( NaN, bits ), bits ) ) );

	} );

	test( `${ name }: random values match a nearest-value search (ties to even)`, () => {

		const table = Array.from( { length: inf }, ( _, code ) => fromUfloat( code, bits ) );
		for ( const value of randomFloats( 20000, bits ) ) {

			const v = Math.min( value, table[ inf - 1 ] );
			let best = 0;
			for ( let code = 1; code < inf; code ++ ) {

				const d = Math.abs( table[ code ] - v ), e = Math.abs( table[ best ] - v );
				if ( d < e || ( d === e && code % 2 === 0 ) ) best = code;

			}

			assert.equal( toUfloat( value, bits ), best, `${ value }` );

		}

	} );

}

test( 'rg11b10ufloat packs r in bits 0–10, g in 11–21, b in 22–31', () => {

	assert.equal( packRg11b10( 1, 0, 0 ), 15 << 6 );
	assert.equal( packRg11b10( 0, 1, 0 ), ( 15 << 6 ) << 11 );
	assert.equal( packRg11b10( 0, 0, 1 ), ( ( 15 << 5 ) << 22 ) >>> 0 );
	assert.deepEqual( unpackRg11b10( packRg11b10( 0.5, 3, 12 ) ), [ 0.5, 3, 12 ] );
	assert.deepEqual( unpackRg11b10( packRg11b10( - 1, 1e9, 1 / 3 ) ), [ 0, rg11b10Max, 0.3359375 ] );

} );

// ---------------------------------------------------------------------------
// rgb9e5ufloat.

/** EXT_texture_shared_exponent restated without log2: the smallest exponent whose rounded largest mantissa fits 9 bits. */
function referenceRgb9e5( r, g, b ) {

	const clamp = v => v > 0 ? Math.min( rgb9e5Max, f32( v ) ) : 0;
	const c = [ r, g, b ].map( clamp );
	const max = Math.max( ...c );
	const round = v => Math.floor( v ) + ( v - Math.floor( v ) >= 0.5 ? 1 : 0 );
	let e = 0;
	while ( round( max / 2 ** ( e - 24 ) ) > 511 ) e ++;
	return c.map( v => round( v / 2 ** ( e - 24 ) ) * 2 ** ( e - 24 ) );

}

test( 'rgb9e5ufloat: every decoded code re-encodes to the same value; canonical codes to themselves', () => {

	// Channels are independent given the exponent, so (exponent, largest
	// mantissa, another mantissa) covers every case; the third channel is 0, and
	// a permutation check moves the channels around.
	let canonical = 0;
	for ( let e = 0; e < 32; e ++ ) for ( let m = 0; m < 512; m ++ ) for ( let other = 0; other <= m; other ++ ) {

		const code = ( m | ( other << 9 ) | ( e << 27 ) ) >>> 0;
		const value = unpackRgb9e5( code );
		const again = packRgb9e5( ...value );
		const [ r, g, b ] = unpackRgb9e5( again );
		if ( r !== value[ 0 ] || g !== value[ 1 ] || b !== value[ 2 ] ) assert.fail( `code ${ code.toString( 16 ) } (e ${ e }, m ${ m }, ${ other }) -> ${ value } -> ${ [ r, g, b ] }` );
		// Canonical: the largest mantissa uses the 9th bit, or the exponent can't go lower.
		if ( ( m >= 256 || e === 0 ) && again !== code && m > 0 ) assert.fail( `canonical code ${ code.toString( 16 ) } re-encoded as ${ again.toString( 16 ) }` );
		if ( m === 0 && again !== 0 ) assert.fail( 'zero encodes as 0' );
		if ( m >= 256 || e === 0 ) canonical ++;

	}

	assert.ok( canonical > 32 * 256 * 256 );
	for ( const [ m, other, e ] of [ [ 300, 7, 20 ], [ 511, 1, 31 ], [ 256, 255, 0 ] ] ) {

		const value = unpackRgb9e5( ( m | ( other << 9 ) | ( e << 27 ) ) >>> 0 );
		for ( const order of [ [ 0, 1, 2 ], [ 1, 2, 0 ], [ 2, 0, 1 ], [ 1, 0, 2 ] ] ) {

			const permuted = order.map( i => [ ...value, 0 ][ i ] );
			assert.deepEqual( unpackRgb9e5( packRgb9e5( ...permuted ) ), permuted );

		}

	}

} );

test( 'rgb9e5ufloat: random full codes round-trip by value', () => {

	let state = 99;
	for ( let i = 0; i < 300000; i ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		const value = unpackRgb9e5( state );
		assert.deepEqual( unpackRgb9e5( packRgb9e5( ...value ) ), value );

	}

} );

test( 'rgb9e5ufloat: shared exponent, the 512 bump and round half up', () => {

	// 1 → exponent 16, mantissa 256.
	assert.equal( packRgb9e5( 1, 0, 0 ), ( 256 | ( 16 << 27 ) ) >>> 0 );
	// Bump: 511.5/256 rounds to mantissa 512 at exponent 16, so it takes 17 (and 256).
	assert.equal( packRgb9e5( 511.5 / 256, 0, 0 ), ( 256 | ( 17 << 27 ) ) >>> 0 );
	assert.equal( packRgb9e5( nextFloat( 511.5 / 256, - 1 ), 0, 0 ), ( 511 | ( 16 << 27 ) ) >>> 0 );
	// Round half up (EXT_texture_shared_exponent's ⌊x + 0.5⌋), not to even: 2.5 → 3.
	assert.equal( ( packRgb9e5( 1, 2.5 / 256, 0 ) >>> 9 ) & 0x1ff, 3 );
	assert.equal( ( packRgb9e5( 1, 3.5 / 256, 0 ) >>> 9 ) & 0x1ff, 4 );
	assert.equal( ( packRgb9e5( 1, nextFloat( 2.5 / 256, - 1 ), 0 ) >>> 9 ) & 0x1ff, 2 );
	// Just below a power of two: ⌊log2⌋ from the exponent bits.
	assert.equal( packRgb9e5( nextFloat( 2, - 1 ), 0, 0 ), ( 256 | ( 17 << 27 ) ) >>> 0 ); // 511.99997 → bump
	assert.equal( packRgb9e5( nextFloat( 2, 1 ), 0, 0 ), ( 256 | ( 17 << 27 ) ) >>> 0 );
	// Smallest values: exponent 0, step 2^-24; half a step rounds up.
	assert.equal( packRgb9e5( 2 ** - 25, 0, 0 ), 1 );
	assert.equal( packRgb9e5( nextFloat( 2 ** - 25, - 1 ), 0, 0 ), 0 );
	assert.deepEqual( unpackRgb9e5( packRgb9e5( 2 ** - 24 * 255, 0, 0 ) ), [ 2 ** - 24 * 255, 0, 0 ] );

} );

test( 'rgb9e5ufloat: clamping of negatives, NaN, overflow and infinity', () => {

	assert.deepEqual( unpackRgb9e5( packRgb9e5( - 1, NaN, 2 ) ), [ 0, 0, 2 ] );
	assert.deepEqual( unpackRgb9e5( packRgb9e5( 1e9, Infinity, - Infinity ) ), [ rgb9e5Max, rgb9e5Max, 0 ] );
	assert.deepEqual( unpackRgb9e5( packRgb9e5( 65408, 65407, 0 ) ), [ 65408, 65408, 0 ] ); // step 128 at the top
	assert.equal( packRgb9e5( 0, 0, 0 ), 0 );
	assert.equal( packRgb9e5( - 0, 0, 0 ), 0 );

} );

test( 'rgb9e5ufloat: random texels match the log2-free restatement of the extension', () => {

	const values = randomFloats( 60000, 7, { minExponent: - 30, maxExponent: 17 } );
	for ( let i = 0; i + 2 < values.length; i += 3 ) {

		// Spread channels across binades, with some negatives.
		const texel = [ values[ i ], values[ i + 1 ] * ( i % 7 === 0 ? - 1 : 1 ), values[ i + 2 ] / 2 ** ( i % 13 ) ];
		assert.deepEqual( unpackRgb9e5( packRgb9e5( ...texel ) ), referenceRgb9e5( ...texel ), `${ texel }` );

	}

} );

// ---------------------------------------------------------------------------
// Quantize-as-stored and the oracle.

test( 'quantizeTexel packs and unpacks, sets alpha to 1, and is idempotent', () => {

	for ( const quantization of [ 'rgb9e5ufloat', 'rg11b10ufloat' ] ) {

		const texel = [ 3.3, 0.01, - 2, 0.25 ];
		quantizeTexel( texel, 0, quantization );
		const unpack = quantization === 'rgb9e5ufloat' ? unpackRgb9e5 : unpackRg11b10, pack = quantization === 'rgb9e5ufloat' ? packRgb9e5 : packRg11b10;
		assert.deepEqual( texel, [ ...unpack( pack( 3.3, 0.01, - 2 ) ), 1 ] );
		const again = texel.slice();
		quantizeTexel( again, 0, quantization );
		assert.deepEqual( again, texel );

	}

	// The shared exponent flushes small channels next to a large one.
	const shared = [ 1000, 0.01, 0, 1 ];
	quantizeTexel( shared, 0, 'rgb9e5ufloat' );
	assert.deepEqual( shared, [ 1000, 0, 0, 1 ] );

} );

test( 'reduceReference quantizes packed texels; float32 pairing matches the f64 average when exact', () => {

	const texels = [ [ 1, 2, 3 ], [ 4, 5, 6 ], [ 0.5, 0.25, 0.125 ], [ 7, 1 / 64, 9 ] ];
	const source = Float32Array.from( texels.flatMap( t => [ ...t, 1 ] ) );
	for ( const quantization of [ 'rgb9e5ufloat', 'rg11b10ufloat' ] ) {

		const level = reduceReference( source, 2, 2, 1, quantization );
		const expected = [ 0, 1, 2 ].map( c => texels.reduce( ( sum, t ) => sum + t[ c ], 0 ) / 4 );
		const stored = [ ...expected, 1 ];
		quantizeTexel( stored, 0, quantization );
		assert.deepEqual( Array.from( level.values ), stored );
		assert.deepEqual( Array.from( reduceReference( source, 2, 2, 1, quantization, 'average', { pairing: 'float32' } ).values ), stored );
		assert.deepEqual( Array.from( copyReference( source, 2, 2, 1, quantization ).values.slice( 0, 4 ) ), ( () => { const t = [ 1, 2, 3, 1 ]; quantizeTexel( t, 0, quantization ); return t; } )() );

	}

	// float32 pairing rounds after each add: 2^24 + 1 + 1 + 0 is 2^24 in f32 (then exact ÷ 4).
	const big = Float32Array.from( [ 2 ** 24, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1 ] );
	assert.equal( reduceReference( big, 2, 2, 1, 'float32', 'average', { pairing: 'float32' } ).values[ 0 ], 2 ** 22 );
	assert.equal( reduceReference( big, 2, 2, 1, 'float32' ).values[ 0 ], f32( ( 2 ** 24 + 2 ) / 4 ) );

} );

test( 'packed tolerances are two steps of the stored value', () => {

	// rgb9e5: the texel's largest channel sets the step (2^(e − 24)).
	assert.equal( packedStep( 'rgb9e5ufloat', 0.01, 1, 3 ), 2 ** - 7 );
	assert.equal( tolerance( 0.01, 'rgb9e5ufloat', 0, 'average', { channel: 1, texelMax: 3 } ), 2 ** - 6 + 2 ** - 13 );
	// rg11b10: per channel, 6 (r, g) or 5 (b) mantissa bits; subnormals step 2^-20 / 2^-19.
	assert.equal( packedStep( 'rg11b10ufloat', 3, 0 ), 2 ** - 5 );
	assert.equal( packedStep( 'rg11b10ufloat', 3, 2 ), 2 ** - 4 );
	assert.equal( packedStep( 'rg11b10ufloat', 2 ** - 16, 0 ), 2 ** - 20 );
	assert.equal( tolerance( 1, 'rg11b10ufloat', 0, 'min', { channel: 0, texelMax: 1 } ), 0 );
	assert.throws( () => tolerance( 1, 'rgb9e5ufloat' ), /texel/ );

} );
