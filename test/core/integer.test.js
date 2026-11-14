import { test } from 'node:test';
import assert from 'node:assert/strict';
import { channelDomain, formats, intRange, intReference, isIntFormat, quantize, reduceReference, roundedMean, sampleKind } from '../../dist/index.js';

// An independent oracle: the exact rational mean with BigInt, rounded half to even.
function bigMean( values, weights, denominator ) {

	let n = 0n;
	values.forEach( ( v, i ) => { n += BigInt( v ) * BigInt( weights[ i ] ); } );
	const d = BigInt( denominator );
	const q = n / d, r = n % d;
	return Number( 2n * r > d || ( 2n * r === d && q % 2n === 1n ) ? q + 1n : q );

}

let state = 7;
const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 );

test( 'roundedMean is the exact mean, ties to even, across the u32 range', () => {

	const extremes = [ 0, 1, 2, 3, 65535, 65536, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 2, 2 ** 32 - 1 ];
	for ( let trial = 0; trial < 20000; trial ++ ) {

		const taps = 1 + ( next() % 9 );
		const values = Array.from( { length: taps }, () => next() % 4 === 0 ? extremes[ next() % extremes.length ] : next() );
		// Weights up to 2^15 per axis, so products up to 2^30 (dimensions up to 32768).
		const weights = Array.from( { length: taps }, () => next() % 4 === 0 ? 0 : 1 + next() % ( 1 << ( next() % 31 ) ) );
		const denominator = weights.reduce( ( a, b ) => a + b, 0 );
		if ( denominator === 0 ) continue;
		assert.equal( roundedMean( values, weights, denominator ), bigMean( values, weights, denominator ), `${ values } ${ weights }` );

	}

	// Ties go to even.
	assert.equal( roundedMean( [ 1, 2 ], [ 1, 1 ], 2 ), 2 );
	assert.equal( roundedMean( [ 2, 3 ], [ 1, 1 ], 2 ), 2 );
	assert.equal( roundedMean( [ 2 ** 32 - 1, 2 ** 32 - 2 ], [ 1, 1 ], 2 ), 2 ** 32 - 2 );
	assert.equal( roundedMean( [ 2 ** 32 - 1, 2 ** 32 - 1, 2 ** 32 - 1, 2 ** 32 - 1 ], [ 1, 1, 1, 1 ], 4 ), 2 ** 32 - 1 );

} );

test( 'integer formats: ranges, sample kinds, and which formats are integer', () => {

	assert.deepEqual( intRange( 'uint8', 0 ), [ 0, 255 ] );
	assert.deepEqual( intRange( 'sint8', 0 ), [ - 128, 127 ] );
	assert.deepEqual( intRange( 'sint32', 3 ), [ - ( 2 ** 31 ), 2 ** 31 - 1 ] );
	assert.deepEqual( intRange( 'uint32', 1 ), [ 0, 2 ** 32 - 1 ] );
	assert.deepEqual( intRange( 'uint10a2', 2 ), [ 0, 1023 ] );
	assert.deepEqual( intRange( 'uint10a2', 3 ), [ 0, 3 ] );
	assert.equal( sampleKind( 'r32uint' ), 'uint' );
	assert.equal( sampleKind( 'rg8sint' ), 'sint' );
	assert.equal( sampleKind( 'rgb10a2uint' ), 'uint' );
	assert.equal( sampleKind( 'rgb10a2unorm' ), 'float' );
	assert.equal( sampleKind( 'depth32float' ), 'float' );
	const ints = Object.keys( formats ).filter( isIntFormat ).sort();
	assert.deepEqual( ints, [ 'r16sint', 'r16uint', 'r32sint', 'r32uint', 'r8sint', 'r8uint', 'rg16sint', 'rg16uint', 'rg32sint', 'rg32uint', 'rg8sint', 'rg8uint', 'rgb10a2uint', 'rgba16sint', 'rgba16uint', 'rgba32sint', 'rgba32uint', 'rgba8sint', 'rgba8uint' ] );
	for ( const format of ints ) assert.equal( sampleKind( format ), formats[ format ].quantization.startsWith( 's' ) ? 'sint' : 'uint', format );
	assert.equal( quantize( 2.5, 'uint8' ), 2 );
	assert.equal( quantize( 300, 'uint8' ), 255 );
	assert.equal( quantize( - 300, 'sint8' ), - 128 );
	assert.equal( quantize( 9, 'uint10a2', 3 ), 3 );

} );

test( 'rgb10a2unorm: 10-bit r, g, b and a 2-bit alpha, every code round-trips, ties to even', () => {

	for ( let c = 0; c < 4; c ++ ) {

		const { scale, min, max } = channelDomain( 'unorm10a2', c );
		assert.equal( scale, c === 3 ? 3 : 1023 );
		for ( let k = min; k <= max; k ++ ) assert.equal( quantize( k / scale, 'unorm10a2', c ) * scale, k, `channel ${ c } ${ k }` );
		assert.equal( quantize( 2.5 / scale, 'unorm10a2', c ) * scale, 2 );
		assert.equal( quantize( 1.5 / scale, 'unorm10a2', c ) * scale, 2 );
		assert.equal( quantize( 2, 'unorm10a2', c ), 1 );
		assert.equal( quantize( - 1, 'unorm10a2', c ), 0 );

	}

	// A 2×2 average: r in 1023 units, alpha in 3 units, each rounded half to even.
	const texel = ( r, a ) => [ r / 1023, 0, 0, a / 3 ];
	const source = Float32Array.from( [ ...texel( 1, 0 ), ...texel( 2, 1 ), ...texel( 2, 1 ), ...texel( 1, 0 ) ] );
	const level = reduceReference( source, 2, 2, 1, 'unorm10a2', 'average', { weights: 'shader' } );
	assert.equal( Math.round( level.values[ 0 ] * 1023 ), 2 ); // 1.5 → 2
	assert.equal( Math.round( level.values[ 3 ] * 3 ), 0 ); // 0.5 → 0

} );

/** Brute force: every output texel from its definition, without the reference's taps. */
function bruteForce( source, width, height, quantization, reduction, area ) {

	const w = Math.max( 1, width >> 1 ), h = Math.max( 1, height >> 1 );
	const signed = quantization.startsWith( 's' );
	const out = [];
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) for ( let c = 0; c < 4; c ++ ) {

		const at = ( sx, sy ) => source[ ( sy * width + sx ) * 4 + c ];
		if ( reduction === 'max' ) {

			// Every texel whose area the output's normalized area touches (independent of footprint()).
			let value = - Infinity;
			for ( let sy = 0; sy < height; sy ++ ) for ( let sx = 0; sx < width; sx ++ ) {

				const touches = ( s, n, t, o ) => t * n < ( o + 1 ) * s && ( t + 1 ) * n > o * s;
				if ( touches( width, w, sx, x ) && touches( height, h, sy, y ) ) value = Math.max( value, at( sx, sy ) );

			}

			out.push( value );
			continue;

		}

		// Per axis weights in units of 1/(2n) (bilinear) or 1/S (area), as BigInt products.
		const axis = ( o, s, n ) => {

			const weights = new Map();
			if ( area ) for ( let t = 0; t < s; t ++ ) {

				const overlap = Math.min( n * ( t + 1 ), ( o + 1 ) * s ) - Math.max( n * t, o * s );
				if ( overlap > 0 ) weights.set( t, overlap );

			} else {

				// Sample position (o + 0.5)·s/n − 0.5 = p / 2n.
				const p = ( 2 * o + 1 ) * s - n, d = 2 * n;
				const t0 = Math.floor( p / d ), r = p - t0 * d;
				const clampT = t => Math.min( s - 1, Math.max( 0, t ) );
				weights.set( clampT( t0 ), ( weights.get( clampT( t0 ) ) ?? 0 ) + d - r );
				weights.set( clampT( t0 + 1 ), ( weights.get( clampT( t0 + 1 ) ) ?? 0 ) + r );

			}

			return { weights, d: area ? s : 2 * n };

		};

		const ax = axis( x, width, w ), ay = axis( y, height, h );
		const values = [], weights = [];
		for ( const [ sy, wy ] of ay.weights ) for ( const [ sx, wx ] of ax.weights ) {

			values.push( at( sx, sy ) + ( signed ? 2 ** 31 : 0 ) );
			weights.push( wx * wy );

		}

		out.push( bigMean( values, weights, ax.d * ay.d ) - ( signed ? 2 ** 31 : 0 ) );

	}

	return out;

}

test( 'intReference matches a brute-force definition at odd, even and unit sizes, signed and unsigned', () => {

	const shapes = [ [ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 7, 4 ], [ 5, 1 ], [ 1, 9 ], [ 15, 17 ], [ 31, 31 ], [ 33, 8 ] ];
	for ( const quantization of [ 'uint8', 'sint16', 'uint32', 'sint32' ] ) for ( const [ width, height ] of shapes ) {

		const [ min, max ] = intRange( quantization, 0 );
		const source = Array.from( { length: width * height * 4 }, () => next() % 3 === 0 ? ( next() & 1 ? min : max ) : min + ( next() % ( max - min + 1 ) ) );
		for ( const reduction of [ 'average', 'max' ] ) for ( const area of [ false, true ] ) {

			const level = intReference( source, width, height, 1, quantization, reduction, { weights: area ? 'area' : 'bilinear' } );
			assert.deepEqual( [ ...level.values ], bruteForce( source, width, height, quantization, reduction, area ), `${ quantization } ${ width }x${ height } ${ reduction } ${ area ? 'area' : 'bilinear' }` );

		}

	}

} );

test( 'intReference: even steps are the 2×2 mean under both weights, a 31×31 box reduces to its rounded mean', () => {

	const source = Array.from( { length: 8 * 6 * 4 }, () => next() );
	const bilinear = intReference( source, 8, 6, 1, 'uint32', 'average' );
	const area = intReference( source, 8, 6, 1, 'uint32', 'average', { weights: 'area' } );
	assert.deepEqual( bilinear.values, area.values );
	const at = ( x, y, c ) => source[ ( y * 8 + x ) * 4 + c ];
	assert.equal( bilinear.values[ 0 ], bigMean( [ at( 0, 0, 0 ), at( 1, 0, 0 ), at( 0, 1, 0 ), at( 1, 1, 0 ) ], [ 1, 1, 1, 1 ], 4 ) );

	// 31 → 15 → 7 → 3 → 1 with area weights: each level keeps the mean, rounded once per level.
	const base = Array.from( { length: 31 * 31 * 4 }, ( _, i ) => ( i * 7919 ) % 1000 );
	let level = { values: base, width: 31, height: 31 };
	while ( level.width > 1 ) level = intReference( level.values, level.width, level.height, 1, 'uint16', 'average', { weights: 'area' } );
	const mean = base.filter( ( _, i ) => i % 4 === 0 ).reduce( ( a, b ) => a + b, 0 ) / ( 31 * 31 );
	assert.ok( Math.abs( level.values[ 0 ] - mean ) <= 2, `${ level.values[ 0 ]} vs ${ mean }` );

} );

test( 'intReference: folds, custom folds on signed values, a wider source clamped, and the rules it rejects', () => {

	const source = [ - 5, 10, 3, 1, 7, - 2, 9, 1, 0, 0, 0, 1, - 9, 4, 4, 1 ];
	assert.deepEqual( [ ...intReference( source, 2, 2, 1, 'sint8', 'min' ).values ], [ - 9, - 2, 0, 1 ] );
	assert.deepEqual( [ ...intReference( source, 2, 2, 1, 'sint8', 'minmax' ).values ], [ - 9, 10, 0, 1 ] );
	assert.deepEqual( [ ...intReference( source, 2, 2, 1, 'sint8', ( a, b ) => a | b ).values ], [ - 5 | 7 | 0 | - 9, 10 | - 2 | 0 | 4, 3 | 9 | 0 | 4, 1 ] );
	// Signed means: −5, 7, 0, −9 → −7/4 → −2 (−1.75); −2.5 → −2 (ties to even).
	assert.equal( intReference( source, 2, 2, 1, 'sint8' ).values[ 0 ], - 2 );
	assert.equal( intReference( [ - 2, 0, 0, 0, - 3, 0, 0, 0 ], 2, 1, 1, 'sint8' ).values[ 0 ], - 2 );
	assert.equal( intReference( [ - 3, 0, 0, 0, - 4, 0, 0, 0 ], 2, 1, 1, 'sint8' ).values[ 0 ], - 4 );
	// A same-size copy from a wider source clamps to the destination's range.
	const copy = intReference( [ 300, 70000, 2 ** 32 - 1, 5 ], 1, 1, 1, 'uint8', 'max', { width: 1, height: 1, source: 'uint32' } );
	assert.deepEqual( [ ...copy.values ], [ 255, 255, 255, 5 ] );
	assert.deepEqual( [ ...intReference( [ 5, 1, 7, 9 ], 1, 1, 1, 'uint10a2', 'average', { width: 1, height: 1, source: 'uint16' } ).values ], [ 5, 1, 7, 3 ] );
	assert.throws( () => intReference( source, 2, 2, 1, 'uint8', 'average', { source: 'sint8' } ), /signed and unsigned/ );
	assert.throws( () => intReference( source, 2, 2, 1, 'uint8', 'average', { width: 3, height: 1 } ), /needs weights 'area'/ );
	assert.throws( () => reduceReference( new Float32Array( 16 ), 2, 2, 1, 'uint8' ), /intReference/ );

} );
