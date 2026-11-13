import { test } from 'node:test';
import assert from 'node:assert/strict';
import { areaReference, boxReference, copyReference, formats, quantize, quantizeSrgb, quantizeTexel, reduceReference, spdReference, srgbBounds, srgbCode, srgbDecode, srgbDecodeTable, srgbEncode, srgbSteps, srgbThresholds, tolerance } from '../../dist/index.js';

const f32 = new Float32Array( 1 ), u32 = new Uint32Array( f32.buffer );
const nextDown = value => { f32[ 0 ] = value; u32[ 0 ] --; return f32[ 0 ]; };
/** The IEC 61966-2-1 encode formula as the standard writes it (not the inverse of our decode). */
const standardEncode = v => v <= 0.0031308 ? 12.92 * v : 1.055 * v ** ( 1 / 2.4 ) - 0.055;

test( 'decode table: the f32 of the exact decode, monotonic, 0 and 1 at the ends, linear segment up to code 10', () => {

	assert.equal( srgbDecodeTable.length, 256 );
	assert.equal( srgbDecodeTable[ 0 ], 0 );
	assert.equal( srgbDecodeTable[ 255 ], 1 );
	for ( let code = 0; code < 256; code ++ ) {

		const c = code / 255;
		const exact = c <= 0.04045 ? c / 12.92 : ( ( c + 0.055 ) / 1.055 ) ** 2.4;
		assert.equal( srgbDecodeTable[ code ], Math.fround( exact ), `code ${ code }` );
		if ( code > 0 ) assert.ok( srgbDecodeTable[ code ] > srgbDecodeTable[ code - 1 ] );

	}

	assert.equal( srgbDecodeTable[ 10 ], Math.fround( 10 / 255 / 12.92 ) );
	assert.equal( srgbDecodeTable[ 11 ], Math.fround( ( ( 11 / 255 + 0.055 ) / 1.055 ) ** 2.4 ) );

} );

test( 'encode inverts the decode table: every code round-trips, so min/max of decoded values select stored codes', () => {

	for ( let code = 0; code < 256; code ++ ) {

		assert.equal( srgbCode( srgbDecodeTable[ code ] ), code );
		assert.equal( quantizeSrgb( srgbDecodeTable[ code ] ), srgbDecodeTable[ code ] );

	}

} );

test( 'encode rounds to nearest: against the standard\'s encode formula at 200000 random values and every threshold neighbourhood', () => {

	let state = 7;
	for ( let i = 0; i < 200000; i ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		// Spread over [0, 1] and densely near 0, where the linear segment is.
		const v = i % 2 ? state / 2 ** 32 : ( state / 2 ** 32 ) ** 4;
		const scaled = 255 * standardEncode( v );
		// The standard's two formulas meet within 1e-8, so only values that close to a tie may differ.
		if ( Math.abs( scaled - Math.floor( scaled ) - 0.5 ) < 1e-6 ) continue;
		assert.equal( srgbCode( v ), Math.round( scaled ), `v ${ v }` );

	}

	for ( let k = 0; k < 255; k ++ ) {

		const t = srgbThresholds[ k ];
		assert.equal( srgbEncode( t ) * 255, srgbEncode( t ) * 255 ); // finite
		assert.ok( Math.abs( srgbEncode( t ) * 255 - ( k + 0.5 ) ) < 1e-9, `threshold ${ k } encodes to k + ½` );
		assert.equal( srgbCode( t * ( 1 + 1e-12 ) ), k + 1 );
		assert.equal( srgbCode( t * ( 1 - 1e-12 ) ), k );

	}

} );

test( 'no ties: no threshold is an f32, and each lies far enough from the f32 grid that its f64 rounding can\'t move a bound', () => {

	for ( let k = 0; k < 255; k ++ ) {

		const t = srgbThresholds[ k ];
		assert.notEqual( Math.fround( t ), t, `threshold ${ k } is an f32` );
		// Distance to the nearest f32, in f64 ulps of t: Math.pow is within an ulp or two.
		const ulp = 2 ** ( Math.floor( Math.log2( t ) ) - 52 );
		const below = Math.fround( t ) > t ? nextDown( Math.fround( t ) ) : Math.fround( t );
		const above = srgbBounds[ k ];
		assert.ok( below < t && t < above, `bound ${ k }` );
		assert.ok( Math.min( t - below, above - t ) > 64 * ulp, `threshold ${ k } is within 64 f64 ulps of an f32` );

	}

} );

test( 'bounds: for f32 values, v ≥ bound(k) exactly when v is above threshold k (what the shader compares)', () => {

	for ( let k = 0; k < 255; k ++ ) {

		const bound = srgbBounds[ k ];
		assert.equal( srgbCode( bound ), k + 1, `bound ${ k }` );
		assert.equal( srgbCode( nextDown( bound ) ), k, `below bound ${ k }` );

	}

} );

test( 'out of range: negatives, −0 and NaN encode to 0, values above 1 and +Inf to 255', () => {

	for ( const v of [ - 1, - 0, 0, NaN, - Infinity, 1e-30 ] ) assert.equal( srgbCode( v ), 0, String( v ) );
	for ( const v of [ 1, 1.5, 1e30, Infinity ] ) assert.equal( srgbCode( v ), 255, String( v ) );

} );

test( 'formats: rgba8unorm-srgb and bgra8unorm-srgb are srgb8, never storage; quantize() asks for whole texels', () => {

	for ( const format of [ 'rgba8unorm-srgb', 'bgra8unorm-srgb' ] ) {

		assert.equal( formats[ format ].quantization, 'srgb8' );
		assert.equal( formats[ format ].storage, false );
		assert.equal( formats[ format ].bytesPerTexel, 4 );

	}

	assert.throws( () => quantize( 0.5, 'srgb8' ), /whole texels/ );
	const texel = [ 0.5, 0.2, 2, 0.5 ];
	quantizeTexel( texel, 0, 'srgb8' );
	assert.deepEqual( texel, [ srgbDecodeTable[ 188 ], srgbDecodeTable[ 124 ], 1, 128 / 255 ] );

} );

test( 'reference averages colour in linear light, alpha as unorm8 (ties to even)', () => {

	// Black and white side by side: the linear mean 0.5 is code 188, not the encoded-space mean (128).
	const source = Float32Array.from( [ 0, 0, 0, 0, 1, 1, 1, 1 ] );
	const level = reduceReference( source, 2, 1, 1, 'srgb8' );
	assert.equal( level.values[ 0 ], srgbDecodeTable[ 188 ] );
	assert.equal( srgbCode( level.values[ 0 ] ), 188 );
	assert.equal( Math.round( standardEncode( 0.5 ) * 255 ), 188 );
	// Alpha 0 and 1: 127.5 units, a tie, to even.
	assert.equal( level.values[ 3 ], Math.fround( 128 / 255 ) );
	const alphaTie = reduceReference( Float32Array.from( [ 0, 0, 0, 1 / 255, 0, 0, 0, 2 / 255, 0, 0, 0, 2 / 255, 0, 0, 0, 2 / 255 ] ), 2, 2, 1, 'srgb8' );
	assert.equal( alphaTie.values[ 3 ], Math.fround( 2 / 255 ) ); // 1.75 → 2
	const alphaHalf = reduceReference( Float32Array.from( [ 0, 0, 0, 1 / 255, 0, 0, 0, 2 / 255 ] ), 2, 1, 1, 'srgb8' );
	assert.equal( alphaHalf.values[ 3 ], Math.fround( 2 / 255 ) ); // 1.5 → 2 (even)

} );

test( 'reference: in-tile levels pair colour in float32 (the kernel\'s order), then encode; folds select codes', () => {

	const codes = [ 3, 200, 17, 255, 90, 91, 0, 128 ];
	const source = Float32Array.from( codes.flatMap( code => [ srgbDecodeTable[ code ], srgbDecodeTable[ 255 - code ], srgbDecodeTable[ code >> 1 ], code / 255 ] ) );
	const paired = reduceReference( source, 4, 2, 1, 'srgb8', 'average', { pairing: 'float32' } );
	const f = Math.fround, at = ( x, y, c ) => source[ ( y * 4 + x ) * 4 + c ];
	for ( let x = 0; x < 2; x ++ ) for ( let c = 0; c < 3; c ++ ) {

		const sum = f( f( f( at( 2 * x, 0, c ) + at( 2 * x + 1, 0, c ) ) + at( 2 * x, 1, c ) ) + at( 2 * x + 1, 1, c ) ) * 0.25;
		assert.equal( paired.values[ x * 4 + c ], srgbDecodeTable[ srgbCode( f( sum ) ) ] );

	}

	const max = reduceReference( source, 4, 2, 1, 'srgb8', 'max' );
	assert.equal( srgbCode( max.values[ 0 ] ), 200 );
	assert.equal( srgbCode( max.values[ 4 ] ), 255 );
	assert.equal( max.values[ 3 ], Math.fround( 200 / 255 ) );
	const min = reduceReference( source, 4, 2, 1, 'srgb8', 'min' );
	assert.equal( srgbCode( min.values[ 1 ] ), 255 - 200 );

} );

test( 'reference: box, area, spd and copy oracles quantize sRGB texels the same way', () => {

	// 3 → 1 under 'box': weights 1/3 each, in linear light.
	const source = Float32Array.from( [ 0, 0, 0, 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 1 ] );
	const box = boxReference( source, 3, 1, 1, 'srgb8' );
	assert.equal( box.values[ 0 ], quantizeSrgb( ( 1 + 0.25 ) / 3 ) );
	const area = areaReference( source, 3, 1, 1, 'srgb8', 'average', 2, 1 );
	assert.equal( area.values[ 0 ], quantizeSrgb( ( 0 + 0.5 * 1 ) / 1.5 ) );
	assert.equal( area.values[ 4 ], quantizeSrgb( ( 0.5 * 1 + 0.25 ) / 1.5 ) );
	const [ spd ] = spdReference( Float32Array.from( [ 0, 0, 0, 1, 1, 1, 1, 1 ] ), 2, 1, 1, 'srgb8' );
	assert.equal( spd.values[ 0 ], srgbDecodeTable[ 188 ] );
	const copy = copyReference( Float32Array.from( [ 0.2, 2, - 1, 0.3 ] ), 1, 1, 1, 'srgb8' );
	assert.deepEqual( Array.from( copy.values ), [ srgbDecodeTable[ srgbCode( 0.2 ) ], 1, 0, Math.fround( 77 / 255 ) ] );

} );

test( 'tolerance: two codes around the value for colour, unorm8 steps for alpha; the texel is required', () => {

	assert.throws( () => tolerance( 0.5, 'srgb8' ), /texel/ );
	const value = srgbDecodeTable[ 188 ];
	assert.ok( Math.abs( tolerance( value, 'srgb8', 0, 'average', { channel: 0 } ) - ( srgbDecodeTable[ 190 ] - value ) ) < 1e-4 );
	assert.ok( tolerance( value, 'srgb8', 0, 'average', { channel: 0 } ) >= srgbSteps( value, 2 ) );
	assert.equal( tolerance( 0.5, 'srgb8', 0, 'average', { channel: 3 } ), 2.01 / 255 );
	assert.equal( tolerance( 0.5, 'srgb8', 0, 'max', { channel: 0 } ), 0 );

} );
