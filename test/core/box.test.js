import { test } from 'node:test';
import assert from 'node:assert/strict';
import { areaOverlap, areaReference, areaSpan, boxChainReference, boxReference, maxViolations, mipLevelCountFor, planBatches, reduceReference } from '../../dist/index.js';
import { createKernel } from '../../dist/webgpu/kernel.js';

globalThis.GPUShaderStage ??= { COMPUTE: 4 };

const rgba = values => Float32Array.from( values.flatMap( value => [ value, value, value, 1 ] ) );
const random = ( count, seed = 1, scale = 1 ) => {

	let state = seed;
	return Array.from( { length: count }, () => {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		return Math.fround( state / 2 ** 32 * scale );

	} );

};
const mean = ( values, channel = 0 ) => {

	let sum = 0, count = 0;
	for ( let i = channel; i < values.length; i += 4, count ++ ) sum += values[ i ];
	return sum / count;

};

test( 'odd sizes take nvpro_pyramid\'s three taps (n − x, n, x + 1) / (2n + 1)', () => {

	for ( let n = 1; n <= 64; n ++ ) for ( let x = 0; x < n; x ++ ) {

		const size = 2 * n + 1;
		assert.deepEqual( areaSpan( x, size, n ), [ 2 * x, 2 * x + 2 ] );
		assert.deepEqual( [ 0, 1, 2 ].map( i => areaOverlap( 2 * x + i, x, size, n ) ), [ n - x, n, x + 1 ] );

	}

	// Even sizes: two taps of n (weight 1/2); a size of 1: one tap.
	assert.deepEqual( areaSpan( 3, 16, 8 ), [ 6, 7 ] );
	assert.deepEqual( [ areaOverlap( 6, 3, 16, 8 ), areaOverlap( 7, 3, 16, 8 ) ], [ 8, 8 ] );
	assert.deepEqual( [ areaSpan( 0, 1, 1 ), areaOverlap( 0, 0, 1, 1 ) ], [ [ 0, 0 ], 1 ] );
	// 3 → 1 averages all three texels (render passes take the centre one alone).
	assert.equal( boxReference( rgba( [ 0, 0.5, 1 ] ), 3, 1, 1, 'float32' ).values[ 0 ], 0.5 );
	assert.equal( boxReference( rgba( [ 0, 0, 1 ] ), 3, 1, 1, 'float32' ).values[ 0 ], Math.fround( 1 / 3 ) );
	assert.equal( reduceReference( rgba( [ 0, 0, 1 ] ), 3, 1, 1, 'float32' ).values[ 0 ], 0 );

} );

test( 'any ratio: the overlaps of each output texel sum to the source size, and every source texel is covered', () => {

	for ( const [ source, size ] of [ [ 1920, 1024 ], [ 1080, 512 ], [ 1129, 512 ], [ 635, 256 ], [ 1920, 2048 ], [ 1080, 1024 ], [ 31, 16 ], [ 3, 8 ], [ 7, 3 ], [ 300, 256 ] ] ) {

		const covered = new Uint8Array( source );
		for ( let x = 0; x < size; x ++ ) {

			const [ first, last ] = areaSpan( x, source, size );
			let total = 0;
			for ( let t = first; t <= last; t ++ ) {

				const overlap = areaOverlap( t, x, source, size );
				assert.ok( overlap > 0, `${ source } → ${ size }: texel ${ t } of output ${ x } has no overlap` );
				total += overlap;
				covered[ t ] = 1;

			}

			assert.equal( total, source, `${ source } → ${ size }, output ${ x }` );
			assert.equal( areaOverlap( first - 1, x, source, size ), 0 );
			assert.equal( areaOverlap( last + 1, x, source, size ), 0 );

		}

		assert.ok( covered.every( Boolean ), `${ source } → ${ size }: a source texel is in no footprint` );

	}

} );

test( 'greggman\'s 31×31: the box chain\'s 1×1 is the true mean; render-pass mipmaps\' is not', () => {

	// Vertical stripes, one texel wide: 16 of 31 columns are 1, so the mean is 16/31.
	const stripes = rgba( Array.from( { length: 31 * 31 }, ( _, i ) => ( i % 31 ) % 2 === 0 ? 1 : 0 ) );
	const box = boxChainReference( stripes, 31, 31, 1, 'float32', 'average', 4 ).pop();
	assert.deepEqual( [ box.width, box.height ], [ 1, 1 ] );
	assert.ok( Math.abs( box.values[ 0 ] - 16 / 31 ) < 1e-6, `box 1×1 ${ box.values[ 0 ] }` );

	let exact = stripes, w = 31, h = 31;
	while ( w > 1 ) { const level = reduceReference( exact, w, h, 1, 'float32' ); exact = level.values; w = level.width; h = level.height; }
	assert.ok( Math.abs( exact[ 0 ] - 16 / 31 ) > 0.1, `'exact' 1×1 ${ exact[ 0 ] }` );

	// Random data: the 1×1 of every channel is the mean, to float32 rounding per level.
	const noise = Float32Array.from( random( 31 * 31 * 4, 7 ) );
	const one = boxChainReference( noise, 31, 31, 1, 'float32', 'average', 4 ).pop();
	for ( let c = 0; c < 4; c ++ ) assert.ok( Math.abs( one.values[ c ] - mean( noise, c ) ) < 1e-6 );

} );

test( 'every level of a box chain keeps the mean at render sizes, layers and strips', () => {

	for ( const [ width, height, layers ] of [ [ 1129, 635, 1 ], [ 1477, 831, 1 ], [ 1, 300, 1 ], [ 299, 1, 1 ], [ 17, 11, 3 ] ] ) {

		const data = Float32Array.from( random( width * height * layers * 4, width, 16 ) );
		const chain = boxChainReference( data, width, height, layers, 'float32', 'average', mipLevelCountFor( width, height ) - 1 );
		for ( const level of chain ) for ( let layer = 0; layer < layers; layer ++ ) {

			const texels = level.width * level.height * 4, base = width * height * 4;
			const got = mean( level.values.subarray( layer * texels, ( layer + 1 ) * texels ) );
			const want = mean( data.subarray( layer * base, ( layer + 1 ) * base ) );
			assert.ok( Math.abs( got - want ) < 2e-5 * Math.abs( want ), `${ width }x${ height}x${ layers } ${ level.width }x${ level.height } layer ${ layer }: ${ got } vs ${ want }` );

		}

		// 'exact' does not: its odd steps weight edge texels by half or less.
		if ( width === 1129 ) {

			let values = data, w = width, h = height;
			while ( w > 1 || h > 1 ) { const level = reduceReference( values, w, h, 1, 'float32' ); values = level.values; w = level.width; h = level.height; }
			assert.ok( Math.abs( values[ 0 ] - mean( data ) ) > 1e-4 );

		}

	}

} );

test( 'normalized formats: exact rational weights, ties to even', () => {

	// 3×2 → 1×1: weights 1/6 each, an exact tie at sum 6k + 3.
	const level = values => Math.round( boxReference( Float32Array.from( values.flatMap( v => [ v / 255, 0, 0, 1 ] ) ), 3, 2, 1, 'unorm8' ).values[ 0 ] * 255 );
	assert.equal( level( [ 1, 1, 1, 0, 0, 0 ] ), 0 ); // 0.5 → 0
	assert.equal( level( [ 3, 3, 3, 0, 0, 0 ] ), 2 ); // 1.5 → 2
	assert.equal( level( [ 4, 0, 0, 0, 0, 0 ] ), 1 ); // 0.67 → 1
	// 5 → 2 (n = 2): weights (2, 2, 1) / 5 and (1, 2, 2) / 5.
	const strip = boxReference( Float32Array.from( [ 10, 20, 30, 40, 50 ].flatMap( v => [ v / 255, 0, 0, 1 ] ) ), 5, 1, 1, 'unorm8' );
	assert.deepEqual( [ strip.values[ 0 ], strip.values[ 4 ] ].map( v => Math.round( v * 255 ) ), [ 18, 42 ] ); // (2·10 + 2·20 + 30) / 5, (30 + 2·40 + 2·50) / 5
	// snorm: negative ties round to even as well.
	const negative = boxReference( Float32Array.from( [ - 1, - 1, - 1, 0, 0, 0 ].flatMap( v => [ v / 127, 0, 0, 1 ] ) ), 3, 2, 1, 'snorm8' );
	assert.equal( Math.round( negative.values[ 0 ] * 127 ), 0 ); // −0.5 → 0

} );

test( 'even and unit steps are the \'exact\' 2×2 box, and 2:1 folds are the \'exact\' folds', () => {

	for ( const [ width, height ] of [ [ 16, 16 ], [ 1, 128 ], [ 64, 1 ], [ 64, 32 ] ] ) for ( const quantization of [ 'unorm8', 'float16', 'float32', 'rgb9e5ufloat' ] ) {

		const data = Float32Array.from( random( width * height * 4, 3, 4 ) );
		assert.deepEqual( boxReference( data, width, height, 1, quantization ), reduceReference( data, width, height, 1, quantization ) );
		assert.deepEqual( boxReference( data, width, height, 1, quantization, 'average', { weights: 'shader' } ), reduceReference( data, width, height, 1, quantization, 'average', { weights: 'shader' } ) );

	}

	for ( const [ width, height ] of [ [ 31, 33 ], [ 3, 7 ], [ 17, 9 ] ] ) for ( const reduction of [ 'min', 'max', 'minmax', ( a, b ) => Math.max( a, b ) ] ) {

		const data = Float32Array.from( random( width * height * 4, 5 ) );
		assert.deepEqual( boxReference( data, width, height, 1, 'float32', reduction ).values, reduceReference( data, width, height, 1, 'float32', reduction ).values );

	}

} );

test( 'area folds are conservative at any ratio; a 2×2 max at the texel centre (a pow2 Hi-Z with a reduction sampler) is not', () => {

	for ( const [ width, height, dw, dh ] of [ [ 1920, 1, 1024, 1 ], [ 1, 1080, 1, 512 ], [ 1129, 1, 512, 1 ], [ 1920, 1, 2048, 1 ], [ 64, 36, 32, 16 ], [ 120, 67, 128, 64 ] ] ) {

		const data = Float32Array.from( random( width * height * 4, 11 ) );
		const level = areaReference( data, width, height, 1, 'float32', 'max', dw, dh );
		assert.equal( maxViolations( data, width, height, level.values, dw, dh ), 0, `${ width }x${ height } → ${ dw }x${ dh }` );
		// Each value is the max of exactly its overlapping texels: none outside them.
		const [ x0, x1 ] = areaSpan( dw - 1, width, dw );
		let max = - Infinity;
		for ( let y = areaSpan( dh - 1, height, dh )[ 0 ]; y < height; y ++ ) for ( let x = x0; x <= x1; x ++ ) max = Math.max( max, data[ ( y * width + x ) * 4 ] );
		assert.equal( level.values[ ( dh * dw - 1 ) * 4 ], max );

	}

	// niagara's layout: previousPow2(1920) = 1024, a min (here max) sampler's 2×2 at each texel centre.
	const width = 1920, dw = 1024, data = Float32Array.from( random( width * 4, 13 ) ), sampled = new Float32Array( dw * 4 );
	for ( let x = 0; x < dw; x ++ ) {

		const c = ( x + 0.5 ) * width / dw - 0.5, lo = Math.max( 0, Math.floor( c ) ), hi = Math.min( width - 1, lo + 1 );
		sampled.fill( Math.max( data[ lo * 4 ], data[ hi * 4 ] ), x * 4, x * 4 + 4 );

	}

	assert.ok( maxViolations( data, width, 1, sampled, dw, 1 ) > 100 );

} );

test( 'area averages: a constant stays constant at any ratio, a same-size step is a copy', () => {

	for ( const [ width, height, dw, dh ] of [ [ 1920, 1080, 64, 32 ].map( v => v / 8 ), [ 113, 64, 51, 25 ], [ 31, 17, 64, 32 ] ] ) {

		const level = areaReference( rgba( Array( width * height ).fill( 0.3 ) ), width, height, 1, 'float32', 'average', dw, dh );
		assert.ok( level.values.every( ( v, i ) => i % 4 === 3 ? v === 1 : Math.abs( v - Math.fround( 0.3 ) ) < 1e-6 ) );

	}

	const data = Float32Array.from( random( 7 * 5 * 4, 2 ) );
	assert.deepEqual( areaReference( data, 7, 5, 1, 'float32', 'average', 7, 5 ).values, data );
	assert.throws( () => areaReference( data, 7, 5, 1, 'float32', 'average', 0, 5 ), /size/ );

} );

test( '\'box\' plans the batches of \'exact\', with and without the tail', () => {

	const counts = [];
	for ( const [ width, height ] of [ [ 960, 540 ], [ 1129, 635 ], [ 1477, 831 ], [ 1920, 1080 ], [ 31, 31 ], [ 1, 300 ], [ 2048, 2048 ] ] ) for ( const tail of [ false, true ] ) {

		const options = { mipLevelCount: mipLevelCountFor( width, height ), levelsPerDispatch: 6, tail };
		const box = planBatches( width, height, { ...options, oddSizes: 'box' } );
		assert.deepEqual( box, planBatches( width, height, options ) );
		counts.push( `${ width }x${ height }${ tail ? ' tail' : '' }: ${ box.length }` );

	}

	assert.deepEqual( counts, [ '960x540: 7', '960x540 tail: 4', '1129x635: 6', '1129x635 tail: 5', '1477x831: 9', '1477x831 tail: 6', '1920x1080: 7', '1920x1080 tail: 4', '31x31: 4', '31x31 tail: 1', '1x300: 4', '1x300 tail: 3', '2048x2048: 2', '2048x2048 tail: 2' ] );
	// A power-of-two destination from a separate source: two dispatches for 12 levels (the source batch, then one).
	const hiZ = planBatches( 2048, 1024, { mipLevelCount: 12, levelsPerDispatch: 6, fromSource: true } );
	assert.deepEqual( hiZ.map( batch => [ batch.sourceMipLevel, batch.firstMipLevel, batch.count ] ), [ [ null, 0, 6 ], [ 5, 6, 6 ] ] );
	// Under 'spd' the resampled source is its own 1-level 'exact' batch (like a copy).
	assert.deepEqual( planBatches( 1024, 512, { mipLevelCount: 11, levelsPerDispatch: 6, fromSource: true, copySource: true, oddSizes: 'spd' } ).map( batch => batch.kernel ), [ 'exact', 'clamp', 'clamp' ] );

} );

test( 'box kernels: area weights, loads only', () => {

	const box = createKernel( { format: 'rgba8unorm', count: 4, box: true, access: 'load' } );
	assert.match( box, /ds_area_weight/ );
	assert.match( createKernel( { format: 'rgba16float', count: 6, box: true, tail: true, tile: 32 } ), /tileAt\(vec2u\(x, y\)\)/ );
	assert.match( createKernel( { format: 'r32float', count: 1, box: true, reduction: 'max' } ), /ds_area_last/ );
	assert.doesNotMatch( createKernel( { format: 'r32float', count: 1, box: true, reduction: 'max' } ), /ds_area_weight/ );
	assert.throws( () => createKernel( { format: 'rgba8unorm', count: 4, box: true, access: 'sample' } ), /textureLoad/ );
	// A resampled source ('area' alone): the first output by area, a tail's steps by the 'exact' rule.
	const area = createKernel( { format: 'rgba16float', count: 6, area: true, tail: true, tile: 32 } );
	assert.match( area, /ds_area_weight/ );
	assert.doesNotMatch( area, /tileAt\(vec2u\(x, y\)\)/ );
	assert.match( area, /let numerator = \(2u \* pos/ );
	assert.throws( () => createKernel( { format: 'rgba8unorm', count: 4, box: true, area: false } ), /area/ );
	assert.doesNotMatch( createKernel( { format: 'rgba8unorm', count: 4 } ), /ds_area/ );

} );

test( 'partial updates of a resampled source: areaDependents is every texel whose span meets the dirty range', async () => {

	const { areaDependents, exactDependents, footprint } = await import( '../../dist/index.js' );
	for ( const [ source, size ] of [ [ 1920, 1024 ], [ 1080, 512 ], [ 1129, 512 ], [ 1920, 2048 ], [ 635, 1024 ], [ 31, 16 ], [ 17, 8 ], [ 16, 8 ], [ 7, 7 ], [ 3, 8 ], [ 300, 256 ] ] ) {

		const ranges = [ [ 0, 0 ], [ source - 1, source - 1 ], [ 0, source - 1 ], [ 1, 2 ], [ source >> 1, ( source >> 1 ) + 3 ].map( v => Math.min( v, source - 1 ) ), [ 5, 9 ].map( v => Math.min( v, source - 1 ) ) ];
		for ( const [ a, b ] of ranges ) {

			let lo = null, hi = null;
			for ( let x = 0; x < size; x ++ ) {

				const [ first, last ] = areaSpan( x, source, size );
				if ( first <= b && last >= a ) { lo ??= x; hi = x; }

			}

			assert.deepEqual( areaDependents( [ a, b ], source, size ), lo === null ? null : [ lo, hi ], `${ source } → ${ size } [${ a }, ${ b }]` );
			// exactDependents takes any ratio through it, and agrees at 2:1 and for copies.
			assert.deepEqual( exactDependents( [ a, b ], source, size ), areaDependents( [ a, b ], source, size ) );

		}

	}

	// At 2:1 the area span is the 'exact' footprint.
	for ( const source of [ 2, 3, 9, 16, 31 ] ) for ( let x = 0; x < source >> 1; x ++ ) assert.deepEqual( areaSpan( x, source, source >> 1 ), footprint( x, source ) );

} );
