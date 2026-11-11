import { test } from 'node:test';
import assert from 'node:assert/strict';
import { levelSize, payloadGroups, reducePyramid, resolveReduction, scalarReference, tileDispatch } from '../../dist/index.js';
import { createKernel, kernelLayoutEntries } from '../../dist/webgpu/kernel.js';

globalThis.GPUShaderStage ??= { COMPUTE: 4 };

// One value per texel, splatted to RGBA (alpha 1).
const row = values => Float64Array.from( values.flatMap( value => [ value, value, value, 1 ] ) );
const reds = level => Array.from( { length: level.values.length / 4 }, ( _, i ) => level.values[ i * 4 ] );

test( 'level sizes: ceil for zero-weight, floor (at least 1) for clamp', () => {

	assert.deepEqual( [ 1, 2, 3, 4 ].map( l => levelSize( 1917, l, 'zero-weight' ) ), [ 959, 480, 240, 120 ] );
	assert.deepEqual( [ 1, 2, 3, 4 ].map( l => levelSize( 1917, l, 'clamp' ) ), [ 958, 479, 239, 119 ] );
	assert.deepEqual( [ 1, 2, 3 ].map( l => levelSize( 1, l, 'zero-weight' ) ), [ 1, 1, 1 ] );
	assert.deepEqual( [ 1, 2, 3 ].map( l => levelSize( 3, l, 'clamp' ) ), [ 1, 1, 1 ] );
	// Six levels: 32×32 level-1 tiles (64×64 source tiles, as SPD).
	const dispatch = tileDispatch( 1920, 1080, 6, 'clamp' );
	assert.deepEqual( [ dispatch.tile, dispatch.side, dispatch.groupsX, dispatch.groupsY ], [ 32, 16, 30, 17 ] );
	assert.deepEqual( [ tileDispatch( 17, 9, 3, 'zero-weight' ).groupsX, tileDispatch( 17, 9, 3, 'zero-weight' ).width ], [ 2, 9 ] );

} );

test( 'zero-weight averages divide by the in-bounds count', () => {

	const [ one, two ] = reducePyramid( row( [ 1, 2, 3 ] ), 3, 1, 1, { reduction: 'average', levels: [ 1, 2 ], edges: 'zero-weight' } );
	assert.deepEqual( reds( one ), [ 1.5, 3 ] );
	assert.deepEqual( reds( two ), [ 2 ] );

} );

test( 'clamp drops the last odd column and repeats the edge past the image', () => {

	const [ one, two ] = reducePyramid( row( [ 1, 2, 3 ] ), 3, 1, 1, { reduction: 'average', levels: [ 1, 2 ], edges: 'clamp' } );
	assert.deepEqual( reds( one ), [ 1.5 ] );
	// Level 2 is 1 texel wide; its block 0…3 reads 1, 2, 3, 3 (and each row twice).
	assert.deepEqual( reds( two ), [ 2.25 ] );

} );

test( 'weighted means: Σw·v / Σw, the weight channel emits Σw, and Σw = 0 emits 0', () => {

	// 4×1 texels (v, v, v, w); level 1 blocks: [ (1, w1), (5, w0) ], [ (9, w0), (2, w0) ].
	const source = Float64Array.of( 1, 1, 1, 1, 5, 5, 5, 0, 9, 9, 9, 0, 2, 2, 2, 0 );
	const [ one, two ] = reducePyramid( source, 4, 1, 1, { reduction: [ 'weighted', 'average', 'sum', 'weight' ], levels: [ 1, 2 ], edges: 'zero-weight' } );
	assert.deepEqual( Array.from( one.values ), [ 1, 3, 6, 1, 0, 5.5, 11, 0 ] );
	assert.deepEqual( Array.from( two.values ), [ 1, 4.25, 17, 1 ] );

} );

test( 'zero-weight sums count every texel exactly once; clamp sums count duplicates', () => {

	const values = Array.from( { length: 7 * 5 }, ( _, i ) => ( i * 37 ) % 11 );
	const source = row( values );
	const total = values.reduce( ( a, b ) => a + b, 0 );
	for ( const level of reducePyramid( source, 7, 5, 1, { reduction: 'sum', levels: [ 1, 2, 3, 4 ], edges: 'zero-weight' } ) ) {

		assert.equal( reds( level ).reduce( ( a, b ) => a + b, 0 ), total, `level ${ level.level }` );

	}

	// 1×1 under clamp: level l reads the one texel 4^l times.
	const [ single ] = reducePyramid( row( [ 3 ] ), 1, 1, 1, { reduction: 'sum', levels: [ 2 ], edges: 'clamp' } );
	assert.equal( single.values[ 0 ], 48 );

} );

test( 'min/max over the policy\'s block; custom folds and custom sums', () => {

	const source = row( [ 4, 8, 1, 6, 9 ] );
	const [ zw ] = reducePyramid( source, 5, 1, 1, { reduction: [ 'min', 'max', 'min', 'max' ], levels: [ 1 ], edges: 'zero-weight' } );
	assert.deepEqual( Array.from( zw.values ), [ 4, 8, 4, 1, 1, 6, 1, 1, 9, 9, 9, 1 ] );
	const [ clamp ] = reducePyramid( source, 5, 1, 1, { reduction: 'minmax', levels: [ 1 ], edges: 'clamp' } );
	assert.deepEqual( reds( clamp ), [ 4, 1 ] );
	// A custom sum combines each counted texel once: n − 1 combines of 'a + b + 1'.
	const [ count ] = reducePyramid( source, 5, 1, 1, { reduction: { reference: ( a, b ) => a + b + 1, kind: 'sum' }, levels: [ 2 ], edges: 'zero-weight' } );
	assert.deepEqual( reds( count ), [ 19 + 3, 9 ] );

} );

// An independent check of the oracle against the tree the kernels run: pad the
// image per the edge policy, then halve it level by level with (sum, count) pairs.
test( 'the oracle equals iterated 2×2 reductions of the padded image', () => {

	let state = 3;
	const random = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;

	for ( const [ width, height ] of [ [ 1, 1 ], [ 5, 3 ], [ 17, 9 ], [ 1, 33 ], [ 64, 31 ] ] ) for ( const edges of [ 'zero-weight', 'clamp' ] ) {

		const base = Array.from( { length: width * height }, () => Math.round( random() * 64 ) );
		const levels = [ 1, 2, 3, 4, 5, 6 ];
		const expected = reducePyramid( Float64Array.from( base.flatMap( v => [ v, v, v, v ] ) ), width, height, 1, { reduction: [ 'average', 'sum', 'min', 'max' ], levels, edges } );
		const side = 2 ** 6, pw = levelSize( width, 6, edges ) * side, ph = levelSize( height, 6, edges ) * side;
		// [ sum, count, min, max ] per padded texel; zero-weight padding counts nothing.
		let grid = Array.from( { length: pw * ph }, ( _, i ) => {

			const x = i % pw, y = Math.floor( i / pw );
			if ( edges === 'zero-weight' && ( x >= width || y >= height ) ) return [ 0, 0, Infinity, - Infinity ];
			const v = base[ Math.min( y, height - 1 ) * width + Math.min( x, width - 1 ) ];
			return [ v, 1, v, v ];

		} );
		let gw = pw, gh = ph;

		for ( const level of levels ) {

			const nw = gw / 2, nh = gh / 2;
			grid = Array.from( { length: nw * nh }, ( _, i ) => {

				const x = i % nw, y = Math.floor( i / nw );
				const cells = [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ].map( ( [ dx, dy ] ) => grid[ ( 2 * y + dy ) * gw + 2 * x + dx ] );
				return [ cells.reduce( ( s, c ) => s + c[ 0 ], 0 ), cells.reduce( ( s, c ) => s + c[ 1 ], 0 ), Math.min( ...cells.map( c => c[ 2 ] ) ), Math.max( ...cells.map( c => c[ 3 ] ) ) ];

			} );
			gw = nw; gh = nh;
			const reference = expected[ level - 1 ];

			for ( let y = 0; y < reference.height; y ++ ) for ( let x = 0; x < reference.width; x ++ ) {

				const [ sum, count, min, max ] = grid[ y * gw + x ];
				const actual = Array.from( reference.values.subarray( ( y * reference.width + x ) * 4, ( y * reference.width + x ) * 4 + 4 ) );
				assert.deepEqual( actual, [ sum / count, sum, min, max ], `${ width }x${ height } ${ edges } level ${ level } texel ${ x },${ y }` );

			}

		}

	}

} );

test( 'tile reductions are validated', () => {

	assert.deepEqual( resolveReduction( 'minmax' ).ops, [ 'min', 'max', 'min', 'max' ] );
	assert.throws( () => resolveReduction( [ 'sum', 'sum', 'sum' ] ), /four of/ );
	assert.throws( () => resolveReduction( [ 'weighted', 'sum', 'sum', 'sum' ] ), /need a 'weight'/ );
	assert.throws( () => resolveReduction( [ 'weight', 'weight', 'sum', 'sum' ] ), /at most one/ );
	assert.throws( () => resolveReduction( { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { return a + b; }', kind: 'product' } ), /kind/ );
	assert.throws( () => reducePyramid( row( [ 1 ] ), 1, 1, 1, { reduction: 'sum', levels: [ 2, 1 ], edges: 'clamp' } ), /increasing/ );
	assert.throws( () => reducePyramid( row( [ 1 ] ), 1, 1, 1, { reduction: 'sum', levels: [ 7 ], edges: 'clamp' } ), /1 \(half size\) to 6/ );

} );

test( '8 channels: ops, builtins and custom reductions resolve to 8; an 8-channel pyramid is its two halves', () => {

	assert.equal( resolveReduction( [ 'sum', 'sum', 'sum', 'sum', 'max', 'max', 'max', 'max' ] ).channels, 8 );
	assert.deepEqual( resolveReduction( 'minmax', 8 ).ops, [ 'min', 'max', 'min', 'max', 'min', 'max', 'min', 'max' ] );
	assert.equal( resolveReduction( { wgsl: 'fn reduce(a: DsWide, b: DsWide) -> DsWide { return a; }' }, 8 ).channels, 8 );
	assert.equal( resolveReduction( 'average' ).channels, 4 );
	assert.throws( () => resolveReduction( [ 'sum', 'sum', 'sum', 'sum' ], 8 ), /channels is 8/ );
	assert.throws( () => resolveReduction( 'sum', 6 ), /4 .* or 8/ );
	assert.throws( () => resolveReduction( Array( 6 ).fill( 'sum' ) ), /four of/ );

	const [ width, height, layers ] = [ 37, 21, 2 ];
	let state = 7;
	const wide = Float64Array.from( { length: width * height * layers * 8 }, ( _, i ) => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) % 1000 / ( i % 8 === 5 ? 1000 : 7 ) );
	const half = from => Float64Array.from( { length: width * height * layers * 4 }, ( _, i ) => wide[ Math.floor( i / 4 ) * 8 + from + i % 4 ] );
	const lo = [ 'sum', 'average', 'min', 'max' ], hi = [ 'weighted', 'weight', 'weighted', 'max' ];
	for ( const edges of [ 'zero-weight', 'clamp' ] ) {

		const levels = [ 1, 2, 5 ];
		const both = reducePyramid( wide, width, height, layers, { reduction: [ ...lo, ...hi ], levels, edges } );
		const parts = [ reducePyramid( half( 0 ), width, height, layers, { reduction: lo, levels, edges } ), reducePyramid( half( 4 ), width, height, layers, { reduction: hi, levels, edges } ) ];
		both.forEach( ( level, l ) => {

			for ( let i = 0; i < level.values.length; i ++ ) {

				const part = parts[ ( i % 8 ) >> 2 ][ l ], j = Math.floor( i / 8 ) * 4 + i % 4;
				assert.equal( level.values[ i ], part.values[ j ], `${ edges } level ${ level.level } value ${ i }` );
				assert.equal( level.magnitude[ i ], part.magnitude[ j ] );

			}

		} );

	}

} );

test( 'kernels bind only emitted levels; exact kernels reject tile options', () => {

	const options = { edges: 'zero-weight', levels: [ 2, 3 ], format: 'rgba32float', reduction: [ 'sum', 'average', 'min', 'max' ] };
	const wgsl = createKernel( options );
	assert.match( wgsl, /@binding\(2\) var ds_out2:/ );
	assert.match( wgsl, /@binding\(3\) var ds_out3:/ );
	assert.doesNotMatch( wgsl, /ds_out1/ );
	assert.deepEqual( kernelLayoutEntries( options ).map( entry => entry.binding ), [ 1, 2, 3 ] );
	// A load hook replaces the source binding with the size uniform; a size hook removes that too.
	const load = 'fn load(texel: vec2i, layer: u32) -> vec4f { return vec4f(1.0); }';
	const store = 'fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {}';
	assert.deepEqual( kernelLayoutEntries( { edges: 'clamp', levels: [ 1 ], load, store } ).map( entry => entry.binding ), [ 8 ] );
	assert.deepEqual( kernelLayoutEntries( { edges: 'clamp', levels: [ 1 ], load, store, size: 'fn size() -> vec2u { return vec2u(4u); }' } ), [] );
	assert.throws( () => createKernel( { edges: 'clamp', levels: [ 1 ], load: 'fn loader() {}', store } ), /load hook/ );
	assert.throws( () => createKernel( { edges: 'clamp', levels: [ 1 ], load } ), /format/ );
	assert.throws( () => createKernel( { format: 'rgba8unorm', count: 2, reduction: [ 'sum', 'sum', 'sum', 'sum' ] } ), /per-channel/ );
	assert.throws( () => createKernel( { format: 'rgba8unorm', count: 2, load } ), /load needs edges/ );

} );

test( 'payload folds: validation, and every level picks the lexicographic winner of its block', () => {

	assert.deepEqual( payloadGroups( [ 'argmin', 'payload', 'argmax', 'payload' ] ), [ { key: 0, kind: 'argmin', payload: [ 1 ] }, { key: 2, kind: 'argmax', payload: [ 3 ] } ] );
	assert.throws( () => resolveReduction( [ 'payload', 'argmin', 'sum', 'sum' ] ), /must follow/ );
	assert.throws( () => resolveReduction( [ 'argmin', 'sum', 'payload', 'sum' ] ), /must follow/ );

	// Keys and the first payload take a few values: ties everywhere. The winner, found by
	// sorting the block's texels (independent of the oracle's scan), must be what it emits.
	const [ width, height ] = [ 23, 13 ];
	let state = 3;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) >>> 8;
	const data = Float64Array.from( { length: width * height * 4 }, ( _, i ) => i % 4 === 0 ? next() % 3 : i % 4 === 1 ? next() % 2 : next() % 1000 );
	for ( const [ kind, sign ] of [ [ 'argmin', 1 ], [ 'argmax', - 1 ] ] ) for ( const edges of [ 'zero-weight', 'clamp' ] ) {

		const result = reducePyramid( data, width, height, 1, { reduction: [ kind, 'payload', 'payload', 'payload' ], levels: [ 1, 2, 3 ], edges } );
		result.forEach( ( { level, width: w, height: h, values } ) => {

			for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

				const texels = [];
				for ( let sy = y << level; sy < ( y + 1 ) << level; sy ++ ) for ( let sx = x << level; sx < ( x + 1 ) << level; sx ++ ) {

					if ( edges === 'zero-weight' && ( sx >= width || sy >= height ) ) continue;
					const i = ( Math.min( sy, height - 1 ) * width + Math.min( sx, width - 1 ) ) * 4;
					texels.push( Array.from( data.subarray( i, i + 4 ) ) );

				}

				texels.sort( ( a, b ) => sign * ( a[ 0 ] - b[ 0 ] ) || a[ 1 ] - b[ 1 ] || a[ 2 ] - b[ 2 ] || a[ 3 ] - b[ 3 ] );
				assert.deepEqual( Array.from( values.subarray( ( y * w + x ) * 4, ( y * w + x ) * 4 + 4 ) ), texels[ 0 ], `${ kind } ${ edges } level ${ level } (${ x }, ${ y })` );

			}

		} );

	}

	// The scalar oracle (its own tree order) agrees with a sort of the whole image.
	const [ scalar ] = scalarReference( Float32Array.from( data ), width, height, 1, [ 'argmax', 'payload', 'payload', 'payload' ] );
	const all = Array.from( { length: width * height }, ( _, i ) => Array.from( data.subarray( i * 4, i * 4 + 4 ) ) ).sort( ( a, b ) => b[ 0 ] - a[ 0 ] || a[ 1 ] - b[ 1 ] || a[ 2 ] - b[ 2 ] || a[ 3 ] - b[ 3 ] );
	assert.deepEqual( scalar.value, all[ 0 ] );
	assert.deepEqual( scalar.exact, all[ 0 ] );

} );
