import { test } from 'node:test';
import assert from 'node:assert/strict';
import { maxViolations, mipLevelCountFor, planBatches, reducePyramid, reduceReference, spdChainReference, spdReference } from '../../dist/index.js';
import { createKernel, kernelLayoutEntries } from '../../dist/webgpu/kernel.js';

globalThis.GPUShaderStage ??= { COMPUTE: 4 };

const plan = ( width, height, levelsPerDispatch, extra = {} ) => planBatches( width, height, { mipLevelCount: mipLevelCountFor( width, height ), levelsPerDispatch, ...extra } );
const shape = batches => batches.map( batch => `${ batch.kernel }:${ batch.count }` );
const renderSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 3840, 2160 ] ];
const rgba = values => Float32Array.from( values.flatMap( value => [ value, value, value, 1 ] ) );

test( 'spd chains take ⌈levels / levelsPerDispatch⌉ clamp batches at any size', () => {

	for ( const [ width, height ] of [ ...renderSizes, [ 4096, 4096 ], [ 1, 128 ], [ 3, 7 ] ] ) for ( const levels of [ 1, 4, 5, 6 ] ) {

		const batches = plan( width, height, levels, { oddSizes: 'spd' } );
		const outputs = mipLevelCountFor( width, height ) - 1;
		assert.equal( batches.length, Math.ceil( outputs / levels ), `${ width }x${ height } at ${ levels }` );
		assert.ok( batches.every( batch => batch.kernel === 'clamp' ) );
		// Each batch reads the level before its first output.
		assert.ok( batches.every( batch => batch.sourceMipLevel === batch.firstMipLevel - 1 ) );

	}

	assert.deepEqual( shape( plan( 1920, 1080, 6, { oddSizes: 'spd' } ) ), [ 'clamp:6', 'clamp:4' ] );
	const [ first ] = plan( 1920, 1080, 6, { oddSizes: 'spd' } );
	assert.deepEqual( [ first.tile, first.groupsX, first.groupsY ], [ 32, 30, 17 ] );

} );

test( 'spd with a separate source: a same-size source is copied by its own exact batch', () => {

	const copy = planBatches( 64, 64, { mipLevelCount: 7, levelsPerDispatch: 6, oddSizes: 'spd', fromSource: true, copySource: true } );
	assert.deepEqual( copy.map( batch => [ batch.kernel, batch.sourceMipLevel, batch.firstMipLevel, batch.count ] ), [ [ 'exact', null, 0, 1 ], [ 'clamp', 0, 1, 6 ] ] );
	const half = planBatches( 64, 64, { mipLevelCount: 7, levelsPerDispatch: 6, oddSizes: 'spd', fromSource: true } );
	assert.deepEqual( half.map( batch => [ batch.kernel, batch.sourceMipLevel, batch.firstMipLevel, batch.count ] ), [ [ 'clamp', null, 0, 6 ], [ 'clamp', 5, 6, 1 ] ] );
	assert.throws( () => plan( 64, 64, 6, { oddSizes: 'spd', tail: true } ), /tail/ );

} );

test( 'the exact tail finishes the chain once a level fits 32×32', () => {

	assert.deepEqual( shape( plan( 1920, 1080, 6 ) ), [ 'exact:3', 'exact:1', 'exact:1', 'exact:2', 'exact:1', 'exact:1', 'exact:1' ] );
	assert.deepEqual( shape( plan( 1920, 1080, 6, { tail: true } ) ), [ 'exact:3', 'exact:1', 'exact:1', 'tail:5' ] );
	assert.deepEqual( shape( plan( 1707, 960, 6, { tail: true } ) ), [ 'exact:1', 'exact:2', 'exact:2', 'tail:5' ] );
	// At four levels per dispatch the tail is capped and continues.
	assert.deepEqual( shape( plan( 1920, 1080, 4, { tail: true } ) ), [ 'exact:3', 'exact:1', 'exact:1', 'tail:4', 'exact:1' ] );
	// Power-of-two plans are unchanged: their batches already reach the limit.
	assert.deepEqual( shape( plan( 4096, 4096, 6, { tail: true } ) ), shape( plan( 4096, 4096, 6 ) ) );
	const tail = plan( 1920, 1080, 6, { tail: true } ).pop();
	assert.deepEqual( [ tail.sourceMipLevel, tail.firstMipLevel, tail.outputWidth, tail.outputHeight, tail.tile, tail.groupsX, tail.groupsY ], [ 5, 6, 30, 16, 32, 1, 1 ] );
	// Every render size: 'exact' 5–9 dispatches, with the tail 3–6 (1477×831: 46×25 doesn't fit).
	for ( const [ width, height ] of renderSizes ) {

		const exact = plan( width, height, 6 ).length, tail = plan( width, height, 6, { tail: true } ).length;
		assert.ok( exact >= 5 && exact <= 9 && tail >= 3 && tail <= 6 && tail < exact, `${ width }x${ height }: ${ exact } → ${ tail }` );

	}

} );

test( 'tail kernels: one 32×32 workgroup, same layout as the exact kernel', () => {

	const exact = createKernel( { format: 'rgba8unorm', count: 5, reduction: 'average' } );
	const tail = createKernel( { format: 'rgba8unorm', count: 5, reduction: 'average', tail: true } );
	assert.doesNotMatch( exact, /tailStep/ );
	assert.match( tail, /fn tailStep/ );
	assert.match( tail, /array<vec4f, 1024>/ );
	assert.match( createKernel( { format: 'r32float', count: 3, reduction: 'max', tail: true } ), /odd\.x && odd\.y/ );
	assert.deepEqual( kernelLayoutEntries( { format: 'rgba8unorm', count: 5, tail: true } ), kernelLayoutEntries( { format: 'rgba8unorm', count: 5 } ) );
	assert.throws( () => createKernel( { format: 'rgba8unorm', edges: 'clamp', count: 2, tail: true } ), /tail/ );

} );

test( 'clamp tile kernels can sample averages at the 2×2 corner', () => {

	const options = { format: 'rgba16float', edges: 'clamp', count: 6, access: 'sample' };
	assert.match( createKernel( options ), /textureSampleLevel\(ds_source, ds_sampler/ );
	assert.deepEqual( kernelLayoutEntries( options ).slice( 0, 2 ).map( entry => [ entry.binding, entry.sampler?.type ?? entry.texture.sampleType ] ), [ [ 0, 'filtering' ], [ 1, 'float' ] ] );
	assert.throws( () => createKernel( { ...options, reduction: 'max' } ), /sample/ );
	assert.throws( () => createKernel( { ...options, edges: 'zero-weight' } ), /sample/ );
	assert.throws( () => createKernel( { ...options, load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return vec4f(1.0); }' } ), /sample/ );

} );

test( 'spd reference: clamp blocks, ⌊n/2⌋ sizes, quantized once per level', () => {

	// 3 → 1: the last column is dropped (render passes sample the centre texel).
	const [ one ] = spdReference( rgba( [ 0, 0.5, 1 ] ), 3, 1, 1, 'float32', 'average', 1 );
	assert.equal( one.values[ 0 ], 0.25 );
	assert.equal( reduceReference( rgba( [ 0, 0.5, 1 ] ), 3, 1, 1, 'float32' ).values[ 0 ], 0.5 );
	// Within a batch nothing is rounded between levels: 8-bit level 2 of 4×1 is
	// the mean of its four texels, not of two rounded level-1 values.
	const units = [ 0, 1, 0, 2 ].map( v => v / 255 );
	const [ , two ] = spdReference( rgba( units ), 4, 1, 1, 'unorm8', 'average', 2 );
	assert.equal( Math.round( two.values[ 0 ] * 255 ), 1 ); // 3/4 → 1; rounded levels (0, 1) would give 0 (tie to even)
	// A single batch equals the tile oracle from the source.
	let state = 5;
	const random = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
	const source = Float32Array.from( { length: 37 * 21 * 4 }, () => Math.fround( random() * 8 - 2 ) );
	const chain = spdChainReference( source, 37, 21, 1, 'float32', 'max', 5, 6 );
	const direct = reducePyramid( source, 37, 21, 1, { reduction: 'max', levels: [ 1, 2, 3, 4, 5 ], edges: 'clamp' } );
	chain.forEach( ( level, i ) => assert.deepEqual( Array.from( level.values ), Array.from( direct[ i ].values, Math.fround ) ) );
	// Two batches: the second reads the first's last level.
	const twoBatches = spdChainReference( source, 37, 21, 1, 'float32', 'average', 5, 3 );
	const second = spdReference( twoBatches[ 2 ].values, 4, 2, 1, 'float32', 'average', 2 );
	assert.deepEqual( Array.from( twoBatches[ 4 ].values ), Array.from( second[ 1 ].values ) );

} );

test( 'spd max is not a conservative Hi-Z; exact max is', () => {

	// A bright last column of an odd width is in no 'spd' block.
	const width = 5, height = 3;
	const base = rgba( Array.from( { length: width * height }, ( _, i ) => i % width === width - 1 ? 1 : 0 ) );
	const spd = spdChainReference( base, width, height, 1, 'float32', 'max', 2, 6 );
	assert.equal( spd[ 0 ].width, 2 );
	assert.ok( spd[ 0 ].values.every( v => v === 0 || v === 1 ) && spd[ 0 ].values[ 4 ] === 0 );
	assert.equal( maxViolations( base, width, height, spd[ 0 ].values, 2, 1 ), 1 );
	const exact = reduceReference( base, width, height, 1, 'float32', 'max' );
	assert.equal( maxViolations( base, width, height, exact.values, exact.width, exact.height ), 0 );

} );
