import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createMipmapKernel, kernelDefaults, mipmapAccess, mipmapFormats, planMipmapBatches } from '../../dist/mipmap.js';
import { createKernel, float16Wgsl } from '../../dist/webgpu/kernel.js';
import { formats, mipLevelCountFor, planBatches } from '../../dist/index.js';
import { readFileSync } from 'node:fs';

// downsampler/mipmap is a pruned copy of the full generator and planner: equal on its whole option space.

test( 'createMipmapKernel equals createKernel( { ...options, f16Pack: false } ) on every option set', () => {

	let compared = 0;
	for ( const format of Object.keys( mipmapFormats ) ) for ( const dimension of [ undefined, '2d', '2d-array' ] ) for ( const access of [ undefined, 'sample', 'load' ] )
	for ( const f16Rounding of [ undefined, 'nearest', 'device' ] ) for ( const tail of [ undefined, false, true ] ) for ( let count = 1; count <= 6; count ++ ) for ( const tile of [ undefined, 8, 16, 32 ] ) {

		const options = { format, count, ...( dimension && { dimension } ), ...( access && { access } ), ...( f16Rounding && { f16Rounding } ), ...( tail !== undefined && { tail } ), ...( tile && { tile } ) };
		let expected;
		try {

			expected = createKernel( { ...options, f16Pack: false } );

		} catch ( error ) {

			assert.throws( () => createMipmapKernel( options ), error.constructor, JSON.stringify( options ) );
			continue;

		}

		assert.equal( createMipmapKernel( options ), expected, JSON.stringify( options ) );
		compared ++;

	}

	assert.ok( compared > 10000, `${ compared } compared` );

} );

test( 'createMipmapKernel rejects what it does not cover', () => {

	for ( const format of [ 'rgba8unorm-srgb', 'rgb10a2unorm', 'rg11b10ufloat', 'rgba8uint', 'depth32float', 'toString' ] ) assert.throws( () => createMipmapKernel( { format, count: 1 } ), /not a mipmap format/ );
	assert.throws( () => createMipmapKernel( { format: 'rgba8unorm', count: 0 } ), /count/ );
	assert.throws( () => createMipmapKernel( { format: 'rgba8unorm', count: 7 } ), /count/ );
	assert.throws( () => createMipmapKernel( { format: 'rgba8unorm', count: 6, tile: 16 } ), /tile of at least 32/ );
	assert.throws( () => createMipmapKernel( { format: 'rgba8unorm', count: 1, access: 'gather' } ), /access/ );
	assert.throws( () => createMipmapKernel( { format: 'rgba16float', count: 1, f16Rounding: 'up' } ), /f16Rounding/ );
	assert.throws( () => createMipmapKernel( { format: 'rgba8unorm', count: 1, dimension: 'cube' } ), /dimension/ );

} );

test( 'float16Wgsl( rounding, false ) is ds_round_f16 alone; the default keeps the pack helpers', () => {

	for ( const rounding of [ 'nearest', 'device' ] ) {

		const round = float16Wgsl( rounding, false );
		assert.match( round, /^fn ds_round_f16/ );
		assert.doesNotMatch( round, /ds_pack_f16|ds_half_bits/ );
		assert.ok( float16Wgsl( rounding ).endsWith( round ) );
		assert.match( float16Wgsl( rounding ), /fn ds_pack_f16/ );

	}

	// The buffer path calls ds_pack_f16, so it keeps the helpers whatever f16Pack says.
	assert.match( createKernel( { format: 'rgba16float', count: 2, output: 'buffer', f16Pack: false } ), /fn ds_pack_f16/ );
	assert.equal( createKernel( { format: 'rgba16float', count: 2 } ), createKernel( { format: 'rgba16float', count: 2, f16Pack: true } ) );
	assert.doesNotMatch( createKernel( { format: 'rgba16float', count: 2, f16Pack: false } ), /ds_half_bits/ );

} );

test( 'planMipmapBatches equals planBatches for in-place exact chains', () => {

	const sizes = [ [ 1, 1 ], [ 2, 1 ], [ 1, 300 ], [ 3, 7 ], [ 31, 31 ], [ 33, 17 ], [ 64, 64 ], [ 128, 8 ], [ 255, 257 ], [ 640, 480 ], [ 960, 540 ], [ 1129, 635 ], [ 1920, 1080 ], [ 2048, 2048 ], [ 96, 4096 ], [ 4097, 3 ] ];
	for ( const [ width, height ] of sizes ) for ( const levelsPerDispatch of [ 1, 2, 4, 5, 6 ] ) for ( const tail of [ false, true ] ) for ( const tailMaxTile of [ 8, 16, 32 ] ) for ( const exactMinTile of [ 8, 16, 32 ] ) {

		const full = mipLevelCountFor( width, height );
		for ( const mipLevelCount of [ full, Math.max( 1, full - 2 ), 1 ] ) {

			const options = { mipLevelCount, levelsPerDispatch, tail, tailMaxTile, exactMinTile };
			assert.deepEqual( planMipmapBatches( width, height, options ), planBatches( width, height, options ), JSON.stringify( [ width, height, options ] ) );

		}

		assert.deepEqual( planMipmapBatches( width, height, { levelsPerDispatch, tail, tailMaxTile, exactTile: exactMinTile } ), planBatches( width, height, { mipLevelCount: full, levelsPerDispatch, tail, tailMaxTile, exactMinTile } ) );

	}

	// Defaults: the full chain, the tail on, 32 and 8 tiles.
	assert.deepEqual( planMipmapBatches( 1920, 1080, { levelsPerDispatch: 6 } ), planBatches( 1920, 1080, { mipLevelCount: 11, levelsPerDispatch: 6, tail: true } ) );
	assert.throws( () => planMipmapBatches( 64, 64, { levelsPerDispatch: 7 } ), /levelsPerDispatch/ );
	assert.throws( () => planMipmapBatches( 64, 64, { levelsPerDispatch: 4, exactMinTile: 12 } ), /exactMinTile/ );
	assert.throws( () => planMipmapBatches( 64, 64, { levelsPerDispatch: 4, exactMinTile: 8, exactTile: 32 } ), /same option/ );

} );

test( 'kernelDefaults: the rules Downsampler resolves its options from', () => {

	const limits = ( invocations, storage ) => ( { maxComputeInvocationsPerWorkgroup: invocations, maxStorageTexturesPerShaderStage: storage } );
	assert.deepEqual( kernelDefaults( limits( 256, 8 ), { vendor: 'nvidia' } ), { levelsPerDispatch: 6, bufferLevelsPerDispatch: 6, exactTile: 32, tailMaxTile: 32, tail: true } );
	assert.deepEqual( kernelDefaults( limits( 1024, 4 ), { vendor: 'apple' } ), { levelsPerDispatch: 4, bufferLevelsPerDispatch: 6, exactTile: 16, tailMaxTile: 32, tail: true } );
	assert.deepEqual( kernelDefaults( limits( 256, 6 ), { vendor: 'google', isFallbackAdapter: true } ), { levelsPerDispatch: 6, bufferLevelsPerDispatch: 6, exactTile: 8, tailMaxTile: 32, tail: true } );
	// Compatibility mode's default: 128 invocations.
	assert.deepEqual( kernelDefaults( limits( 128, 4 ), { vendor: 'apple' } ), { levelsPerDispatch: 4, bufferLevelsPerDispatch: 4, exactTile: 8, tailMaxTile: 8, tail: true } );
	assert.deepEqual( kernelDefaults( limits( 256, 4 ) ), { levelsPerDispatch: 4, bufferLevelsPerDispatch: 6, exactTile: 32, tailMaxTile: 32, tail: true } );
	assert.deepEqual( kernelDefaults( limits( 256, 4 ), null ), kernelDefaults( limits( 256, 4 ) ) );

	// Downsampler takes them from kernelDefaults, not from rules of its own.
	const source = readFileSync( new URL( '../../src/webgpu/downsampler.ts', import.meta.url ), 'utf8' );
	assert.match( source, /kernelDefaults\( device\.limits/ );
	assert.doesNotMatch( source, /vendor === 'apple'|isFallbackAdapter/ );

} );

test( 'mipmapFormats is a subset of formats, with the same quantization, filterability and feature', () => {

	for ( const [ format, info ] of Object.entries( mipmapFormats ) ) {

		const full = formats[ format ];
		assert.ok( full, format );
		assert.equal( info.quantization, full.quantization, format );
		assert.equal( info.filterable, full.filterable, format );
		assert.equal( info.feature, full.feature, format );
		assert.notEqual( full.storage, false, format );

	}

	// Every storage-capable float-view format of those quantizations is included.
	const families = new Set( Object.values( mipmapFormats ).map( info => info.quantization ) );
	for ( const [ format, info ] of Object.entries( formats ) ) if ( families.has( info.quantization ) && info.storage !== false ) assert.ok( format in mipmapFormats, format );

	const features = ( ...names ) => new Set( names );
	assert.equal( mipmapAccess( 'rgba8unorm' ), 'sample' );
	assert.equal( mipmapAccess( 'rgba16unorm' ), 'load' );
	assert.equal( mipmapAccess( 'rgba32float' ), 'load' );
	assert.equal( mipmapAccess( 'rgba32float', features( 'float32-filterable' ) ), 'sample' );
	assert.equal( mipmapAccess( 'r16snorm', features( 'float32-filterable' ) ), 'load' );

} );

test( 'dist/mipmap.js imports nothing', () => {

	const code = readFileSync( new URL( '../../dist/mipmap.js', import.meta.url ), 'utf8' );
	assert.doesNotMatch( code, /^\s*import\b|\bimport\s*\(|\brequire\s*\(/m );

} );
