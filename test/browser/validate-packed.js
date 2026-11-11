// Packed float formats and the buffer output path (SPEC §6.1), beyond the main
// matrix: the WGSL codecs against the core ones bit for bit (every code point,
// every rounding boundary), buffer against storage output, the default storage
// texture limit, and API behaviour.
import { fromUfloat, packRgb9e5, quantizeTexel, reducePyramid, reduceReference, toUfloat, unpackRgb9e5 } from '/dist/index.js';
import { Downsampler, packedFloatWgsl, Reducer } from '/dist/webgpu/index.js';
import { createTexture, makeData, poison, readMips, toFloats, upload } from './gpu.js';
import { check, checkPyramid, reducerData } from './validate.js';

const f32Bits = value => new Uint32Array( new Float32Array( [ value ] ).buffer )[ 0 ];
const bitsF32 = bits => new Float32Array( new Uint32Array( [ bits ] ).buffer )[ 0 ];
const nextFloat = ( value, direction ) => bitsF32( f32Bits( value ) + direction );

/** Runs `fn ds_test(i: u32)` over `count` invocations: `input` and `output` are u32 arrays. */
async function runCodec( device, helpers, body, input, outputWords, count ) {

	const code = `${ helpers }
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	let i = id.y * 65536u + id.x;
	if (i >= ${ count }u) { return; }
${ body }
}`;
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module: device.createShaderModule( { label: 'packed codec test', code } ), entryPoint: 'main' } } );
	const inBuffer = device.createBuffer( { size: Math.max( 16, input.byteLength ), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( inBuffer, 0, input );
	const outBuffer = device.createBuffer( { size: outputWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
	const readback = device.createBuffer( { size: outputWords * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline( pipeline );
	pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: { buffer: inBuffer } }, { binding: 1, resource: { buffer: outBuffer } } ] } ) );
	const groups = Math.ceil( count / 64 );
	pass.dispatchWorkgroups( Math.min( groups, 1024 ), Math.ceil( groups / 1024 ) );
	pass.end();
	encoder.copyBufferToBuffer( outBuffer, 0, readback, 0, outputWords * 4 );
	device.queue.submit( [ encoder.finish() ] );
	await readback.mapAsync( GPUMapMode.READ );
	const result = new Uint32Array( readback.getMappedRange().slice( 0 ) );
	readback.unmap();
	for ( const buffer of [ inBuffer, outBuffer, readback ] ) buffer.destroy();
	return result;

}

/** Float32 inputs at every rounding boundary of a ufloat, plus specials and random values. */
function ufloatInputs( bits ) {

	const inf = 0x1f << bits;
	const values = [ 0, - 0, - 1, - Infinity, Infinity, NaN, 2 ** - 140, 1e30, 3.4e38, - ( 2 ** - 30 ) ];
	for ( let code = 0; code < inf; code ++ ) {

		const value = fromUfloat( code, bits );
		values.push( value );
		if ( code + 1 < inf ) {

			const mid = ( value + fromUfloat( code + 1, bits ) ) / 2;
			values.push( mid, nextFloat( mid, - 1 ), nextFloat( mid, 1 ) );

		}

	}

	let state = bits;
	for ( let i = 0; i < 20000; i ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		values.push( bitsF32( ( state & 0x807fffff ) | ( ( 90 + state % 80 ) << 23 ) ) );

	}

	return values;

}

/** RGB texels at the rgb9e5 boundaries: bumps, half-up ties, powers of two, clamps, NaN. */
function rgb9e5Inputs() {

	const texels = [ [ 0, 0, 0 ], [ - 1, NaN, 2 ], [ 1e9, Infinity, - Infinity ], [ 65408, 65407, 0 ], [ 2 ** - 25, 0, 0 ], [ nextFloat( 2 ** - 25, - 1 ), 0, 0 ] ];
	for ( let k = - 26; k <= 16; k ++ ) {

		const p = 2 ** k;
		texels.push( [ p, p / 3, p / 7 ], [ nextFloat( p, - 1 ), 0, p / 2 ], [ nextFloat( p, 1 ), p / 512, 0 ] );
		// The bump boundary: 511.5 · 2^(k − 8) rounds to 512 at the first exponent.
		const bump = 511.5 * 2 ** ( k - 8 );
		texels.push( [ bump, 0, 0 ], [ 0, nextFloat( bump, - 1 ), 0 ], [ 0, 0, nextFloat( bump, 1 ) ] );
		// Half-up ties of a small channel next to a large one (2.5 and 3.5 steps).
		texels.push( [ p, 2.5 * p / 256, 3.5 * p / 256 ], [ p, nextFloat( 2.5 * p / 256, - 1 ), nextFloat( 3.5 * p / 256, 1 ) ] );

	}

	let state = 5;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 );
	for ( let i = 0; i < 30000; i ++ ) texels.push( [ 0, 1, 2 ].map( () => ( ( next() & 0xffff ) / 65536 + 1 ) * 2 ** ( next() % 46 - 30 ) ) );
	return texels;

}

async function validateCodecs( device ) {

	const results = [];
	const helpers = packedFloatWgsl( 'rg11b10ufloat' );

	for ( const bits of [ 6, 5 ] ) {

		// Every code decodes as in core; every boundary encodes as in core.
		const codes = Uint32Array.from( { length: 1 << ( bits + 5 ) }, ( _, i ) => i );
		const decoded = await runCodec( device, helpers, `	output[i] = bitcast<u32>(ds_from_ufloat(input[i], ${ bits }u));`, codes, codes.length, codes.length );
		codes.forEach( code => {

			const expected = fromUfloat( code, bits ), actual = bitsF32( decoded[ code ] );
			if ( ! ( Object.is( actual, expected ) || ( Number.isNaN( expected ) && Number.isNaN( actual ) ) ) ) throw new Error( `ds_from_ufloat(${ code }, ${ bits }) = ${ actual }, expected ${ expected }` );

		} );

		const values = ufloatInputs( bits );
		const input = Uint32Array.from( values, f32Bits );
		const encoded = await runCodec( device, helpers, `	output[i] = ds_ufloat(bitcast<f32>(input[i]), ${ bits }u);`, input, input.length, input.length );
		values.forEach( ( value, i ) => {

			// Float32 subnormal inputs may be flushed to zero on the GPU; they encode to 0 either way.
			const expected = toUfloat( value, bits );
			if ( encoded[ i ] !== expected ) throw new Error( `ds_ufloat(${ value } [0x${ input[ i ].toString( 16 ) }], ${ bits }) = ${ encoded[ i ] }, expected ${ expected }` );

		} );
		results.push( { codec: `ufloat ${ bits + 5 }-bit`, decoded: codes.length, encoded: values.length } );

	}

	// rgb9e5: decode a code for every exponent and mantissa; encode boundary and random texels.
	const helpers9e5 = packedFloatWgsl( 'rgb9e5ufloat' );
	const codes = new Uint32Array( 32 * 512 );
	for ( let e = 0; e < 32; e ++ ) for ( let m = 0; m < 512; m ++ ) codes[ e * 512 + m ] = ( m | ( ( ( m * 7 ) & 0x1ff ) << 9 ) | ( ( m ^ 0x155 ) << 18 ) | ( e << 27 ) ) >>> 0;
	const decoded = await runCodec( device, helpers9e5, `	let v = bitcast<vec3u>(ds_unpack9e5(input[i]));
	output[3u * i] = v.x; output[3u * i + 1u] = v.y; output[3u * i + 2u] = v.z;`, codes, codes.length * 3, codes.length );
	codes.forEach( ( code, i ) => unpackRgb9e5( code ).forEach( ( expected, c ) => {

		if ( bitsF32( decoded[ 3 * i + c ] ) !== expected ) throw new Error( `ds_unpack9e5(0x${ code.toString( 16 ) }) channel ${ c } = ${ bitsF32( decoded[ 3 * i + c ] ) }, expected ${ expected }` );

	} ) );

	const texels = rgb9e5Inputs();
	const input = Uint32Array.from( texels.flat(), f32Bits );
	const encoded = await runCodec( device, helpers9e5, '	output[i] = ds_pack9e5(bitcast<vec3f>(vec3u(input[3u * i], input[3u * i + 1u], input[3u * i + 2u])));', input, texels.length, texels.length );
	texels.forEach( ( texel, i ) => {

		const expected = packRgb9e5( ...texel );
		if ( encoded[ i ] !== expected ) throw new Error( `ds_pack9e5(${ texel }) = 0x${ encoded[ i ].toString( 16 ) }, expected 0x${ expected.toString( 16 ) }` );

	} );
	results.push( { codec: 'rgb9e5', decoded: codes.length, encoded: texels.length } );
	return results;

}

function expectThrow( run, message, label ) {

	try { run(); } catch ( error ) {

		if ( ! String( error.message ).includes( message ) ) throw new Error( `${ label }: expected "${ message }", got "${ error.message }"` );
		return;

	}

	throw new Error( `${ label }: expected an error "${ message }"` );

}

/** Generates `texture`'s chain and checks every level against the oracle (from the GPU's previous level). */
async function generateAndCheck( device, downsampler, texture, data, label, reduction = 'average' ) {

	const { width, height, depthOrArrayLayers: layers, format } = texture;
	upload( device, texture, data );
	poison( device, texture );
	downsampler.generate( texture, { reduction } );
	const mips = await readMips( device, texture, 1 );
	const quantization = format === 'rgb9e5ufloat' || format === 'rg11b10ufloat' ? format : null;
	const firstOutputs = new Set( downsampler.prepare( texture, { reduction } ).map( batch => batch.firstMipLevel ) );
	let previous = toFloats( data, format ), w = width, h = height;
	for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

		const inTile = ! firstOutputs.has( level );
		const expected = reduceReference( previous, w, h, layers, quantization, reduction, inTile ? { pairing: 'float32' } : {} );
		check( mips[ level - 1 ], expected, quantization, `${ label } mip ${ level }`, reduction, 3, 0, inTile );
		previous = mips[ level - 1 ]; w = expected.width; h = expected.height;

	}

	return mips;

}

async function validateBufferPath( device ) {

	const results = [];

	// The buffer and storage paths write identical bits (same kernel arithmetic).
	const storage = new Downsampler( device, { output: 'storage' } ), buffer = new Downsampler( device, { output: 'buffer' } );
	for ( const format of [ 'rgba16float', 'rg11b10ufloat', 'rgba8unorm', 'rgba32float' ] ) {

		if ( storage.unsupportedReason( createTexture( device, { width: 1, height: 1, format } ) ) !== null ) continue;
		for ( const reduction of [ 'average', 'max' ] ) {

			const [ a, b ] = [ 0, 1 ].map( () => createTexture( device, { width: 333, height: 201, layers: 2, format } ) );
			const data = makeData( 333, 201, 2, format, 'random', 21 );
			for ( const texture of [ a, b ] ) { upload( device, texture, data ); poison( device, texture ); }
			storage.generate( a, { reduction } );
			buffer.generate( b, { reduction } );
			if ( storage.outputPath( a ) !== 'storage' || buffer.outputPath( b ) !== 'buffer' ) throw new Error( `${ format }: unexpected output paths` );
			const [ ma, mb ] = [ await readMips( device, a, 1 ), await readMips( device, b, 1 ) ];
			ma.forEach( ( level, l ) => { if ( level.some( ( value, i ) => ! Object.is( value, mb[ l ][ i ] ) ) ) throw new Error( `${ format } ${ reduction }: buffer path level ${ l + 1 } differs from storage path` ); } );
			a.destroy(); b.destroy();

		}

		results.push( { option: `buffer path equals storage path: ${ format }` } );

	}

	// One Downsampler, both paths and a separate source in one submit: auto picks per texture.
	{

		const auto = new Downsampler( device );
		const gi = createTexture( device, { width: 96, height: 64, format: 'rgb9e5ufloat' } ), hdr = createTexture( device, { width: 96, height: 64, format: 'rgba16float' } );
		if ( auto.outputPath( gi ) !== 'buffer' || auto.outputPath( hdr ) !== 'storage' ) throw new Error( 'auto output paths' );
		upload( device, hdr, makeData( 96, 64, 1, 'rgba16float', 'random', 4 ) );
		const encoder = device.createCommandEncoder();
		auto.encode( encoder, hdr );
		auto.encode( encoder, gi, { source: { texture: hdr } } );
		device.queue.submit( [ encoder.finish() ] );
		const [ source, mips ] = [ await readMips( device, hdr, 0 ), await readMips( device, gi, 0 ) ];
		// Level 0 (same size as the source) is a converting copy; the rest reduce it.
		const copy = source[ 0 ].slice();
		for ( let i = 0; i < copy.length; i += 4 ) copy.set( [ ...unpackRgb9e5( packRgb9e5( copy[ i ], copy[ i + 1 ], copy[ i + 2 ] ) ), 1 ], i );
		if ( mips[ 0 ].some( ( value, i ) => value !== copy[ i ] ) ) throw new Error( 'rgb9e5 from a separate source: level 0 is not the quantized copy' );
		for ( let l = 1; l < mips.length; l ++ ) check( mips[ l ], reduceReference( mips[ l - 1 ], Math.max( 1, 96 >> ( l - 1 ) ), Math.max( 1, 64 >> ( l - 1 ) ), 1, 'rgb9e5ufloat' ), 'rgb9e5ufloat', `auto rgb9e5 level ${ l }`, 'average', 3 );
		gi.destroy(); hdr.destroy();
		results.push( { option: 'auto output: storage and buffer textures in one encoder' } );

	}

	// API: reasons, encodePass, levelsPerDispatch, release.
	const auto = new Downsampler( device );
	const noCopy = device.createTexture( { size: [ 16, 16 ], format: 'rgb9e5ufloat', mipLevelCount: 5, usage: GPUTextureUsage.TEXTURE_BINDING } );
	const reason = auto.unsupportedReason( noCopy );
	if ( ! reason?.includes( 'COPY_DST' ) || ! reason.includes( 'storage' ) ) throw new Error( `rgb9e5 without COPY_DST: reason ${ reason }` );
	if ( ! storage.unsupportedReason( createTexture( device, { width: 4, height: 4, format: 'rgb9e5ufloat' } ) )?.includes( 'can\'t be a storage texture' ) ) throw new Error( 'output storage accepted rgb9e5ufloat' );
	if ( device.features.has( 'texture-formats-tier1' ) && ! buffer.unsupportedReason( createTexture( device, { width: 4, height: 4, format: 'r8unorm' } ) )?.includes( '4-byte words' ) ) throw new Error( 'output buffer accepted r8unorm' );
	noCopy.destroy();
	const gi = createTexture( device, { width: 64, height: 64, format: 'rgb9e5ufloat' } );
	const pass = device.createCommandEncoder().beginComputePass();
	expectThrow( () => auto.encodePass( pass, gi ), 'use encode()', 'encodePass on the buffer path' );
	pass.end();
	expectThrow( () => new Downsampler( device, { output: 'pixels' } ), 'output must be', 'unknown output' );
	// release() destroys the plan's buffers; the next generate prepares new ones.
	const data = makeData( 64, 64, 1, 'rgb9e5ufloat', 'random', 8 );
	await generateAndCheck( device, auto, gi, data, 'rgb9e5 before release' );
	auto.release( gi );
	await generateAndCheck( device, auto, gi, data, 'rgb9e5 after release' );
	gi.destroy();
	results.push( { option: 'buffer path: reasons, encodePass, release' } );

	// A device with default limits (4 storage textures) and no optional features:
	// the buffer path still reduces 6 levels per dispatch, and rg11b10ufloat falls back to it.
	const plain = await ( await navigator.gpu.requestAdapter() ).requestDevice();
	plain.pushErrorScope( 'validation' );
	try {

		const ds = new Downsampler( plain );
		const limit = Math.min( 6, plain.limits.maxStorageTexturesPerShaderStage );
		if ( limit < 6 ) expectThrow( () => new Downsampler( plain, { levelsPerDispatch: 6 } ), 'output \'buffer\'', 'six levels at the storage limit' );
		new Downsampler( plain, { levelsPerDispatch: 6, output: 'buffer' } );
		for ( const format of [ 'rgb9e5ufloat', 'rg11b10ufloat' ] ) {

			const texture = createTexture( plain, { width: 512, height: 256, format } );
			if ( ds.outputPath( texture ) !== 'buffer' ) throw new Error( `${ format } on a plain device: expected the buffer path` );
			const counts = ds.prepare( texture ).map( batch => batch.count );
			if ( counts.join() !== '6,3' ) throw new Error( `${ format } 512×256 on a plain device: batches ${ counts }, expected 6,3` );
			await generateAndCheck( plain, ds, texture, makeData( 512, 256, 1, format, 'random', 3 ), `${ format } plain device` );
			await generateAndCheck( plain, ds, texture, makeData( 512, 256, 1, format, 'random', 5 ), `${ format } plain device max`, 'max' );
			texture.destroy();

		}

		const noFeature = plain.createTexture( { size: [ 8, 8 ], format: 'rg11b10ufloat', mipLevelCount: 4, usage: GPUTextureUsage.TEXTURE_BINDING } );
		const featureReason = ds.unsupportedReason( noFeature );
		if ( ! featureReason?.includes( 'texture-formats-tier1' ) || ! featureReason.includes( 'COPY_DST' ) ) throw new Error( `rg11b10 without tier1 or COPY_DST: reason ${ featureReason }` );

	} finally {

		const error = await plain.popErrorScope();
		plain.destroy();
		if ( error ) throw new Error( `GPU validation error (plain device): ${ error.message }` );

	}

	results.push( { option: 'buffer path: 6 levels per dispatch at the default storage texture limit; rg11b10ufloat without tier1' } );
	return results;

}

/** Output textures at exactly each level's size; `storage` adds STORAGE_BINDING. */
function levelTextures( device, reducer, width, height, layers, format, storage ) {

	return reducer.levels.map( level => {

		const [ w, h ] = reducer.levelSize( level, width, height );
		return device.createTexture( { size: [ w, h, layers ], format, textureBindingViewDimension: layers > 1 ? '2d-array' : '2d', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | ( storage ? GPUTextureUsage.STORAGE_BINDING : 0 ) } );

	} );

}

async function runReducer( device, reducer, source, outputs, mode = 'encode' ) {

	for ( const texture of outputs ) poison( device, texture, 0 );
	const encoder = device.createCommandEncoder();
	if ( mode === 'encode' ) reducer.encode( encoder, { source, outputs } );
	else {

		const pass = encoder.beginComputePass();
		reducer.encodePass( pass, { source, outputs } );
		pass.end();
		reducer.encodeCopies( encoder, { source, outputs } );

	}

	device.queue.submit( [ encoder.finish() ] );
	return Promise.all( outputs.map( async texture => ( await readMips( device, texture, 0, 1 ) )[ 0 ] ) );

}

const sameBits = ( a, b ) => a.every( ( level, l ) => level.every( ( value, i ) => Object.is( value, b[ l ][ i ] ) ) );

/** Reducer (tile kernels) on the buffer output path: against the oracle, against the storage path, packed outputs, API. */
async function validateReducerBuffer( device ) {

	const results = [];
	const tier1 = device.features.has( 'texture-formats-tier1' );

	for ( const [ width, height, layers = 1 ] of [ [ 17, 9 ], [ 192, 108 ], [ 65, 129, 3 ], [ 1917, 1079 ] ] ) {

		const dimension = layers > 1 ? '2d-array' : '2d';
		const source = createTexture( device, { width, height, layers, format: 'rgba32float', dimension, storage: false } );
		const data = reducerData( width, height, layers, 'random', width + height );
		upload( device, source, data );

		for ( const edges of [ 'zero-weight', 'clamp' ] ) {

			// rgba32float: the oracle, and bit-identical to the storage path. Six outputs even at the default limit.
			const reduction = [ 'sum', 'average', 'min', 'max' ];
			const levels = [ 1, 2, 3, 4, 5, 6 ];
			const buffered = new Reducer( device, { reduction, levels, edges, format: 'rgba32float', dimension, output: 'buffer' } );
			const outputs = levelTextures( device, buffered, width, height, layers, 'rgba32float', true );
			const actual = await runReducer( device, buffered, source, outputs );
			const expected = reducePyramid( data, width, height, layers, { reduction, levels, edges } );
			expected.forEach( ( level, i ) => checkPyramid( actual[ i ], level, `reducer buffer ${ width }x${ height }x${ layers } ${ edges } level ${ level.level }` ) );
			if ( ! sameBits( actual, await runReducer( device, buffered, source, outputs, 'encodePass' ) ) ) throw new Error( 'reducer buffer: encodePass + encodeCopies differs from encode' );
			if ( device.limits.maxStorageTexturesPerShaderStage >= 6 ) {

				const stored = new Reducer( device, { reduction, levels, edges, format: 'rgba32float', dimension } );
				if ( ! sameBits( actual, await runReducer( device, stored, source, outputs ) ) ) throw new Error( `reducer ${ width }x${ height } ${ edges }: buffer output differs from storage output` );

			}

			// Packed and half outputs: rounded once, by our encoder (or pack2x16float).
			for ( const format of [ 'rgb9e5ufloat', 'rg11b10ufloat', 'rgba16float' ] ) {

				const packedLevels = [ 1, 2, 3 ];
				const reducer = new Reducer( device, { reduction: 'average', levels: packedLevels, edges, format, dimension, output: 'buffer' } );
				const textures = levelTextures( device, reducer, width, height, layers, format, false );
				const values = await runReducer( device, reducer, source, textures );
				const oracle = reducePyramid( data, width, height, layers, { reduction: 'average', levels: packedLevels, edges } );
				oracle.forEach( ( level, i ) => {

					const quantization = format === 'rgba16float' ? 'float16' : format;
					const reference = { values: Float32Array.from( level.values ), filterTolerance: new Float32Array( level.values.length ) };
					for ( let t = 0; t < reference.values.length; t += 4 ) quantizeTexel( reference.values, t, quantization );
					check( values[ i ], reference, quantization, `reducer ${ format } ${ width }x${ height }x${ layers } ${ edges } level ${ level.level }`, 'average', 3 );

				} );

				// rg11b10ufloat: the same bits as the storage path, since both quantize with our encoder.
				// Not rgba16float: Apple's textureStore truncates f32 → f16 where pack2x16float rounds to
				// nearest even (results/probes/f16-store-rounding.mjs), so only the oracle check applies.
				if ( format === 'rg11b10ufloat' && tier1 ) {

					const stored = new Reducer( device, { reduction: 'average', levels: packedLevels, edges, format, dimension, output: 'storage' } );
					const storageTextures = levelTextures( device, reducer, width, height, layers, format, true );
					if ( ! sameBits( values, await runReducer( device, stored, source, storageTextures ) ) ) throw new Error( `reducer ${ format } ${ width }x${ height } ${ edges }: buffer output differs from storage output` );
					storageTextures.forEach( texture => texture.destroy() );

				}

				textures.forEach( texture => texture.destroy() );

			}

			outputs.forEach( texture => texture.destroy() );

		}

		source.destroy();
		results.push( { option: `reducer buffer output ${ width }x${ height }x${ layers }` } );

	}

	// API: the default output follows the format; outputs need COPY_DST; store hooks take no output path.
	if ( new Reducer( device, { levels: [ 1 ], format: 'rgb9e5ufloat' } ).output !== 'buffer' ) throw new Error( 'reducer: rgb9e5ufloat should default to the buffer path' );
	if ( new Reducer( device, { levels: [ 1 ], format: 'rgba16float' } ).output !== 'storage' ) throw new Error( 'reducer: rgba16float should default to storage' );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], format: 'rgb9e5ufloat', output: 'storage' } ), 'can\'t be a storage texture', 'reducer rgb9e5 storage' );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return vec4f(0.0); }', store: 'fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {}', size: 'fn size() -> vec2u { return vec2u(4u); }', output: 'buffer' } ), 'store hook', 'reducer store hook with output' );
	const source = createTexture( device, { width: 8, height: 8, format: 'rgba32float', storage: false } );
	const noCopy = device.createTexture( { size: [ 4, 4 ], format: 'rgb9e5ufloat', usage: GPUTextureUsage.TEXTURE_BINDING } );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], format: 'rgb9e5ufloat' } ).encode( device.createCommandEncoder(), { source, outputs: [ noCopy ] } ), 'COPY_DST', 'reducer output without COPY_DST' );
	source.destroy(); noCopy.destroy();
	results.push( { option: 'reducer buffer output: defaults and validation' } );
	return results;

}

export async function validatePacked( device, { onProgress = () => {} } = {} ) {

	const results = await validateCodecs( device );
	onProgress( 'validated packed codecs' );
	results.push( ...await validateBufferPath( device ) );
	onProgress( 'validated buffer output path' );
	results.push( ...await validateReducerBuffer( device ) );
	onProgress( 'validated reducer buffer output' );
	return results;

}
