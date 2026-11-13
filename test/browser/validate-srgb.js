// sRGB destinations (M2.3.5, SPEC §6.1 sRGB), beyond the main matrix (which runs
// rgba8unorm-srgb on every shape and bgra8unorm-srgb on the subset, against the
// core oracle with the declared tolerance, and in-tile levels bit for bit):
// - the WGSL encoder against core `srgbCode` bit for bit (every threshold, every
//   decode table value, specials, random values), and the decode table;
// - the hardware's own decode, reported (Apple decodes to 12 bits), and checked to
//   re-encode to its code (what snapping loads and folding codes rely on);
// - linear light: black and white average to code 188, not 128;
// - near-exact chains on the load path: every level's codes equal the oracle's except
//   where the exact linear value lies within float error of a rounding threshold
//   (catches unsnapped loads, encoded-space averaging and rounding errors that the
//   two-code tolerance would hide);
// - composition: the compatibility path, a caller's scratch buffer shared by two
//   chains, bgra8unorm-srgb against rgba8unorm-srgb, the kernel's storage output (an
//   rgba8unorm scratch copied into the level) against the buffer path, and the API.
import { boxReference, quantizeTexel, reducePyramid, reduceReference, srgbBounds, srgbCode, srgbDecodeTable, srgbThresholds } from '/dist/index.js';
import { Downsampler, Reducer, srgbWgsl } from '/dist/webgpu/index.js';
import { createTexture, poison, readMips, upload } from './gpu.js';
import { compatibility, expectThrow } from './validate.js';
import { levelTextures, runCodec, runReducer } from './validate-packed.js';
import { scratchPrototype } from '../bench/srgb.js';

const f32Bits = value => new Uint32Array( new Float32Array( [ value ] ).buffer )[ 0 ];
const bitsF32 = bits => new Float32Array( new Uint32Array( [ bits ] ).buffer )[ 0 ];
const mipSize = ( size, level ) => Math.max( 1, size >> level );

/** Random sRGB fixtures: codes in [ lo, hi ] for colour, any alpha. */
function codes( width, height, layers, seed, lo = 0, hi = 255 ) {

	const data = new Uint8Array( width * height * layers * 4 );
	let state = seed;
	for ( let i = 0; i < data.length; i ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		data[ i ] = i % 4 === 3 ? state >>> 24 : lo + ( state >>> 24 ) % ( hi - lo + 1 );

	}

	return data;

}

export async function validateSrgb( device, { quick = false, onProgress = () => {}, select } ) {

	const results = [];
	results.push( ...await select.run( 'srgb/codec', () => validateCodec( device ) ) );
	results.push( ...await select.run( 'srgb/decode', () => validateDecode( device ) ) );
	results.push( ...await select.run( 'srgb/near-exact', () => validateNearExact( device, { quick, onProgress } ) ) );
	results.push( ...await select.run( 'srgb/api', () => validateApi( device ) ) );
	results.push( ...await select.run( 'srgb/reducer', () => validateReducer( device ) ) );
	return results;

}

/**
 * A Reducer from an sRGB source into sRGB outputs (the tile kernel's buffer output encodes;
 * its loads snap to the exact decode): code for code against `reducePyramid` of the decoded
 * source, one code apart only at near-ties (alpha: pack4x8unorm's round half up against the
 * oracle's ties to even).
 */
async function validateReducer( device ) {

	const results = [];
	for ( const [ width, height, layers ] of [ [ 61, 37, 1 ], [ 32, 32, 2 ] ] ) {

		const dimension = layers > 1 ? '2d-array' : '2d';
		const source = createTexture( device, { width, height, layers, format: 'rgba8unorm-srgb', dimension } );
		const data = codes( width, height, layers, 21 );
		upload( device, source, data );
		const floats = ( await readMips( device, source, 0, 1 ) )[ 0 ];
		try {

			for ( const edges of [ 'zero-weight', 'clamp' ] ) {

				const levels = [ 1, 2, 3 ];
				const reducer = new Reducer( device, { reduction: 'average', levels, edges, format: 'rgba8unorm-srgb', dimension } );
				const outputs = levelTextures( device, reducer, width, height, layers, 'rgba8unorm-srgb', false );
				const values = await runReducer( device, reducer, source, outputs );
				reducePyramid( floats, width, height, layers, { reduction: 'average', levels, edges } ).forEach( ( level, i ) => {

					const expected = { values: Float32Array.from( level.values ) };
					for ( let t = 0; t < expected.values.length; t += 4 ) quantizeTexel( expected.values, t, 'srgb8' );
					nearExact( values[ i ], expected, { values: level.values }, `srgb reducer ${ width }x${ height }x${ layers } ${ edges } level ${ level.level }`, 1e-5 );

				} );
				outputs.forEach( texture => texture.destroy() );
				reducer.release();

			}

		} finally {

			source.destroy();

		}

		results.push( { srgb: 'reducer', width, height, layers } );

	}

	return results;

}

/** `ds_srgb_code` and `ds_srgb_quantize` against core, bit for bit. */
async function validateCodec( device ) {

	const inputs = [];
	for ( let k = 0; k < 255; k ++ ) {

		const bound = f32Bits( srgbBounds[ k ] );
		for ( let d = - 3; d <= 3; d ++ ) inputs.push( bound + d );

	}

	for ( const value of srgbDecodeTable ) inputs.push( f32Bits( value ) );
	for ( const value of [ - 0, - 1, - Infinity, Infinity, NaN, 1, 1.0000001, 2, 1e30, 1e-30, 1e-45, 0.0031308, 0.04045 / 12.92 ] ) inputs.push( f32Bits( value ) );
	let state = 99;
	for ( let i = 0; i < 40000; i ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		const u = state / 2 ** 32;
		inputs.push( f32Bits( i % 3 === 0 ? u * 1.2 - 0.1 : i % 3 === 1 ? u ** 4 : u * 0.004 ) );

	}

	const input = Uint32Array.from( inputs );
	const output = await runCodec( device, srgbWgsl(), `	let v = bitcast<f32>(input[i]);
	output[2u * i] = ds_srgb_code(v);
	output[2u * i + 1u] = bitcast<u32>(ds_srgb_quantize(vec3f(v)).x);`, input, input.length * 2, input.length );
	let mismatches = 0, first = null;
	for ( let i = 0; i < input.length; i ++ ) {

		const v = bitsF32( input[ i ] ), code = srgbCode( v );
		if ( output[ 2 * i ] !== code || output[ 2 * i + 1 ] !== f32Bits( srgbDecodeTable[ code ] ) ) {

			mismatches ++;
			first ??= `ds_srgb_code(${ v }) = ${ output[ 2 * i ] } (quantized ${ bitsF32( output[ 2 * i + 1 ] ) }), core ${ code }`;

		}

	}

	if ( mismatches ) throw new Error( `srgb codec: ${ mismatches } of ${ input.length } inputs differ from core; first: ${ first }` );
	return [ { srgb: 'codec', inputs: input.length } ];

}

/** The hardware decode (textureLoad of an sRGB view) of every code: reported, and it must re-encode to its code. */
async function validateDecode( device ) {

	const texture = device.createTexture( { size: [ 256, 1 ], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
	const data = new Uint8Array( 1024 );
	for ( let i = 0; i < 256; i ++ ) data.set( [ i, 255 - i, i, i ], i * 4 );
	upload( device, texture, data );
	const input = Array.from( { length: 256 }, ( _, i ) => i );
	// The codec harness has no texture binding, so read through a tiny pipeline of its own.
	const module = device.createShaderModule( { code: `${ srgbWgsl() }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	let t = textureLoad(image, vec2i(i32(id.x), 0), 0);
	output[4u * id.x] = bitcast<u32>(t.x);
	output[4u * id.x + 1u] = ds_srgb_code(t.x);
	output[4u * id.x + 2u] = ds_srgb_code(t.y);
	output[4u * id.x + 3u] = bitcast<u32>(t.w);
}` } );
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const out = device.createBuffer( { size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
	const readback = device.createBuffer( { size: 4096, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline( pipeline );
	pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: out } } ] } ) );
	pass.dispatchWorkgroups( 4 );
	pass.end();
	encoder.copyBufferToBuffer( out, 0, readback, 0, 4096 );
	device.queue.submit( [ encoder.finish() ] );
	await readback.mapAsync( GPUMapMode.READ );
	const words = new Uint32Array( readback.getMappedRange().slice( 0 ) );
	readback.unmap();
	for ( const resource of [ texture, out, readback ] ) resource.destroy();
	let inexact = 0, worst = 0;
	for ( const i of input ) {

		const decoded = bitsF32( words[ 4 * i ] );
		if ( decoded !== srgbDecodeTable[ i ] ) inexact ++;
		const gap = srgbDecodeTable[ Math.min( 255, i + 1 ) ] - srgbDecodeTable[ Math.max( 0, i - 1 ) ];
		worst = Math.max( worst, Math.abs( decoded - srgbDecodeTable[ i ] ) / ( gap / 2 ) );
		if ( words[ 4 * i + 1 ] !== i || words[ 4 * i + 2 ] !== 255 - i ) throw new Error( `srgb decode: the hardware's decode of code ${ i } (${ decoded }) re-encodes to ${ words[ 4 * i + 1 ] }` );
		if ( bitsF32( words[ 4 * i + 3 ] ) !== Math.fround( i / 255 ) ) throw new Error( `srgb decode: alpha ${ i } reads ${ bitsF32( words[ 4 * i + 3 ] ) }, not linear` );

	}

	// Informational: how far the device's decode is from the exact table (M5 Pro: 254 codes, 0.39 of a step).
	return [ { srgb: 'decode', hardwareInexact: inexact, worstStepFraction: Number( worst.toFixed( 3 ) ) } ];

}

/**
 * Codes of `actual` (RGBA floats read back) against the oracle: equal, except one code apart
 * where the exact linear value (`linear`, unrounded) lies within float error of the threshold
 * between them. Alpha likewise in unorm8 units against ties. Returns the near-tie count.
 */
function nearExact( actual, expected, linear, label, window = 1e-6 ) {

	let nearTies = 0;
	for ( let i = 0; i < expected.values.length; i ++ ) {

		const alpha = i % 4 === 3;
		const a = alpha ? Math.round( actual[ i ] * 255 ) : srgbCode( actual[ i ] );
		const e = alpha ? Math.round( expected.values[ i ] * 255 ) : srgbCode( expected.values[ i ] );
		if ( a === e ) continue;
		const v = linear.values[ i ];
		const near = Math.abs( a - e ) === 1 && ( alpha ? Math.abs( v * 255 - ( Math.min( a, e ) + 0.5 ) ) < 100 * window : Math.abs( v - srgbThresholds[ Math.min( a, e ) ] ) <= window * v );
		if ( ! near ) throw new Error( `${ label }: value ${ i } is code ${ a }, the oracle's ${ e } (linear ${ v })` );
		nearTies ++;

	}

	return nearTies;

}

/** Load-path chains (sampler-free) against the oracle, code for code. */
async function validateNearExact( device, { quick, onProgress } ) {

	const shapes = [ [ 255, 257 ], [ 192, 108 ], [ 7, 1 ], [ 1, 128 ], [ 33, 17, 3 ], ...( quick ? [] : [ [ 1920, 1080 ], [ 1129, 635 ] ] ) ];
	const modes = [
		{ name: 'load', downsampler: new Downsampler( device, { averageAccess: 'load', tail: false } ), oddSizes: 'exact' },
		{ name: 'tail load', downsampler: new Downsampler( device, { averageAccess: 'load' } ), oddSizes: 'exact' },
		{ name: 'box load', downsampler: new Downsampler( device, { averageAccess: 'load', tail: false } ), oddSizes: 'box' }
	];
	const results = [];
	let nearTies = 0, values = 0;
	for ( const [ width, height, layers = 1 ] of shapes ) for ( const format of [ 'rgba8unorm-srgb', ...( width === 255 ? [ 'bgra8unorm-srgb' ] : [] ) ] ) {

		const texture = createTexture( device, { width, height, layers, format } );
		try {

			// All codes, then dark codes only (the linear segment, where Apple's 12-bit decode is furthest off).
			for ( const [ seed, lo, hi ] of [ [ 5, 0, 255 ], [ 6, 0, 24 ] ] ) {

				const data = codes( width, height, layers, seed, lo, hi );
				for ( const { name, downsampler, oddSizes } of modes ) {

					upload( device, texture, data );
					poison( device, texture );
					downsampler.generate( texture, { oddSizes } );
					const mips = await readMips( device, texture, 0 );
					const step = oddSizes === 'box' ? boxReference : reduceReference;
					for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

						const previous = mips[ level - 1 ], w = mipSize( width, level - 1 ), h = mipSize( height, level - 1 );
						const expected = step( previous, w, h, layers, 'srgb8', 'average', { weights: 'shader' } );
						const linear = step( previous, w, h, layers, 'float32', 'average', { quantizedInput: false, weights: 'shader' } );
						// Alpha: the unrounded value in units, from the same f64 arithmetic.
						const alpha = step( previous.map( ( value, i ) => i % 4 === 3 ? value * 255 : value ), w, h, layers, 'float32', 'average', { quantizedInput: false, weights: 'shader' } );
						for ( let i = 3; i < linear.values.length; i += 4 ) linear.values[ i ] = alpha.values[ i ] / 255;
						nearTies += nearExact( mips[ level ], expected, linear, `srgb near-exact ${ name } ${ width }x${ height }x${ layers } ${ format } codes ${ lo }–${ hi } mip ${ level }` );
						values += expected.values.length;

					}

				}

			}

			onProgress( `validated srgb near-exact ${ width }×${ height }×${ layers } ${ format }` );

		} finally {

			texture.destroy();

		}

	}

	results.push( { srgb: 'near-exact', values, nearTies } );
	return results;

}

async function validateApi( device ) {

	const results = [];
	const downsampler = new Downsampler( device );
	const make = ( format, width = 37, height = 21, layers = 2, usage ) => usage === undefined ? createTexture( device, { width, height, layers, format } ) : device.createTexture( { size: [ width, height, layers ], format, mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1, usage } );
	const textures = [];
	const track = texture => ( textures.push( texture ), texture );
	const chain = async ( owner, texture, data, options = {}, encoder ) => {

		upload( device, texture, data );
		poison( device, texture );
		if ( encoder ) owner.encode( encoder, texture, options );
		else owner.generate( texture, options );
		return encoder ? null : readMips( device, texture, 0 );

	};
	const same = ( a, b, label ) => a.forEach( ( level, l ) => level.forEach( ( value, i ) => { if ( ! Object.is( value, b[ l ][ i ] ) ) throw new Error( `${ label }: level ${ l } value ${ i }: ${ value } vs ${ b[ l ][ i ] }` ); } ) );

	try {

		// Linear light: a black and white checkerboard averages to 0.5, code 188 (encoded space would give 128),
		// and alpha 0 and 255 to 127.5 units, a tie, to even (128).
		const checker = track( make( 'rgba8unorm-srgb', 16, 16, 1 ) );
		const data = new Uint8Array( 16 * 16 * 4 );
		for ( let y = 0; y < 16; y ++ ) for ( let x = 0; x < 16; x ++ ) data.set( ( x + y ) % 2 ? [ 255, 255, 255, 255 ] : [ 0, 0, 0, 0 ], ( y * 16 + x ) * 4 );
		const mips = await chain( downsampler, checker, data );
		for ( let level = 1; level < mips.length; level ++ ) for ( let i = 0; i < mips[ level ].length; i ++ ) {

			const expected = i % 4 === 3 ? Math.fround( 128 / 255 ) : srgbDecodeTable[ 188 ];
			if ( mips[ level ][ i ] !== expected ) throw new Error( `srgb linear light: level ${ level } value ${ i } is ${ mips[ level ][ i ] }, expected ${ expected } (code 188 / alpha 128)` );

		}

		// The same bits through the compatibility path (skipped on the buffer path), output 'buffer' and a shared scratch buffer.
		const data2 = codes( 37, 21, 2, 11 );
		const texture = track( make( 'rgba8unorm-srgb' ) ), other = track( make( 'rgba8unorm-srgb' ) );
		const reference = await chain( downsampler, texture, data2 );
		same( await chain( new Downsampler( device, { compatibility: true } ), texture, data2 ), reference, 'srgb compatibility path' );
		same( await chain( new Downsampler( device, { output: 'buffer' } ), texture, data2 ), reference, 'srgb output buffer' );
		const size = downsampler.scratchSize( texture );
		const scratch = device.createBuffer( { size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		try {

			// Two chains through one scratch buffer in one encoder: each chain's copies come before the next dispatch.
			const encoder = device.createCommandEncoder();
			const data3 = codes( 37, 21, 2, 12 );
			await chain( downsampler, texture, data2, { scratch }, encoder );
			await chain( downsampler, other, data3, { scratch }, encoder );
			device.queue.submit( [ encoder.finish() ] );
			same( await readMips( device, texture, 0 ), reference, 'srgb shared scratch, first chain' );
			same( await readMips( device, other, 0 ), await chain( new Downsampler( device ), other, data3 ), 'srgb shared scratch, second chain' );
			if ( downsampler.prepare( texture, { scratch } )[ 0 ].buffer !== scratch ) throw new Error( 'srgb scratch: the plan does not write the caller\'s buffer' );
			if ( downsampler.prepare( texture )[ 0 ].buffer === scratch ) throw new Error( 'srgb scratch: a plan without scratch reused the caller\'s buffer' );
			// The sizes a scratch must have: at least the largest batch, so a smaller one is rejected with the size.
			const needed = Math.max( ...downsampler.prepare( texture ).map( batch => batch.buffer.size ) );
			if ( size < needed ) throw new Error( `srgb scratchSize ${ size } is below the plan's buffer (${ needed })` );
			const small = device.createBuffer( { size: needed - 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
			expectThrow( () => downsampler.prepare( texture, { scratch: small } ), `this chain needs ${ needed }`, 'srgb scratch too small' );
			small.destroy();
			const wrong = device.createBuffer( { size, usage: GPUBufferUsage.STORAGE } );
			expectThrow( () => downsampler.prepare( texture, { scratch: wrong } ), 'COPY_SRC', 'srgb scratch usage' );
			wrong.destroy();
			// Storage-path textures ignore it, and need none.
			const storable = track( createTexture( device, { width: 8, height: 8, format: 'rgba8unorm' } ) );
			if ( downsampler.scratchSize( storable ) !== 0 ) throw new Error( 'srgb scratchSize: a storage-path texture needs no scratch' );
			downsampler.prepare( storable, { scratch } );
			downsampler.release( texture );

		} finally {

			scratch.destroy();

		}

		// bgra8unorm-srgb holds the same bytes in another channel order, and every channel reduces alike:
		// the same bytes in, the same bytes out.
		const bgra = track( make( 'bgra8unorm-srgb' ) );
		same( toBytes( await chain( downsampler, bgra, data2 ) ), toBytes( reference ), 'srgb bgra8unorm-srgb vs rgba8unorm-srgb bytes' );

		// The kernel's storage output (codes into an rgba8unorm scratch, copied into the level) writes the buffer path's bits.
		if ( ! compatibility( device ) ) {

			const viaScratch = track( make( 'rgba8unorm-srgb' ) );
			upload( device, viaScratch, data2 );
			poison( device, viaScratch );
			const prototype = scratchPrototype( device, downsampler );
			const encoder = device.createCommandEncoder();
			prototype.encode( encoder, viaScratch );
			device.queue.submit( [ encoder.finish() ] );
			same( await readMips( device, viaScratch, 0 ), reference, 'srgb storage output into an rgba8unorm scratch' );

		}

		// API: the buffer path is the only path, so encodePass and a texture without COPY_DST say so.
		const pass = device.createCommandEncoder().beginComputePass();
		expectThrow( () => downsampler.encodePass( pass, texture ), 'use encode()', 'srgb encodePass' );
		pass.end();
		const noCopy = track( make( 'rgba8unorm-srgb', 8, 8, 1, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC ) );
		const reason = downsampler.unsupportedReason( noCopy );
		if ( ! reason?.includes( 'COPY_DST' ) || ! reason.includes( "can't be a storage texture" ) ) throw new Error( `srgb reason without COPY_DST: ${ reason }` );
		if ( ! new Downsampler( device, { output: 'storage' } ).unsupportedReason( texture )?.includes( "can't be a storage texture" ) ) throw new Error( 'srgb with output \'storage\' must be unsupported' );
		expectThrow( () => downsampler.prepare( texture, { oddSizes: 'spd', reduction: 'max' } ), 'average', 'srgb spd max' );
		results.push( { srgb: 'api' } );

	} finally {

		for ( const texture of textures ) {

			downsampler.release( texture );
			texture.destroy();

		}

	}

	return results;

}

/** Levels of RGBA floats (sRGB colour, linear alpha) as the bytes they were read from, in memory order. */
function toBytes( levels ) {

	return levels.map( level => Array.from( level, ( value, i ) => i % 4 === 3 ? Math.round( value * 255 ) : srgbCode( value ) ) );

}

