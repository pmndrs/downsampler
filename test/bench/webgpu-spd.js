// Comparison with webgpu-spd (JolifantoBambla/webgpu-spd, npm `webgpu-spd`
// 3.0.0, git adc666e, vendored test-only in test/vendor/webgpu-spd/ with its
// MIT licence; never a dependency of the library). Results and method:
// results/compare/webgpu-spd.md.
//
// - `webgpuSpdVariants( device )`: the library's configurations we compare.
//   `wspd` is its default on this device (min(maxStorageTexturesPerShaderStage,
//   12) mips per pass, so the atomic last-workgroup handoff above 6 mips;
//   subgroup quads when the device has 'subgroups'). `wspd6` is 6 mips per
//   pass, the limit its own `setPreferredLimits` asks for: no handoff,
//   ⌈mips/6⌉ dispatches, the same structure as our `oddSizes: 'spd'`.
//   `wspdNoSg` is the default without subgroups.
// - `checkWebgpuSpd( device, … )`: correctness against our CPU oracles (SPD
//   semantics per pass, from the GPU's own pass source level), missing and
//   stale levels over many back-to-back frames with changing inputs, Hi-Z
//   conservativeness, and targeted probes (see `probes`).
import { fromHalf, maxViolations, quantizeTexel, tolerance, toHalf } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { WebGPUSinglePassDownsampler, SPDFilters } from '../vendor/webgpu-spd/index.js';
import { createTexture, makeData, poison, readMips, upload } from '../browser/gpu.js';

export const compareSpdSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1707, 960 ], [ 1920, 1080 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ], [ 256, 256 ] ];

const quantizations = { rgba8unorm: 'unorm8', rgba16float: 'float16', r32float: 'float32', rg32float: 'float32' };

/** One webgpu-spd configuration: its own instance (resources are per instance and device). */
export function webgpuSpdVariant( device, { mipsPerPass, subgroups = true } = {} ) {

	const lib = new WebGPUSinglePassDownsampler();
	lib.prepareDeviceResources( { device, maxMipsPerPass: mipsPerPass, disableSubgroups: ! subgroups } );
	const perPass = Math.min( device.limits.maxStorageTexturesPerShaderStage, mipsPerPass ?? 12 );
	const passes = new WeakMap();
	const key = config => `${ config.filter ?? 'average' }:${ config.halfPrecision ?? false }`;
	return {
		lib, perPass, subgroups: subgroups && device.features.has( 'subgroups' ),
		/** A cached SPDPass per texture and config, as the library's docs recommend for per-frame use. */
		pass( texture, config = {} ) {

			// Sub-rectangles, targets and mip counts are one-off probes: prepared fresh.
			if ( config.offset || config.size || config.target || config.numMips ) return lib.preparePass( device, texture, config );
			let map = passes.get( texture );
			if ( ! map ) passes.set( texture, map = new Map() );
			if ( ! map.has( key( config ) ) ) map.set( key( config ), lib.preparePass( device, texture, config ) );
			return map.get( key( config ) );

		},
		encode( encoder, texture, config ) {

			const pass = encoder.beginComputePass();
			this.pass( texture, config ).encode( pass );
			pass.end();

		},
		/** webgpu-spd's pass structure for `mips` output levels: [ baseMip, count ] per dispatch. */
		structure( mips ) {

			const out = [];
			for ( let base = 0; base < mips; base += perPass ) out.push( [ base, Math.min( perPass, mips - base ) ] );
			return out;

		}
	};

}

export function webgpuSpdVariants( device ) {

	const variants = { wspd: webgpuSpdVariant( device ), wspd6: webgpuSpdVariant( device, { mipsPerPass: 6 } ) };
	if ( device.features.has( 'subgroups' ) ) variants.wspdNoSg = webgpuSpdVariant( device, { subgroups: false } );
	return variants;

}

const size = ( n, level ) => Math.max( 1, n >> level );

/**
 * SPD semantics for one pass, fast and iterative (float64, no rounding between
 * levels, each level quantized as stored): level l texel x averages (or folds)
 * the source block [x·2^l, (x+1)·2^l). `pad` is what an out-of-bounds read of
 * the pass source returns (it matters only from the first unit step): a vec4
 * (WGSL textureLoad's permitted zero results, (0,0,0,0) or (0,0,0,1)), or
 * 'clamp' (the edge texel, also permitted: "some texel within bounds"; Dawn's
 * robustness clamps coordinates). Levels are computed on the padded grid, so
 * deep levels see exactly the padded block.
 */
function spdPass( source, width, height, count, quantization, reduction = 'average', pad = [ 0, 0, 0, 0 ] ) {

	const combine = reduction === 'max' ? Math.max : reduction === 'min' ? Math.min : null;
	const out = [];
	let level = Float64Array.from( source ), w = width, h = height;
	if ( pad === 'clamp' ) {

		// Pad the source to whole 2^count blocks with its edge texels.
		const pw = Math.ceil( width / 2 ** count ) * 2 ** count, ph = Math.ceil( height / 2 ** count ) * 2 ** count;
		level = new Float64Array( pw * ph * 4 );
		for ( let y = 0; y < ph; y ++ ) for ( let x = 0; x < pw; x ++ ) for ( let c = 0; c < 4; c ++ ) level[ ( y * pw + x ) * 4 + c ] = source[ ( Math.min( y, height - 1 ) * width + Math.min( x, width - 1 ) ) * 4 + c ];
		w = pw; h = ph;

	}
	for ( let l = 1; l <= count; l ++ ) {

		const nw = Math.max( 1, Math.ceil( w / 2 ) ), nh = Math.max( 1, Math.ceil( h / 2 ) );
		const next = new Float64Array( nw * nh * 4 );
		const at = ( x, y, c ) => x < w && y < h ? level[ ( y * w + x ) * 4 + c ] : pad[ c ];
		for ( let y = 0; y < nh; y ++ ) for ( let x = 0; x < nw; x ++ ) for ( let c = 0; c < 4; c ++ ) {

			const a = at( 2 * x, 2 * y, c ), b = at( 2 * x + 1, 2 * y, c ), d = at( 2 * x, 2 * y + 1, c ), e = at( 2 * x + 1, 2 * y + 1, c );
			next[ ( y * nw + x ) * 4 + c ] = combine ? combine( combine( a, b ), combine( d, e ) ) : ( a + b + d + e ) / 4;

		}

		level = next; w = nw; h = nh;
		// Stored size ⌊n/2^l⌋ (at least 1): crop the top-left of the padded grid.
		const sw = size( width, l ), sh = size( height, l );
		const values = new Float32Array( sw * sh * 4 );
		for ( let y = 0; y < sh; y ++ ) for ( let x = 0; x < sw; x ++ ) for ( let c = 0; c < 4; c ++ ) values[ ( y * sw + x ) * 4 + c ] = level[ ( y * w + x ) * 4 + c ];
		for ( let i = 0; i < values.length; i += 4 ) quantizeTexel( values, i, quantization );
		out.push( { width: sw, height: sh, values } );

	}

	return out;

}

/**
 * Levels (1-based, absolute) whose SPD value reads outside its source: from
 * the first 2:1 step along an axis that is already 1 texel. Above them every
 * read is in bounds, so drop semantics (SPD) and our 'clamp' oracle agree;
 * from them on, WGSL leaves the result implementation-defined (textureLoad
 * out of bounds returns zero or any in-bounds texel).
 */
export function firstUnitStep( width, height, mips ) {

	for ( let l = 1; l <= mips; l ++ ) if ( size( width, l - 1 ) === 1 || size( height, l - 1 ) === 1 ) return l;
	return Infinity;

}

/** The colour `poison()` writes, as read back (it encodes with Math.round, not quantizeTexel's ties to even). */
function poisonTexel( format ) {

	const v = [ 0.9, 0.1, 0.7, 0.3 ];
	if ( format === 'r32float' ) return [ Math.fround( 0.9 ), 0, 0, 1 ];
	if ( format === 'rg32float' ) return [ Math.fround( 0.9 ), Math.fround( 0.1 ), 0, 1 ];
	if ( format === 'rgba16float' ) return v.map( x => Math.fround( fromHalf( toHalf( x ) ) ) );
	return v.map( x => Math.fround( Math.round( x * 255 ) / 255 ) );

}

/** Compares one GPU level with expected values; counts mismatches, exact values and poisoned (unwritten) texels. */
function compareLevel( actual, expected, quantization, reduction, poisonValue, channels = 4 ) {

	let mismatches = 0, exact = 0, worst = 0, poisoned = 0;
	const step = quantization === 'unorm8' ? 1 / 255 : 1;
	for ( let t = 0; t < actual.length; t += 4 ) {

		if ( poisonValue.every( ( p, c ) => c >= channels || Object.is( actual[ t + c ], p ) ) ) poisoned ++;
		for ( let c = 0; c < channels; c ++ ) {

			const a = actual[ t + c ], e = expected[ t + c ], error = Math.abs( a - e );
			if ( Object.is( a, e ) ) exact ++;
			if ( ! ( error <= tolerance( e, quantization, 0, reduction ) ) ) mismatches ++;
			worst = Math.max( worst, error / step );

		}

	}

	return { mismatches, exact, values: actual.length / 4 * channels, worst, poisoned };

}

/**
 * One webgpu-spd chain checked per pass against SPD semantics from the GPU's
 * own pass source level. In-bounds levels: the iterative SPD oracle (the same
 * values as `spdReference`, whose 'clamp' never triggers there). Unit-step
 * levels: the zero models, and our 'spd' (clamp) output for scale.
 */
async function checkChain( device, variant, texture, base, format, reduction, ours ) {

	const quantization = quantizations[ format ];
	const mips = texture.mipLevelCount - 1;
	const unit = firstUnitStep( texture.width, texture.height, mips );
	const levels = await readMips( device, texture, 1 );
	const p = poisonTexel( format );
	const channels = format === 'r32float' ? 1 : 4;
	const result = { inBounds: { mismatches: 0, exact: 0, values: 0, worst: 0, poisoned: 0 }, unit: [] };

	for ( const [ baseMip, count ] of variant.structure( mips ) ) {

		const source = baseMip === 0 ? base : levels[ baseMip - 1 ];
		const sw = size( texture.width, baseMip ), sh = size( texture.height, baseMip );
		// Out-of-bounds models only where the pass reaches a unit step.
		const pads = baseMip + count >= unit ? [ [ 0, 0, 0, 0 ], [ 0, 0, 0, 1 ], 'clamp' ] : [ [ 0, 0, 0, 0 ] ];
		const models = pads.map( pad => spdPass( source, sw, sh, count, quantization, reduction, pad ) );
		for ( let i = 0; i < count; i ++ ) {

			const level = baseMip + i + 1, actual = levels[ level - 1 ];
			if ( level < unit ) {

				const c = compareLevel( actual, models[ 0 ][ i ].values, quantization, reduction, p, channels );
				for ( const k of Object.keys( c ) ) result.inBounds[ k ] = k === 'worst' ? Math.max( result.inBounds[ k ], c[ k ] ) : result.inBounds[ k ] + c[ k ];

			} else {

				const [ zero, zeroAlpha1, clamp ] = models.map( m => compareLevel( actual, m[ i ].values, quantization, reduction, p, channels ) );
				const entry = { level, size: `${ size( texture.width, level ) }x${ size( texture.height, level ) }`, values: zero.values, poisoned: zero.poisoned, zeroMismatches: zero.mismatches, zeroAlpha1Mismatches: zeroAlpha1.mismatches, clampMismatches: clamp.mismatches };
				if ( ours ) {

					const o = ours[ level - 1 ];
					let worst = 0, ratio = 0;
					for ( let t = 0; t < o.length; t ++ ) if ( t % 4 < 3 ) {

						worst = Math.max( worst, Math.abs( actual[ t ] - o[ t ] ) );
						if ( o[ t ] !== 0 ) ratio = Math.max( ratio, Math.abs( 1 - actual[ t ] / o[ t ] ) );

					}
					entry.vsOursSpd = { worst, relative: ratio };

				}

				result.unit.push( entry );

			}

		}

	}

	return { result, levels };

}

/**
 * Back-to-back frames with changing inputs: copy one of four pre-uploaded
 * sources into level 0, poison the deep levels (from level 5: the handoff's
 * levels 7+ and mip 6 it reads), run the pass, copy the deep levels out. Every
 * frame must equal, bit for bit, the majority of three isolated runs on the same
 * input: a stale payload, a missed handoff or a lost reset shows as a
 * difference or as poison. Evidence for one device, not a proof.
 */
async function raceStress( device, variant, width, height, format, frames, config = {} ) {

	const sources = [ 11, 12, 13, 14 ].map( seed => {

		const t = createTexture( device, { width, height, format } );
		upload( device, t, makeData( width, height, 1, format, seed % 2 ? 'random' : 'tiles', seed ) );
		return t;

	} );
	const work = createTexture( device, { width, height, format } );
	const mips = work.mipLevelCount - 1;
	const first = Math.max( 1, Math.min( 5, mips ) );
	const bytes = { rgba8unorm: 4, rgba16float: 8, r32float: 4 }[ format ];
	const layout = [];
	let frameBytes = 0;
	for ( let level = first; level <= mips; level ++ ) {

		const w = size( width, level ), h = size( height, level ), bytesPerRow = Math.ceil( w * bytes / 256 ) * 256;
		layout.push( { level, w, h, bytesPerRow, offset: frameBytes } );
		frameBytes += bytesPerRow * h;

	}

	const poisonBuffer = device.createBuffer( { size: frameBytes, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST } );
	{

		const words = new Uint8Array( frameBytes );
		const texel = { rgba8unorm: [ 230, 26, 179, 77 ], rgba16float: new Uint8Array( Uint16Array.from( [ 0x3b33, 0x2e66, 0x399a, 0x34cd ] ).buffer ), r32float: new Uint8Array( Float32Array.from( [ 0.9 ] ).buffer ) }[ format ];
		for ( let i = 0; i < frameBytes; i ++ ) words[ i ] = texel[ i % texel.length ];
		device.queue.writeBuffer( poisonBuffer, 0, words );

	}

	const record = ( encoder, k, readback, slot ) => {

		encoder.copyTextureToTexture( { texture: sources[ k ] }, { texture: work }, [ width, height ] );
		for ( const { level, w, h, bytesPerRow, offset } of layout ) encoder.copyBufferToTexture( { buffer: poisonBuffer, offset, bytesPerRow }, { texture: work, mipLevel: level }, [ w, h ] );
		variant.encode( encoder, work, config );
		for ( const { level, w, h, bytesPerRow, offset } of layout ) encoder.copyTextureToBuffer( { texture: work, mipLevel: level }, { buffer: readback, offset: slot * frameBytes + offset, bytesPerRow }, [ w, h ] );

	};

	const read = async ( buffer, n ) => {

		await buffer.mapAsync( GPUMapMode.READ );
		const copy = new Uint8Array( buffer.getMappedRange() ).slice();
		buffer.unmap(); buffer.destroy();
		return Array.from( { length: n }, ( _, i ) => copy.subarray( i * frameBytes, ( i + 1 ) * frameBytes ) );

	};

	// Texel bytes only (row padding is never written).
	const differences = ( x, y ) => {

		let n = 0;
		for ( const { w, h, bytesPerRow, offset } of layout ) for ( let r = 0; r < h; r ++ ) for ( let j = 0; j < w * bytes; j ++ ) if ( x[ offset + r * bytesPerRow + j ] !== y[ offset + r * bytesPerRow + j ] ) n ++;
		return n;

	};

	// Isolated references: each input alone, three times with the queue idle in between;
	// the reference is the byte-wise majority, and any run that differs from it is recorded.
	const references = [], isolatedDiffer = [];
	for ( let k = 0; k < 4; k ++ ) {

		const runs = [];
		for ( let r = 0; r < 3; r ++ ) {

			const buffer = device.createBuffer( { size: frameBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
			const encoder = device.createCommandEncoder();
			record( encoder, k, buffer, 0 );
			device.queue.submit( [ encoder.finish() ] );
			runs.push( ( await read( buffer, 1 ) )[ 0 ] );
			await device.queue.onSubmittedWorkDone();

		}

		const majority = runs[ 0 ].map( ( v, i ) => v === runs[ 1 ][ i ] || v === runs[ 2 ][ i ] ? v : runs[ 1 ][ i ] );
		runs.forEach( ( run, r ) => {

			const n = differences( run, majority );
			if ( n ) isolatedDiffer.push( { input: k, run: r, bytes: n } );

		} );
		references.push( majority );

	}

	// Frames in one encoder per chunk of 32, or one submit per frame (as an app records a
	// frame), read back per chunk.
	const result = { frames, levels: `${ first }-${ mips }`, isolatedDiffer, isolatedRuns: 12 };
	for ( const mode of [ 'encoder', 'submits' ] ) {

		let badFrames = 0, badBytes = 0, state = 7;
		const chunk = 32;
		for ( let start = 0; start < frames; start += chunk ) {

			const n = Math.min( chunk, frames - start );
			const buffer = device.createBuffer( { size: n * frameBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
			const ks = [];
			let encoder = device.createCommandEncoder();
			for ( let i = 0; i < n; i ++ ) {

				state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
				ks.push( ( state >>> 24 ) % 4 );
				record( encoder, ks[ i ], buffer, i );
				if ( mode === 'submits' ) { device.queue.submit( [ encoder.finish() ] ); encoder = device.createCommandEncoder(); }

			}

			device.queue.submit( [ encoder.finish() ] );
			( await read( buffer, n ) ).forEach( ( frame, i ) => {

				const bad = differences( frame, references[ ks[ i ] ] );
				if ( bad ) { badFrames ++; badBytes += bad; }

			} );

		}

		result[ mode ] = { badFrames, badBytes };

	}

	for ( const t of [ ...sources, work ] ) t.destroy();
	poisonBuffer.destroy();
	return result;

}

/** Texels of `level` smaller (max) or larger (min) than a base texel in their normalized footprint. */
function violations( base, width, height, level, w, h, reduction ) {

	if ( reduction === 'max' ) return maxViolations( base, width, height, level, w, h );
	const neg = a => Float32Array.from( a, v => - v );
	return maxViolations( neg( base ), width, height, neg( level ), w, h );

}

/** Targeted probes from reading the source (results/compare/webgpu-spd.md, "Findings"). */
async function probes( device, variants, log ) {

	const out = {};
	const { wspd, wspd6 } = variants;

	// 1. Separate target, as the README's own example sizes it (mipLevelCount = maxMipLevelCount − 1):
	//    numMips defaults to target.mipLevelCount, and preparePass generates numMips − 1 levels.
	{

		const W = 1024, H = 1024;
		const source = createTexture( device, { width: W, height: H, format: 'r32float' } );
		const target = device.createTexture( { size: [ W / 2, H / 2 ], format: 'rg32float', mipLevelCount: 10, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		upload( device, source, makeData( W, H, 1, 'r32float', 'random', 5, { unit: true } ) );
		poison( device, target, 0 );
		const run = config => { const e = device.createCommandEncoder(); wspd.encode( e, source, { target, filter: SPDFilters.MinMax, ...config } ); device.queue.submit( [ e.finish() ] ); };
		run( {} );
		const last = ( await readMips( device, target, 9, 10 ) )[ 0 ];
		const p = poisonTexel( 'rg32float' );
		out.targetDefault = { lastLevelUnwritten: last[ 0 ] === p[ 0 ] && last[ 1 ] === p[ 1 ], value: Array.from( last.slice( 0, 2 ) ) };
		poison( device, target, 0 );
		const fresh = webgpuSpdVariant( device );
		{ const e = device.createCommandEncoder(); fresh.encode( e, source, { target, filter: SPDFilters.MinMax, numMips: 11 } ); device.queue.submit( [ e.finish() ] ); }
		const fixed = ( await readMips( device, target, 9, 10 ) )[ 0 ];
		out.targetNumMips11 = { lastLevelUnwritten: fixed[ 0 ] === p[ 0 ] && fixed[ 1 ] === p[ 1 ], value: Array.from( fixed.slice( 0, 2 ) ) };
		// MinMax on a single-channel source: max = max(r, g = 0) on the first level, so negative inputs read 0.
		upload( device, source, Float32Array.from( { length: W * H }, ( _, i ) => - 1 - ( i % 7 ) ) );
		{ const e = device.createCommandEncoder(); fresh.encode( e, source, { target, filter: SPDFilters.MinMax, numMips: 11 } ); device.queue.submit( [ e.finish() ] ); }
		const neg = ( await readMips( device, target, 0, 1 ) )[ 0 ];
		out.minMaxNegative = { inputRange: [ - 7, - 1 ], level1Max: Math.max( ...Array.from( { length: 64 }, ( _, i ) => neg[ i * 4 + 1 ] ) ) };
		source.destroy(); target.destroy();
		log( 'probe: separate target' );

	}

	// 2. halfPrecision: sanitizeScalarType picks f16 only when 'shader-f16' is NOT enabled
	//    (index.ts:818), and the f16 shader declares spd_intermediate_bg but uses _ba (157, 216, 227).
	{

		const W = 512, H = 512;
		const a = createTexture( device, { width: W, height: H, format: 'rgba16float' } ), b = createTexture( device, { width: W, height: H, format: 'rgba16float' } );
		const data = makeData( W, H, 1, 'rgba16float', 'random', 9 );
		upload( device, a, data ); upload( device, b, data );
		device.pushErrorScope( 'validation' );
		const warnings = [];
		const warn = console.warn; console.warn = ( ...args ) => warnings.push( args.join( ' ' ) );
		let error = null;
		try {

			const e = device.createCommandEncoder();
			wspd6.encode( e, a, {} ); wspd6.encode( e, b, { halfPrecision: true } );
			device.queue.submit( [ e.finish() ] );

		} finally { console.warn = warn; error = await device.popErrorScope(); }

		const [ ma, mb ] = [ await readMips( device, a, 1 ), await readMips( device, b, 1 ) ];
		out.halfPrecisionWithF16 = { deviceHasShaderF16: device.features.has( 'shader-f16' ), validationError: error?.message ?? null, warnings, bitIdenticalToF32: ma.every( ( l, i ) => l.every( ( v, j ) => Object.is( v, mb[ i ][ j ] ) ) ) };
		a.destroy(); b.destroy();

		// A device without shader-f16: the library now emits `enable f16;`.
		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
		const plain = await adapter.requestDevice();
		const v = webgpuSpdVariant( plain, { mipsPerPass: 6 } );
		const t = createTexture( plain, { width: W, height: H, format: 'rgba16float' } );
		plain.pushErrorScope( 'validation' );
		const warn2 = console.warn; const w2 = []; console.warn = ( ...args ) => w2.push( args.join( ' ' ) );
		let thrown = null;
		try { const e = plain.createCommandEncoder(); v.encode( e, t, { halfPrecision: true } ); plain.queue.submit( [ e.finish() ] ); } catch ( err ) { thrown = String( err.message || err ); } finally { console.warn = warn2; }
		const error2 = await plain.popErrorScope();
		out.halfPrecisionWithoutF16 = { validationError: error2?.message?.slice( 0, 300 ) ?? null, thrown, warnings: w2 };
		plain.destroy();
		log( 'probe: halfPrecision' );

	}

	// 3. The mid-mip buffer (mip 6 for the handoff when the format has no read-write storage)
	//    is shared by every texture of a DevicePipelines and only partly rewritten per dispatch.
	//    a) A strip whose level 7 is a unit step reads mip-6 entries no workgroup of this
	//       dispatch wrote: the result depends on the previous texture.
	//    b) A sub-rectangle update recomputes levels 7+ from the whole 64×64 mip-6 buffer,
	//       but only the rectangle's tiles were rewritten: the rest is the previous texture's.
	if ( wspd.perPass > 6 ) {

		const fill = ( t, value ) => upload( device, t, makeData( t.width, t.height, 1, 'rgba8unorm', 'constant', 1 ).map( () => value ) );
		const big = createTexture( device, { width: 4096, height: 4096, format: 'rgba8unorm' } );
		const strip = createTexture( device, { width: 2048, height: 64, format: 'rgba8unorm' } );
		// Level 7 texel 0 reads mip-6 (0..1, 0..1): row 0 is this dispatch's own (128), row 1 no
		// workgroup writes. With row 0 visible: (128 + 128 + r1 + r1) / 4, so 64 after black, 191
		// after white. 32 trials, alternating, in one encoder and in separate submits.
		const observed = {};
		for ( const separate of [ false, true ] ) for ( let trial = 0; trial < 32; trial ++ ) {

			const value = trial % 2 ? 255 : 0;
			fill( big, value ); fill( strip, 128 );
			if ( separate ) {

				{ const e = device.createCommandEncoder(); wspd.encode( e, big ); device.queue.submit( [ e.finish() ] ); }
				await device.queue.onSubmittedWorkDone();
				{ const e = device.createCommandEncoder(); wspd.encode( e, strip ); device.queue.submit( [ e.finish() ] ); }

			} else {

				const e = device.createCommandEncoder(); wspd.encode( e, big ); wspd.encode( e, strip ); device.queue.submit( [ e.finish() ] );

			}

			const level7 = ( await readMips( device, strip, 7, 8 ) )[ 0 ];
			const key = `${ separate ? 'separate submits' : 'one encoder' }, after ${ value ? 'white' : 'black' }`;
			observed[ key ] ??= {};
			for ( let x = 0; x < level7.length / 4; x ++ ) {

				const v = Math.round( level7[ x * 4 ] * 255 );
				observed[ key ][ v ] = ( observed[ key ][ v ] ?? 0 ) + 1;

			}

		}
		out.stripAfterOtherTexture = { size: '2048x64', level: '7 (16x1; histogram of red over 16 texels x 16 trials)', expectedIfOwnRowVisible: { black: 64, white: 191 }, observed };

		const A = createTexture( device, { width: 1024, height: 1024, format: 'rgba8unorm' } ), B = createTexture( device, { width: 1024, height: 1024, format: 'rgba8unorm' } );
		fill( B, 255 ); fill( A, 0 );
		{ const e = device.createCommandEncoder(); wspd.encode( e, B ); wspd.encode( e, A ); wspd.encode( e, B, { offset: [ 0, 0 ], size: [ 512, 512 ] } ); device.queue.submit( [ e.finish() ] ); }
		const level7 = ( await readMips( device, B, 7, 8 ) )[ 0 ];
		let black = 0;
		for ( let i = 0; i < level7.length; i += 4 ) if ( level7[ i ] < 0.5 ) black ++;
		out.subRectAfterOtherTexture = { size: '1024x1024, rect [0,0]+[512,512]', level: 7, texels: level7.length / 4, fromOtherTexture: black, note: 'B is white everywhere; A (black) ran in between' };
		for ( const t of [ big, strip, A, B ] ) t.destroy();
		log( 'probe: mid-mip buffer' );

	}

	// 4. Above 4096²: mip 6 is wider than the 64×64 tile the last workgroup reduces.
	if ( wspd.perPass > 6 && device.limits.maxTextureDimension2D >= 8192 ) {

		const W = 8192;
		const a = createTexture( device, { width: W, height: W, format: 'rgba8unorm' } ), b = createTexture( device, { width: W, height: W, format: 'rgba8unorm' } );
		const data = new Uint8Array( W * W * 4 );
		for ( let i = 0; i < data.length; i ++ ) data[ i ] = Math.imul( i, 2654435761 ) >>> 24;
		upload( device, a, data ); upload( device, b, data );
		poison( device, a, 7 ); poison( device, b, 7 );
		const e = device.createCommandEncoder(); wspd.encode( e, a ); wspd6.encode( e, b ); device.queue.submit( [ e.finish() ] );
		const [ la, lb ] = [ await readMips( device, a, 7 ), await readMips( device, b, 7 ) ];
		const p = poisonTexel( 'rgba8unorm' );
		out.over4096 = la.map( ( level, i ) => {

			let poisoned = 0, differs = 0;
			for ( let t = 0; t < level.length; t += 4 ) {

				if ( p.every( ( v, c ) => level[ t + c ] === v ) ) poisoned ++;
				if ( Math.abs( level[ t ] - lb[ i ][ t ] ) > 2.01 / 255 ) differs ++;

			}
			return { level: 7 + i, size: `${ size( W, 7 + i ) }²`, texels: level.length / 4, unwritten: poisoned, differFromWspd6: differs };

		} );
		a.destroy(); b.destroy();
		log( 'probe: 8192²' );

	}

	return out;

}

export async function checkWebgpuSpd( device, { sizes = compareSpdSizes, formats = [ 'rgba8unorm', 'rgba16float' ], frames = 64, onProgress = () => {} } = {} ) {

	const variants = webgpuSpdVariants( device );
	const ours = new Downsampler( device );
	const log = message => onProgress( message );
	const report = { variants: Object.fromEntries( Object.entries( variants ).map( ( [ k, v ] ) => [ k, { mipsPerPass: v.perPass, subgroups: v.subgroups } ] ) ), average: [], hiz: [], stress: [], probes: null };
	device.pushErrorScope( 'validation' );

	try {

		// Averages: every variant vs the SPD oracle; unit-step levels vs zero models and ours.
		for ( const format of formats ) for ( const [ width, height ] of sizes ) {

			const quantization = quantizations[ format ];
			const data = makeData( width, height, 1, format, 'random', 21 );
			const mine = createTexture( device, { width, height, format } );
			upload( device, mine, data );
			{ const e = device.createCommandEncoder(); ours.encode( e, mine, { oddSizes: 'spd' } ); device.queue.submit( [ e.finish() ] ); }
			const oursLevels = await readMips( device, mine, 1 );
			const base = ( await readMips( device, mine, 0, 1 ) )[ 0 ];
			const row = { width, height, format, mips: mine.mipLevelCount - 1, firstUnitStep: firstUnitStep( width, height, mine.mipLevelCount - 1 ), variants: {} };

			for ( const [ name, variant ] of Object.entries( variants ) ) {

				const t = createTexture( device, { width, height, format } );
				upload( device, t, data );
				poison( device, t, 1 );
				const e = device.createCommandEncoder(); variant.encode( e, t ); device.queue.submit( [ e.finish() ] );
				const { result, levels } = await checkChain( device, variant, t, base, format, 'average', oursLevels );
				// Bit-identical to our 'spd' on the in-bounds levels?
				let same = 0, total = 0, worst = 0;
				for ( let l = 1; l < Math.min( row.firstUnitStep, row.mips + 1 ); l ++ ) for ( let i = 0; i < levels[ l - 1 ].length; i ++ ) {

					total ++;
					if ( Object.is( levels[ l - 1 ][ i ], oursLevels[ l - 1 ][ i ] ) ) same ++;
					worst = Math.max( worst, Math.abs( levels[ l - 1 ][ i ] - oursLevels[ l - 1 ][ i ] ) );

				}
				result.vsOursSpd = { identical: same, values: total, worst: quantization === 'unorm8' ? worst * 255 : worst };
				result.dispatches = variant.pass( t ).numPasses;
				row.variants[ name ] = result;
				t.destroy();

			}

			mine.destroy();
			report.average.push( row );
			log( `average ${ width }x${ height } ${ format }` );

		}

		// Hi-Z: r32float min and max, in place. Conservativeness against the base level.
		for ( const [ width, height ] of sizes ) {

			const data = makeData( width, height, 1, 'r32float', 'random', 31, { unit: true } );
			const base = new Float32Array( width * height * 4 );
			for ( let i = 0; i < width * height; i ++ ) base.set( [ data[ i ], 0, 0, 1 ], i * 4 );
			const mips = createTexture( device, { width, height, format: 'r32float' } ).mipLevelCount - 1;
			const row = { width, height, mips, firstUnitStep: firstUnitStep( width, height, mips ), reductions: {} };
			for ( const reduction of [ 'max', 'min' ] ) {

				const entry = {};
				const run = async ( name, encode, variant ) => {

					const t = createTexture( device, { width, height, format: 'r32float' } );
					upload( device, t, data ); poison( device, t, 1 );
					const e = device.createCommandEncoder(); encode( e, t ); device.queue.submit( [ e.finish() ] );
					const levels = variant ? ( await checkChain( device, variant, t, base, 'r32float', reduction, null ) ) : { levels: await readMips( device, t, 1 ) };
					let bad = 0, total = 0, worstLevel = 0;
					levels.levels.forEach( ( level, i ) => {

						const v = violations( base, width, height, level, size( width, i + 1 ), size( height, i + 1 ), reduction );
						bad += v; total += level.length / 4;
						if ( v ) worstLevel = Math.max( worstLevel, v / ( level.length / 4 ) );

					} );
					entry[ name ] = { violations: bad, texels: total, worstLevelFraction: worstLevel, ...( levels.result ? { oracle: levels.result.inBounds, unit: levels.result.unit } : {} ) };
					t.destroy();

				};

				await run( 'exact', ( e, t ) => ours.encode( e, t, { reduction } ) );
				for ( const [ name, variant ] of Object.entries( variants ) ) await run( name, ( e, t ) => variant.encode( e, t, { filter: reduction === 'max' ? SPDFilters.Max : SPDFilters.Min } ), variant );
				row.reductions[ reduction ] = entry;

			}

			report.hiz.push( row );
			log( `hi-z ${ width }x${ height }` );

		}

		// Missing / stale levels over back-to-back frames.
		for ( const format of [ ...formats, 'r32float' ] ) for ( const [ width, height ] of sizes ) {

			const config = format === 'r32float' ? { filter: SPDFilters.Max } : {};
			const row = { width, height, format, variants: {} };
			for ( const [ name, variant ] of Object.entries( variants ) ) row.variants[ name ] = await raceStress( device, variant, width, height, format, frames, config );
			report.stress.push( row );
			log( `stress ${ width }x${ height } ${ format }` );

		}

		report.probes = await probes( device, variants, log );

	} finally {

		const error = await device.popErrorScope();
		if ( error ) report.validationError = error.message;

	}

	return report;

}
