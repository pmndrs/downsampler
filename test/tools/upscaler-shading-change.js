// pmndrs/upscaler's fused shading-change pass ('upscaler/shading-change'):
// one 8×8 workgroup per 16×16 render tile, block sums at 4×4 and 8×8, block
// memory, luma history. Dispatched as UpscalerCore does: ⌈⌈w/2⌉/8⌉ × ⌈⌈h/2⌉/8⌉.
import { SHADING_CHANGE_SHADER } from './upscaler-wgsl.js';
import { compile, createTexture, readTexture, uniformBuffer } from './gpu.js';
import { f16, f32, fromHalfBits, halfUlp, toHalfBits } from './half.js';
import { exposureOracle, previousExposure, upscalerConstants } from './upscaler-exposure.js';

export function memorySize( width, height ) {

	return [ Math.max( 1, Math.ceil( width / 4 ) ), Math.max( 1, Math.ceil( height / 4 ) + Math.ceil( height / 8 ) ) ];

}

/**
 * A plausible block memory: every 4×4 and 8×8 block holds 8 means near its
 * previous-frame mean (±1-4 %), slot 0 newest. Exercises the memory range and
 * jump gate instead of the all-zero restart path.
 */
export function prefillMemory( scene ) {

	const { width, height, lumaHistory } = scene;
	const [ mw, mh ] = memorySize( width, height );
	const words = new Uint32Array( mw * mh * 4 );
	const midRows = Math.ceil( height / 4 );
	const fill = ( bx, by, size, row ) => {

		let sum = 0;
		for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) sum += lumaHistory[ Math.min( by * size + y, height - 1 ) * width + Math.min( bx * size + x, width - 1 ) ];
		const mean = sum / ( size * size ), slots = [];
		for ( let k = 0; k < 8; k ++ ) slots.push( toHalfBits( Math.min( mean * ( 1 + ( k % 2 ? - 0.01 : 0.01 ) * ( 1 + ( k >> 1 ) ) ), 65000 ) ) );
		const i = ( row * mw + bx ) * 4;
		for ( let w = 0; w < 4; w ++ ) words[ i + w ] = ( slots[ 2 * w ] | ( slots[ 2 * w + 1 ] << 16 ) ) >>> 0;

	};

	for ( let by = 0; by < midRows; by ++ ) for ( let bx = 0; bx < Math.ceil( width / 4 ); bx ++ ) fill( bx, by, 4, by );
	for ( let by = 0; by < Math.ceil( height / 8 ); by ++ ) for ( let bx = 0; bx < Math.ceil( width / 8 ); bx ++ ) fill( bx, by, 8, midRows + by );
	return words;

}

/** The exposure texel this frame's exposure pass leaves (oracle value, float16). */
function currentExposure( scene ) {

	return exposureOracle( scene ).value.map( f16 );

}

export async function createUpscalerShadingChange( device, scene, colour ) {

	const { width, height } = scene;
	const module = await compile( device, 'upscaler shading change', SHADING_CHANGE_SHADER );
	const pipeline = await device.createComputePipelineAsync( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const [ mw, mh ] = memorySize( width, height );
	const outputSize = [ Math.max( 1, Math.ceil( width / 2 ) ), Math.max( 1, Math.ceil( height / 2 ) ) ];
	const memoryIn = prefillMemory( scene );
	const resources = {
		constants: uniformBuffer( device, 'FsrConstants', upscalerConstants( scene ) ),
		lumaHistoryIn: createTexture( device, { label: 'shading-luma-in', width, height, format: 'r32float', data: scene.lumaHistory } ),
		motion: createTexture( device, { label: 'dilated-motion', width, height, format: 'rgba16float', data: scene.upscalerMotion } ),
		exposure: createTexture( device, { label: 'exposure', width: 1, height: 1, format: 'rgba16float', data: currentExposure( scene ) } ),
		exposurePrev: createTexture( device, { label: 'exposure prev', width: 1, height: 1, format: 'rgba16float', data: previousExposure } ),
		lumaHistoryOut: createTexture( device, { label: 'shading-luma-out', width, height, format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING } ),
		signal: createTexture( device, { label: 'shading-signal', width: outputSize[ 0 ], height: outputSize[ 1 ], format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING } ),
		masks: device.createTexture( { label: 'masks', size: [ width, height ], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } ),
		memoryIn: createTexture( device, { label: 'shading-memory-in', width: mw, height: mh, format: 'rgba32uint', data: memoryIn } ),
		memoryOut: createTexture( device, { label: 'shading-memory-out', width: mw, height: mh, format: 'rgba32uint', usage: GPUTextureUsage.STORAGE_BINDING } )
	};
	device.queue.writeTexture( { texture: resources.masks }, scene.masks, { bytesPerRow: width * 4, rowsPerImage: height }, [ width, height ] );
	const group = device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [
		{ buffer: resources.constants }, colour.createView(), resources.lumaHistoryIn.createView(), resources.motion.createView(),
		resources.exposure.createView(), resources.exposurePrev.createView(), resources.lumaHistoryOut.createView(), resources.signal.createView(),
		resources.masks.createView(), resources.memoryIn.createView(), resources.memoryOut.createView()
	].map( ( resource, binding ) => ( { binding, resource } ) ) } );
	const groups = [ Math.ceil( Math.max( 1, Math.ceil( width / 2 ) ) / 8 ), Math.ceil( Math.max( 1, Math.ceil( height / 2 ) ) / 8 ) ];

	return {
		name: 'upscaler/shading-change',
		pass: 'shading',
		dispatches: 1,
		encodePass( pass ) { pass.setPipeline( pipeline ); pass.setBindGroup( 0, group ); pass.dispatchWorkgroups( ...groups ); },
		async check() {

			const output = {
				signal: await readTexture( device, resources.signal ),
				lumaHistory: await readTexture( device, resources.lumaHistoryOut ),
				memory: await readTexture( device, resources.memoryOut )
			};
			return checkUpscalerShading( scene, output, memoryIn );

		},
		destroy() { for ( const resource of Object.values( resources ) ) resource.destroy(); }
	};

}

// ---------------------------------------------------------------------------
// CPU oracle: shadingChange.ts transcribed to float32 JavaScript, workgroup by
// workgroup (threads past the image still feed the block sums, clamped).

const SHADING_FLOOR_MID = f32( 0.08 ), SHADING_FLOOR_COARSE = f32( 0.04 ), SHADING_FLOOR_CV = f32( 0.35 );
const SHADING_MEMORY_SLOTS = 8, SHADING_MEMORY_JUMP_GAIN = f32( 1.5 ), SHADING_MEMORY_STILL_PX = f32( 0.05 );

const mix = ( a, b, t ) => f32( f32( a * f32( 1 - t ) ) + f32( b * t ) );
const clamp = ( x, lo, hi ) => Math.min( Math.max( x, lo ), hi );

function relativeDistance( a, b ) {

	const maximum = Math.max( a, b );
	if ( maximum <= f32( 1e-8 ) ) return 0;
	return f32( 1 - f32( Math.min( a, b ) / maximum ) );

}

const slot = ( words, index ) => fromHalfBits( ( words[ index >> 1 ] >>> ( ( index & 1 ) * 16 ) ) & 0xffff );

function memoryDistance( words, current ) {

	let lo = 3.0e38, hi = - 3.0e38, flicker = 0, newer = slot( words, 0 );
	for ( let i = 0; i < SHADING_MEMORY_SLOTS; i ++ ) {

		const value = slot( words, i );
		lo = Math.min( lo, value ); hi = Math.max( hi, value );
		flicker = Math.max( flicker, relativeDistance( newer, value ) );
		newer = value;

	}

	const jump = relativeDistance( current, slot( words, 0 ) );
	return { value: jump > f32( SHADING_MEMORY_JUMP_GAIN * flicker ) ? 1 : relativeDistance( current, clamp( current, lo, hi ) ), margin: Math.abs( jump - SHADING_MEMORY_JUMP_GAIN * flicker ) };

}

function memoryUpdate( words, current, restart ) {

	const bits = toHalfBits( Math.min( current, 65000 ) );
	if ( restart || ( words[ 0 ] === 0 && words[ 1 ] === 0 && words[ 2 ] === 0 && words[ 3 ] === 0 ) ) {

		const both = ( bits | ( bits << 16 ) ) >>> 0;
		return [ both, both, both, both ];

	}

	return [
		( ( words[ 0 ] << 16 ) | bits ) >>> 0,
		( ( words[ 1 ] << 16 ) | ( words[ 0 ] >>> 16 ) ) >>> 0,
		( ( words[ 2 ] << 16 ) | ( words[ 1 ] >>> 16 ) ) >>> 0,
		( ( words[ 3 ] << 16 ) | ( words[ 2 ] >>> 16 ) ) >>> 0
	];

}

function smoothstep( e0, e1, x ) {

	const t = clamp( f32( f32( x - e0 ) / f32( e1 - e0 ) ), 0, 1 );
	return f32( f32( t * t ) * f32( 3 - 2 * t ) );

}

function scaleResponse( sums, count, floorBase, memory ) {

	const maximum = Math.max( sums[ 0 ], sums[ 1 ] );
	if ( maximum <= f32( 1e-5 ) ) return 0;
	const mx = f32( sums[ 0 ] / count ), my = f32( sums[ 1 ] / count );
	const vx = Math.max( f32( f32( sums[ 2 ] / count ) - f32( mx * mx ) ), 0 ), vy = Math.max( f32( f32( sums[ 3 ] / count ) - f32( my * my ) ), 0 );
	const cv = f32( f32( Math.sqrt( f32( 0.5 * f32( vx + vy ) ) ) ) / Math.max( f32( 0.5 * f32( mx + my ) ), f32( 1e-4 ) ) );
	const floorValue = f32( floorBase + f32( SHADING_FLOOR_CV * cv ) );
	const relative = Math.min( f32( 1 - f32( Math.min( sums[ 0 ], sums[ 1 ] ) / maximum ) ), memory );
	return smoothstep( floorValue, f32( floorValue * 3 ), relative );

}

export function upscalerShadingOracle( scene, memoryIn ) {

	const { width: W, height: H, colour, lumaHistory, upscalerMotion: motionData, masks, jitter, previousJitter } = scene;
	const [ mw, mh ] = memorySize( W, H );
	const ow = Math.max( 1, Math.ceil( W / 2 ) ), oh = Math.max( 1, Math.ceil( H / 2 ) );
	const signal = new Float32Array( ow * oh ), memory = new Uint32Array( mw * mh * 4 ), written = new Uint8Array( mw * mh );
	const decisions = new Float32Array( ow * oh ).fill( Infinity ); // distance of a discrete decision from its threshold
	const frameInfo = exposureOracle( scene ).value.map( f16 );
	const hostPrev = previousExposure[ 2 ];
	const hostRatio = hostPrev > 1e-4 && frameInfo[ 2 ] > 1e-4 ? f32( frameInfo[ 2 ] / hostPrev ) : 1;
	const conditioning = Math.max( frameInfo[ 0 ], f32( 1e-4 ) );
	const memoryScale = f32( 1 / f32( conditioning * ( frameInfo[ 2 ] > 1e-4 ? frameInfo[ 2 ] : 1 ) ) );
	const inv = [ f32( 1 / W ), f32( 1 / H ) ];
	const jitterDelta = [ f32( jitter[ 0 ] - previousJitter[ 0 ] ), f32( jitter[ 1 ] - previousJitter[ 1 ] ) ];
	const history = ( x, y ) => lumaHistory[ y * W + x ];
	const scale = f32( hostRatio * conditioning );

	function lumaPair( cx, cy, current, motion, disocclusion, out ) {

		const currentSq = f32( current * current );
		const u = f32( f32( cx + 0.5 ) * inv[ 0 ] ), v = f32( f32( cy + 0.5 ) * inv[ 1 ] );
		const mu = f32( u - motion[ 0 ] ), mv = f32( v - motion[ 1 ] );
		if ( mu < 0 || mv < 0 || mu > 1 || mv > 1 ) { out[ 0 ] = f32( out[ 0 ] + current ); out[ 1 ] = f32( out[ 1 ] + current ); out[ 2 ] = f32( out[ 2 ] + currentSq ); out[ 3 ] = f32( out[ 3 ] + currentSq ); return; }
		const pu = f32( mu + f32( jitterDelta[ 0 ] * inv[ 0 ] ) ), pv = f32( mv + f32( jitterDelta[ 1 ] * inv[ 1 ] ) );
		const px = f32( f32( pu * W ) - 0.5 ), py = f32( f32( pv * H ) - 0.5 );
		const bx = Math.floor( px ), by = Math.floor( py );
		const x0 = clamp( bx, 0, W - 1 ), y0 = clamp( by, 0, H - 1 ), x1 = clamp( bx + 1, 0, W - 1 ), y1 = clamp( by + 1, 0, H - 1 );
		const l00 = history( x0, y0 ), l10 = history( x1, y0 ), l01 = history( x0, y1 ), l11 = history( x1, y1 );
		const tapMin = f32( Math.min( l00, l10, l01, l11 ) * scale ), tapMax = f32( Math.max( l00, l10, l01, l11 ) * scale );
		const previousLuma = mix( clamp( current, tapMin, tapMax ), current, disocclusion );
		const fx = f32( px - bx ), fy = f32( py - by );
		const reprojected = f32( mix( mix( l00, l10, fx ), mix( l01, l11, fx ), fy ) * scale );
		const previousSq = mix( f32( reprojected * reprojected ), currentSq, disocclusion );
		out[ 0 ] = f32( out[ 0 ] + current ); out[ 1 ] = f32( out[ 1 ] + previousLuma ); out[ 2 ] = f32( out[ 2 ] + currentSq ); out[ 3 ] = f32( out[ 3 ] + previousSq );

	}

	const groupsX = Math.ceil( ow / 8 ), groupsY = Math.ceil( oh / 8 );
	const tileSums = Array.from( { length: 64 }, () => [ 0, 0, 0, 0 ] ), tileMoving = new Float32Array( 64 );
	const midRows = Math.ceil( H / 4 );

	for ( let gy = 0; gy < groupsY; gy ++ ) for ( let gx = 0; gx < groupsX; gx ++ ) {

		for ( let ly = 0; ly < 8; ly ++ ) for ( let lx = 0; lx < 8; lx ++ ) {

			const sums = tileSums[ ly * 8 + lx ];
			sums.fill( 0 );
			let moving = 0;
			const ox = ( gx * 8 + lx ) * 2, oy = ( gy * 8 + ly ) * 2;
			for ( let y = 0; y < 2; y ++ ) for ( let x = 0; x < 2; x ++ ) {

				const cx = clamp( ox + x, 0, W - 1 ), cy = clamp( oy + y, 0, H - 1 ), i = cy * W + cx;
				const hostLuma = f32( f32( f32( colour[ i * 4 ] * f32( 0.2126 ) ) + f32( colour[ i * 4 + 1 ] * f32( 0.7152 ) ) ) + f32( colour[ i * 4 + 2 ] * f32( 0.0722 ) ) );
				const motion = [ motionData[ i * 4 ], motionData[ i * 4 + 1 ] ];
				const disocclusion = clamp( masks[ i * 4 ] / 255, 0, 1 );
				const still = Math.hypot( motion[ 0 ] * W, motion[ 1 ] * H ) < SHADING_MEMORY_STILL_PX && disocclusion < 0.5;
				moving = Math.max( moving, still ? 0 : 1 );
				lumaPair( cx, cy, f32( hostLuma * conditioning ), motion, disocclusion, sums );

			}

			tileMoving[ ly * 8 + lx ] = moving;

		}

		for ( let ly = 0; ly < 8; ly ++ ) for ( let lx = 0; lx < 8; lx ++ ) {

			const gidX = gx * 8 + lx, gidY = gy * 8 + ly;
			if ( gidX >= ow || gidY >= oh ) continue;
			const block = ( size ) => {

				const bx = Math.floor( lx / size ) * size, by = Math.floor( ly / size ) * size, sums = [ 0, 0, 0, 0 ];
				let moving = 0;
				for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

					const index = ( by + y ) * 8 + bx + x;
					for ( let c = 0; c < 4; c ++ ) sums[ c ] = f32( sums[ c ] + tileSums[ index ][ c ] );
					moving = Math.max( moving, tileMoving[ index ] );

				}

				return { sums, moving };

			};

			const mid = block( 2 ), coarse = block( 4 );
			const midCoord = [ gidX >> 1, gidY >> 1 ], coarseCoord = [ gidX >> 2, ( gidY >> 2 ) + midRows ];
			const wordsAt = ( [ x, y ] ) => memoryIn.subarray( ( y * mw + x ) * 4, ( y * mw + x ) * 4 + 4 );
			const midWords = wordsAt( midCoord ), coarseWords = wordsAt( coarseCoord );
			const midMean = f32( f32( mid.sums[ 0 ] / 16 ) * memoryScale ), coarseMean = f32( f32( coarse.sums[ 0 ] / 64 ) * memoryScale );
			const midDistance = mid.moving > 0.5 ? { value: 1, margin: Infinity } : memoryDistance( midWords, midMean );
			const coarseDistance = coarse.moving > 0.5 ? { value: 1, margin: Infinity } : memoryDistance( coarseWords, coarseMean );
			if ( ( lx & 1 ) === 0 && ( ly & 1 ) === 0 ) { const i = midCoord[ 1 ] * mw + midCoord[ 0 ]; memory.set( memoryUpdate( midWords, midMean, mid.moving > 0.5 ), i * 4 ); written[ i ] = 1; }
			if ( ( lx & 3 ) === 0 && ( ly & 3 ) === 0 ) { const i = coarseCoord[ 1 ] * mw + coarseCoord[ 0 ]; memory.set( memoryUpdate( coarseWords, coarseMean, coarse.moving > 0.5 ), i * 4 ); written[ i ] = 1; }
			const out = gidY * ow + gidX;
			signal[ out ] = Math.max( scaleResponse( mid.sums, 16, SHADING_FLOOR_MID, midDistance.value ), scaleResponse( coarse.sums, 64, SHADING_FLOOR_COARSE, coarseDistance.value ) );
			decisions[ out ] = Math.min( midDistance.margin, coarseDistance.margin );

		}

	}

	return { signal, memory, written, decisions, size: [ ow, oh ], memorySize: [ mw, mh ] };

}

/**
 * Tolerances:
 * - luma history: 2 float32 ulps (a dot product; FMA contraction may differ).
 * - signal: 2e-3 absolute. Block variances are E[x²] − E[x]² in float32,
 *   which cancels on flat blocks: a few ulps of the squared sums become
 *   ~1e-4 of cv, then ×0.35 in the floor, then up to ~19× through
 *   smoothstep's slope (1.5 / (2·floor), floor ≥ 0.04). A texel whose memory
 *   jump gate sat within 1e-4 of its threshold (a discrete flip) is not counted.
 * - memory: one float16 ulp per stored mean (pack2x16float rounding is implementation-defined).
 * At most 1e-4 of the signal texels may fall outside (other discrete decisions:
 * floor() of the reprojection, the off-screen test).
 */
export function checkUpscalerShading( scene, output, memoryIn ) {

	const expected = upscalerShadingOracle( scene, memoryIn );
	const checks = [];
	const [ ow, oh ] = expected.size;

	let worst = 0, outside = 0, worstOutside = 0, flips = 0, firing = 0;
	for ( let i = 0; i < ow * oh; i ++ ) {

		const error = Math.abs( output.signal.data[ i ] - expected.signal[ i ] );
		if ( expected.signal[ i ] > 0.5 ) firing ++;
		if ( error <= 2e-3 ) { worst = Math.max( worst, error ); continue; }
		if ( expected.decisions[ i ] < 1e-4 ) { flips ++; continue; }
		outside ++; worstOutside = Math.max( worstOutside, error );

	}

	const budget = Math.floor( 1e-4 * ow * oh );
	checks.push( { output: 'shading signal (r32float)', texels: ow * oh, maxError: worst, unit: 'absolute', tolerance: 2e-3, outside, budget, worstOutside, gateFlips: flips, firing: firing / ( ow * oh ), ok: outside <= budget } );

	let worstLuma = 0, badLuma = 0;
	for ( let i = 0; i < scene.width * scene.height; i ++ ) {

		const want = scene.hostLuma[ i ], got = output.lumaHistory.data[ i ];
		const error = Math.abs( got - want ) / ( Math.abs( want ) * 2 ** - 23 + 1e-30 );
		worstLuma = Math.max( worstLuma, error );
		if ( error > 2 ) badLuma ++;

	}

	checks.push( { output: 'luma history (r32float)', texels: scene.width * scene.height, maxError: worstLuma, unit: 'f32 ulp', tolerance: 2, outside: badLuma, ok: badLuma === 0 } );

	let worstMemory = 0, badMemory = 0, compared = 0;
	for ( let i = 0; i < expected.written.length; i ++ ) {

		if ( ! expected.written[ i ] ) continue;
		for ( let w = 0; w < 4; w ++ ) for ( const shift of [ 0, 16 ] ) {

			const want = fromHalfBits( ( expected.memory[ i * 4 + w ] >>> shift ) & 0xffff ), got = fromHalfBits( ( output.memory.data[ i * 4 + w ] >>> shift ) & 0xffff );
			const error = Math.abs( got - want ) / halfUlp( want );
			worstMemory = Math.max( worstMemory, error );
			compared ++;
			if ( error > 1 ) badMemory ++;

		}

	}

	// A block mean within one ulp of a float16 rounding point can round either way; the
	// slot then differs by one ulp in this frame and shifts down with the rest of the memory.
	const memoryBudget = Math.floor( 1e-4 * compared );
	checks.push( { output: 'block memory (rgba32uint, 8 × f16)', texels: compared, maxError: worstMemory, unit: 'f16 ulp', tolerance: 1, outside: badMemory, budget: memoryBudget, ok: badMemory <= memoryBudget } );

	return { checks, ok: checks.every( check => check.ok ) };

}
