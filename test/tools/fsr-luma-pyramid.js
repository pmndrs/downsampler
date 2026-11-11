// FSR 3.1 luma pyramid (auto-exposure + FarthestDepthMip1), two fixtures:
// - 'fsr/luma-pyramid':         AMD's pass as shipped, SPD structure (tile + tail dispatch).
// - 'fsr/luma-pyramid-correct': the same load and epilogue over every texel
//                               (weighted sums, ceil-sized mip 1), the M2T.2 target.
//
// Tool code ported from FidelityFX SDK FSR3UPSCALER 3.1.6:
// ffx_fsr3upscaler_luma_pyramid.h (SpdLoadSourceImage, SpdLoad, SpdStore),
// ffx_fsr3upscaler_common.h (constants, ComputeAutoExposureFromLavg) and the
// resource callbacks of ffx_fsr3upscaler_callbacks_hlsl.h.
// Copyright (C) 2026 Advanced Micro Devices, Inc. MIT licence; full notice in ./spd.js.
import { spdConstants, spdReference, spdSetup, spdWgsl } from './spd.js';
import { compile, createTexture, readBuffer, readTexture, uniformBuffer } from './gpu.js';
import { f16, f32, fromHalfBits, halfUlp } from './half.js';

const EPSILON = 6.10e-05;

// AMD clears FrameInfo to (-1, 1, 0, 0) on reset (ffx_fsr3upscaler.cpp, "Clear Frame Info").
export const resetFrameInfo = [ - 1, 1, 0, 0 ];
// A previous log luma of 1e4 or more skips the temporal smoothing, so FrameInfo holds this frame's pyramid result.
export const unsmoothedFrameInfo = [ 0, 1e4, 0, 0 ];

/**
 * The tool half of the pass, in @group(1) (SPEC §14.1): FSR's callbacks as
 * hooks. `load` is SpdLoadSourceImage without its ClampLoad; `store` is
 * SpdStore (level = SPD mip index + 1); `spdLoad` is SpdLoad (mip 6 read back
 * by the tail); `finish` is the scalar epilogue for the every-texel variant.
 */
export function lumaToolParts( { half1 } ) {

	const bindings = /* wgsl */ `
struct LumaConstants {
	mips: u32,          // MipCount(); 0 disables the SPD last-mip epilogue
	deltaTime: f32,     // DeltaTime()
	mip6Size: vec2u,    // dimensions of SPD mip 6 (rw_spd_mip5: max(1, (render / 2) >> 5))
}
@group(1) @binding(0) var<uniform> fsr: LumaConstants;
@group(1) @binding(1) var r_current_luma: texture_2d<f32>;
@group(1) @binding(2) var r_farthest_depth: texture_2d<f32>;
@group(1) @binding(3) var rw_farthest_depth_mip1: texture_storage_2d<${ half1 }, write>;
// rw_spd_mip5 (RG16F) as packed halves: WebGPU can't read and write one storage texture of this format.
@group(1) @binding(4) var<storage, read_write> rw_spd_mip5: array<u32>;
@group(1) @binding(5) var<storage, read_write> rw_frame_info: array<vec4f, 1>;

const FSR3UPSCALER_EPSILON: f32 = 6.10e-05;
const resetAutoExposureAverageSmoothing: f32 = 1e4;
const FRAME_INFO_EXPOSURE = 0;
const FRAME_INFO_LOG_LUMA = 1;
const FRAME_INFO_SCENE_AVERAGE_LUMA = 2;
const LOG_LUMA = 0;
const LUMA = 1;
const DEPTH_IN_METERS = 2;

fn ComputeAutoExposureFromLavg(LavgIn: f32) -> f32 {
	let Lavg = exp(LavgIn);
	let S = 100.0; //ISO arithmetic speed
	let K = 12.5;
	let ExposureISO100 = log2((Lavg * S) / K);
	let q = 0.65;
	let Lmax = (78.0 / (q * S)) * pow(2.0, ExposureISO100);
	return 1.0 / Lmax;
}

`;
	const load = /* wgsl */ `
fn load(texel: vec2i, layer: u32) -> vec4f {
	let fLuma = textureLoad(r_current_luma, texel, 0).r;
	let fLogLuma = max(FSR3UPSCALER_EPSILON, log(fLuma));
	let fFarthestDepthInMeters = textureLoad(r_farthest_depth, texel, 0).r;
	var fOutput = vec4f(0.0);
	fOutput[LOG_LUMA] = fLogLuma;
	fOutput[LUMA] = fLuma;
	fOutput[DEPTH_IN_METERS] = fFarthestDepthInMeters;
	return fOutput;
}

`;
	const helpers = /* wgsl */ `
// Texture writes outside the resource are dropped (D3D UAV semantics), and out-of-bounds reads return 0.
fn StoreFarthestDepthMip1(pix: vec2i, depth: f32) {
	if (all(vec2u(pix) < textureDimensions(rw_farthest_depth_mip1))) { textureStore(rw_farthest_depth_mip1, pix, vec4f(depth, 0.0, 0.0, 0.0)); }
}
fn StorePyramidMip5(pix: vec2i, value: vec2f) {
	if (all(vec2u(pix) < fsr.mip6Size)) { rw_spd_mip5[u32(pix.y) * fsr.mip6Size.x + u32(pix.x)] = pack2x16float(value); }
}
fn spdLoad(tex: vec2i) -> vec4f {
	if (any(vec2u(tex) >= fsr.mip6Size)) { return vec4f(0.0); }
	return vec4f(unpack2x16float(rw_spd_mip5[u32(tex.y) * fsr.mip6Size.x + u32(tex.x)]), 0.0, 0.0);
}

fn updateFrameInfo(outValue: vec4f) {
	var frameInfo = rw_frame_info[0];
	let fSceneAvgLuma = outValue[LUMA];
	let fPrevLogLuma = frameInfo[FRAME_INFO_LOG_LUMA];
	var fLogLuma = outValue[LOG_LUMA];

	if (fPrevLogLuma < resetAutoExposureAverageSmoothing) { // Compare Lavg, so small or negative values
		fLogLuma = fPrevLogLuma + (fLogLuma - fPrevLogLuma) * (1.0 - exp(-fsr.deltaTime));
		fLogLuma = max(0.0, fLogLuma);
	}

	frameInfo[FRAME_INFO_EXPOSURE] = ComputeAutoExposureFromLavg(fLogLuma);
	frameInfo[FRAME_INFO_LOG_LUMA] = fLogLuma;
	frameInfo[FRAME_INFO_SCENE_AVERAGE_LUMA] = fSceneAvgLuma;
	rw_frame_info[0] = frameInfo;
}

`;
	const store = /* wgsl */ `
fn store(level: u32, pix: vec2i, layer: u32, outValue: vec4f) {
	if (level == 6u) {
		StorePyramidMip5(pix, outValue.xy);
	} else if (level == 1u) {
		StoreFarthestDepthMip1(pix, outValue[DEPTH_IN_METERS]);
	}
	if (level == fsr.mips) { //accumulate on 1x1 level
		if (all(pix == vec2i(0, 0))) { updateFrameInfo(outValue); }
	}
}

`;
	const finish = /* wgsl */ `
fn finish(value: vec4f, weight: f32) { updateFrameInfo(value); }
`;
	return { bindings: bindings + helpers, load, store, finish };

}

/** All of the tool half in one string (the SPD port and the every-texel kernel compile it whole). */
export function lumaToolWgsl( formats ) {

	const { bindings, load, store, finish } = lumaToolParts( formats );
	return bindings + load + store + finish;

}

// The every-texel variant: per-tile sums with an in-bounds count, then one
// workgroup combines the tile partials and calls finish(). Mip 1 is ceil-sized
// and averages only its in-bounds texels ('zero-weight' of SPEC §14.1).
const correctWgsl = /* wgsl */ `
struct Reduce { renderSize: vec2u, tiles: vec2u }
@group(0) @binding(0) var<uniform> totals: Reduce;
@group(0) @binding(1) var<storage, read_write> partials: array<vec4f>;

var<workgroup> scratch: array<vec4f, 256>;

fn tree(index: u32) {
	for (var stride = 128u; stride > 0u; stride >>= 1u) {
		if (index < stride) { scratch[index] += scratch[index + stride]; }
		workgroupBarrier();
	}
}

// SPD's access pattern: each thread owns mip-1 texels (x, y), (x + 16, y),
// (x, y + 16), (x + 16, y + 16) of the tile, so neighbouring threads read
// neighbouring source texels.
@compute @workgroup_size(256)
fn tile(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) index: u32) {
	let xy = vec2u(index % 16u, index / 16u);
	var sum = vec4f(0.0);
	for (var quadrant = 0u; quadrant < 4u; quadrant++) {
		let cell = wg.xy * 32u + xy + vec2u(quadrant % 2u, quadrant / 2u) * 16u;
		var block = vec4f(0.0);
		var count = 0.0;
		for (var dy = 0u; dy < 2u; dy++) {
			for (var dx = 0u; dx < 2u; dx++) {
				let texel = cell * 2u + vec2u(dx, dy);
				if (all(texel < totals.renderSize)) { block += load(vec2i(texel), 0u); count += 1.0; }
			}
		}
		if (count > 0.0) { store(1u, vec2i(cell), 0u, block / count); }
		sum += vec4f(block.xyz, count);
	}
	scratch[index] = sum;
	workgroupBarrier();
	tree(index);
	if (index == 0u) { partials[wg.y * totals.tiles.x + wg.x] = scratch[0]; }
}

@compute @workgroup_size(256)
fn combine(@builtin(local_invocation_index) index: u32) {
	var sum = vec4f(0.0);
	for (var i = index; i < totals.tiles.x * totals.tiles.y; i += 256u) { sum += partials[i]; }
	scratch[index] = sum;
	workgroupBarrier();
	tree(index);
	if (index == 0u) { finish(vec4f(scratch[0].xyz / scratch[0].w, 0.0), scratch[0].w); }
}
`;

function toolEntries( half1 ) {

	return [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: half1 } },
		{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		{ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
	];

}

/** Inputs shared by both variants: current luma and farthest depth (R16F). */
export function createLumaInputs( device, scene ) {

	const { width, height } = scene;
	return {
		currentLuma: createTexture( device, { label: 'r_current_luma', width, height, format: 'r16float', data: scene.currentLuma } ),
		farthestDepth: createTexture( device, { label: 'r_farthest_depth', width, height, format: 'r16float', data: scene.farthestDepth } )
	};

}

/**
 * The FSR luma tool as SPEC §14.1 hands it to a generator: WGSL for
 * @group(1), its layout and bind group, and the outputs. 'amd' sizes mip 1
 * ⌊w/2⌋ and enables the last-mip epilogue; 'correct' sizes it ⌈w/2⌉ and
 * leaves the scalar to `finish`. This is what a generator-based fixture plugs in.
 */
export function createLumaTool( device, scene, inputs, formats, { variant = 'amd', frameInfo = unsmoothedFrameInfo } = {} ) {

	const { width, height } = scene;
	const setup = spdSetup( width, height );
	const correct = variant === 'correct';
	const mip1 = correct ? [ Math.ceil( width / 2 ), Math.ceil( height / 2 ) ] : [ Math.floor( width / 2 ), Math.floor( height / 2 ) ];
	const mip6 = [ Math.max( 1, Math.floor( width / 2 ) >> 5 ), Math.max( 1, Math.floor( height / 2 ) >> 5 ) ];
	const depthMip1 = createTexture( device, { label: 'rw_farthest_depth_mip1', width: mip1[ 0 ], height: mip1[ 1 ], format: formats.half1, usage: GPUTextureUsage.STORAGE_BINDING } );
	const mip5 = device.createBuffer( { label: 'rw_spd_mip5', size: mip6[ 0 ] * mip6[ 1 ] * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
	const frameInfoBuffer = device.createBuffer( { label: 'rw_frame_info', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST } );
	const constants = new ArrayBuffer( 16 );
	new Uint32Array( constants, 0, 1 )[ 0 ] = correct ? 0 : setup.mips;
	new Float32Array( constants, 4, 1 )[ 0 ] = scene.deltaTime;
	new Uint32Array( constants, 8, 2 ).set( mip6 );
	const uniforms = uniformBuffer( device, 'fsr luma constants', constants );
	const entries = toolEntries( formats.half1 );
	const layout = device.createBindGroupLayout( { label: 'fsr luma tool (group 1)', entries } );
	const groupEntries = [
		{ binding: 0, resource: { buffer: uniforms } },
		{ binding: 1, resource: inputs.currentLuma.createView() },
		{ binding: 2, resource: inputs.farthestDepth.createView() },
		{ binding: 3, resource: depthMip1.createView() },
		{ binding: 4, resource: { buffer: mip5 } },
		{ binding: 5, resource: { buffer: frameInfoBuffer } }
	];
	const group = device.createBindGroup( { layout, entries: groupEntries } );
	const reset = ( state = frameInfo ) => device.queue.writeBuffer( frameInfoBuffer, 0, new Float32Array( state ) );
	reset();

	return {
		wgsl: lumaToolWgsl( formats ), layout, group, variant, mips: setup.mips, mip6Size: mip6, reset,
		// For generators that compose their own group 1 (SPEC §14.1): the hooks
		// separately, and the layout and bind group entries to extend.
		parts: lumaToolParts( formats ), entries, groupEntries,
		async read() {

			return {
				frameInfo: Array.from( new Float32Array( await readBuffer( device, frameInfoBuffer ) ) ),
				depthMip1: await readTexture( device, depthMip1 ),
				mip6: correct ? null : new Uint32Array( await readBuffer( device, mip5 ) )
			};

		},
		/** Compares the last run with the oracle; `state` is the FrameInfo the run started from. */
		async check( state = frameInfo ) { return checkLuma( scene, await this.read(), { variant, frameInfo: state } ); },
		destroy() { for ( const resource of [ depthMip1, mip5, frameInfoBuffer, uniforms ] ) resource.destroy(); }
	};

}

/** @returns {Promise<import('./harness.js').Fixture>} */
export async function createFsrLumaPyramid( device, scene, inputs, formats, options = {} ) {

	const { width, height } = scene;
	const setup = spdSetup( width, height );
	const tool = createLumaTool( device, scene, inputs, formats, options );
	const correct = tool.variant === 'correct';
	const module = await compile( device, `fsr luma pyramid (${ tool.variant })`, ( correct ? correctWgsl : spdWgsl ) + tool.wgsl );

	let layout0, entries0, stages;
	const owned = [];
	if ( correct ) {

		const tiles = setup.groups;
		const partials = device.createBuffer( { label: 'luma partials', size: tiles[ 0 ] * tiles[ 1 ] * 16, usage: GPUBufferUsage.STORAGE } );
		const uniforms = uniformBuffer( device, 'reduce constants', new Uint32Array( [ width, height, ...tiles ] ) );
		owned.push( partials, uniforms );
		layout0 = device.createBindGroupLayout( { entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
		] } );
		entries0 = [ { binding: 0, resource: { buffer: uniforms } }, { binding: 1, resource: { buffer: partials } } ];
		stages = [ [ 'tile', tiles ], [ 'combine', [ 1, 1 ] ] ];

	} else {

		const uniforms = uniformBuffer( device, 'cbSPD', spdConstants( setup, width, height ) );
		owned.push( uniforms );
		layout0 = device.createBindGroupLayout( { entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } ] } );
		entries0 = [ { binding: 0, resource: { buffer: uniforms } } ];
		stages = [ [ 'spdTile', setup.groups ], [ 'spdTail', [ 1, 1 ] ] ];

	}

	const group0 = device.createBindGroup( { layout: layout0, entries: entries0 } );
	const pipelineLayout = device.createPipelineLayout( { bindGroupLayouts: [ layout0, tool.layout ] } );
	const pipelines = await Promise.all( stages.map( ( [ entryPoint ] ) => device.createComputePipelineAsync( { layout: pipelineLayout, compute: { module, entryPoint } } ) ) );

	return {
		name: correct ? 'fsr/luma-pyramid-correct' : 'fsr/luma-pyramid',
		pass: 'exposure',
		dispatches: stages.length,
		encodePass( pass ) {

			pass.setBindGroup( 0, group0 );
			pass.setBindGroup( 1, tool.group );
			stages.forEach( ( [ , groups ], i ) => { pass.setPipeline( pipelines[ i ] ); pass.dispatchWorkgroups( groups[ 0 ], groups[ 1 ] ); } );

		},
		reset: () => tool.reset(),
		// The harness's run starts unsmoothed (FrameInfo = this frame's pyramid);
		// a second run from AMD's reset state checks the temporal smoothing.
		async check() {

			const unsmoothed = await tool.check();
			tool.reset( resetFrameInfo );
			const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
			this.encodePass( pass );
			pass.end();
			device.queue.submit( [ encoder.finish() ] );
			const smoothed = await tool.check( resetFrameInfo );
			tool.reset();
			const frameInfoOnly = smoothed.checks.filter( check => check.output.startsWith( 'FrameInfo' ) ).map( check => ( { ...check, output: `${ check.output } (from reset)` } ) );
			return { ...unsmoothed, checks: [ ...unsmoothed.checks, ...frameInfoOnly ], ok: unsmoothed.ok && smoothed.ok };

		},
		destroy() { tool.destroy(); for ( const resource of owned ) resource.destroy(); }
	};

}

// ---------------------------------------------------------------------------
// CPU oracles

function computeAutoExposureFromLavg( Lavg ) {

	const L = Math.exp( Lavg );
	const exposureISO100 = Math.log2( ( L * 100 ) / 12.5 );
	const Lmax = ( 78 / ( 0.65 * 100 ) ) * Math.pow( 2, exposureISO100 );
	return 1 / Lmax;

}

/** The FSR epilogue (SpdStore's last-mip branch) on (log luma, luma). */
export function frameInfoFrom( logLuma, sceneAverageLuma, previous, deltaTime ) {

	const prevLogLuma = previous[ 1 ];
	let L = logLuma;
	if ( prevLogLuma < 1e4 ) L = Math.max( 0, prevLogLuma + ( L - prevLogLuma ) * ( 1 - Math.exp( - deltaTime ) ) );
	return [ computeAutoExposureFromLavg( L ), L, sceneAverageLuma, previous[ 3 ] ];

}

/** The tool's load for every texel: (max(ε, log luma), luma, farthest depth, 0) in float32. */
export function lumaSource( scene ) {

	const count = scene.width * scene.height, source = new Float32Array( count * 4 );
	for ( let i = 0; i < count; i ++ ) {

		const luma = scene.currentLuma[ i ];
		source[ i * 4 ] = Math.max( f32( EPSILON ), f32( Math.log( luma ) ) );
		source[ i * 4 + 1 ] = luma;
		source[ i * 4 + 2 ] = scene.farthestDepth[ i ];

	}

	return source;

}

/**
 * AMD as shipped: mip 1 of depth at ⌊w/2⌋×⌊h/2⌋, mip 6 through RG16F (writes
 * outside the resource dropped, reads outside return 0), and the scalar taken
 * from texel (0, 0) of mip `mips`, which covers only [0, 2^mips)².
 */
export function lumaOracleAmd( scene, previous = resetFrameInfo ) {

	const { width, height, deltaTime } = scene;
	const { mips } = spdSetup( width, height );
	const mip1Size = [ Math.floor( width / 2 ), Math.floor( height / 2 ) ];
	const mip6Size = [ Math.max( 1, mip1Size[ 0 ] >> 5 ), Math.max( 1, mip1Size[ 1 ] >> 5 ) ];
	const depthMip1 = new Float32Array( mip1Size[ 0 ] * mip1Size[ 1 ] );
	const mip6 = new Float32Array( mip6Size[ 0 ] * mip6Size[ 1 ] * 2 );
	let last = null;

	spdReference( {
		width, height, source: lumaSource( scene ), mips,
		store( level, x, y, value ) {

			if ( level === 6 && x < mip6Size[ 0 ] && y < mip6Size[ 1 ] ) { const i = y * mip6Size[ 0 ] + x; mip6[ i * 2 ] = f16( value[ 0 ] ); mip6[ i * 2 + 1 ] = f16( value[ 1 ] ); }
			if ( level === 1 && x < mip1Size[ 0 ] && y < mip1Size[ 1 ] ) depthMip1[ y * mip1Size[ 0 ] + x ] = f16( value[ 2 ] );
			if ( level === mips && x === 0 && y === 0 ) last = value.slice();

		},
		spdLoad( x, y, out ) {

			const inside = x < mip6Size[ 0 ] && y < mip6Size[ 1 ], i = y * mip6Size[ 0 ] + x;
			out[ 0 ] = inside ? mip6[ i * 2 ] : 0; out[ 1 ] = inside ? mip6[ i * 2 + 1 ] : 0; out[ 2 ] = 0; out[ 3 ] = 0;

		}
	} );

	return { mips, mip6, mip6Size, depthMip1, depthMip1Size: mip1Size, logLuma: last[ 0 ], luma: last[ 1 ], frameInfo: frameInfoFrom( last[ 0 ], last[ 1 ], previous, deltaTime ) };

}

/** Every texel, equally weighted; mip 1 ceil-sized over in-bounds texels. */
export function lumaOracleCorrect( scene, previous = resetFrameInfo ) {

	const { width, height, deltaTime } = scene;
	const source = lumaSource( scene );
	let logSum = 0, lumaSum = 0;
	for ( let i = 0; i < width * height; i ++ ) { logSum += source[ i * 4 ]; lumaSum += source[ i * 4 + 1 ]; }
	const size = [ Math.ceil( width / 2 ), Math.ceil( height / 2 ) ];
	const depthMip1 = new Float32Array( size[ 0 ] * size[ 1 ] );
	for ( let y = 0; y < size[ 1 ]; y ++ ) for ( let x = 0; x < size[ 0 ]; x ++ ) {

		let sum = 0, count = 0;
		for ( let dy = 0; dy < 2; dy ++ ) for ( let dx = 0; dx < 2; dx ++ ) {

			const sx = 2 * x + dx, sy = 2 * y + dy;
			if ( sx < width && sy < height ) { sum += source[ ( sy * width + sx ) * 4 + 2 ]; count ++; }

		}

		depthMip1[ y * size[ 0 ] + x ] = f16( sum / count );

	}

	const logLuma = logSum / ( width * height ), luma = lumaSum / ( width * height );
	return { depthMip1, depthMip1Size: size, logLuma, luma, frameInfo: frameInfoFrom( logLuma, luma, previous, deltaTime ) };

}

const oracleCache = new WeakMap();
export function lumaOracle( scene, variant, previous = resetFrameInfo ) {

	let cache = oracleCache.get( scene );
	if ( ! cache ) oracleCache.set( scene, cache = {} );
	const key = `${ variant } ${ previous.join( ',' ) }`;
	return cache[ key ] ??= variant === 'correct' ? lumaOracleCorrect( scene, previous ) : lumaOracleAmd( scene, previous );

}

/**
 * Tolerances:
 * - depth mip 1: one float16 ulp of the value. The four-texel average is the
 *   same float32 sum on both sides; a reassociating compiler can move it by an
 *   f32 ulp, which can only flip the float16 rounding by one step.
 * - frame info: relative 1e-4. GPU log/exp are not correctly rounded (WGSL: 3
 *   ulp or 2^-21 absolute) and a reassociated sum, or a flipped float16
 *   rounding of one mip-6 texel (2^-11 relative, then /256 in the tail),
 *   moves log luma by ~1e-6 relative; exposure is exp(-L)-proportional, so
 *   its relative error is |ΔL|. 1e-4 leaves two decades of margin and still
 *   catches a single dropped or doubled tile (≥ 1/510 at 1080p).
 */
export function checkLuma( scene, output, { variant, frameInfo } ) {

	const expected = lumaOracle( scene, variant, frameInfo );
	const checks = [];
	const [ w, h ] = expected.depthMip1Size;
	if ( output.depthMip1.width !== w || output.depthMip1.height !== h ) throw new Error( `depth mip 1 is ${ output.depthMip1.width }x${ output.depthMip1.height }, expected ${ w }x${ h }` );
	let worst = 0, bad = 0;
	for ( let i = 0; i < w * h; i ++ ) {

		const actual = output.depthMip1.data[ i * output.depthMip1.channels ], want = expected.depthMip1[ i ];
		const error = Math.abs( actual - want ) / halfUlp( want );
		worst = Math.max( worst, error );
		if ( ! ( error <= 1 ) ) bad ++;

	}

	checks.push( { output: 'FarthestDepthMip1', texels: w * h, maxError: worst, unit: 'f16 ulp', tolerance: 1, outside: bad, ok: bad === 0 } );

	for ( const [ index, name ] of [ [ 0, 'exposure' ], [ 1, 'logLuma' ], [ 2, 'sceneAverageLuma' ] ] ) {

		const actual = output.frameInfo[ index ], want = expected.frameInfo[ index ];
		const error = Math.abs( actual - want ) / Math.max( Math.abs( want ), 1e-30 );
		checks.push( { output: `FrameInfo.${ name }`, actual, expected: want, maxError: error, unit: 'relative', tolerance: 1e-4, ok: error <= 1e-4 } );

	}

	if ( output.mip6 ) {

		let worstMip = 0, badMip = 0;
		for ( let i = 0; i < output.mip6.length; i ++ ) {

			const word = output.mip6[ i ];
			for ( const [ k, bits ] of [ [ 0, word & 0xffff ], [ 1, word >>> 16 ] ] ) {

				const want = expected.mip6[ i * 2 + k ], actual = fromHalfBits( bits );
				const error = Math.abs( actual - want ) / halfUlp( want );
				worstMip = Math.max( worstMip, error );
				if ( ! ( error <= 1 ) ) badMip ++;

			}

		}

		checks.push( { output: 'SPD mip 6 (.xy)', texels: output.mip6.length, maxError: worstMip, unit: 'f16 ulp', tolerance: 1, outside: badMip, ok: badMip === 0 } );

	}

	return { checks, ok: checks.every( check => check.ok ), notes: variant === 'correct' ? null : { mips: expected.mips, coverage: `[0, ${ 2 ** expected.mips })² of ${ scene.width }×${ scene.height }` } };

}
