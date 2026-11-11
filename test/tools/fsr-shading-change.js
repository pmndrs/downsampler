// FSR 3.1 shading-change pyramid and its ShadingChange consumer.
// - 'fsr/shading-change-pyramid':     SPD with the custom load, mips 1-3 stored (all the consumer reads).
// - 'fsr/shading-change-pyramid-amd': as AMD dispatches it: mips 1-6 stored (the tail stores nothing and is not run).
// - 'fsr/shading-change':             the consumer: max over mips 1-3 of |diff × sign|, R8.
// - 'fsr/shading-change-total':       pyramid (mips 1-3) + consumer, the pair the upscaler's fused pass replaces.
//
// Tool code ported from FidelityFX SDK FSR3UPSCALER 3.1.6:
// ffx_fsr3upscaler_shading_change_pyramid.h (SpdLoadSourceImage and helpers, SpdStore),
// ffx_fsr3upscaler_shading_change.h (ShadingChange), ffx_fsr3upscaler_common.h
// (constants, ClampLoad, ClampUv, IsUvInside, MinDividedByMax) and the
// resource callbacks of ffx_fsr3upscaler_callbacks_hlsl.h.
// Copyright (C) 2026 Advanced Micro Devices, Inc. MIT licence; full notice in ./spd.js.
import { spdConstants, spdReference, spdSetup, spdWgsl } from './spd.js';
import { compile, createTexture, readTexture, uniformBuffer } from './gpu.js';
import { f16, f32, halfUlp } from './half.js';

/** The tool half of the pyramid in @group(1); `levels` mips are stored (AMD: 6). */
export function shadingToolWgsl( { half2 }, levels ) {

	const mips = Array.from( { length: levels }, ( _, i ) => i );
	return /* wgsl */ `
struct ShadingConstants {
	renderSize: vec2i,          // RenderSize()
	previousRenderSize: vec2i,  // PreviousFrameRenderSize()
	jitter: vec2f,              // Jitter()
	previousJitter: vec2f,      // PreviousFrameJitter()
	deltaPreExposure: f32,      // DeltaPreExposure()
}
@group(1) @binding(0) var<uniform> fsr: ShadingConstants;
@group(1) @binding(1) var r_current_luma: texture_2d<f32>;
@group(1) @binding(2) var r_previous_luma: texture_2d<f32>;
@group(1) @binding(3) var r_dilated_motion_vectors: texture_2d<f32>;
@group(1) @binding(4) var<storage, read> r_input_exposure: array<vec4f, 1>; // FrameInfo (auto-exposure on)
${ mips.map( i => `@group(1) @binding(${ 5 + i }) var rw_spd_mip${ i }: texture_storage_2d<${ half2 }, write>;` ).join( '\n' ) }

const FSR3UPSCALER_FP16_MIN: f32 = 6.10e-05;
const FSR3UPSCALER_FP16_MAX: f32 = 65504.0;
const FSR3UPSCALER_EPSILON: f32 = FSR3UPSCALER_FP16_MIN;
const FSR3UPSCALER_FP32_MIN: f32 = 1.175494351e-38;
const SHADING_CHANGE_SET_SIZE = 5;
const iShadingChangeMipStart = 0u;
// fShadingChangeSamplePow is 1.0 / 1.0, so ffxPow(x, fShadingChangeSamplePow) is the identity
// and is left out (WGSL pow is exp2(y·log2(x)), which would add error a compiler folding pow(x, 1) does not).
const DIFFERENCE = 0;
const SIGN_SUM = 1;
const MIP0_INDICATOR = 2;
const iSampleOffsets = array<vec2i, 5>(vec2i(0, 0), vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));

fn RenderSize() -> vec2i { return fsr.renderSize; }
fn PreviousFrameRenderSize() -> vec2i { return fsr.previousRenderSize; }

fn Exposure() -> f32 {
	var exposure = r_input_exposure[0].x;
	if (exposure == 0.0) { exposure = 1.0; }
	return exposure;
}

fn ClampLoad(iPxSample: vec2i, iPxOffset: vec2i, iTextureSize: vec2i) -> vec2i {
	var result = iPxSample + iPxOffset;
	result.x = max(0, min(result.x, iTextureSize.x - 1));
	result.y = max(0, min(result.y, iTextureSize.y - 1));
	return result;
}

fn IsUvInside(fUv: vec2f) -> bool {
	return (fUv.x >= 0.0 && fUv.x <= 1.0) && (fUv.y >= 0.0 && fUv.y <= 1.0);
}

fn MinDividedByMax(v0: f32, v1: f32) -> f32 {
	let m = max(v0, v1);
	if (m != 0.0) { return min(v0, v1) / m; }
	return 0.0;
}

fn CompareSwap(s: ptr<function, array<f32, 5>>, i: u32, j: u32) {
	let fTmp = min((*s)[i], (*s)[j]);
	(*s)[j] = max((*s)[i], (*s)[j]);
	(*s)[i] = fTmp;
}

fn SortSet(s: ptr<function, array<f32, 5>>) {
	CompareSwap(s, 0u, 3u);
	CompareSwap(s, 1u, 4u);
	CompareSwap(s, 0u, 2u);
	CompareSwap(s, 1u, 3u);
	CompareSwap(s, 0u, 1u);
	CompareSwap(s, 2u, 4u);
	CompareSwap(s, 1u, 2u);
	CompareSwap(s, 3u, 4u);
	CompareSwap(s, 2u, 3u);
}

fn ComputeMinimumDifference(iPxPos: vec2i, set0: array<f32, 5>, set1: array<f32, 5>) -> f32 {
	var fMinDiff = FSR3UPSCALER_FP16_MAX - 1.0;
	var a = 0;
	var b = 0;
	var fSet0 = set0;
	var fSet1 = set1;

	SortSet(&fSet0);
	SortSet(&fSet1);

	let fMax = min(fSet0[SHADING_CHANGE_SET_SIZE - 1], fSet1[SHADING_CHANGE_SET_SIZE - 1]);

	if (fMax > FSR3UPSCALER_FP32_MIN) {
		for (var i = 0; i < SHADING_CHANGE_SET_SIZE && (fMinDiff < FSR3UPSCALER_FP16_MAX); i++) {
			var fDiff = fSet0[a] - fSet1[b];
			if (abs(fDiff) > FSR3UPSCALER_FP16_MIN) {
				fDiff = sign(fDiff) * (1.0 - MinDividedByMax(fSet0[a], fSet1[b]));
				fMinDiff = select(fMinDiff, fDiff, abs(fDiff) < abs(fMinDiff));
				a += i32(fSet0[a] < fSet1[b]);
				// As in AMD: b compares against the already advanced a (index 5 on the last pass; b is unused after it).
				b += i32(fSet0[min(a, SHADING_CHANGE_SET_SIZE - 1)] >= fSet1[b]);
			} else {
				fMinDiff = FSR3UPSCALER_FP16_MAX;
			}
		}
	}

	return fMinDiff * f32(fMinDiff < (FSR3UPSCALER_FP16_MAX - 1.0));
}

fn GetCurrentLumaBilinearSamples(fUv: vec2f) -> array<f32, 5> {
	let fUvJittered = fUv + fsr.jitter / vec2f(RenderSize());
	let iBasePos = vec2i(floor(fUvJittered * vec2f(RenderSize())));
	var fSet: array<f32, 5>;
	for (var iSampleIndex = 0; iSampleIndex < SHADING_CHANGE_SET_SIZE; iSampleIndex++) {
		let iSamplePos = ClampLoad(iBasePos, iSampleOffsets[iSampleIndex], RenderSize());
		fSet[iSampleIndex] = textureLoad(r_current_luma, iSamplePos, 0).r * Exposure();
		fSet[iSampleIndex] = max(fSet[iSampleIndex], FSR3UPSCALER_EPSILON);
	}
	return fSet;
}

struct PreviousLumaBilinearSamplesData {
	fSet: array<f32, 5>,
	bIsExistingSample: bool,
}

fn GetPreviousLumaBilinearSamples(fUv: vec2f, fMotionVector: vec2f) -> PreviousLumaBilinearSamplesData {
	var data: PreviousLumaBilinearSamplesData;
	let fUvJittered = fUv + fsr.previousJitter / vec2f(PreviousFrameRenderSize());
	let fReprojectedUv = fUvJittered + fMotionVector;
	data.bIsExistingSample = IsUvInside(fReprojectedUv);
	if (data.bIsExistingSample) {
		let iBasePos = vec2i(floor(fReprojectedUv * vec2f(PreviousFrameRenderSize())));
		for (var iSampleIndex = 0; iSampleIndex < SHADING_CHANGE_SET_SIZE; iSampleIndex++) {
			let iSamplePos = ClampLoad(iBasePos, iSampleOffsets[iSampleIndex], PreviousFrameRenderSize());
			data.fSet[iSampleIndex] = textureLoad(r_previous_luma, iSamplePos, 0).r * fsr.deltaPreExposure * Exposure();
			data.fSet[iSampleIndex] = max(data.fSet[iSampleIndex], FSR3UPSCALER_EPSILON);
		}
	}
	return data;
}

fn ComputeDiff(iPxPos: vec2i, fUv: vec2f, fMotionVector: vec2f) -> f32 {
	var fMinDiff = 0.0;
	let fCurrentSamples = GetCurrentLumaBilinearSamples(fUv);
	let previousData = GetPreviousLumaBilinearSamples(fUv, fMotionVector);
	if (previousData.bIsExistingSample) {
		fMinDiff = ComputeMinimumDifference(iPxPos, fCurrentSamples, previousData.fSet);
	}
	return fMinDiff;
}

// SpdLoadSourceImage without its ClampLoad (the skeleton clamps).
fn load(iPxSamplePos: vec2i, layer: u32) -> vec4f {
	let fDilatedMotionVector = textureLoad(r_dilated_motion_vectors, iPxSamplePos, 0).xy;
	let fUv = (vec2f(iPxSamplePos) + 0.5) / vec2f(RenderSize());
	let fScaledAndSignedLumaDiff = ComputeDiff(iPxSamplePos, fUv, fDilatedMotionVector);
	var fOutput = vec4f(0.0);
	fOutput[DIFFERENCE] = fScaledAndSignedLumaDiff;
	fOutput[SIGN_SUM] = select(0.0, sign(fScaledAndSignedLumaDiff), fScaledAndSignedLumaDiff != 0.0);
	fOutput[MIP0_INDICATOR] = 1.0;
	return fOutput;
}

// StorePyramid: writes outside a mip are dropped (D3D UAV semantics).
fn StorePyramid(pix: vec2i, outValue: vec2f, index: u32) {
${ mips.map( i => `	if (index == ${ i }u && all(vec2u(pix) < textureDimensions(rw_spd_mip${ i }))) { textureStore(rw_spd_mip${ i }, pix, vec4f(outValue, 0.0, 0.0)); }` ).join( '\n' ) }
}

fn store(level: u32, pix: vec2i, layer: u32, outValue: vec4f) {
	if (level - 1u >= iShadingChangeMipStart) { StorePyramid(pix, outValue.xy, level - 1u); }
}

// SpdLoad (RWLoadPyramid(tex, 5)) feeds only the tail, whose mips this pass never stores; the tail is not run.
fn spdLoad(tex: vec2i) -> vec4f { return vec4f(0.0); }
`;

}

export function consumerWgsl( { unorm1 } ) {

	return /* wgsl */ `
struct ConsumerConstants {
	renderSize: vec2i,  // RenderSize()
	jitter: vec2f,      // Jitter()
}
@group(0) @binding(0) var<uniform> c: ConsumerConstants;
@group(0) @binding(1) var r_spd_mips: texture_2d<f32>;
@group(0) @binding(2) var s_LinearClamp: sampler;
@group(0) @binding(3) var rw_shading_change: texture_storage_2d<${ unorm1 }, write>;

const s_MipLevelsToUse = 3;
const iShadingChangeMipStart = 0;

fn ShadingChangeRenderSize() -> vec2i { return vec2i(vec2f(c.renderSize) * 0.5); }

fn ClampUv(fUv: vec2f, iTextureSize: vec2i, iResourceSize: vec2i) -> vec2f {
	let fSampleLocation = fUv * vec2f(iTextureSize);
	let fClampedLocation = max(vec2f(0.5, 0.5), min(fSampleLocation, vec2f(iTextureSize) - vec2f(0.5, 0.5)));
	return fClampedLocation / vec2f(iResourceSize);
}

fn IsOnScreen(pos: vec2i, size: vec2i) -> bool { return all(vec2u(pos) < vec2u(size)); }

fn ShadingChange(iPxPos: vec2i) {
	if (IsOnScreen(iPxPos, ShadingChangeRenderSize())) {
		let fUv = (vec2f(iPxPos) + 0.5) / vec2f(ShadingChangeRenderSize());
		let fUvJittered = fUv + c.jitter / vec2f(c.renderSize);

		// ComputeShadingChangeLuma
		var fSamples: array<f32, 3>;
		let fMipUv = ClampUv(fUvJittered, ShadingChangeRenderSize(), vec2i(textureDimensions(r_spd_mips, 0)));
		for (var iMipLevel = iShadingChangeMipStart; iMipLevel < s_MipLevelsToUse; iMipLevel++) {
			let fSample = textureSampleLevel(r_spd_mips, s_LinearClamp, fMipUv, f32(iMipLevel)).xy;
			fSamples[iMipLevel] = abs(fSample.x * fSample.y);
		}

		let fScale = 1.0 + f32(iShadingChangeMipStart / s_MipLevelsToUse);
		var fShadingChange = 0.0;
		for (var iMipLevel = iShadingChangeMipStart; iMipLevel < s_MipLevelsToUse; iMipLevel++) {
			if (fSamples[iMipLevel] > 0.0) {
				fShadingChange = max(fShadingChange, fSamples[iMipLevel]) * fScale;
			}
		}

		textureStore(rw_shading_change, iPxPos, vec4f(saturate(fShadingChange)));
	}
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) { ShadingChange(vec2i(gid.xy)); }
`;

}

function toolLayout( device, half2, levels ) {

	return device.createBindGroupLayout( { label: 'fsr shading tool (group 1)', entries: [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
		...Array.from( { length: levels }, ( _, i ) => ( { binding: 5 + i, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: half2 } } ) )
	] } );

}

/** Inputs: current and previous luma (R16F), dilated motion (RG16F), FrameInfo with this frame's exposure. */
export function createShadingInputs( device, scene, exposure ) {

	const { width, height } = scene;
	const frameInfo = device.createBuffer( { label: 'FrameInfo', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( frameInfo, 0, new Float32Array( [ exposure, 0, 0, 0 ] ) );
	const constants = new ArrayBuffer( 48 );
	new Int32Array( constants, 0, 4 ).set( [ width, height, width, height ] );
	new Float32Array( constants, 16, 5 ).set( [ ...scene.jitter, ...scene.previousJitter, 1 ] );
	const inputs = {
		exposure, frameInfo,
		uniforms: uniformBuffer( device, 'fsr shading constants', constants ),
		currentLuma: createTexture( device, { label: 'r_current_luma', width, height, format: 'r16float', data: scene.currentLuma } ),
		previousLuma: createTexture( device, { label: 'r_previous_luma', width, height, format: 'r16float', data: scene.previousLuma } ),
		motion: createTexture( device, { label: 'r_dilated_motion_vectors', width, height, format: 'rg16float', data: scene.fsrMotion } )
	};
	inputs.destroy = () => { for ( const resource of [ frameInfo, inputs.uniforms, inputs.currentLuma, inputs.previousLuma, inputs.motion ] ) resource.destroy(); };
	return inputs;

}

/**
 * The FSR shading-change tool as SPEC §14.1 hands it to a generator: WGSL for
 * @group(1), its layout and bind group, and the stored mips (`levels` of
 * them, at ⌊w/2^k⌋ × ⌊h/2^k⌋ as AMD's SpdMips). What a generator fixture plugs in.
 */
export function createShadingTool( device, scene, inputs, formats, { levels = 3 } = {} ) {

	const { width, height } = scene;
	const texture = createTexture( device, { label: `r_spd_mips (${ levels })`, width: Math.floor( width / 2 ), height: Math.floor( height / 2 ), format: formats.half2, mipLevelCount: levels, usage: GPUTextureUsage.STORAGE_BINDING } );
	const layout = toolLayout( device, formats.half2, levels );
	const group = device.createBindGroup( { layout, entries: [
		{ binding: 0, resource: { buffer: inputs.uniforms } },
		{ binding: 1, resource: inputs.currentLuma.createView() },
		{ binding: 2, resource: inputs.previousLuma.createView() },
		{ binding: 3, resource: inputs.motion.createView() },
		{ binding: 4, resource: { buffer: inputs.frameInfo } },
		...Array.from( { length: levels }, ( _, i ) => ( { binding: 5 + i, resource: texture.createView( { baseMipLevel: i, mipLevelCount: 1 } ) } ) )
	] } );
	return {
		wgsl: shadingToolWgsl( formats, levels ), layout, group, texture, levels,
		async read() { return Promise.all( Array.from( { length: levels }, ( _, i ) => readTexture( device, texture, i ) ) ); },
		async check() { return checkShadingPyramid( scene, await this.read(), { exposure: inputs.exposure, levels } ); },
		destroy() { texture.destroy(); }
	};

}

/**
 * Shading-change fixtures for one scene. The inputs' exposure is FrameInfo.x as
 * this frame's luma pyramid leaves it (the oracle's), so the fixtures stay independent.
 * @returns {Promise<import('./harness.js').Fixture[]>}
 */
export async function createFsrShadingChange( device, scene, inputs, formats ) {

	const { width, height } = scene;
	const setup = spdSetup( width, height );
	const half = [ Math.floor( width / 2 ), Math.floor( height / 2 ) ];
	const owned = [];

	async function pyramid( levels, mips ) {

		const tool = createShadingTool( device, scene, inputs, formats, { levels } );
		const module = await compile( device, `fsr shading change pyramid (${ levels } mips)`, spdWgsl + tool.wgsl );
		const layout0 = device.createBindGroupLayout( { entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } ] } );
		const spdUniforms = uniformBuffer( device, 'cbSPD', spdConstants( { ...setup, mips }, width, height ) );
		owned.push( tool, spdUniforms );
		const group0 = device.createBindGroup( { layout: layout0, entries: [ { binding: 0, resource: { buffer: spdUniforms } } ] } );
		const pipeline = await device.createComputePipelineAsync( { layout: device.createPipelineLayout( { bindGroupLayouts: [ layout0, tool.layout ] } ), compute: { module, entryPoint: 'spdTile' } } );
		return { tool, encodePass( pass ) { pass.setPipeline( pipeline ); pass.setBindGroup( 0, group0 ); pass.setBindGroup( 1, tool.group ); pass.dispatchWorkgroups( ...setup.groups ); } };

	}

	// Mips 1-3 only (SPD stops after mip 3), and AMD's mips 1-6.
	const short = await pyramid( 3, 3 );
	const full = device.limits.maxStorageTexturesPerShaderStage >= 6 ? await pyramid( 6, setup.mips ) : null;

	const consumerModule = await compile( device, 'fsr shading change', consumerWgsl( formats ) );
	const output = createTexture( device, { label: 'rw_shading_change', width: half[ 0 ], height: half[ 1 ], format: formats.unorm1, usage: GPUTextureUsage.STORAGE_BINDING } );
	const consumerUniforms = new ArrayBuffer( 16 );
	new Int32Array( consumerUniforms, 0, 2 ).set( [ width, height ] );
	new Float32Array( consumerUniforms, 8, 2 ).set( scene.jitter );
	const consumerBuffer = uniformBuffer( device, 'shading change constants', consumerUniforms );
	owned.push( output, consumerBuffer );
	const consumerPipeline = await device.createComputePipelineAsync( { layout: 'auto', compute: { module: consumerModule, entryPoint: 'main' } } );
	const sampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' } );
	const consumerGroup = device.createBindGroup( { layout: consumerPipeline.getBindGroupLayout( 0 ), entries: [
		{ binding: 0, resource: { buffer: consumerBuffer } },
		{ binding: 1, resource: short.tool.texture.createView() },
		{ binding: 2, resource: sampler },
		{ binding: 3, resource: output.createView() }
	] } );
	const consumerGroups = [ Math.ceil( half[ 0 ] / 8 ), Math.ceil( half[ 1 ] / 8 ) ];
	const consumer = pass => { pass.setPipeline( consumerPipeline ); pass.setBindGroup( 0, consumerGroup ); pass.dispatchWorkgroups( ...consumerGroups ); };
	const checkConsumerOutput = async () => checkConsumer( scene, await short.tool.read(), await readTexture( device, output ) );

	const fixtures = [
		{ name: 'fsr/shading-change-pyramid', pass: 'shading', dispatches: 1, encodePass: short.encodePass, check: () => short.tool.check() },
		full && { name: 'fsr/shading-change-pyramid-amd', pass: 'shading', dispatches: 1, encodePass: full.encodePass, check: () => full.tool.check() },
		// Reads the mips fsr/shading-change-pyramid stored (it runs first).
		{ name: 'fsr/shading-change', pass: 'shading', dispatches: 1, encodePass: consumer, check: checkConsumerOutput },
		{ name: 'fsr/shading-change-total', pass: 'shading', dispatches: 2, encodePass( pass ) { short.encodePass( pass ); consumer( pass ); }, check: checkConsumerOutput }
	].filter( Boolean );

	let destroyed = false;
	for ( const fixture of fixtures ) fixture.destroy = () => { if ( destroyed ) return; destroyed = true; for ( const resource of owned ) resource.destroy(); };
	return fixtures;

}

// ---------------------------------------------------------------------------
// CPU oracles

const FP16_MIN = f32( 6.10e-05 ), FP16_MAX = 65504, EPSILON = FP16_MIN, FP32_MIN = f32( 1.175494351e-38 );
const offsets = [ [ 0, 0 ], [ - 1, 0 ], [ 1, 0 ], [ 0, - 1 ], [ 0, 1 ] ];
const network = [ [ 0, 3 ], [ 1, 4 ], [ 0, 2 ], [ 1, 3 ], [ 0, 1 ], [ 2, 4 ], [ 1, 2 ], [ 3, 4 ], [ 2, 3 ] ];

function sortSet( set ) {

	for ( const [ i, j ] of network ) { const low = Math.min( set[ i ], set[ j ] ); set[ j ] = Math.max( set[ i ], set[ j ] ); set[ i ] = low; }

}

function minimumDifference( set0, set1 ) {

	let minDiff = FP16_MAX - 1, a = 0, b = 0;
	sortSet( set0 ); sortSet( set1 );
	if ( Math.min( set0[ 4 ], set1[ 4 ] ) > FP32_MIN ) {

		for ( let i = 0; i < 5 && minDiff < FP16_MAX; i ++ ) {

			let diff = f32( set0[ a ] - set1[ b ] );
			if ( Math.abs( diff ) > FP16_MIN ) {

				const m = Math.max( set0[ a ], set1[ b ] );
				const ratio = m !== 0 ? f32( Math.min( set0[ a ], set1[ b ] ) / m ) : 0;
				diff = Math.sign( diff ) * f32( 1 - ratio );
				if ( Math.abs( diff ) < Math.abs( minDiff ) ) minDiff = diff;
				a += set0[ a ] < set1[ b ] ? 1 : 0;
				b += set0[ Math.min( a, 4 ) ] >= set1[ b ] ? 1 : 0;

			} else {

				minDiff = FP16_MAX;

			}

		}

	}

	return minDiff < FP16_MAX - 1 ? minDiff : 0;

}

/** The tool's load (diff, sign, 1, 0) for every texel, in float32. */
export function shadingSource( scene, exposure ) {

	const { width: W, height: H, currentLuma, previousLuma, fsrMotion, jitter, previousJitter } = scene;
	const source = new Float32Array( W * H * 4 );
	const set0 = [ 0, 0, 0, 0, 0 ], set1 = [ 0, 0, 0, 0, 0 ];
	const e = f32( exposure === 0 ? 1 : exposure );
	const jx = f32( f32( jitter[ 0 ] ) / W ), jy = f32( f32( jitter[ 1 ] ) / H );
	const px = f32( f32( previousJitter[ 0 ] ) / W ), py = f32( f32( previousJitter[ 1 ] ) / H );
	const at = ( data, x, y ) => data[ Math.max( 0, Math.min( y, H - 1 ) ) * W + Math.max( 0, Math.min( x, W - 1 ) ) ];

	for ( let y = 0; y < H; y ++ ) for ( let x = 0; x < W; x ++ ) {

		const i = y * W + x;
		const u = f32( f32( x + 0.5 ) / W ), v = f32( f32( y + 0.5 ) / H );
		const bx = Math.floor( f32( f32( u + jx ) * W ) ), by = Math.floor( f32( f32( v + jy ) * H ) );
		for ( let s = 0; s < 5; s ++ ) set0[ s ] = Math.max( f32( at( currentLuma, bx + offsets[ s ][ 0 ], by + offsets[ s ][ 1 ] ) * e ), EPSILON );
		const ru = f32( f32( u + px ) + fsrMotion[ i * 2 ] ), rv = f32( f32( v + py ) + fsrMotion[ i * 2 + 1 ] );
		let diff = 0;
		if ( ru >= 0 && ru <= 1 && rv >= 0 && rv <= 1 ) {

			const qx = Math.floor( f32( ru * W ) ), qy = Math.floor( f32( rv * H ) );
			for ( let s = 0; s < 5; s ++ ) set1[ s ] = Math.max( f32( f32( at( previousLuma, qx + offsets[ s ][ 0 ], qy + offsets[ s ][ 1 ] ) * 1 ) * e ), EPSILON );
			diff = minimumDifference( set0, set1 );

		}

		source[ i * 4 ] = diff; source[ i * 4 + 1 ] = diff !== 0 ? Math.sign( diff ) : 0; source[ i * 4 + 2 ] = 1;

	}

	return source;

}

const cache = new WeakMap();
/** Stored mips 1..levels (float16 .xy at ⌊w/2^k⌋ × ⌊h/2^k⌋), AMD semantics. */
export function shadingPyramidOracle( scene, exposure, levels ) {

	let entry = cache.get( scene );
	if ( ! entry || entry.exposure !== exposure ) cache.set( scene, entry = { exposure, source: shadingSource( scene, exposure ), pyramids: {} } );
	if ( entry.pyramids[ levels ] ) return entry.pyramids[ levels ];
	const { width, height } = scene;
	const sizes = Array.from( { length: levels + 1 }, ( _, k ) => [ Math.max( 1, Math.floor( width / 2 ) >> Math.max( 0, k - 1 ) ), Math.max( 1, Math.floor( height / 2 ) >> Math.max( 0, k - 1 ) ) ] );
	const stored = sizes.map( ( [ w, h ] ) => new Float32Array( w * h * 2 ) );
	spdReference( { width, height, source: entry.source, mips: levels, store( level, x, y, value ) {

		if ( level > levels ) return;
		const [ w, h ] = sizes[ level ];
		if ( x >= w || y >= h ) return;
		stored[ level ][ ( y * w + x ) * 2 ] = f16( value[ 0 ] ); stored[ level ][ ( y * w + x ) * 2 + 1 ] = f16( value[ 1 ] );

	} } );
	return entry.pyramids[ levels ] = { sizes, stored };

}

// Values are averages of signed differences in [-1, 1]; where + and - cancel,
// a reassociated float32 sum differs by up to ~4 · 2^-24 · 2 (2^-21) absolute,
// which is far below one float16 ulp everywhere except near zero.
const ABSOLUTE_FLOOR = 2 ** - 21;

/**
 * Tolerance: max(one float16 ulp, 2^-21) per stored value (same float32 arithmetic, then
 * one rounding to RG16F; a reassociated average or a 1-2 ulp division
 * difference can flip that rounding once). A texel outside it is counted:
 * ComputeMinimumDifference makes discrete choices (sort, |d| < |min|,
 * a/b advance, the IsUvInside and floor() edges), so an f32 ulp can change the
 * selected difference. Allowed: as many values outside as 1e-4 of mip 1 (per mip).
 */
export function checkShadingPyramid( scene, levels, { exposure, levels: count } ) {

	const { sizes, stored } = shadingPyramidOracle( scene, exposure, count );
	const checks = [];
	for ( let level = 1; level <= count; level ++ ) {

		const [ w, h ] = sizes[ level ], actual = levels[ level - 1 ];
		if ( actual.width !== w || actual.height !== h ) throw new Error( `mip ${ level } is ${ actual.width }x${ actual.height }, expected ${ w }x${ h }` );
		let worst = 0, outside = 0, worstAbsolute = 0;
		for ( let i = 0; i < w * h; i ++ ) for ( let c = 0; c < 2; c ++ ) {

			const want = stored[ level ][ i * 2 + c ], got = actual.data[ i * actual.channels + c ];
			const error = Math.abs( got - want ) / Math.max( halfUlp( want ), ABSOLUTE_FLOOR );
			if ( error > 1 || ! ( error === error ) ) { outside ++; worstAbsolute = Math.max( worstAbsolute, Math.abs( got - want ) ); } else worst = Math.max( worst, error );

		}

		// A flipped source texel moves one texel per mip (by 1/4 as much each mip down), so every mip gets mip 1's budget.
		const budget = Math.floor( 1e-4 * sizes[ 1 ][ 0 ] * sizes[ 1 ][ 1 ] * 2 );
		checks.push( { output: `SpdMips mip ${ level } (.xy)`, texels: w * h, maxError: worst, unit: 'max(f16 ulp, 2^-21)', tolerance: 1, outside, budget, worstOutside: worstAbsolute, ok: outside <= budget } );

	}

	return { checks, ok: checks.every( check => check.ok ) };

}

// GPU bilinear weights are fixed point; 8 fractional bits is the D3D/Vulkan floor.
const WEIGHT_STEP = 2 ** - 8;

function bilinear( level, u, v, channel ) {

	const { width: w, height: h, data, channels } = level;
	const x = u * w - 0.5, y = v * h - 0.5, x0 = Math.floor( x ), y0 = Math.floor( y ), fx = x - x0, fy = y - y0;
	const at = ( i, j ) => data[ ( Math.min( Math.max( j, 0 ), h - 1 ) * w + Math.min( Math.max( i, 0 ), w - 1 ) ) * channels + channel ];
	const a = at( x0, y0 ), b = at( x0 + 1, y0 ), c = at( x0, y0 + 1 ), d = at( x0 + 1, y0 + 1 );
	const value = ( a * ( 1 - fx ) + b * fx ) * ( 1 - fy ) + ( c * ( 1 - fx ) + d * fx ) * fy;
	const range = Math.max( a, b, c, d ) - Math.min( a, b, c, d );
	return { value, bound: 2 * WEIGHT_STEP * range + 1e-7 * Math.max( Math.abs( a ), Math.abs( b ), Math.abs( c ), Math.abs( d ) ) };

}

/**
 * The consumer, from the pyramid the GPU stored (so this check is independent
 * of the pyramid's). Per texel the tolerance is the bilinear-weight bound
 * (2·2^-8 of each tap range, propagated through |x·y|) plus one R8 step for
 * the output rounding.
 */
export function checkConsumer( scene, levels, output ) {

	const { width, height, jitter } = scene;
	const sw = Math.floor( width * 0.5 ), sh = Math.floor( height * 0.5 );
	if ( output.width !== sw || output.height !== sh ) throw new Error( `shading change is ${ output.width }x${ output.height }, expected ${ sw }x${ sh }` );
	const jx = f32( f32( jitter[ 0 ] ) / width ), jy = f32( f32( jitter[ 1 ] ) / height );
	let worst = 0, outside = 0, worstOutside = 0, nonZero = 0;

	for ( let y = 0; y < sh; y ++ ) for ( let x = 0; x < sw; x ++ ) {

		const u = f32( f32( f32( x + 0.5 ) / sw ) + jx ), v = f32( f32( f32( y + 0.5 ) / sh ) + jy );
		const mu = Math.max( 0.5, Math.min( f32( u * sw ), sw - 0.5 ) ) / levels[ 0 ].width, mv = Math.max( 0.5, Math.min( f32( v * sh ), sh - 0.5 ) ) / levels[ 0 ].height;
		let value = 0, bound = 0;
		for ( let level = 0; level < 3; level ++ ) {

			const sx = bilinear( levels[ level ], mu, mv, 0 ), sy = bilinear( levels[ level ], mu, mv, 1 );
			const sample = Math.abs( sx.value * sy.value );
			bound = Math.max( bound, Math.abs( sy.value ) * sx.bound + Math.abs( sx.value ) * sy.bound + sx.bound * sy.bound );
			if ( sample > 0 ) value = Math.max( value, sample );

		}

		value = Math.min( 1, Math.max( 0, value ) );
		const want = Math.round( value * 255 ) / 255, got = output.data[ ( y * sw + x ) * output.channels ];
		if ( got > 0 ) nonZero ++;
		const error = Math.abs( got - want ), tolerance = bound + 1 / 255 + 1e-6;
		if ( error > tolerance ) { outside ++; worstOutside = Math.max( worstOutside, error ); } else worst = Math.max( worst, error );

	}

	const budget = Math.floor( 1e-4 * sw * sh );
	return { checks: [ { output: 'ShadingChange (R8)', texels: sw * sh, maxError: worst, unit: 'absolute', tolerance: 'bilinear bound + 1/255', outside, budget, worstOutside, nonZero: nonZero / ( sw * sh ), ok: outside <= budget } ], ok: outside <= budget };

}
