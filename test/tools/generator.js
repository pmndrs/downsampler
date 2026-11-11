// Downsampler-generator versions of the upscaler's pyramid passes: the third
// column of the comparison (upscaler-current vs FSR-reference vs generator).
// Every one is built from the library (`Reducer`, `ScalarReduction` from
// dist/), with only tool code (hooks and bindings) written here.
//
// Each entry's `create( device, context )` returns a Fixture (see harness.js).
// `context` holds:
//   scene     the synthetic frame (scene.js)
//   formats   storage formats for the FSR outputs (gpu.js outputFormats)
//   colour    the scene colour texture (rgba16float) the upscaler passes read
//   tools     factories for the tool halves:
//               tools.luma( { variant } )      FSR luma pyramid    → { parts, entries, groupEntries, layout, group, reset, check, destroy }
//               tools.shading( { levels } )    FSR shading change  → { parts, entries, layout, group, texture, check, destroy }
//               tools.upscalerExposure()       upscaler exposure   → { groupEntries, check, destroy }
//               tools.upscalerShading()        upscaler shading    → { groupEntries, outputSize, check, destroy }
//             `parts` are WGSL strings for @group(1): `bindings` (declarations
//             and helpers), `load`, `store` (and `finish`).
//
// A fixture returns { name, pass, dispatches, encodePass, reset, check, destroy }.
// It appears in the report next to each of `compareWith`, and the ratio
// generator ÷ compareWith is printed.
import { Reducer, ScalarReduction } from '../../dist/webgpu/index.js';
import { compile, uniformBuffer } from './gpu.js';
import { resetFrameInfo } from './fsr-luma-pyramid.js';
import { LUMINANCE_PYRAMID_SHADER, SHADING_CHANGE_SHADER } from './upscaler-wgsl.js';
import { exposureOracleEveryTexel } from './upscaler-exposure.js';

const visibility = GPUShaderStage.COMPUTE;

/**
 * Builds library objects inside a validation scope. The library compiles
 * synchronously, so on an error the WGSL is recompiled here to name the line.
 */
async function built( device, label, make, sources ) {

	device.pushErrorScope( 'validation' );
	const result = make();
	const error = await device.popErrorScope();
	if ( error ) {

		for ( const code of sources( result ) ) await compile( device, label, code );
		throw new Error( `${ label }: ${ error.message }` );

	}

	return result;

}

const reducer = ( device, label, options ) => built( device, label, () => new Reducer( device, { label, ...options } ), () => [] );

function scalar( device, label, options ) {

	return built( device, label, () => {

		const reduction = new ScalarReduction( device, options );
		reduction.kernel(); // compiles both pipelines
		return reduction;

	}, reduction => [ reduction.kernel().wgsl ] );

}

/** The every-texel and AMD luma fixtures check twice: unsmoothed, then from AMD's reset state (as fsr/luma-pyramid). */
function lumaCheck( device, tool, encodePass ) {

	return async () => {

		const unsmoothed = await tool.check();
		tool.reset( resetFrameInfo );
		const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
		encodePass( pass );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );
		const smoothed = await tool.check( resetFrameInfo );
		tool.reset();
		const frameInfoOnly = smoothed.checks.filter( check => check.output.startsWith( 'FrameInfo' ) ).map( check => ( { ...check, output: `${ check.output } (from reset)` } ) );
		return { ...unsmoothed, checks: [ ...unsmoothed.checks, ...frameInfoOnly ], ok: unsmoothed.ok && smoothed.ok };

	};

}

// ---------------------------------------------------------------------------
// FSR 3.1 luma pyramid

/** AMD parity: Reducer 'clamp' [1, 6] with FSR's own hooks, then the top-left 2^mips window of mip 6 → FrameInfo. */
async function lumaPyramidClamp( device, { scene, tools } ) {

	const tool = tools.luma( { variant: 'amd' } );
	const { bindings, load, store } = tool.parts;
	const group1 = { wgsl: bindings, entries: tool.entries };
	// AMD keeps texel (0, 0) of mip `mips`: the average of mip 6 over [0, 2^(mips-6))², where
	// texels past rw_spd_mip5 read 0 (spdLoad). One f16 rounding (mip 6), then f32 as AMD's tail.
	if ( tool.mips < 7 ) throw new Error( `generator/luma-pyramid-clamp: mips ${ tool.mips } < 7 would take FrameInfo from mip 6 itself, unrounded` );
	const window = 2 ** ( tool.mips - 6 );
	const pyramid = await reducer( device, 'generator luma pyramid (clamp)', { reduction: 'average', levels: [ 1, 6 ], edges: 'clamp', bindings: group1, load, store } );
	const exposure = await scalar( device, 'generator luma window', {
		reduce: 'average', bindings: group1,
		load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return spdLoad(texel); }',
		finish: 'fn finish(value: vec4f, weight: f32) { updateFrameInfo(value); }'
	} );
	const dispatch = { width: scene.width, height: scene.height, bindGroup: tool.group };
	const run = { size: [ window, window ], bindGroup: tool.group };
	const encodePass = pass => { pyramid.encodePass( pass, dispatch ); exposure.encodePass( pass, run ); };
	return { name: 'generator/luma-pyramid-clamp', pass: 'exposure', dispatches: 3, encodePass, reset: () => tool.reset(), check: lumaCheck( device, tool, encodePass ), destroy: () => tool.destroy() };

}

/** Every texel: Reducer 'zero-weight' [1] for depth mip 1, ScalarReduction over every texel with FSR's `finish`. */
async function lumaPyramid( device, { scene, tools } ) {

	const tool = tools.luma( { variant: 'correct' } );
	const { bindings, load, store, finish } = tool.parts;
	const group1 = { wgsl: bindings, entries: tool.entries };
	const depth = await reducer( device, 'generator depth mip 1', { reduction: 'average', levels: [ 1 ], edges: 'zero-weight', bindings: group1, load, store } );
	const exposure = await scalar( device, 'generator luma scalar', { reduce: 'average', bindings: group1, load, finish } );
	const dispatch = { width: scene.width, height: scene.height, bindGroup: tool.group };
	const run = { size: [ scene.width, scene.height ], bindGroup: tool.group };
	const encodePass = pass => { depth.encodePass( pass, dispatch ); exposure.encodePass( pass, run ); };
	return { name: 'generator/luma-pyramid', pass: 'exposure', dispatches: 3, encodePass, reset: () => tool.reset(), check: lumaCheck( device, tool, encodePass ), destroy: () => tool.destroy() };

}

/**
 * Every texel, one read of the source: Reducer 'zero-weight' [1, 6] with
 * (Σ log luma, Σ luma, mean depth, count) per channel; level 1 stores depth,
 * level 6 one partial per 64×64 tile; a ScalarReduction sums the partials and
 * divides in `finish`.
 */
async function lumaPyramidFused( device, { scene, tools } ) {

	const tool = tools.luma( { variant: 'correct' } );
	const { bindings, load, finish } = tool.parts;
	const grid = [ Math.ceil( scene.width / 64 ), Math.ceil( scene.height / 64 ) ];
	const partials = device.createBuffer( { label: 'luma tile partials', size: grid[ 0 ] * grid[ 1 ] * 16, usage: GPUBufferUsage.STORAGE } );
	const constants = uniformBuffer( device, 'luma tile grid', new Uint32Array( [ grid[ 0 ], grid[ 1 ], 0, 0 ] ) );
	const entries = [ ...tool.entries, { binding: 6, visibility, buffer: { type: 'uniform' } }, { binding: 7, visibility, buffer: { type: 'storage' } } ];
	const wgsl = `${ bindings }
@group(1) @binding(6) var<uniform> lumaGrid: vec4u;
@group(1) @binding(7) var<storage, read_write> lumaPartials: array<vec4f>;
${ load.replace( 'fn load(', 'fn fsrLoad(' ) }
${ finish.replace( 'fn finish(', 'fn fsrFinish(' ) }`;
	const group = device.createBindGroup( { layout: device.createBindGroupLayout( { entries } ), entries: [ ...tool.groupEntries, { binding: 6, resource: { buffer: constants } }, { binding: 7, resource: { buffer: partials } } ] } );
	const pyramid = await reducer( device, 'generator luma pyramid (fused)', {
		reduction: [ 'sum', 'sum', 'average', 'sum' ], levels: [ 1, 6 ], edges: 'zero-weight', bindings: { wgsl, entries },
		load: 'fn load(texel: vec2i, layer: u32) -> vec4f { var v = fsrLoad(texel, layer); v.w = 1.0; return v; }',
		store: `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
	if (level == 1u) { StoreFarthestDepthMip1(texel, value.z); } else { lumaPartials[u32(texel.y) * lumaGrid.x + u32(texel.x)] = value; }
}`
	} );
	const exposure = await scalar( device, 'generator luma partials', {
		reduce: 'sum', bindings: { wgsl, entries },
		load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return lumaPartials[u32(texel.y) * lumaGrid.x + u32(texel.x)]; }',
		finish: 'fn finish(value: vec4f, weight: f32) { fsrFinish(vec4f(value.xy / value.w, 0.0, 0.0), value.w); }'
	} );
	const dispatch = { width: scene.width, height: scene.height, bindGroup: group };
	const run = { size: grid, bindGroup: group };
	const encodePass = pass => { pyramid.encodePass( pass, dispatch ); exposure.encodePass( pass, run ); };
	return {
		name: 'generator/luma-pyramid-fused', pass: 'exposure', dispatches: 3, encodePass, reset: () => tool.reset(), check: lumaCheck( device, tool, encodePass ),
		destroy() { tool.destroy(); partials.destroy(); constants.destroy(); }
	};

}

// ---------------------------------------------------------------------------
// FSR 3.1 shading-change pyramid: FSR's load and store hooks, mips 1-3, 'clamp'.

async function shadingChangePyramid( device, { scene, tools } ) {

	const tool = tools.shading( { levels: 3 } );
	const { bindings, load, store, spdLoad } = tool.parts;
	const pyramid = await reducer( device, 'generator shading change pyramid', {
		reduction: 'average', levels: [ 1, 2, 3 ], edges: 'clamp', bindings: { wgsl: bindings + spdLoad, entries: tool.entries }, load, store
	} );
	const dispatch = { width: scene.width, height: scene.height, bindGroup: tool.group };
	return { name: 'generator/shading-change-pyramid', pass: 'shading', dispatches: 1, encodePass: pass => pyramid.encodePass( pass, dispatch ), check: () => tool.check(), destroy: () => tool.destroy() };

}

// ---------------------------------------------------------------------------
// pmndrs/upscaler exposure (luminancePyramid.ts): the metering as a
// ScalarReduction, the epilogue (host pre-exposure, target, dead band, easing,
// flags, store) verbatim as `finish`.

const exposureShader = LUMINANCE_PYRAMID_SHADER;
const exposureBindings = exposureShader.slice( 0, exposureShader.indexOf( '@compute' ) ).replaceAll( '@group(0)', '@group(1)' );
const exposureEpilogue = exposureShader.slice( exposureShader.indexOf( '    // Meter host-invariantly' ), exposureShader.lastIndexOf( '}' ) );
if ( ! exposureEpilogue.includes( 'exp2(logSum / f32(EXPOSURE_TAPS * EXPOSURE_TAPS))' ) ) throw new Error( 'luminancePyramid.ts changed: update the generator exposure epilogue' );
const exposureFinish = `fn finish(value: vec4f, weight: f32) {\n${ exposureEpilogue.replace( 'exp2(logSum / f32(EXPOSURE_TAPS * EXPOSURE_TAPS))', 'exp2(value.x)' ) }}\n`;
const exposureEntries = [
	{ binding: 0, visibility, buffer: { type: 'uniform' } },
	{ binding: 1, visibility, texture: { sampleType: 'float' } },
	{ binding: 2, visibility, sampler: { type: 'filtering' } },
	{ binding: 3, visibility, texture: { sampleType: 'float' } },
	{ binding: 4, visibility, storageTexture: { access: 'write-only', format: 'rgba16float' } },
	{ binding: 5, visibility, texture: { sampleType: 'float' } },
	{ binding: 6, visibility, texture: { sampleType: 'float' } }
];

async function upscalerExposure( device, { scene, tools }, { taps } ) {

	const shared = tools.upscalerExposure();
	const group = device.createBindGroup( { layout: device.createBindGroupLayout( { entries: exposureEntries } ), entries: shared.groupEntries() } );
	// Every texel by textureLoad, or the upscaler's own 32×32 bilinear taps (one 32×32 tile: block 2).
	const load = taps
		? 'fn load(texel: vec2i, layer: u32) -> vec4f {\n\tlet uv = (vec2f(texel) + 0.5) / f32(EXPOSURE_TAPS);\n\treturn vec4f(log2(max(luma(textureSampleLevel(inputColor, linearSampler, uv, 0.0).rgb), 1.0e-4)), 0.0, 0.0, 0.0);\n}'
		: 'fn load(texel: vec2i, layer: u32) -> vec4f {\n\treturn vec4f(log2(max(luma(textureLoad(inputColor, texel, 0).rgb), 1.0e-4)), 0.0, 0.0, 0.0);\n}';
	const exposure = await scalar( device, `generator upscaler exposure${ taps ? ' (taps)' : '' }`, {
		reduce: [ 'average', 'sum', 'sum', 'sum' ], bindings: { wgsl: exposureBindings, entries: exposureEntries }, load, finish: exposureFinish, ...( taps ? { block: 2 } : {} )
	} );
	const run = { size: taps ? [ 32, 32 ] : [ scene.width, scene.height ], bindGroup: group };
	return {
		name: taps ? 'generator/upscaler-exposure-taps' : 'generator/upscaler-exposure', pass: 'exposure', dispatches: 2,
		encodePass: pass => exposure.encodePass( pass, run ),
		check: () => taps ? shared.check() : shared.check( exposureOracleEveryTexel ),
		destroy: () => shared.destroy()
	};

}

// ---------------------------------------------------------------------------
// pmndrs/upscaler shading change (shadingChange.ts) as a Reducer: the load is
// the upscaler's per-texel work (luma, history store, motion/disocclusion
// "moving" flag, lumaPair's reprojection and clamp into the bilinear taps);
// levels 2 and 3 are its 4×4 and 8×8 blocks; the store hook is its block
// memory update and response, written at the signal's 2×2 resolution.
//
// Five quantities per block: Σ cur, Σ prev, Σ cur², Σ prev² and max moving.
// The kernel carries one vec4f, so the flag rides in the sign of the Σ prev²
// channel (a sum of non-negative terms): a custom 'sum' reduction adds
// magnitudes and ORs signs. A moving texel contributes −(prev² + 1e-30), so a
// zero prev² still carries its flag; 1e-30 vanishes in the f32 rounding of
// any non-black block. (Adding the two variances in one channel does not
// work: Σ prev² is of the interpolated reprojection while the mean is of the
// clamped value, so that variance is often negative and the upscaler clamps
// it at 0 on its own. Measured: 2368 signal texels off at 960×540.)
//
// Edges: the upscaler clamps loads to the image and emits ⌈n/2^l⌉ blocks
// (partial blocks repeat the edge texel; 'clamp' with ceil sizes). Expressed
// here by running the Reducer on the size rounded up to whole source tiles
// (16² for levels ≤ 4) and clamping in the load hook; the store hook drops
// blocks past the outputs.

const shadingShader = SHADING_CHANGE_SHADER;
const shadingBindings = shadingShader.slice( 0, shadingShader.indexOf( '// Per-thread block state' ) ).replaceAll( '@group(0)', '@group(1)' );
const shadingHelpers = shadingShader.slice( shadingShader.indexOf( '// Per-scale base noise floors' ), shadingShader.indexOf( '@compute' ) );
const shadingEntries = [
	{ binding: 0, visibility, buffer: { type: 'uniform' } },
	...[ 1, 2, 3, 4, 5 ].map( binding => ( { binding, visibility, texture: { sampleType: 'unfilterable-float' } } ) ),
	{ binding: 6, visibility, storageTexture: { access: 'write-only', format: 'r32float' } },
	{ binding: 7, visibility, storageTexture: { access: 'write-only', format: 'r32float' } },
	{ binding: 8, visibility, texture: { sampleType: 'unfilterable-float' } },
	{ binding: 9, visibility, texture: { sampleType: 'uint' } },
	{ binding: 10, visibility, storageTexture: { access: 'write-only', format: 'rgba32uint' } }
];

const shadingTool = /* wgsl */ `
${ shadingBindings }
${ shadingHelpers }

// The exposure texels, as the upscaler hoists them per invocation.
struct Conditioning { hostRatio: f32, conditioning: f32, memoryScale: f32 }
fn conditioningLoad() -> Conditioning {
    let frameInfo = textureLoad(exposureTex, vec2i(0), 0);
    let hostPrev = textureLoad(exposurePrevTex, vec2i(0), 0).b;
    var c: Conditioning;
    c.hostRatio = select(1.0, frameInfo.b / hostPrev, hostPrev > 1.0e-4 && frameInfo.b > 1.0e-4);
    c.conditioning = max(frameInfo.r, 1.0e-4);
    c.memoryScale = 1.0 / (c.conditioning * select(1.0, frameInfo.b, frameInfo.b > 1.0e-4));
    return c;
}

HOIST

// The block's sums with the moving flag taken out of the sign of .w.
fn blockSums(value : vec4f) -> vec4f { return vec4f(value.xyz, abs(value.w)); }
`;

// Σ of x, y, z; Σ |w|, negative if either side is (the moving flag).
const shadingReduce = /* wgsl */ `fn reduce(a : vec4f, b : vec4f) -> vec4f {
    let squares = abs(a.w) + abs(b.w);
    return vec4f(a.xyz + b.xyz, select(squares, -squares, a.w < 0.0 || b.w < 0.0));
}`;

const shadingLoad = /* wgsl */ `
fn load(texel : vec2i, layer : u32) -> vec4f {
    let maxCoord = vec2i(C.renderSize) - 1;
    let coord = min(texel, maxCoord);
    let c = conditioningOf();
    let hostLuma = luma(textureLoad(inputColor, coord, 0).rgb);
    // The padded size is whole tiles, so every texel is loaded exactly once: the upscaler's one history store per texel.
    if (all(texel <= maxCoord)) { textureStore(lumaHistoryOut, texel, vec4f(hostLuma, 0.0, 0.0, 0.0)); }
    let motion = textureLoad(dilatedMotion, coord, 0).xy;
    let disocclusion = clamp(textureLoad(masks, coord, 0).r, 0.0, 1.0);
    let still = length(motion * C.renderSize) < SHADING_MEMORY_STILL_PX && disocclusion < 0.5;
    let moving = select(select(1.0, 0.0, still), 1.0, hasFlag(FLAG_RESET));
    let sums = lumaPair(coord, hostLuma * c.conditioning, c.hostRatio, c.conditioning, motion, disocclusion);
    return vec4f(sums.xyz, select(sums.w, -(sums.w + 1.0e-30), moving > 0.5));
}
`;

// Level 2 (4×4 blocks): mid memory and response, kept in workgroup memory for level 3.
// Level 3 (8×8 blocks): coarse memory and response, then the signal of the block's
// 4×4 output texels: max(mid, coarse). The Reducer emits levels in order with a
// workgroup barrier between them, and a workgroup emits every 4×4 block of its 8×8 ones.
const shadingStore = /* wgsl */ `
var<workgroup> midResponse : array<f32, 256>;

fn store(level : u32, texel : vec2i, layer : u32, value : vec4f) {
    if (level > 3u) { return; }
    let outputSize = vec2i(textureDimensions(shadingChangeOut));
    let memoryScale = conditioningOf().memoryScale;
    let moving = value.w < 0.0;
    if (level == 2u) {
        if (any(texel * 2 >= outputSize)) { return; }
        let words = textureLoad(blockMemoryIn, texel, 0);
        let mean = value.x / 16.0 * memoryScale;
        let memory = select(memoryDistance(words, mean), 1.0, moving);
        textureStore(blockMemoryOut, texel, memoryUpdate(words, mean, moving));
        midResponse[u32(texel.y & 15) * 16u + u32(texel.x & 15)] = scaleResponse(blockSums(value), 16.0, SHADING_FLOOR_MID, memory);
        return;
    }
    if (any(texel * 4 >= outputSize)) { return; }
    let coord = texel + vec2i(0, i32(ceil(C.renderSize.y / 4.0)));
    let words = textureLoad(blockMemoryIn, coord, 0);
    let mean = value.x / 64.0 * memoryScale;
    let memory = select(memoryDistance(words, mean), 1.0, moving);
    textureStore(blockMemoryOut, coord, memoryUpdate(words, mean, moving));
    let coarse = scaleResponse(blockSums(value), 64.0, SHADING_FLOOR_COARSE, memory);
    for (var j = 0; j < 4; j++) {
        let mid = texel * 2 + vec2i(j & 1, j >> 1);
        let response = max(midResponse[u32(mid.y & 15) * 16u + u32(mid.x & 15)], coarse);
        for (var k = 0; k < 4; k++) {
            let pixel = mid * 2 + vec2i(k & 1, k >> 1);
            if (all(pixel < outputSize)) { textureStore(shadingChangeOut, pixel, vec4f(response, 0.0, 0.0, 1.0)); }
        }
    }
}
`;

// Per call (the load and store hooks have no per-invocation prologue), or cached in private memory.
const conditioningPerCall = 'fn conditioningOf() -> Conditioning { return conditioningLoad(); }';
const conditioningCached = `var<private> conditioningLoaded : bool;
var<private> conditioningCache : Conditioning;
fn conditioningOf() -> Conditioning {
    if (!conditioningLoaded) { conditioningCache = conditioningLoad(); conditioningLoaded = true; }
    return conditioningCache;
}`;

/**
 * Options are for the experiments (`--only=experiment/`): `levels` adds unemitted
 * deeper levels to grow the tile (the store hook ignores them), `cache` hoists
 * the exposure texels, `patch` rewrites the kernel WGSL (self-dispatch).
 */
async function upscalerShadingChange( device, { scene, tools }, { name = 'generator/upscaler-shading-change', levels = [ 2, 3 ], cache = false, patch = null } = {} ) {

	const shared = tools.upscalerShading();
	const group = device.createBindGroup( { layout: device.createBindGroupLayout( { entries: shadingEntries } ), entries: shared.groupEntries() } );
	const pyramid = await reducer( device, name, {
		reduction: { kind: 'sum', wgsl: shadingReduce }, levels, edges: 'clamp',
		bindings: { wgsl: shadingTool.replace( 'HOIST', cache ? conditioningCached : conditioningPerCall ), entries: shadingEntries }, load: shadingLoad, store: shadingStore
	} );
	const sourceTile = Math.max( 16, 2 ** levels[ levels.length - 1 ] ), pad = n => Math.ceil( n / sourceTile ) * sourceTile;
	const dispatch = { width: pad( scene.width ), height: pad( scene.height ), bindGroup: group };
	let encodePass = pass => pyramid.encodePass( pass, dispatch );
	if ( patch ) {

		const code = patch( pyramid.wgsl );
		const module = await compile( device, name, code );
		const pipeline = await device.createComputePipelineAsync( { layout: pyramid.layout, compute: { module, entryPoint: 'main' } } );
		const bindGroup = pyramid.bindGroup( dispatch ), workgroups = pyramid.workgroups( dispatch.width, dispatch.height );
		encodePass = pass => { pass.setPipeline( pipeline ); pass.setBindGroup( 0, bindGroup ); pass.setBindGroup( 1, group ); pass.dispatchWorkgroups( ...workgroups ); };

	}

	return { name, pass: 'shading', dispatches: 1, encodePass, check: () => shared.check(), destroy: () => shared.destroy() };

}

/**
 * Kernel experiment (levels [ 2, 3 ]): level 2 goes to a second workgroup
 * region instead of overwriting level 1 in place, which drops the barrier
 * between reading level 1 and writing level 2.
 */
function pingPong( wgsl ) {

	const at = wgsl.lastIndexOf( '\tworkgroupBarrier();\n\tif (local.x < 2u && local.y < 2u) {' );
	const steps = [
		[ 'array<vec4f, 64>', 'array<vec4f, 80>' ],
		[ '\t// Every reader must finish before the shared tile is overwritten.\n\tworkgroupBarrier();\n', '' ],
		[ 'ds_tile[local.y * 8u + local.x] = value;', 'ds_tile[64u + local.y * 4u + local.x] = value;' ]
	];
	let head = wgsl.slice( 0, at );
	for ( const [ from, to ] of steps ) {

		if ( ! head.includes( from ) ) throw new Error( `pingPong: kernel changed (${ JSON.stringify( from ) })` );
		head = head.replace( from, to );

	}

	const tail = wgsl.slice( at ).replace( /ds_tile\[\(lo\.y \+ (\d)u\) \* 8u \+ lo\.x \+ (\d)u\]/g, 'ds_tile[64u + (lo.y + $1u) * 4u + lo.x + $2u]' );
	if ( tail === wgsl.slice( at ) ) throw new Error( 'pingPong: kernel changed (level 3 reads)' );
	return head + tail;

}

const shadingExperiments = [
	{ name: 'experiment/shading-pingpong', patch: pingPong },
	{ name: 'experiment/shading-cached', cache: true },
	{ name: 'experiment/shading-tile16', levels: [ 2, 3, 5 ] },
	{ name: 'experiment/shading-tile16-cached', levels: [ 2, 3, 5 ], cache: true }
];

export const generatorPasses = [
	{ name: 'generator/luma-pyramid', pass: 'exposure', compareWith: [ 'fsr/luma-pyramid-correct', 'upscaler/exposure' ], milestone: 'M2T.3', create: lumaPyramid },
	{ name: 'generator/luma-pyramid-fused', pass: 'exposure', compareWith: [ 'fsr/luma-pyramid-correct', 'upscaler/exposure' ], milestone: 'M2T.3', create: lumaPyramidFused },
	{ name: 'generator/luma-pyramid-clamp', pass: 'exposure', compareWith: [ 'fsr/luma-pyramid' ], milestone: 'M2T.3', create: lumaPyramidClamp },
	{ name: 'generator/upscaler-exposure', pass: 'exposure', compareWith: [ 'upscaler/exposure', 'fsr/luma-pyramid-correct' ], milestone: 'M2T.3', create: ( device, context ) => upscalerExposure( device, context, { taps: false } ) },
	{ name: 'generator/upscaler-exposure-taps', pass: 'exposure', compareWith: [ 'upscaler/exposure' ], milestone: 'M2T.3', create: ( device, context ) => upscalerExposure( device, context, { taps: true } ) },
	{ name: 'generator/shading-change-pyramid', pass: 'shading', compareWith: [ 'fsr/shading-change-pyramid', 'upscaler/shading-change' ], milestone: 'M2T.3', create: shadingChangePyramid },
	{ name: 'generator/upscaler-shading-change', pass: 'shading', compareWith: [ 'upscaler/shading-change', 'fsr/shading-change-total' ], milestone: 'M2T.3', create: ( device, context ) => upscalerShadingChange( device, context ) },
	// Built only when --only names them.
	...shadingExperiments.map( options => ( { name: options.name, pass: 'shading', compareWith: [ 'upscaler/shading-change', 'generator/upscaler-shading-change' ], milestone: 'experiment', experiment: true, create: ( device, context ) => upscalerShadingChange( device, context, options ) } ) )
];
