// pmndrs/upscaler's exposure and shading-change passes, copied verbatim as
// baselines. Do not edit the WGSL below: re-copy it when the upscaler changes.
//
// Source: github.com/pmndrs/upscaler @ f38e9ac (shaders last changed in 8ffb3fd),
// packages/core/src/shaders/common.ts (WGSL_CONSTANTS, WGSL_COLOR),
// luminancePyramid.ts (LUMINANCE_PYRAMID_SHADER), shadingChange.ts
// (SHADING_CHANGE_SHADER) and wgsl.ts (assembleShader). MIT licence,
// Copyright (c) 2026 Dennis Smolek. The bindings are the upscaler's own
// (@group(0), bindings 0-10); nothing needed adapting.

/** assembleShader (wgsl.ts): trimmed parts joined by blank lines, duplicates dropped. */
function assembleShader( ...parts ) {

	const seen = new Set();
	const out = [];
	for ( const part of parts ) {

		const key = part.trim();
		if ( key.length === 0 || seen.has( key ) ) continue;
		seen.add( key );
		out.push( key );

	}

	return out.join( '\n\n' ) + '\n';

}

export const WGSL_CONSTANTS = /* wgsl */ `
struct FsrConstants {
    renderSize      : vec2f,  // offset  0 — jittered render resolution (px)
    displaySize     : vec2f,  // offset  8 — output resolution (px)
    renderSizeInv   : vec2f,  // offset 16
    displaySizeInv  : vec2f,  // offset 24
    jitter          : vec2f,  // offset 32 — current sub-pixel offset (render px, top-left origin)
    jitterPrev      : vec2f,  // offset 40
    motionScale     : vec2f,  // offset 48 — NDC velocity delta -> UV delta (0.5, -0.5)
    depthNearFar    : vec2f,  // offset 56 — camera near/far for linearization
    sharpness       : f32,    // offset 64 — RCAS attenuation 0..1 (1 = sharpest)
    maxAccumulation : f32,    // offset 68 — max history sample count
    exposure        : f32,    // offset 72 — pre-exposure before invertible tonemap
    deltaTime       : f32,    // offset 76 — seconds
    flags           : u32,    // offset 80 — FLAG_* bits
    frameIndex      : u32,    // offset 84
    debugMode       : u32,    // offset 88 — DebugView
    _pad            : u32,    // offset 92
}
@group(0) @binding(0) var<uniform> C : FsrConstants;

const FLAG_RESET : u32 = 1u;
const FLAG_REVERSED_DEPTH : u32 = 2u;
const FLAG_PERSPECTIVE : u32 = 4u;
const FLAG_INPUT_REINHARD : u32 = 8u;
const FLAG_INPUT_DISPLAY : u32 = 16u;
const FLAG_LOCKS : u32 = 32u;
const FLAG_AUTO_EXPOSURE : u32 = 64u;
const FLAG_SHADING_CHANGE : u32 = 128u;
const FLAG_REACTIVE : u32 = 256u;
const FLAG_RCAS_DENOISE : u32 = 512u;
const FLAG_EXTERNAL_EXPOSURE : u32 = 1024u;

fn hasFlag(bit : u32) -> bool { return (C.flags & bit) != 0u; }
`;

export const WGSL_COLOR = /* wgsl */ `
fn luma(c : vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn rgbToYCoCg(c : vec3f) -> vec3f {
    return vec3f(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r             - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn yCoCgToRgb(c : vec3f) -> vec3f {
    return vec3f(
        c.x + c.y - c.z,
        c.x       + c.z,
        c.x - c.y - c.z,
    );
}
`;

export const LUMINANCE_PYRAMID_SHADER = assembleShader(
	// Exposure names the final reserved slot of the shared 96-byte layout.
	// Other passes keep it as padding; the writer always supplies 32 here.
	WGSL_CONSTANTS.replace( '_pad            : u32,    // offset 92',
		'exposureTaps    : u32,    // offset 92 — fixed metering grid, uniform loop bound' ),
	WGSL_COLOR,
	/* wgsl */ `
@group(0) @binding(1) var inputColor : texture_2d<f32>;
@group(0) @binding(2) var linearSampler : sampler;
@group(0) @binding(3) var prevExposure : texture_2d<f32>;
@group(0) @binding(4) var exposureOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var externalExposure : texture_2d<f32>;
@group(0) @binding(6) var hostPreExposure : texture_2d<f32>;

// 32×32 = 1024 bilinear taps across the whole frame — a coarse but stable
// average for exposure (each tap already averages 4 texels).
const EXPOSURE_TAPS : u32 = 32u;
// Middle-grey target: the exposure maps average scene luma to this.
const EXPOSURE_KEY : f32 = 0.18;
// Clamp so a pitch-black or fully blown-out frame can't drive exposure to
// infinity/zero and destabilize the accumulation it is meant to steady.
const EXPOSURE_MIN : f32 = 0.02;
// The brightening cap is also the dark-scene highlight ceiling. History is
// stored in f16 as x/(1+x), x the pre-exposed color, and tonemapInvert clamps
// at 0.999: nothing resolves above 999 / exposure, and below that a stored
// value moves in relative steps of ~(1 + x)/2048. A mostly-black frame drives the
// log-average to its floor, so the old cap (80) put every highlight above
// ~12.5 linear on the ceiling and averaged sub-pixel ones to the same value
// whatever their brightness (issue #49). 8 (three stops) lifts the ceiling to
// ~125 and stays above every bench scene's metered target (2.4–6.8), so
// ordinary scenes are unchanged. Highlights can't key the exposure instead:
// the 32×32 taps miss small and sub-pixel lights. See bench/docs/NEXT-STEPS.md.
const EXPOSURE_MAX : f32 = 8.0;
// Eye-adaptation rate (per second) toward the target exposure.
const ADAPT_SPEED : f32 = 2.5;
// Adaptation dead-band, in stops. The metering taps read the JITTERED input, so
// on a still scene the log-average wobbles with the jitter phase (edges and
// sub-pixel detail land on different texels each phase). Easing toward that
// moving target re-conditions the stored history every frame (plus f16 storage
// rounding of the exposure itself), which reads as low-level contour shimmer
// across smooth gradients. Holding the stored exposure exactly while the target
// stays within 1/16 stop (~4.4%) freezes a still scene's conditioning
// bit-for-bit; genuine lighting changes are far larger and adapt as before.
// Exposure is divided back out before display, so resting up to 1/16 stop off
// target changes nothing visible.
const ADAPT_DEADBAND_STOPS : f32 = 0.0625;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= C.displaySize)) { return; }
    // Single-thread reduction — the other 63 lanes of the lone workgroup idle.
    if (any(gid.xy != vec2u(0u))) { return; }

    // Geometric mean of luminance (average in log space) resists a few bright
    // pixels dragging the whole exposure, matching FSR2's log-average.
    var logSum = 0.0;
    for (var y = 0u; y < C.exposureTaps; y = y + 1u) {
        for (var x = 0u; x < EXPOSURE_TAPS; x = x + 1u) {
            let uv = (vec2f(f32(x), f32(y)) + 0.5) / f32(EXPOSURE_TAPS);
            let c = textureSampleLevel(inputColor, linearSampler, uv, 0.0).rgb;
            logSum = logSum + log2(max(luma(c), 1.0e-4));
        }
    }
    // Meter host-invariantly (FSR2 divides pre-exposure out of every input
    // load): the app already metered what it baked in, so a host pre-exposure
    // step must not send auto-exposure re-adapting — that multi-frame
    // conditioning drift would desynchronize history from the current frame
    // and read as a full-screen shading change.
    let hostRaw = textureLoad(hostPreExposure, vec2i(0), 0).r;
    let host = select(1.0, hostRaw, hostRaw > 0.0);
    let avgLum = exp2(logSum / f32(EXPOSURE_TAPS * EXPOSURE_TAPS)) / host;

    let targetExposure = clamp(EXPOSURE_KEY / max(avgLum, 1.0e-4), EXPOSURE_MIN, EXPOSURE_MAX);

    // Ease toward the target for eye-adaptation; snap on reset so a stale value
    // doesn't slowly fade in after a camera cut / resize.
    let prev = textureLoad(prevExposure, vec2i(0), 0).r;
    var exposure = targetExposure;
    if (!hasFlag(FLAG_RESET) && prev > 0.0) {
        exposure = prev;
        if (abs(log2(targetExposure / prev)) > ADAPT_DEADBAND_STOPS) {
            let rate = clamp(1.0 - exp2(-C.deltaTime * ADAPT_SPEED), 0.0, 1.0);
            exposure = prev + (targetExposure - prev) * rate;
        }
    }

    // Manual override: when auto-exposure is off, publish the fixed setting so
    // downstream passes read exposure from one place regardless of mode.
    exposure = select(C.exposure, exposure, hasFlag(FLAG_AUTO_EXPOSURE));

    // App-supplied exposure wins over both: a pipeline that already computes
    // exposure (its own metering pass) feeds it here, and every downstream
    // pass keeps reading this one 1×1 value. That includes the shading-change
    // detector, which scales both frames by this conditioning exposure (.r).
    // avgLum (.g) stays our own measurement, but no shader reads it.
    let ext = textureLoad(externalExposure, vec2i(0), 0).r;
    exposure = select(exposure, ext, hasFlag(FLAG_EXTERNAL_EXPOSURE));

    // Host pre-exposure rides along in .b: 0 (the dummy) means "not supplied"
    // and publishes as 1.0 so the accumulate-side ratio correction is inert.
    textureStore(exposureOut, vec2i(0), vec4f(exposure, avgLum, host, 0.0));
}
`,
);

export const SHADING_CHANGE_SHADER = assembleShader(
	WGSL_CONSTANTS,
	WGSL_COLOR,
	/* wgsl */ `
@group(0) @binding(1) var inputColor : texture_2d<f32>;
@group(0) @binding(2) var lumaHistoryIn : texture_2d<f32>;
@group(0) @binding(3) var dilatedMotion : texture_2d<f32>;
@group(0) @binding(4) var exposureTex : texture_2d<f32>;
@group(0) @binding(5) var exposurePrevTex : texture_2d<f32>;
@group(0) @binding(6) var lumaHistoryOut : texture_storage_2d<r32float, write>;
@group(0) @binding(7) var shadingChangeOut : texture_storage_2d<r32float, write>;
@group(0) @binding(8) var masks : texture_2d<f32>;
@group(0) @binding(9) var blockMemoryIn : texture_2d<u32>;
@group(0) @binding(10) var blockMemoryOut : texture_storage_2d<rgba32uint, write>;

// Per-thread block state: x = current-luma sum, y = reprojected previous-luma
// sum, z = current-luma² sum, w = previous-luma² sum (each frame's own
// within-block spread, for the block's coefficient of variation).
// Luma is averaged BEFORE taking any ratio: per-texel relative differences are
// asymmetric (the darker side of any jitter/alias residue always yields the
// larger ratio), so their signed mean carries a coherent bias on
// high-frequency content — measured as a 0.07–0.10 still-scene floor.
var<workgroup> tileSums : array<vec4f, 64>;
// 1 where any texel of the thread's 2×2 moved, was disoccluded or reset: the
// block's memory describes other content and restarts.
var<workgroup> tileMoving : array<f32, 64>;

// Per-scale base noise floors for the relative difference of block means.
// Only the 4×4 and 8×8 scales contribute to the response — this is the
// "coarse mip" of the roadmap item: 2×2 means of a thin feature still swing
// under sub-pixel jitter no matter the floor (measured as per-block speckle on
// grid intersections and silhouettes), while a genuinely changing small
// feature still moves its containing 4×4 mean.
const SHADING_FLOOR_MID : f32 = 0.08;    // 4×4 render-texel means
const SHADING_FLOOR_COARSE : f32 = 0.04; // 8×8
// Adaptive part: jitter/alias flicker of a block mean scales with the block's
// own luma contrast, so the floor grows with its coefficient of variation.
// Flat regions (cv ≈ 0) stay maximally sensitive; a checkerboard block
// (cv ≈ 1) is inherently ambiguous and defers to the variance-clip path. A
// block whose luma sits in a few texels over black (sub-texel wires, issue
// #22) reads cv ≈ √(N/m − 1) for m lit texels of N — its whole mean can come
// and go with the jitter phase, and this term keeps it from firing.
const SHADING_FLOOR_CV : f32 = 0.35;

//* Block memory (NEXT-STEPS §14, measured on Q18).
// 8 means: 4 left a quarter of the Q18 false positives (0.52 % of the frame vs
// 0.12 %), because 4 consecutive jitter phases rarely span a block's full
// alias swing.
const SHADING_MEMORY_SLOTS : u32 = 8u;
// The range is trusted only when this frame's jump from the previous mean is
// at most this × the largest jump between stored means. Alias flicker repeats
// jumps of its own size; a ramp's are small, so a step right after one (Q9
// f180) is not hidden inside the ramp's range (without the gate it lost two
// thirds of its response). At 1.0 a sixth of the false positives come back on
// Q18 (0.75 % vs 0.12 %) — the newest of 8 jumps is the largest one time in 8.
const SHADING_MEMORY_JUMP_GAIN : f32 = 1.5;
// Render-px motion above which a block's stored means describe other content.
const SHADING_MEMORY_STILL_PX : f32 = 0.05;

// Sums this texel's current luma, the previous frame's luma reprojected to the
// same world position (jitter-delta compensated; the compared value is clamped
// into the bilinear footprint's tap range — see below), and both squared.
// Reset/offscreen texels contribute neutrally
// (prev = cur), and disoccluded texels are neutralized toward it — their
// previous luma belongs to another surface, and disocclusion already discards
// that history downstream.
fn lumaPair(
    coord : vec2i,
    currentLuma : f32,
    hostRatio : f32,
    conditioning : f32,
    motion : vec2f,
    disocclusion : f32,
) -> vec4f {
    let currentSq = currentLuma * currentLuma;
    let neutral = vec4f(currentLuma, currentLuma, currentSq, currentSq);
    if (hasFlag(FLAG_RESET)) { return neutral; }
    let uv = (vec2f(coord) + 0.5) * C.renderSizeInv;
    // Off-screen tested on the motion-only reprojection, as in reconstruct.ts:
    // the jitter shift alone must not push border texels out of detection.
    let motionUv = uv - motion;
    if (any(motionUv < vec2f(0.0)) || any(motionUv > vec2f(1.0))) {
        return neutral;
    }
    // Texel i samples the scene at i + jitter, so the previous frame's
    // equivalent position shifts by the jitter delta.
    let previousUv = motionUv + (C.jitter - C.jitterPrev) * C.renderSizeInv;
    let pos = previousUv * C.renderSize - 0.5;
    let base = floor(pos);
    let maxCoord = vec2i(C.renderSize) - 1;
    let p00 = clamp(vec2i(base), vec2i(0), maxCoord);
    let p11 = clamp(vec2i(base) + 1, vec2i(0), maxCoord);
    let l00 = textureLoad(lumaHistoryIn, p00, 0).r;
    let l10 = textureLoad(lumaHistoryIn, vec2i(p11.x, p00.y), 0).r;
    let l01 = textureLoad(lumaHistoryIn, vec2i(p00.x, p11.y), 0).r;
    let l11 = textureLoad(lumaHistoryIn, p11, 0).r;
    // Last frame never sampled this exact position — only the four texels
    // around it, under a different jitter. An aliased edge between them can
    // land on either side, so no interpolation recovers the true value; the
    // honest prior is the whole tap range. Comparing against the closest value
    // in it reads 0 wherever jitter alone explains the difference, while a
    // genuine change (current outside its neighbours' range) still registers.
    // Interpolating instead left a measured ~0.7% still-scene firing floor on
    // edges and ~2% under SSGI's rotating pattern (example 09); this reads 0%
    // and 0.06%, with light-step detection unchanged.
    let scale = hostRatio * conditioning;
    let tapMin = min(min(l00, l10), min(l01, l11)) * scale;
    let tapMax = max(max(l00, l10), max(l01, l11)) * scale;
    let previousLuma = mix(clamp(currentLuma, tapMin, tapMax), currentLuma, disocclusion);
    // The spread term keeps the interpolated value: it measures last frame's
    // own within-block contrast (issue #22's two-sided floor), which the
    // closest-value prior would collapse toward the current frame.
    let fraction = pos - base;
    let reprojected = mix(mix(l00, l10, fraction.x), mix(l01, l11, fraction.x), fraction.y) * scale;
    let previousSq = mix(reprojected * reprojected, currentSq, disocclusion);
    return vec4f(currentLuma, previousLuma, currentSq, previousSq);
}

// Relative distance of two non-negative values (1 − min/max); 0 for two blacks.
fn relativeDistance(a : f32, b : f32) -> f32 {
    let maximum = max(a, b);
    if (maximum <= 1.0e-8) { return 0.0; }
    return 1.0 - min(a, b) / maximum;
}

// Slot i of a block's memory: 8 f16 means, two per word, slot 0 the newest.
fn memorySlot(words : vec4u, index : u32) -> f32 {
    let pair = unpack2x16float(words[index / 2u]);
    return select(pair.x, pair.y, (index & 1u) == 1u);
}

// How far the block's current mean is from what its memory explains: the
// distance to the stored means' [min, max], or 1 (memory does not apply, the
// previous-frame comparison alone decides) when this frame's jump exceeds
// SHADING_MEMORY_JUMP_GAIN × the largest jump the memory holds. An all-zero
// memory (allocation, or the detector re-enabled) suppresses nothing on its
// first frame, since every jump from 0 is 1 and it holds none; memoryUpdate
// then restarts it rather than keeping its zeros.
fn memoryDistance(words : vec4u, current : f32) -> f32 {
    var lo = 3.0e38;
    var hi = -3.0e38;
    var flicker = 0.0;
    var newer = memorySlot(words, 0u);
    for (var i = 0u; i < SHADING_MEMORY_SLOTS; i++) {
        let value = memorySlot(words, i);
        lo = min(lo, value);
        hi = max(hi, value);
        flicker = max(flicker, relativeDistance(newer, value));
        newer = value;
    }
    if (relativeDistance(current, memorySlot(words, 0u)) > SHADING_MEMORY_JUMP_GAIN * flicker) {
        return 1.0;
    }
    return relativeDistance(current, clamp(current, lo, hi));
}

fn halfBits(value : f32) -> u32 {
    return pack2x16float(vec2f(min(value, 65000.0), 0.0)) & 0xffffu;
}

// Pushes the current mean into slot 0 — a 16-bit funnel shift across the four
// words drops the oldest — or restarts the memory from it. An all-zero memory
// is empty and restarts too: pushed instead, its zeros would stay in the range
// for 8 frames with a stored jump of 1 (m vs 0), so the jump gate never trips
// and any darkening inside [0, m] reads as no change (measured: Q18's light
// step fully hidden 2–4 frames after re-enabling the detector, NEXT-STEPS §14).
// For a genuinely black block a restart to zeros is the same memory.
fn memoryUpdate(words : vec4u, current : f32, restart : bool) -> vec4u {
    let bits = halfBits(current);
    if (restart || all(words == vec4u(0u))) {
        let both = bits | (bits << 16u);
        return vec4u(both, both, both, both);
    }
    return vec4u(
        (words.x << 16u) | bits,
        (words.y << 16u) | (words.x >> 16u),
        (words.z << 16u) | (words.y >> 16u),
        (words.w << 16u) | (words.z >> 16u),
    );
}

// Relative difference of two block means, gated by the scale's base floor
// plus the block's own contrast-scaled flicker allowance. The contrast is the
// within-frame spread of BOTH frames over their joint mean. Measuring it on
// the current frame alone made the gate one-sided: a thin bright feature that
// this jitter phase missed leaves a block with no spread (cv = 0, base floor
// only) against a previous frame that hit it, so the block fired at full
// strength exactly when its content vanished (issue #22, NEXT-STEPS §9 — the
// same artefact was most of the still-scene block speckle on Q1/Q12 and the
// motion false positives on Q4). The between-frame shift is deliberately left
// out of the spread, so a genuine change on a flat surface does not raise its
// own floor. The memory distance can only lower the difference.
fn scaleResponse(sums : vec4f, count : f32, floorBase : f32, memory : f32) -> f32 {
    let maximum = max(sums.x, sums.y);
    if (maximum <= 1.0e-5) { return 0.0; }
    let means = sums.xy / count;
    let variances = max(sums.zw / count - means * means, vec2f(0.0));
    let cv = sqrt(0.5 * (variances.x + variances.y)) / max(0.5 * (means.x + means.y), 1.0e-4);
    let floorValue = floorBase + SHADING_FLOOR_CV * cv;
    let relative = min(1.0 - min(sums.x, sums.y) / maximum, memory);
    return smoothstep(floorValue, floorValue * 3.0, relative);
}

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid : vec3u,
    @builtin(local_invocation_id) lid : vec3u,
    @builtin(local_invocation_index) lidx : u32,
) {
    // Hoisted once per invocation — the source-style candidate reloaded these
    // 1×1 texels inside every reduction tap, which dominated its cost.
    let frameInfo = textureLoad(exposureTex, vec2i(0), 0);
    let hostPrev = textureLoad(exposurePrevTex, vec2i(0), 0).b;
    let hostRatio = select(1.0, frameInfo.b / hostPrev, hostPrev > 1.0e-4 && frameInfo.b > 1.0e-4);
    let conditioning = max(frameInfo.r, 1.0e-4);
    // Block memory is kept free of both exposures, so it survives their changes.
    let memoryScale = 1.0 / (conditioning * select(1.0, frameInfo.b, frameInfo.b > 1.0e-4));

    //* Fine Sums + Luma History (2×2 render block per thread)
    let origin = vec2i(gid.xy) * 2;
    let maxCoord = vec2i(C.renderSize) - 1;
    var sums0 = vec4f(0.0);
    var moving = select(0.0, 1.0, hasFlag(FLAG_RESET));
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let coord = clamp(origin + vec2i(x, y), vec2i(0), maxCoord);
            // History stays in the caller's host domain; host + conditioning
            // are applied to both sides of the comparison only.
            let hostLuma = luma(textureLoad(inputColor, coord, 0).rgb);
            textureStore(lumaHistoryOut, origin + vec2i(x, y), vec4f(hostLuma, 0.0, 0.0, 0.0));
            let motion = textureLoad(dilatedMotion, coord, 0).xy;
            let disocclusion = clamp(textureLoad(masks, coord, 0).r, 0.0, 1.0);
            let still = length(motion * C.renderSize) < SHADING_MEMORY_STILL_PX && disocclusion < 0.5;
            moving = max(moving, select(1.0, 0.0, still));
            sums0 += lumaPair(coord, hostLuma * conditioning, hostRatio, conditioning, motion, disocclusion);
        }
    }
    tileSums[lidx] = sums0;
    tileMoving[lidx] = moving;
    workgroupBarrier();

    // Grid guards sit AFTER the barrier: every invocation must reach it
    // (uniform control flow), and out-of-range texture writes above are no-ops.
    if (any(vec2f(gid.xy) >= C.renderSize)) { return; }
    let outputSize = vec2u(textureDimensions(shadingChangeOut));
    if (any(gid.xy >= outputSize)) { return; }

    //* Coarse Sums (workgroup-local: 4×4 render per mid, 8×8 per coarse)
    let base1 = (lid.xy / 2u) * 2u;
    var sums1 = vec4f(0.0);
    var moving1 = 0.0;
    for (var y = 0u; y < 2u; y++) {
        for (var x = 0u; x < 2u; x++) {
            let index = (base1.y + y) * 8u + base1.x + x;
            sums1 += tileSums[index];
            moving1 = max(moving1, tileMoving[index]);
        }
    }

    let base2 = (lid.xy / 4u) * 4u;
    var sums2 = vec4f(0.0);
    var moving2 = 0.0;
    for (var y = 0u; y < 4u; y++) {
        for (var x = 0u; x < 4u; x++) {
            let index = (base2.y + y) * 8u + base2.x + x;
            sums2 += tileSums[index];
            moving2 = max(moving2, tileMoving[index]);
        }
    }

    //* Block Memory — every thread reads its blocks; one thread per block
    //* (the block's top-left) advances them. Ping-ponged, so no read races a write.
    let midRows = i32(ceil(C.renderSize.y / 4.0));
    let midCoord = vec2i(gid.xy / 2u);
    let coarseCoord = vec2i(gid.xy / 4u) + vec2i(0, midRows);
    let midWords = textureLoad(blockMemoryIn, midCoord, 0);
    let coarseWords = textureLoad(blockMemoryIn, coarseCoord, 0);
    let midMean = sums1.x / 16.0 * memoryScale;
    let coarseMean = sums2.x / 64.0 * memoryScale;
    let midMemory = select(memoryDistance(midWords, midMean), 1.0, moving1 > 0.5);
    let coarseMemory = select(memoryDistance(coarseWords, coarseMean), 1.0, moving2 > 0.5);
    if (all((lid.xy & vec2u(1u)) == vec2u(0u))) {
        textureStore(blockMemoryOut, midCoord, memoryUpdate(midWords, midMean, moving1 > 0.5));
    }
    if (all((lid.xy & vec2u(3u)) == vec2u(0u))) {
        textureStore(blockMemoryOut, coarseCoord, memoryUpdate(coarseWords, coarseMean, moving2 > 0.5));
    }

    //* Resolve — strongest floor-gated mean-ratio across the coarse scales.
    let response = max(
        scaleResponse(sums1, 16.0, SHADING_FLOOR_MID, midMemory),
        scaleResponse(sums2, 64.0, SHADING_FLOOR_COARSE, coarseMemory),
    );
    textureStore(shadingChangeOut, vec2i(gid.xy), vec4f(response, 0.0, 0.0, 1.0));
}
`,
);
