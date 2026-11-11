# Tool fixtures: the upscaler's pyramid passes

Reference fixtures and a benchmark for the two SPD passes of the FSR 3.1 upscaler, so a downsampler-generator version (M2T.1 hooks, M2T.2 `reduceToScalar`) can be proven against them. Each pass exists in up to three forms:

1. **upscaler**: what pmndrs/upscaler runs today, copied verbatim.
2. **fsr**: AMD's pass, ported to WGSL with its quirks reproduced.
3. **generator**: the downsampler-generator version. These are plug-in points for now (`generator.js`).

Every fixture is checked against a CPU oracle before it is timed.

```
npm run tools                                   # build, check + benchmark all sizes → results/tools/<date>.json
node test/tools/run.js --check                  # oracles only
node test/tools/run.js --mutate                 # fail-before proof: perturbs the oracles' inputs; every check must fail
node test/tools/run.js --sizes=1920x1080 --only=fsr/luma,upscaler/exposure --rounds=11 --repeats=16 --output=results/tools/x.json --verbose
node test/tools/run.js --adapter=swiftshader --check   # software adapter: 4 storage textures, so pyramid-amd is skipped
```

The same harness runs in a browser at `/test/tools/` (`npm run lab`, then open the page).

## Inputs (`scene.js`)

The inputs are synthetic and deterministic (seeded), at render sizes 960×540, 1280×720, 1129×635, 1707×960 and 1920×1080. They are generated on the CPU and rounded to the format they are uploaded in, so the oracles read exactly what the GPU reads.

The scene:
- A static camera looks at an HDR sky with a sun (luma ~55) over a value-noise ground plane.
- Six objects move between the previous frame (t = 0) and the current one (t = 1). One is a small emissive light with luma ~95.
- A ground region is lit 2.5× brighter in the current frame. This is a genuine shading change.
- A 1.3 px checker aliases under the jitter.
- Both frames carry ±3 % render noise.

Derived per texel:
- Current and previous colour (rgba16float).
- FSR current and previous luma (`RGBToLuma(max(0, rgb))`, R16F).
- Farthest depth in metres (R16F).
- Motion dilated by the nearest depth of the 3×3 neighbourhood:
  - FSR convention, `prev = uv + mv`, RG16F.
  - Upscaler convention, `prev = uv − motion`, rgba16float.
- The upscaler's luma history (r32float).
- A disocclusion mask (rgba8unorm `.r`).

Jitters are (−0.375, −0.1111) for the current frame and (0.125, 0.2222) for the previous one. Velocities are ±1.25, ±2.25, 3.25 or 0.25 px. With these, no reprojected position lands within 0.02 px of a texel edge, so `floor()` decisions are stable against float16 motion.

## Fixtures

| Fixture | What it is | Dispatches |
|---|---|---|
| `upscaler/exposure` | `luminancePyramid.ts` verbatim: a single invocation takes 32×32 bilinear taps of the colour, averages log2 luma and eases toward the target. One 8×8 workgroup, as `UpscalerCore` dispatches it | 1 |
| `fsr/luma-pyramid` | FSR 3.1 luma pyramid as shipped. Load `(max(ε, log luma), luma, farthest depth)`; SPD average; mip 1 depth → `FarthestDepthMip1` (R16F, ⌊w/2⌋×⌊h/2⌋); mip 6 `.xy` → `rw_spd_mip5` (RG16F); texel (0,0) of mip `mips` → `FrameInfo` with temporal smoothing and `ComputeAutoExposureFromLavg` | 2 (tile, tail) |
| `fsr/luma-pyramid-correct` | The same load and epilogue over **every** texel, equally weighted: per-tile sums with an in-bounds count, one workgroup combines them, then `finish()`. Mip 1 is ⌈w/2⌉×⌈h/2⌉ over in-bounds texels only. This is what M2T.2 should match | 2 (tile, combine) |
| `upscaler/shading-change` | `shadingChange.ts` verbatim: a fused 16×16-tile pass with 4×4/8×8 block means, block memory and luma history. ⌈⌈w/2⌉/8⌉ × ⌈⌈h/2⌉/8⌉ workgroups | 1 |
| `fsr/shading-change-pyramid` | FSR 3.1 shading-change pyramid. Custom load: 5-tap crosses of current luma × exposure and previous luma reprojected by dilated motion, sorted, signed min-difference → `(diff, sign, 1)`; SPD average; mips 1–3 `.xy` stored (RG16F), which is all the consumer reads | 1 |
| `fsr/shading-change-pyramid-amd` | The same, storing mips 1–6 as AMD does. Needs 6 storage textures | 1 |
| `fsr/shading-change` | The `ShadingChange` consumer: bilinear samples of mips 1–3 at the jittered, `ClampUv`'d position; max of `\|x·y\|`; saturate → R8. Reads the mips `fsr/shading-change-pyramid` stored | 1 |
| `fsr/shading-change-total` | pyramid (mips 1–3) + consumer: the pair the upscaler's fused pass replaces | 2 |
| `floor` | One single-invocation dispatch: the device's per-dispatch cost floor | 1 |

The FSR fixtures read the exposure the oracle computes for `fsr/luma-pyramid` on the same scene, so the fixtures stay independent. The upscaler fixtures read the oracle's `upscaler/exposure` output.

### Provenance

- **AMD**: FidelityFX SDK at `7539ae8` (FSR3UPSCALER 3.1.6), `Kits/FidelityFX/upscalers/fsr3/`. The sources are:
  - `include/gpu/spd/ffx_spd.h`: the SPD skeleton, ported in `spd.js`.
  - `ffx_fsr3upscaler_luma_pyramid.h`, `_shading_change_pyramid.h`, `_shading_change.h`, `_common.h` and `_callbacks_hlsl.h`: the tool code, in `fsr-*.js`.
  - `internal/ffx_fsr3upscaler.cpp`: the setup, resource sizes and clears.

  MIT licence. The notice is kept in `spd.js`, and the `fsr-*.js` headers point to it. Function and constant names follow the HLSL.
- **pmndrs/upscaler**: `f38e9ac`; the shaders last changed in `8ffb3fd`. `upscaler-wgsl.js` holds the WGSL of `common.ts`, `luminancePyramid.ts` and `shadingChange.ts`, extracted verbatim, plus `assembleShader`. Re-extract it when the upscaler changes; don't edit it by hand. The bindings were already plain `@group(0)`, so nothing was adapted. MIT licence.

### How the FSR port differs from AMD (deliberately)

- **The atomic tail is a second dispatch.** It is one workgroup running AMD's own `SpdDownsampleMips_6_7` and `SpdDownsampleNextFour(…, 8, …)` on the same mip-6 data. WGSL has no globally coherent memory or forward-progress guarantee for the "last workgroup continues" pattern (SPEC §2.2). The shading-change pyramid's tail is not run at all: it stores nothing (`StorePyramid` handles indices 0–5 only).
- **The non-wave LDS path** (`FFX_SPD_NO_WAVE_OPERATIONS`) is used instead of quad intrinsics. `ffxRemapForWaveReduction` is reconstructed from `ffx_core` (which is not in the sparse clone). It has the property the LDS path needs: threads 0..15 cover 4×4 and threads 0..3 cover 2×2.
- **Hook-shaped callbacks** (SPEC §14.1):
  - `load( texel, layer )` is `SpdLoadSourceImage` without its `ClampLoad`. The skeleton clamps, which is edge policy `'clamp'`.
  - `store( level, … )` is `SpdStore` with level = mip index + 1.
  - `spdLoad` is `SpdLoad`.
  - The tool bindings live in `@group(1)`.
- **Resources**:
  - `rw_spd_mip5` is a packed-halves `u32` buffer (`pack2x16float`, so the RG16F quantization is the same). WebGPU can't read and write a storage texture of that format in one pass.
  - `FrameInfo` is a 16-byte storage buffer rather than a 1×1 RGBA32F UAV.
  - R16F/RG16F/R8 outputs need `texture-formats-tier1`. Without it the fixtures write rgba16float/rgba8unorm: same values, more bytes.
- **D3D out-of-bounds semantics are explicit.** WebGPU leaves out-of-bounds texture access implementation-defined, so stores outside a resource are dropped by a bounds test, and `SpdLoad` outside mip 6 returns 0.
- **`ffxPow(x, fShadingChangeSamplePow)` is left out.** The exponent is 1/1, so it is the identity. WGSL `pow` would add error that a compiler folding `pow(x, 1)` does not.
- **In `ComputeMinimumDifference`, `b`'s update reads `fSet0[a]` after `a` advanced**, which is index 5 (out of bounds) on the last iteration. The WGSL clamps that index. `b` is dead after the last iteration, so results are unchanged.

### AMD quirks reproduced (with what they cost on these scenes)

- **Exposure covers `[0, 2^mips)²` only.** `mips = min(⌊log2 max(w,h)⌋, 12)`, and only texel (0,0) of mip `mips` is used.
- **That window also reads zeros.** The mip-6 resource is `(⌊w/2⌋ >> 5) × (⌊h/2⌋ >> 5)`, so:
  - Tiles past it are dropped on store.
  - The tail's out-of-bounds `SpdLoad`s return 0 (the D3D UAV rule).
  - At 1280×720, rows 704–1023 of the 1024² window contribute zero: tile row 11 is dropped, since `360 >> 5 = 11`.

  Unsmoothed log luma, linear scene luma and exposure, relative to the every-texel result (from `results/tools/m5-pro.json`):

  | Size | Window, real rows in it | log luma | scene luma | exposure |
  |---|---|---:|---:|---:|
  | 960×540 | 512², all real | +1.8 % | +5.5 % | −0.8 % |
  | 1280×720 | 1024², 704 real rows | −27.9 % | −28.5 % | +13.5 % |
  | 1129×635 | 1024², 576 real rows | −38.4 % | −40.4 % | +19.0 % |
  | 1707×960 | 1024², 960 real rows | −8.7 % | −4.4 % | +4.0 % |
  | 1920×1080 | 1024², all real | +1.7 % | +5.5 % | −0.8 % |

  Where the window is all real, it is the sky-heavy top-left that biases the result upward.
  Exposure is ∝ exp(−L), so its relative change is −ΔL.
- **`max(ε, log luma)`.** Every texel darker than luma 1 counts as log 0, not negative. The "log average" is the average of `max(0, log L)`.
- **The reset clears `FrameInfo` to (−1, 1, 0, 0).** `1 < resetAutoExposureAverageSmoothing (1e4)`, so the first frame after a reset eases from log luma 1 at `1 − exp(−Δt)` (1.65 % at 60 Hz) instead of jumping. The check runs both from this state and from an unsmoothed one (previous log luma = 1e4).
- **Clamped edge loads.** Partial tiles repeat the edge texel. `FarthestDepthMip1` and SpdMips are ⌊w/2⌋ wide, so the last odd row or column is dropped. Every stored texel of both pyramids covers in-bounds source only; the clamps matter for the 5-tap crosses and for tiles beyond the stored range.
- **The fp16 paths (`SpdLoadSourceImageH` etc.) are stubs returning 0.** They are not ported; the fixtures are the float path.
- **(Input side, not a pyramid quirk.)** `FindDepthExtents` only grows the farthest depth from a sample that is also nearer than the nearest so far, so it always equals the centre depth. `scene.js` therefore takes the centre depth.

## Oracles and tolerances

The oracles redo the GPU's float32 arithmetic in JavaScript (`Math.fround` at each step, in the kernel's addition order). Most outputs therefore match to within one rounding of the stored format. The checks:

| Output | Tolerance | Why |
|---|---|---|
| `FarthestDepthMip1`, SPD mip 6 | 1 f16 ulp | Same f32 sum; a reassociating compiler moves it by an f32 ulp, which can flip one f16 rounding |
| `FrameInfo` (exposure, log luma, scene luma) | 1e-4 relative | GPU `log`/`exp` are not correctly rounded (3 ulp, or 2^-21 absolute). A flipped mip-6 rounding moves L by ~2^-11/256. Exposure ∝ exp(−L). 1e-4 still catches one dropped tile (≥ 1/510 at 1080p). Measured: ≤ 1e-7 |
| shading SpdMips 1–6 | max(1 f16 ulp, 2^-21) per value; up to 1e-4 × mip-1 values outside | `ComputeMinimumDifference` makes discrete choices (sort, ties in `\|d\| < \|min\|`, a/b advance, `IsUvInside`, `floor`), so one f32 ulp can change the selected difference. A flipped source texel moves one texel per mip, by ¼ as much each mip down. Measured: 0–2 flipped texels per size |
| FSR `ShadingChange` (R8) | per texel: bilinear-weight bound + 1/255 | GPU bilinear weights are fixed point (≥ 8 fractional bits), so each tap range is allowed 2·2^-8, propagated through `\|x·y\|`. The consumer check reads the GPU's own mips, so it is independent of the pyramid check |
| `upscaler/exposure` | per output: computed bound | Same bilinear bound per tap, propagated through log2 and averaged over the 1024 taps, plus 1 f16 ulp (rgba16float output) and 1e-5 relative. Also: the dead-band decision must be > 0.01 stops from its threshold |
| upscaler signal | 2e-3 absolute; up to 1e-4 outside; jump-gate flips within 1e-4 of the threshold excluded | `E[x²] − E[x]²` in f32 cancels on flat blocks, then goes through ×0.35 and up to ~19× smoothstep slope. Measured: ≤ 3e-5 |
| upscaler luma history | 2 f32 ulp | A dot product; FMA contraction may differ |
| upscaler block memory | 1 f16 ulp; up to 1e-4 outside | `pack2x16float` rounding is implementation-defined. Measured: exact |

`--mutate` is the fail-before proof. It perturbs the oracles' copy of the scene after upload: colour, current luma and depth ×(1 + 2^-7), and jitter +0.25 px. Every fixture's check then fails.

## Timing

The method follows SPEC §8, adapted to compute:
- Each sample is one compute pass holding 16 back-to-back copies of a fixture's dispatches. The pass's own `timestampWrites` (begin, end) bracket it, and the total is divided by 16.
- The passes are never empty; empty compute passes were what gave invalid timestamps on Metal.
- 11 rounds, with the fixture order rotated each round, after 4 warm-up runs. Each reported value is the median.
- `floor` gives the dispatch-cost floor.
- Ratios are reported for same-run pairs only (`ratioPairs` in `harness.js`).
- The runner records the host's load average before and after. Under load, compare ratios, not absolute times.

### Baseline (`results/tools/m5-pro.json`)

Apple M5 Pro, Metal 3, Chrome (puppeteer), 2026-11-11. One run, 11 rounds × 16 repeats. **Under heavy load:** load average 62.6 before and 64.6 after on 18 CPUs, with other agents' GPU suites running at the same time, so absolute times are unreliable. Same-run ratios:

| Ratio | 960×540 | 1280×720 | 1129×635 | 1707×960 | 1920×1080 |
|---|---:|---:|---:|---:|---:|
| `fsr/luma-pyramid` ÷ `upscaler/exposure` | 0.32× | 0.42× | 0.35× | 0.57× | 0.68× |
| `fsr/luma-pyramid-correct` ÷ `fsr/luma-pyramid` | 1.36× | 1.05× | 1.18× | 1.14× | 1.21× |
| `fsr/shading-change-total` ÷ `upscaler/shading-change` | 2.36× | 2.22× | 2.31× | 2.14× | 2.01× |
| `fsr/shading-change-pyramid-amd` ÷ `fsr/shading-change-pyramid` | 1.04× | 1.00× | 1.01× | 1.05× | 1.07× |

How to read these:
- The upscaler's single-invocation exposure costs the same at every size (1024 serial taps); the 2-dispatch SPD pyramid grows with the frame.
- The FSR shading-change pair costs about twice the upscaler's fused pass. Almost all of it is the pyramid's 10-load, two-sort custom load; the consumer is 6–8 %.
- Storing mips 4–6 costs within noise.
- The every-texel luma variant is a straightforward reduction (tree over 256 lanes), not tuned. It is the M2T.2 target to beat or match, not a lower bound.

## Plug-in points (`generator.js`)

`generatorPasses` lists the generator versions, each with `create( device, context )`. Today every `create` is `null`, and the report lists those versions as pending. `context.tools` hands over the FSR tool halves in SPEC §14.1 shape:

- **`tools.luma( { variant: 'correct' | 'amd' } )`** → `{ wgsl, layout, group, reset, check, destroy }`. `wgsl` defines `load`, `store`, `spdLoad` and `finish` for `@group(1)`.
  - `generator/luma-pyramid` (M2T.2): `reduceToScalar` with edges `'zero-weight'`, level `[1]` stored by the hook, and `finish` for FrameInfo. Check against `variant: 'correct'`.
  - `generator/luma-pyramid-clamp` (M2T.1): edges `'clamp'` and levels `[1, 6, mips]`. This is parity with AMD; check against `variant: 'amd'`.
- **`tools.shading( { levels: 3 } )`** → `{ wgsl, layout, group, texture, check, destroy }`.
  - `generator/shading-change-pyramid` (M2T.1): the custom load, per-channel average, levels `[1, 2, 3]`, edges `'clamp'`. 1 dispatch.

A generator fixture compiles its kernel around `tool.wgsl` (the generator owns `@group(0)`), binds `tool.group` at 1, and records its dispatches in `encodePass( pass )`. It returns `{ name, pass, dispatches, encodePass, reset: tool.reset, check: () => tool.check(), destroy }`. The harness then checks it against the same oracle as the FSR fixture, and reports `generator ÷ compareWith` next to the existing ratios.

One contract detail surfaced here. The scalar of AMD's luma pyramid comes from texel (0,0) of a mip whose ⌊h/2^mips⌋ is 0 at 1280×720. A generator that calls `store` only for in-range texels (as §14.1 specifies for `'clamp'`) never emits it. The `'clamp'` parity variant therefore needs `finish`, or an explicit "(0,0) of the last level" rule.
