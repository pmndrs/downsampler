# Divergences from the primary sources

Where this library does something different from the sources in SPEC §13: AMD FidelityFX SPD, nvpro_pyramid (vk_compute_mipmaps), the WebGPU community ports (webgpu-spd, `greggman`, `davidar`, the `Volcomix` fiddle) and the requests in gpuweb #386/#5253.

Each entry says what they do, what we do, why, what it would take to converge, and the expected effect. Performance figures marked *measured* come from an Apple M5 Pro (SPEC §2.2, §4, §8; `results/m3/`), an NVIDIA RTX 3060 Laptop and an Intel UHD (D3D12; `results/windows/`). AMD and mobile GPUs are unmeasured; anything else is an *estimate* or *unknown*.

This is the design record. For a plain-language summary for users, see [docs/differences.md](docs/differences.md).

**Verdicts**
- **Keep:** a deliberate difference, with no plan to converge.
- **Measured, not built:** we tried the source's approach and the data said no.
- **Done:** we converged, or built an opt-in with the source's behavior.
- **Candidate:** worth doing, but not scheduled.
- **Open:** an unresolved question that could change a default or the code.

Effort: **S** is under a day, **M** a few days including oracle and GPU cases, **L** a week or more.

Last updated: 2026-11-15 (review of every entry after Windows round 2 and M3: subgroups, fp16, `STORAGE_BINDING`, throughput, `'box'` timings, the `round()` quantize; entries 2, 3, 7, 8, 10–13, 16–19 updated, 25 added. Before that, 2026-11-13: rgb10a2unorm and integer formats, sRGB destinations, `'box'`, resampled sources, the first Windows runs.)

## Summary

| # | Divergence | Source | Verdict | Quality | Performance |
|---|---|---|---|---|---|
| 1 | No single dispatch / atomic handoff | SPD, webgpu-spd | Keep | Converging risks stale levels (seen on Apple) | 2 dispatches beat the atomic probe (M5) and tie natively (D3D12) |
| 2 | Batch ends at every odd size | SPD, nvpro | Done in part: tail kernel default; halo batches opt-in. **Open:** halo default (needs the M5) | None | Tail 0.82–0.90× (RTX), 0.97× (Intel), 0.81–1.02× (M5); halo 0.85–1.00× of that (RTX), neutral (Intel) |
| 3 | Odd sizes match render passes | SPD, nvpro | Keep default; `'spd'` and `'box'` opt-ins (Done) | `'spd'` drifts and isn't a conservative Hi-Z | `'spd'` 1.3–2× faster at render sizes; `'box'` 1.45–2.15× of `'exact'` (Windows), 1.23–1.28× (M5) |
| 4 | Conservative min/max at odd edges | SPD | Keep | SPD's Hi-Z can false-cull | One extra load per odd edge texel |
| 5 | Clamped edges, no zero reads | SPD | Keep | AMD's own SPD reads zeros at the 1×1 | ~None |
| 6 | No wrap edge mode | gpuweb#5253 | Candidate | Faint seam on tiling textures at odd sizes | ~None |
| 7 | Every level rounded as stored (`'exact'`, `'box'`) | SPD | Keep; `'spd'` keeps SPD's unrounded in-tile values | Ours equals level-by-level generation | Tie rounding now one `round()` |
| 8 | Binary reductions, not `SpdReduce4` | SPD | Done (sums, weights, argmin/argmax with payloads) | — | None |
| 9 | No alpha-coverage preservation | gpuweb#5253 | Candidate (with the deferred transform hooks) | Cutout foliage thins with distance | One extra pass if added |
| 10 | No subgroup / wave ops | SPD, nvpro | Measured, not built | — | Bound 8 % (RTX rgba8), 1.7–5.5 % (M5); the prototype lost 8–84 % |
| 11 | No packed fp16 | SPD | Measured, not built | f16 loses precision (Intel rgba16f up to 512 steps) | ~2–5 %, not worth the precision |
| 12 | One `vec4f` workgroup array | SPD | Keep (bounded) | None | All workgroup-memory traffic bounded at 8 % (RTX rgba8) |
| 13 | float32 averages sample only with `float32-filterable` | nvpro | Done: `requiredFeatures( adapter )` asks for it (2026-11-17) | Sampled levels follow the GPU's sampler weights, as for every filterable format | Sampling 0.74–1.00× of loading (RTX), 0.69–1.00× (Intel), 0.71–1.01× (M5 Pro) |
| 14 | No `texture_external` source | fiddle | Candidate | None | Saves one frame copy for video |
| 15 | Sub-rectangles: exact per-level regions | SPD | Done, tighter | None | A rect costs the dispatch floor (7–18 µs, RTX) |
| 16 | sRGB written by render passes (or the buffer path) | SPD | Done: the render path is the sRGB default with `RENDER_ATTACHMENT` (Built 2026-11-16); `colorSpace: 'srgb'` and `srgbRounding: 'device'` (Built; default stays `'exact'`, owner 2026-11-16). By size (owner, 2026-11-17): `'auto'` renders from `renderPathMinTexels` (1920 × 1080; the buffer path below), since on the M5 Pro the render path loses below 1080p | Linear-light, correctly rounded (`'device'`: up to 3 codes off) | Render path 0.63–0.79× of the buffer path (RTX), 0.74–0.92× (Intel), 1.25–2.18× (M5 Pro below 1080p, 0.79–0.83× from 1440p); plain render passes 1.05–1.6× faster (RTX), ~5× (Intel), the same speed (M5); `'device'` matches them; `colorSpace: 'srgb'` 0.40–0.68× of the buffer path on the M5, fastest there |
| 17 | Compute only, no render fallback | `greggman`, `toji` | Keep (resolved by measurement), except sRGB: a render path (#16) | None | See entry: at or below render passes on NVIDIA and Intel, 1.3–27× faster on the M5 |
| 18 | Needs `STORAGE_BINDING` | gpuweb#5253 (`Kangz`) | Keep (measured: free) | None | No cost on M5, RTX or Intel |
| 19 | Benchmark method | `greggman`, nvpro | Done (timestamps + throughput agree) | — | — |
| 20 | Mip count, levels written, no counter buffer | SPD | Keep (documented in docs/spd-users.md) | — | — |
| 21 | No size or level cap | SPD | Keep | — | — |
| 22 | webgpu-spd 3.0.0 side by side | webgpu-spd | Keep | Its handoff gave stale levels on Apple; non-conservative Hi-Z | Our `'spd'` 1.2–3.5× faster than its default |
| 23 | Power-of-two Hi-Z from a render-size source | niagara, SSSR, Capsaicin | Keep | Ours conservative at any ratio | Not timed |
| 24 | Integer averages exact, ties to even | webgpu-spd | Keep | Truncation drifts ~4 units by 1×1 | Not timed |
| 25 | float16 rounded to nearest, not truncated | D3D12/Metal `textureStore`, naive ports | Keep (`f16Rounding: 'device'` opt-out) | D3D12 and Apple stores truncate | 5 % (RTX), 8.5 % (Intel), ~3 % (M5) of an rgba16f chain |

## Dispatch structure

### 1. No single dispatch or atomic handoff
- **They:** SPD and webgpu-spd write every level in one dispatch. The last workgroup to finish, found with a global atomic counter, reduces the final levels.
- **We:** use one dispatch per batch of up to 6 levels: two dispatches up to 4096² at powers of two and under `oddSizes: 'spd'`.
- **Why:**
  - The WGSL memory model doesn't guarantee the handoff: atomics are relaxed and barriers are workgroup-scoped (SPEC §2.2).
  - webgpu-spd#7 saw missing mips on M1 and "fixed" it with a `storageBarrier`, which isn't guaranteed in WGSL either.
  - Observed since (2026-11-12, entry 22): webgpu-spd 3.0.0's handoff, with plain-store payloads, read stale mip-6 data on the M5 Pro in up to 77 % of frames.
  - It is slower, measured twice (below).
- **Effort to converge:** S. A measurement-only probe is in this repo: `test/bench/atomic-spd.js` (M2T.4). It is the library's 6-level `'clamp'` tile kernel plus an atomic level-6 payload, a self-resetting ticket and a tail reduced by the last workgroup.
- **Quality:** converging could produce stale or missing levels on hardware that reorders. On the M5 Pro, the probe reproduced `'spd'` on 4 changing inputs at all 22 size/format pairs, with its counter reset. That is evidence for one device, not a proof.
- **Performance (measured):**
  - **2026-11-10, three.js prototype, ≤ 1024²:** single dispatch vs ours, in ms per chain: 0.0099 vs 0.0068 at 256² rgba8, 0.0305 vs 0.0151 at 1024² rgba8, 0.0592 vs 0.0246 at 1024² rgba16f. The atomic tail cost about 20 µs; the dispatch boundary it replaces 1–2 µs.
  - **2026-11-11, M2T.4 probe, render sizes** (M5 Pro under heavy load, 7 runs, same-run ratios to `'exact'`; SPEC §2.1):
    - rgba8 at 960×540 … 2560×1440: atomic 0.88–1.05×, against 0.47–0.73× for the same kernel in two dispatches (`'spd'`).
    - rgba16f: atomic 1.00–1.60×, against 0.62–0.94×.
    - 1024²: atomic 1.5–1.6×, against 0.81–0.97×. 4096²: 1.01–1.02×, against 0.99–1.00×.
    - Without copying its tail levels into the texture, it is still behind `'spd'`: 0.73–0.88× for rgba8, 0.90–1.50× for rgba16f.
  - **2026-11-12, native AMD SPD on Windows** (D3D12, `native/d3d12-mips`, results/windows/timings.md): AMD's atomic single dispatch and the same shader split in two dispatches are within noise of each other on the RTX 3060 and on Intel UHD, every size and format. The handoff buys nothing natively either. webgpu-spd's handoff showed no stale frames on either GPU (entry 22): the race was observed on Apple only.
- **Verdict:** keep. Revisit only if WGSL gains device-scope acquire/release, and measure first.

### 2. A batch ends at every odd size
- **They:**
  - SPD carries every level through odd sizes in one pass by dropping the last row or column.
  - nvpro carries 2 levels per dispatch through odd sizes, recomputing a 1-texel halo.
- **We:** end the batch where the next reduction is inexact: 5–9 dispatches at render sizes (1920×1080: 7).
- **Why:** it is the simplest structure that keeps `'exact'` results identical to render passes.
- **Effort to converge:**
  - **Tail kernel: built (M2T.4), the default since 2026-11-13** (`tail: false` turns it off). Once a first output fits one 32×32 workgroup, that workgroup finishes the chain with exact weights. Render sizes drop to 3–6 dispatches (1920×1080: 4; 1477×831: 6, since 46×25 doesn't fit).
  - **Halo batches, built (opt-in `halo`, 2026-11-14):** our odd taps fall in `2x … 2x+2`, the same support as nvpro's, so their halo structure carries odd steps without changing results (bit-identical with shader weights; within one step with the sampler, as the tail). The planner takes a halo only where it saves a dispatch, on first outputs up to 128²: RTX 3060 0.85–1.00× of the default chain (8–15 % at 960×540, 1129×635, 1477×831, 1920×1080), Intel UHD neutral; on large levels Intel lost 24–52 % (16-texel tiles), hence the cap. Not the default: the rule is ≥ 5 % on both GPUs; the M5 decides ([results/windows/round2/halo.md](results/windows/round2/halo.md)).
- **Quality:** none. The tail matches `'exact'` within one 8-bit step (its shader bilinear weights replace the sampler's 8-bit ones). It passed the full GPU matrix against the `'exact'` oracle.
- **Performance (measured, M5 Pro, loaded; SPEC §2.1):**
  - The tail is 0.83–1.07× of `'exact'` on a loaded M5 Pro (2026-11-11). Idle re-runs (2026-11-13): RTX 3060 0.82 / 0.90 (rgba8 / rgba16f), Intel UHD 0.97 / 0.97, M5 Pro 0.81–1.02 with 16×16 exact tiles (`results/render/m5-pro-render-tiles-*.json`). Owner rule: default if faster on most devices, so it became the default.
  - The gap is real, though. The `'spd'` structure, 2 dispatches with no batch ends, is 0.47–0.73× for rgba8 and 0.62–0.94× for rgba16f at render sizes up to 1440p.
  - It isn't pass overhead: one compute pass for all batches (`encodePass`) measured 0.94–1.03×.
  - Part of it was the exact kernel's arithmetic: `'spd'` was 0.81× at 1024² rgba8 with the same 2 dispatches, and 0.63× for max. The tie rounding is now one WGSL `round()` (2026-11-15, same bits): rgba8 chains 0.91× / 0.96× (squares / render sizes) on the RTX, 0.76× / 0.73× on Intel ([results/windows/round2/trim.md](results/windows/round2/trim.md)).
- **Open:** halo batches as the default. The rule is ≥ 5 % on both measured vendors; the RTX qualifies, Intel is neutral, the M5 is unmeasured (owner, PROGRESS *Waiting on owner*).
  - In absolute terms, `'spd'` saves 13–29 µs per render-size chain up to 1080p on the loaded machine.

## Odd sizes and edges

### 3. Odd sizes match render passes
- **They:**
  - SPD drops the last row or column.
  - nvpro weights by area, `(n−x, n, x+1)/(2n+1)` for size `2n+1`, so 3→1 averages all three texels.
- **We:** `'exact'` by default: normalized-coordinate bilinear, identical to render-pass and WebGL mipmaps.
  - **`oddSizes: 'spd'` (M2T.4, opt-in):** SPD's semantics, built from the `Reducer`'s `'clamp'` tile kernel. It has its own oracle (`spdReference`) and runs in the full GPU matrix.
  - **`oddSizes: 'box'` (M2.5, opt-in, 2026-11-13):** nvpro's weights, checked against `nvpro_pyramid.glsl`. Its own oracle (`areaReference`, `boxReference`) and GPU cases (main matrix, render sizes, 31×31, strips, layers). Folds under it are the `'exact'` folds.
  - **Still different from nvpro in structure:** nvpro carries 2 levels per dispatch through odd sizes with a halo; `'box'` ends a batch at an odd size like `'exact'`, and the tail kernel (default on) carries the small levels (entry 2).
- **Why:** three.js and other engines can switch to us without any visual change, and Hi-Z stays conservative.
- **Effort to converge:** done for SPD and nvpro, as opt-ins. Converging the default would break render-pass parity.
- **Quality (measured, `results/spd/m5-pro-quality.json`; SPEC §2.1):**
  - `'exact'` shares the render-pass flaw: a 31×31 texture can reduce to an unrepresentative 1×1. `'box'` is area-correct; recommend it for GI, irradiance and roughness chains.
  - **`'spd'` against `'exact'`:** the two agree to one 8-bit step until the first odd size, then differ. On the scene image the largest difference is 0.24–0.64 (rgba8) over the render sizes, with texel-weighted means of 0.0004–0.010.
  - **Drift:** when level 0 or 1 is odd (1129×635, 1477×831, 1707×960), `'spd'`'s footprints drift by up to a texel towards the far edge, and every level is off.
  - **Against the area-weighted ideal** (scene, levels ≥ 4), `'exact'` is 1.3–7× closer than `'spd'`. On uniform noise `'spd'` is closer, because a box averages away more noise than a 2-tap bilinear.
  - **Hi-Z:** `'spd'` max leaves 0.01–15 % of texels below a base texel in their footprint (scene), up to 52 % on noise. `'exact'` and the tail have none.
- **Performance (measured):**
  - M5 Pro (loaded): `'spd'` is 1.4–2.1× faster than `'exact'` at render sizes up to 1440p in rgba8, 1.2–1.6× in rgba16f up to 1080p, within 3–9 % at 4K and at powers of two from 2048².
  - Windows (2026-11-15, `--compare=render`, tail kernel on): `'spd'` takes 0.56–0.81× of render passes on the RTX, 0.55–1.00× on Intel.
  - `'box'` has `'exact'`'s dispatches; its odd-level batches load up to 9 texels per output instead of one sampler fetch: 1.45–2.15× of `'exact'` on Windows, 1.23–1.28× on the M5 Pro. The tail stays on under `'box'`: turning it off is 1.06–2.25× slower on the RTX at render sizes ([results/windows/round2/defaults.md](results/windows/round2/defaults.md)).
- **Verdict:** keep `'exact'` as the default; `'box'` is the opt-in for quality (GI, irradiance, roughness, UI). `'spd'` stays a public opt-in for `'average'` only; folds under it are rejected, with an error pointing to `'exact'` or a `Reducer` with `'zero-weight'` edges (coordinator, 2026-11-11; the owner can overrule). It works on the buffer output path and in compatibility mode (SPEC §2.1).

### 4. Conservative min/max footprint at odd edges
- **They:** SPD's min/max skip the last row or column at odd sizes.
- **We:** each fold covers texels `2x … 2x+2` along an odd dimension, the smallest footprint that is conservative.
- **Why:** Hi-Z occlusion culling needs it; a non-conservative max can cull visible objects.
- **Effort to converge:** S, but it would reintroduce the bug.
- **Performance:** one extra load per output texel along odd edges, in the first output of a batch only. Not measured separately.
- **Verdict:** keep.

### 5. Clamped edges, no zero reads
- **They:** SPD's unclamped loads read zeros past the texture edge unless the texture is padded.
- **We:** clamp every read inside the texture.
- **Quality:** no darkened borders, and no zero depth leaking into Hi-Z.
- **Measured in AMD's own code** (native D3D12 SPD at 1129×635, RTX 3060, results/windows/native-semantics-rtx3060.json): the 1×1 level reads zeros past the 2×1 edge, 0.28 against 0.50 clamped, in every variant that loads (atomic or split). The linear-sampler split variant clamps and matches. It affects every size where ⌊log2 h⌋ < ⌊log2 w⌋ (1129×635, 1280×720, 1707×960, 2560×1440).
- **Verdict:** keep.

### 6. No wrap edge mode
- **They:** gpuweb#5253 asked for wrap or clamp edge handling.
- **We:** clamp only.
- **Why it rarely matters:** even sizes never read past an edge. Wrap only changes odd-size footprints at the border.
- **Effort:** S for folds and `'box'`. Under `'exact'` it would break render-pass parity, so only offer it with `'box'`.
- **Quality:** tiling textures with odd sizes get a faint seam at the border.
- **Verdict:** candidate. Add to SPEC §12 if a user asks.

## Precision

### 7. Every level is rounded as stored
- **They:** SPD keeps unrounded values in workgroup memory between levels and rounds once on store.
- **We:** under `'exact'` and `'box'`, quantize each level exactly as stored. For 8-bit and 16-bit normalized formats, the math runs in exact integer units with ties to even (one WGSL `round()` since 2026-11-15). Under `'spd'` we keep SPD's behavior: in-tile values stay unrounded in workgroup memory and each level is rounded only when stored (`spdReference` models it; GPU-checked by `packed/f16-ties`).
- **Why:** a chain equals level-by-level generation, so results match render passes and are deterministic across batch sizes and devices.
- **Effort to converge:** S, as an opt-in that skips intermediate quantization. It would need an oracle mode.
- **Quality:** SPD's output is closer to a true box average of the base level on deep 8-bit chains. Ours matches what every render-pass generator produces, and doesn't depend on the batch layout.
- **Performance:** unknown; a few ALU ops per texel. Likely noise at bandwidth-bound sizes.
- **Verdict:** keep as the default; the opt-in is a candidate.

### 24. Integer averages are exact, rounded to nearest
- **They:** webgpu-spd averages integer formats as `(v0 + v1 + v2 + v3) / 4` in the format's integer type: integer division truncates (toward zero for sint), and 32-bit sums wrap, so averages of values from 2³⁰ (u32) or 2²⁹ (i32) up are garbage. SPD (the FidelityFX sample) has no integer path; render-pass mipmaps can't make integer levels at all (integer formats aren't filterable).
- **We:** the exact weighted mean, rounded to nearest with ties to even, at every width, 32-bit included: the 2×2 mean at even sizes, render-pass bilinear weights as exact rationals at odd ones (`'exact'`), area weights under `'box'` and for resampled sources. Signed formats are offset by 2³¹, so min, max and tie parity work unchanged (SPEC §6.1 Integer formats).
- **Why:** the same rule as every normalized format here (§4 Quantization: integer units, ties to even), no drift along a chain, and no overflow. The oracle (`intReference`) is exact, so every level is checked bit for bit.
- **Effort to converge:** S, a truncating mode would be a different `ds_mean`. Not planned.
- **Quality:** truncation biases every level down by 3/8 of a unit on average (remainders 0–3 equally likely), compounding along the chain: about 4 units by the 1×1 of a 2048² chain.
- **Performance:** not timed. The 2×2 step costs a few more integer ops than a sum and a shift; odd steps split each value into 16-bit halves and carry the division as quotient and remainder. Integer chains are rare (IDs and occupancy use folds, which cost the same as for other formats).
- **Verdict:** keep.

## Reductions

### 8. Binary reductions, not a 4-way `SpdReduce4`
- **They:** SPD's `SpdReduce4(a, b, c, d)` is arbitrary, so it can be a weighted average, a sum or a count.
- **We (since M2T.1):** `Reducer` takes a reduction per channel: sum, average, min, max, a weight channel for weighted means, or a custom binary `{ wgsl, kind: 'fold' | 'sum' }`. `Downsampler` keeps idempotent folds only, because its `'exact'` odd sizes have no defined meaning for sums.
- **Why binary:** a pairwise reduction composes inside the tile, and normalization happens once when a level is emitted. A 4-way custom reduction adds nothing that a binary sum plus a weight channel can't express.
- **Payload folds (built):** `'argmin'` / `'argmax'` keys with `'payload'` channels carried from the winning texel (ssgiDev's normal taken from the min-depth texel), ties to the smaller payload (SPEC §4; `validate-payload.js`).
- **Still missing:** a non-commutative "first valid" scan (GI-1 probe masks). No user has asked.
- **Verdict:** done.

### 9. No alpha-coverage preservation
- **They:** gpuweb#5253 lists it as a requirement for a core API.
- **We:** don't offer it.
- **Effort:** M. Each level needs a search for the alpha scale that keeps the base level's coverage at a cutoff, which means an extra pass or a per-level reduction.
- **Quality:** alpha-tested foliage and fences thin out in the distance.
- **Verdict:** candidate for after M4. It is a natural fit for the transform hooks.

## Hardware features

### 10. No subgroup / wave operations
- **They:**
  - SPD uses wave ops with a fixed quad layout (`ARmpRed8x8` and `WaveReadLaneAt`).
  - nvpro uses `subgroupShuffleXor`, assuming lane equals local index.
  - The fiddle uses quad ops in compute.
- **We:** use workgroup memory only.
- **Why:**
  - WGSL defines quads only in fragment shaders and gives no mapping between lanes and `local_invocation_index`, so those layouts are undefined behavior.
  - Firefox lacks `subgroups`; Chrome and Safari 27.3 have it.
- **Effort:** L. The portable scheme in SPEC §13 builds a linear id from `subgroup_id` and uses `subgroupShuffleXor`. It ships with the workgroup-memory fallback.
- **Quality:** butterfly order pairs sums as `(a+b)+(c+d)`. That doesn't matter for integer units, but the float oracle must pair the same way.
- **Performance (measured, M3, `--compare=subgroups`):** the upper bound (no barriers or workgroup memory at all, wrong values) is 1.7 % (rgba8) / 5.5 % (rgba16f) on the M5 Pro and 8 % on the RTX in rgba8. The only real subgroup structure measured (`shuffle`) lost 8–20 % on the RTX and 47–84 % on Intel, and its quad mapping gave different results on both; one barrier per level without subgroups (`slot`) was 0.99× ([results/windows/round2/m3.md](results/windows/round2/m3.md), `results/m3/`).
- **Verdict:** measured, not built. A kernel would have to beat `slot` by 5 % where the bound itself is 8 %. Revisit if WGSL defines a lane mapping in compute.

### 11. No packed fp16 math
- **They:** SPD has a packed half-precision path.
- **We:** compute in f32.
- **Effort:** M. It needs a `shader-f16` kernel variant and oracle pairing.
- **Performance (measured, M3, `--compare=f16`):** about 2 % on the M5 Pro, under 5 % on the RTX and Intel in the overflow-safe form, except Intel rgba16f (9 %, with values up to 512 f16 steps off). nvpro saw no gain on an RTX 3090 either.
- **Verdict:** measured, not built: too little gain for the lost precision.

### 12. Workgroup memory layout
- **They:** SPD stores intermediates in separate per-channel arrays.
- **We:** use one `array<vec4f>`.
- **Effort:** S.
- **Performance:** not measured directly. The subgroup bound (entry 10) caps everything workgroup memory costs: removing it entirely would save at most 8 % (RTX rgba8), 1.7–5.5 % on the M5 Pro, so a layout change can win only a fraction of that.
- **Verdict:** keep.

### 13. float32 formats load instead of sampling
- **They:** nvpro measured the bilinear sampler 26–32 % faster than loads on NVIDIA.
- **We:** sample float32 `'average'` chains when the device has `float32-filterable`, like every filterable format, and load them otherwise.
- **Measured (2026-11-17, `--compare=float32`, [float32.md](results/windows/round2/float32.md)):** sampling takes 0.74–1.00× the time of loading on the RTX 3060 (VALID, locked clocks) and 0.69–1.00× on Intel UHD; never slower. Most for r32float and rg32float at render sizes; rgba32float and 4096² are about even.
- **Measured on the M5 Pro (2026-11-17, [srgb-m5.md](results/m5/srgb-m5.md), [JSON](results/m5/m5-pro-float32.json)):** 0.71–1.01× of loading, the same shape (r32float 0.71–0.83× at render sizes, rgba32float and 4096² about even). `downsampler/float32-sample` passes on Metal (72 chains, 495 levels).
- **Tested:** `downsampler/float32-sample`, a device with the feature, each level against the oracle with the sampler's tolerance (RTX: 72 chains, 495 levels).
- **What changed:** most apps never asked for the feature, so float32 chains loaded. `requiredFeatures( adapter )` (next to `requiredLimits`) asks for it and the other optional features the library uses, and the setup docs use it.
- **Who it affects:** `'average'` chains of r32float, rg32float and rgba32float (HDR, GI). Folds (Hi-Z) load anyway.
- **Verdict:** done: sampling is never slower on any of the three GPUs.

## Sources and inputs

### 14. No `texture_external` source
- **They:** the fiddle samples level 1 straight from a video frame's `texture_external`.
- **We:** need the frame copied first with `copyExternalImageToTexture`, into level 0 or a separate source texture.
- **Effort:** M. Add a source-batch variant that binds `texture_external` on the sample path. That gives level 1 onward, with no level 0.
- **Performance (estimate):** saves one full-frame write and read per frame for video pyramids.
- **Verdict:** candidate. It matters for video, the upscaler or ML preprocessing.

### 15. Sub-rectangles: exact per-level regions
- **They:** SPD's `rectInfo` snaps the rect to 64-texel source tiles, and recomputes levels below 64×64 for the whole texture.
- **We:** `rect` (M2.4, SPEC §4 Partial updates) rewrites, per level, exactly the texels whose footprint touches the previous level's rewritten ones (`2x … 2x+2` at odd sizes under `'exact'`, clamped blocks under `'spd'`). Dispatches cover the workgroups holding those texels; stores outside them are skipped. The result is bit-identical to a full recompute, and every other texel is untouched. Layer ranges (`layers`) and every path (tail, buffer, compatibility, separate sources) compose with it.
- **Cost of converging:** none needed. SPD's snapping would write more texels for the same result; matching it would only matter for a caller relying on SPD rewriting whole 64-texel tiles.
- **Verdict:** resolved, tighter by design. Not timed.

### 23. Power-of-two Hi-Z from a render-size source
- **They:**
  - SPD-based setups keep Hi-Z level 0 at the depth buffer's size: FidelityFX SSSR (SDK 1.1.4, `ffx_sssr_depth_downsample.h`) copies depth and runs SPD's min, which drops odd rows and columns (entry 4). Capsaicin's HZB (`visibility_buffer.cpp`, `914b915`) copies depth, then `GPUMip` DepthMin: `2x … 2x+2` at odd sizes, SPD once a level is a power of two.
  - zeux's niagara (`567ebe3`) uses a power-of-two pyramid, `previousPow2` of the swapchain size, filled by one min-reduction-sampler tap per texel center. Its comment calls that conservative because each reduction is "at most by 2x2". At 1920 → 1024 a texel's area spans up to 3 source texels, so the 2×2 tap misses one.
- **We:** `source: { texture, resample: 'area' }` (M2.5): any destination size, up, or down to 8× per axis by default (the `maxResampleRatio` option; validation only, so raising it costs no pipeline). Averages are area-weighted; folds combine every source texel the destination texel overlaps, so the pyramid is conservative at every level (GPU-checked against every source texel). A 2048×1024 pyramid takes 2 dispatches for 12 levels.
- **Why:** Hi-Z must not false-cull, and WebGPU has no reduction sampler, so the fold is ours anyway.
- **Quality:** a core test counts over 100 non-conservative texels on one 1920-texel row for the 2×2 tap; ours has none.
- **Performance:** the first batch loads the overlap (3×3 texels at 1920×1080 → 1024×512, 4×4 at 1129×635 → 512×256) instead of a sampler fetch. Not timed.
- **Verdict:** keep.

### 16. sRGB through the buffer output path
- **They:** SPD samples an sRGB view (hardware decode and linear filtering) and writes a UNORM alias of the same texture, so the encode happens in its store.
- **We (M2.3.5, 2026-11-13):** WebGPU forbids sRGB storage textures even through a view, and a UNORM view of an sRGB texture needs `viewFormats` at creation, which three.js and most apps don't set. So each batch reads the sRGB view (linear), reduces in linear, encodes each level in the shader and writes the codes through the buffer output path (`copyBufferToTexture` per level, SPEC §6.1 sRGB).
  - The encode is correctly rounded for the shader's f32 value (a `pow` estimate corrected against the exact thresholds), and in-tile levels read the exact decode table, so a chain equals level-by-level generation (SPD keeps unrounded values between levels, #7).
  - Loads snap the hardware's decode to the exact table: the M5 Pro decodes sRGB to 12-bit linear values (254 of 256 codes inexact, up to 0.39 of a code step), which SPD would average as is. Samples keep it, as render passes do.
- **The alternative measured:** the same kernel writing an rgba8unorm scratch texture (the linear alias, as storage) and `copyTextureToTexture` per level, which WebGPU allows between formats that differ only in sRGB-ness. Same bits (GPU-tested). Not faster: scratch ÷ buffer medians 0.95, 1.00 and 1.00 over 8 sizes in three runs, the quietest within 0.96–1.08 (M5 Pro, never idle; SPEC §6.1 sRGB). The buffer path was already built, composes with compatibility mode, `rect`, `layers` and separate sources, writes 6 levels per dispatch at the default storage limit, and needs no `bgra8unorm-storage` for bgra8unorm-srgb, so it won on simplicity.
- **Effort to converge:** none possible (WebGPU). A scratch-texture variant is a kernel option (`output: 'storage'` writes the alias format) for tools that dispatch it themselves.
- **Quality:** linear-light averages, as SPD; ours are rounded per level and identical on every GPU with `averageAccess: 'load'`.
- **Performance:** one extra write and copy of the written levels, as for every buffer-path format. `encodePass` can't record the copies.
  - Against render passes (three.js's path for sRGB textures today), M5 Pro at load average 14: ours is 1.19–1.98× faster up to 1280×720 and at 1024², a tie at 1707×960, and slower from 1080p up (render passes take 0.55–0.79× our time).
  - RTX 3060 (2026-11-15, [rtx3060-srgb-paths.json](results/windows/round2/rtx3060-srgb-paths.json)): render passes take 0.32–0.51× our time at every size. The scratch-texture path doesn't help there either (1.05–1.13× of the buffer path). The tables were a large part (moved to a storage buffer in round 2: 3.8–4× faster, [srgb.md](results/windows/round2/srgb.md)); with no lookups at all (wrong bits, timing only) the chain was still about 2× render passes at 1080p.
  - Where the time goes ([srgb-cost.md](results/windows/round2/srgb-cost.md), RTX, 1080p): shader sRGB maths 32 % (mostly the correction steps), reading an sRGB source ~17 %, the buffer path's write and copy 25 %. With no sRGB maths at all the chain is still 1.4× render passes, so no shader change reaches them on NVIDIA. Stores now skip the correction step, because every stored value is already an exact decoded code: same bits, 16–28 % less time, so render passes take 0.37–0.63× our time.
- **Verdict:** done, by necessity. On NVIDIA render passes stay faster, and the buffer path WebGPU forces on us bounds how close we can get (above), so nothing is left to try in the shader. The docs say so, and three.js keeps render passes for sRGB textures (the fork's `mipmapGeneration: 'compute'` already does). *Superseded 2026-11-16 by the render path (below).*
- **Other options considered (2026-11-16), not pursued; maybe later:**
  - *Reduce in linear, convert once at the end:* the same number of encodes, and the final write still has to be sRGB, which compute can't do. A render pass per level to convert would cost more than render-pass mips. Skipping the per-level rounding (`noSnap`, 18 % faster) would make results depend on how the chain is split into dispatches.
  - *A linear-only mode where shaders decode:* averaging sRGB-encoded bytes darkens (black + white → 22 % brightness, not 50 %), and 8-bit linear values band in the darks.
  - *Store color textures linear from load (rgba16float), encode only at display:* correct, no shader changes, and our fast path; but twice the memory and per-frame sampling bandwidth for a one-time ~0.1 ms saving per texture. Worth revisiting as an opt-in for color textures re-mipped every frame (canvas, video).
  - *Write sRGB codes through an rgba8unorm view of the texture:* WebGPU rejects `STORAGE_BINDING` on sRGB textures even with `viewFormats`. If the spec relaxes that, the scratch-texture kernel option becomes a direct path with no copies.
- **Measured (2026-11-16), the other direction works: an rgba8unorm texture with `viewFormats: ['rgba8unorm-srgb']`** ([srgb-alias.md](results/windows/round2/srgb-alias.md)). It reads through the sRGB view and writes codes into the rgba8unorm levels, as SPD and native engines do.
  - Valid in Chrome 148. The sRGB view needs an explicit `usage: TEXTURE_BINDING`. Compatibility mode rejects it.
  - Same bits as the buffer path, every level.
  - RTX: 19–30 % less time than the buffer path. Render passes take 0.52–0.94× its time: about half the old gap, nearly a tie at 1440p.
  - Sampling or rendering into the aliased texture costs nothing on the RTX (1.000×).
  - **Built (2026-11-16): `EncodeOptions.colorSpace: 'srgb'`** for rgba8unorm and bgra8unorm textures with their sRGB view format. Same bytes as an sRGB texture across every kernel and option (`srgb/alias`, 86 chains), 0.70–0.80× of the buffer path on the RTX, the prototype's speed ([rtx3060-srgb-colorspace.json](results/windows/round2/rtx3060-srgb-colorspace.json)); `encodePass` works. Not with `'spd'` (its tile kernel has no sRGB storage store) or the compatibility path.
  - Intel UHD (2026-11-16, [intel-srgb-renderpath.json](results/windows/round2/intel-srgb-renderpath.json)): 0.83–0.93× of the buffer path.
  - **Windows timing-marker check.** With markers on the aliased output, `colorSpace`/buffer was 0.726–0.841 on the RTX 3060 Laptop and 0.771–0.927 on Intel UHD (eight sizes, paired per round). At 1440p the RTX ratio moved from 0.797 to 0.745. Intel moved from 0.848 to 0.771 at 960×540 and from 0.906 to 0.805 at 720p. [RTX run](results/windows/round3/review-rtx3060-srgb.json) · [Intel run](results/windows/round3/review-intel-srgb.json)
  - **M5 Pro (2026-11-17, [srgb-m5.md](results/m5/srgb-m5.md)): the fastest sRGB path at every size**: 0.40–0.68× of the buffer path, 0.26–0.91× of render passes and of the render path (median 0.35–0.38). Sampling the aliased texture through its sRGB view: 0.97–1.04× of a plain sRGB texture (medians 1.000–1.005), so Metal loses nothing by it. By the handoff's rule (beats the render path somewhere, sampling ≈ 1.00×) the guide now recommends it for apps that create their own 8-bit color textures.
  - **Open:** three.js would create 8-bit color textures this way to use it (owner). Unmeasured on AMD.
- **Built (2026-11-16): the render path** ([srgb-renderpath.md](results/windows/round2/srgb-renderpath.md)). One render pass per level and layer, as three.js does, with the exact compute kernel's first output as the fragment shader (`firstWgsl`: the same 'exact' or 'box' rule, reads and rounding). For sRGB it returns the exact decode of each level's rounded code, so the attachment's encode writes that code: the compute path's bits per level, with no buffer, no copies and no `viewFormats`.
  - `output: 'auto'` takes it for sRGB textures with `RENDER_ATTACHMENT` (and `colorSpace: 'srgb'` textures that have it), for `'average'` in place under `'exact'` or `'box'`; folds, `'spd'` and separate sources keep the buffer path. `output: 'render'` asks for it on any color-renderable float format. `prepare()` and `encodePass` can't express render passes and throw.
  - Tested (`srgb/render`, `srgb/render-api`): value for value against the compute path with one-level batches (the same rule per level) for sRGB, `colorSpace: 'srgb'` and nine other formats across sample, 'load', 'box', sub-ranges, `rect`, `layers` and `f16Rounding: 'device'` (float32 within a few ulp: no rounding step to absorb how each stage contracts its arithmetic); the load path's codes against the oracle. Passes on the RTX, Intel and in compatibility mode (where the layer index needs `@interpolate(flat, either)`).
  - Speed: 0.63–0.79× of the buffer path on the RTX (VALID, locked clocks), 0.74–0.92× on Intel; as fast as `colorSpace: 'srgb'` or faster. Plain render passes are still 1.05–1.64× faster on the RTX and about 5× on Intel: the difference is the exact rounding (`pow` and table lookups per texel), most of an sRGB chain's time on Intel.
  - **M5 Pro (2026-11-17, [srgb-m5.md](results/m5/srgb-m5.md), two runs): slower than the buffer path below 1080p**: 1.25–1.58× from 960×540 to 1707×960 and 2.18× at 1024², a tie at 1080p (1.01×), faster only at 2560×1440 and 4096² (0.79–0.83×). It runs at plain render passes' speed (0.99–1.05× of them), and they cost about 0.01 ms per pass almost regardless of size on the M5 (0.10–0.14 ms for 9–11 passes up to 1440p), where the compute paths scale with texels. Exact rounding is nearly free here: `'device'` takes 0.96–1.02× of the render path and changes 0.3–1.0 % of the bytes (up to 2 codes). Passes `srgb/render`, `srgb/render-api` and the full matrix on Metal (35,908 cases).
  - **`srgbRounding: 'device'`** (Built, opt-in): the attachment encodes and the sampler's decode is used as is, with no tables (so the render path also works where a fragment shader can't read a storage buffer). Plain render passes' speed (within 0–7 % on the RTX, 3–12 % on Intel), but about 16 % of the written bytes differ from the exact codes on the RTX (up to 3 codes) and 6 % on Intel (up to 2), so results differ between GPUs, as with `f16Rounding: 'device'` (#25).
  - **Native D3D12 (2026-11-17, [native-srgb.md](results/windows/round2/native-srgb.md)):** SPD on a typeless texture (`_SRGB` SRV, `UNORM` UAV, encode in the shader), render passes and `GenerateMips`, VALID on the RTX. **RTX:** native SPD sRGB takes 0.77–0.86× of native sRGB render passes at every size (1080p: 0.042 against 0.051 ms), so natively compute wins for sRGB too; what WebGPU forbids (an sRGB texture with a `UNORM` storage view) is what costs us. Our render path is 1.5–2.4× native render passes at render sizes; at 1080p about half of that is WebGPU's per-pass cost (plain WebGPU render passes 0.063) and half the exact rounding (0.079). **Intel:** native render passes win (SPD sRGB 1.5–1.7× of them), and our render path is 6.7–8.7× native, mostly the exact rounding (`'device'` is 1.3–1.7×). AMD's own SPD sRGB handling averages encoded values after level 1 and mis-encodes codes 0–10: not a correct sRGB chain.
  - **Decided (owner, 2026-11-17): by size, not by vendor.** `'auto'` takes the render path only for chains whose first level has at least `renderPathMinTexels` texels per layer (default 1920 × 1080, where the M5 ties); smaller ones take the buffer path (`colorSpace: 'srgb'` its storage path) unless the texture can only render. Nothing is slower than the buffer path everywhere; the RTX and Intel give up their gain below 1080p (RTX: 1.7–2.6× of render passes there instead of 1.3–1.6×), which `renderPathMinTexels: 0` restores. Tested in `srgb/render-api` (boundary at 1920 × 1079 / 1080, sub-ranges, rects, render-only textures, `colorSpace`, both paths against the oracle: 2 values a code apart at 1920 × 1080 on the M5, both near-ties).
  - **Open:** AMD and mobile (the same bench and `srgb/render`; tile-based mobile GPUs may behave like Apple's, which would argue for the threshold, or a higher one). `srgbRounding` stays `'exact'` by default (owner, 2026-11-16): identical results are the library's promise, `'device'` is the documented opt-in for display-only color, and three.js keeps its render passes for sRGB, which is what `'device'` does.

## Compute vs render, and measurement

### 17. Compute only, with no per-device render fallback
- **They:** `greggman` found compute generators slower than render passes on NVIDIA/D3D12, up to 10× for SPD-style ports, and `toji` reported a partner seeing the same. The community default is render passes.
- **We:** are compute only. The three.js adapter falls back to render passes only for textures we can't handle.
- **Effort:** M. Collect cross-vendor data (M3), then either choose the path per device in the adapters or add a render-pass backend here (SPEC §3 leaves room for one).
- **Performance (measured):**
  - **Apple M5 Pro:** 1.3–27× faster than render passes.
  - **Windows, now** (2026-11-15, Chrome 148, D3D12, `--compare=render`, after the `round()` quantize; [rtx3060-render-round.json](results/windows/round2/rtx3060-render-round.json), [intel-uhd-render-round.json](results/windows/round2/intel-uhd-render-round.json)): the default chain (tail kernel on) takes **0.77–0.95× of render passes in rgba8 and 0.77–1.01× in rgba16f on the RTX 3060 Laptop**, and **0.48–1.05× / 0.62–0.96× on Intel UHD**, over 960×540 … 2560×1440, 1024² and 4096². `'spd'` 0.56–0.81× (RTX), 0.55–1.00× (Intel).
  - **History:** with 8 × 8 exact workgroups (round 1) `'exact'` ÷ render was 1.25 / 1.06 (RTX) and 1.71 / 0.94 (Intel); 32 × 32 tiles, the tail default and the `round()` quantize closed the gap.
  - **Render passes for the large levels, then `'exact'`** (same results) cost as much as `'exact'` or more: render passes aren't cheaper per texel. A dependent compute dispatch costs about 3–4 µs on the RTX and Intel (2026-11-15, `--compare=dispatch`; [dispatch.md](results/windows/round2/dispatch.md)); round 1's estimate of ~10 µs predated the 32 × 32 tiles.
  - **Native platform paths** (D3D12): on the RTX, AMD SPD beats render passes by 27–31 % and `GenerateMips` by 15–20 %; on Intel, render passes and `GenerateMips` beat every SPD variant by ~27 % in rgba8 (level in rgba16f). nvpro's Vulkan compute is 3–5× slower than `vkCmdBlitImage` on Intel. Nothing like `greggman`'s 10× on NVIDIA. sRGB natively (2026-11-17): SPD 0.77–0.86× of render passes on the RTX; render passes win on Intel (SPD 1.5–1.7×) ([native-srgb.md](results/windows/round2/native-srgb.md)).
- **Verdict:** keep, resolved by measurement: compute for every format that can be a storage texture. The exception is sRGB (entry 16): since 2026-11-16 the library has its own render path, the default for sRGB textures that can render, so engines don't each need a render-pass mipmapper for sRGB. A render fallback stays for textures we can't write. AMD and mobile are unmeasured.

### 18. Textures need `STORAGE_BINDING`
- **They:** render-pass generators need only `RENDER_ATTACHMENT`. `Kangz` flagged that adding storage usage to sampled textures has an unmeasured cost.
- **We:** need `STORAGE_BINDING` on every destination; the three.js integration would add it automatically.
- **Effort:** none. Avoiding it would require the buffer output path for every format.
- **Performance:** some drivers may disable compression on storage textures. Measured natively on Windows (D3D12, results/windows/native-tables.md): the same render-pass chain on a texture that also allows UAV access vs a render-target-only texture is 0.99–1.01× on the RTX 3060 and 0.96–1.08× on Intel UHD (no consistent direction). The flag itself costs nothing measurable there; still unmeasured on Apple and AMD.
- **Measured in WebGPU (M3, `--compare=storage-usage`):** sampling a texture with `STORAGE_BINDING` against one without costs nothing on the M5 Pro (0.999–1.006×), the RTX or Intel.
- **Verdict:** keep: no cost on Apple, NVIDIA or Intel. AMD unmeasured.

### 19. Benchmark method
- **They:**
  - `greggman` measures work per frame, because he found timestamp queries unreliable.
  - nvpro interleaves alternatives, takes the min and median over 256 batches, and compares against a speed-of-light kernel.
- **We:** use timestamps bracketing 16 repeats, with medians of 11 rotated rounds, plus a throughput mode (`--compare=throughput`, chains per second without timestamps) as a cross-check.
- **Measured (M3):** timestamps order strategies like throughput: 80 of 83 pairs on the M5 Pro (Kendall τ 0.81), τ 0.83 on Intel, 0.46 on the RTX when hot ([m3.md](results/windows/round2/m3.md)). No speed-of-light kernel.
- **Fixed (2026-11-17, [srgb-m5.md](results/m5/srgb-m5.md)):** the markers depend on the texture they bracket, so a strategy that writes another texture (`colorSpace`'s aliased twin, the `alias` prototype, `srgb-cost`'s linear textures) now names it (`target`) and the markers bracket that. Before, Metal ran the end marker before the work (`colorSpace` timed at 0 ms); D3D12's ordering apparently hid it.
- **Verdict:** done.

## API conventions

### 20. Mip count, levels written, no counter buffer
- **They:** SPD counts `mips` excluding the source level and always halves, and needs an atomic counter buffer allocated and reset.
- **We:**
  - Count `mipLevelCount` including the source level in place, so SPD's `mips = N` is `N + 1` here.
  - With a separate source, write destination `baseMipLevel` at the same size (a copy) or half.
  - Need no counter buffer.
- **Effort:** none: documented in [docs/spd-users.md](docs/spd-users.md).
- **Verdict:** keep.

### 21. No size or level cap
- **They:** SPD handles at most 4096² and 12 levels.
- **We:** handle any size, by chaining batches.
- **Verdict:** keep.

## Side-by-side comparisons

### 22. webgpu-spd 3.0.0, measured side by side
- **They:** webgpu-spd (`JolifantoBambla`, npm 3.0.0, git `adc666e`) is a WGSL port of SPD.
  - It generates min(`maxStorageTexturesPerShaderStage`, 12) mips per pass. Above 6 mips per pass it uses SPD's atomic last-workgroup handoff, with mip 6 in a shared f32 buffer, or in a read-write texture when the format allows one.
  - It uses subgroup quads when available.
  - It has a simple API (`generateMipmaps`, `preparePass` → `SPDPass.encode( pass )`), integer and tier1 formats including rgb10a2, sub-rectangles, and a 4-way custom filter.
- **We:** the `'exact'` default, `oddSizes: 'spd'` in ⌈mips/6⌉ dispatches, and no cross-workgroup communication (entries 1–3).
- **Measured** (Apple M5 Pro, Chrome, 2026-11-12; details, tables and method in [results/compare/webgpu-spd.md](results/compare/webgpu-spd.md)):
  - **Correctness:** with ≤ 6 mips per pass it matches our SPD oracle on every in-bounds level, at all 9 sizes, in both formats.
  - **Stale levels:** its default here (8 mips per pass) returned stale levels 7+ in up to 5 % of frames at render sizes, 25–41 % at 4096² and up to 77 % at 256² (256 frames per case). Isolated runs disagreed too, and the bench's parity check caught it.
    - The read-write texture path (r32float) and 6 mips per pass had no failures.
    - This confirms entry 1's memory-model argument on real hardware.
  - **Hi-Z:** SPD odd sizes leave 0.07–52 % of min/max texels non-conservative at render sizes; ours has 0 under `'exact'`.
  - **Other defects found:**
    - `halfPrecision` is non-functional: a no-op with `shader-f16`, a shader error without it.
    - A separate target's last level is unwritten by default.
    - The shared mip-6 buffer leaks other textures' data into strips and sub-rect updates.
    - Above 4096² with more than 6 mips per pass, 3/4 of levels 7+ are never written.
  - **Performance:** same-run ratios to `'exact'`, load average 11–20.
    - Its default: 1.16–1.32× (rgba8) and 0.95–1.41× (rgba16f) at render sizes, 1.5–2.0× at 1024².
    - Its handoff-free mode: 0.81–1.25×.
    - Our `'spd'`, with the same semantics and dispatch count: 0.50–0.85×, so 1.2–2.1× faster.
    - The handoff costs 1.1–1.25× over 6 mips per pass. Its subgroup path is 1.05–1.36× slower than its own non-subgroup path.
    - Its one dispatch (256²): 0.91×, against 0.90× for our `'spd'` in two.
- **Where it is ahead:** a simpler API, a smaller bundle (8.3 KB gzipped against 13.9 KB for `Downsampler`), and it is released. Integer formats, rgb10a2 and sub-rectangles were ahead until we built them (M2.3.3, M2.3.7, M2.4; its integer averages truncate and overflow, entry 24).
- **Effort to converge:** none planned.
- **Measured on Windows** (RTX 3060 Laptop and Intel UHD, Chrome 148, D3D12, 2026-11-12; results/windows/timings.md):
  - **Stale levels:** none: 0 of 41,472 frames on each GPU (27 size/format rows × 3 variants × 256 frames, one encoder and one submit per frame). The race is real on Apple, not observed on NVIDIA or Intel.
  - **Speed:** its default ÷ our `'spd'` is 1.64 (rgba8) / 1.07 (rgba16f) on the RTX and 3.52 / 2.68 on Intel, where its subgroup path is the cost (1.78 / 1.35 without). Against native AMD SPD: its default 2.0× / 1.36× on the RTX (locked clocks), ours 1.26× / 1.27×.
- **Verdict:** keep. For webgpu-spd users: set `maxMipsPerPass: 6`, and don't use it for Hi-Z at odd sizes. This is input for the planned webgpu-spd#7 comment (SPEC §13, Consequences item 5).

## Precision on Windows and Apple

### 25. float16 levels are rounded to nearest, not truncated
- **They:** a plain `textureStore` of an f32 into rgba16float truncates toward zero on D3D12 (NVIDIA and Intel) and on Apple (probe `results/probes/f16-store-rounding.mjs`), so a naive WGSL port, webgpu-spd included, truncates there. Vulkan and some drivers round.
- **We:** round every float16 level to nearest even in the shader before the store (`f16Rounding: 'nearest'`, the default), with integer math on the float's bits, so it survives fast math. `f16Rounding: 'device'` keeps the device's conversion.
- **Why:** identical results on every GPU, matching the CPU oracle; truncation biases every level down by half a step on average, compounding along a chain.
- **Performance (measured):** 5 % of an rgba16float chain on the RTX 3060, 8.5 % on Intel UHD, about 3 % on the M5 Pro ([results/windows/round2/defaults.md](results/windows/round2/defaults.md)).
- **Verdict:** keep (owner, 2026-11-13: `'nearest'` everywhere; the Windows costs are small, so no per-device default).
