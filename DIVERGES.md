# Divergences from the primary sources

Where this library does something different from the sources in SPEC §13: AMD FidelityFX SPD, nvpro_pyramid (vk_compute_mipmaps), the WebGPU community ports (webgpu-spd, `greggman`, `davidar`, the `Volcomix` fiddle) and the requests in gpuweb #386/#5253.

Each entry says what they do, what we do, why, what it would take to converge, and the expected effect. Performance figures marked *measured* come from the Apple M5 Pro runs in SPEC §2.2, §4 and §8. Nothing has been measured on other vendors yet, so anything else is an *estimate* or *unknown*.

**Verdicts**
- **Keep:** a deliberate difference, with no plan to converge.
- **Planned:** convergence or an alternative is on the milestone list (SPEC §11).
- **Candidate:** worth doing, but not scheduled.
- **Risk:** an open question that could change a default.

Effort: **S** is under a day, **M** a few days including oracle and GPU cases, **L** a week or more.

Last updated: 2026-11-13 (M2.3.3/M2.3.7: rgb10a2unorm and integer formats, entries 22 and 24. M2.3.5: sRGB destinations, entry 16 done. Before that, M2.5: `oddSizes: 'box'` and resampled sources, entries 3 and 23. Before that, Windows: RTX 3060 Laptop and Intel UHD, native D3D12/Vulkan baselines; entries 1, 5, 17, 18, 22; results/windows/README.md)

## Summary

| # | Divergence | Source | Verdict | Effort to converge | Quality | Performance |
|---|---|---|---|---|---|---|
| 1 | No single dispatch / atomic handoff | SPD, webgpu-spd | Keep | S (probe exists) | Converging risks missing mips | Measured: 2 dispatches beat the atomic probe, 1.1–2.2× at render sizes up to 1440p; tie at 4096² |
| 2 | Batch ends at every odd size | SPD, nvpro | Tail kernel built, default since 2026-11-13; halo batches built, opt-in (`halo`, 2026-11-14) | — / M–L | None | Measured, idle: tail 0.82–0.90× (RTX 3060), 0.81–1.02× (M5 Pro), 0.97× (Intel UHD); halo 0.85–1.00× of the tail (RTX), neutral (Intel); `'spd'` structure 0.47–0.73× |
| 3 | Odd sizes match render passes, not area-weighted or dropped | SPD, nvpro | Keep default; `'spd'` and `'box'` opt-ins | Done for both (M2T.4, M2.5) | `'spd'`: drifting footprints, non-conservative Hi-Z | `'spd'` 1.4–2.1× faster at render sizes (rgba8); `'box'` same dispatches as `'exact'`, not timed |
| 4 | Conservative min/max footprint at odd edges | SPD | Keep | S (but wrong) | SPD's Hi-Z can false-cull | One extra load per odd edge texel |
| 5 | Clamp at edges, no zero reads | SPD | Keep | S | Ours: no dark or zero borders | ~None |
| 6 | Clamp only, no wrap edges | gpuweb#5253 | Candidate | S–M | Seams on tiling textures at odd sizes | ~None |
| 7 | Every level rounded as stored | SPD | Keep; unrounded opt-in is a candidate | S | Ours equals level-by-level generation | Unknown, likely noise |
| 8 | Binary `reduce(a, b)` (fold or sum), not 4-way `SpdReduce4` | SPD | Mostly converged (M2T.1) | — | Sums, counts and weighted means now supported | None |
| 9 | No alpha-coverage preservation | gpuweb#5253 | Candidate | M | Cutout foliage thins out with distance | One extra pass if added |
| 10 | No subgroup / wave ops | SPD, nvpro, fiddle | Planned (M3, gated) | L | Butterfly order changes float sums | Unknown; plausible gain at ≤512² only |
| 11 | No packed fp16 | SPD | Planned (M3) | M | f16 math, slightly less precise | Unknown; nvpro saw no gain |
| 12 | One `vec4f` workgroup array, not per channel | SPD | Planned (M3) | S | None | Unknown, vendor-dependent |
| 13 | float32 loads instead of sampling | nvpro | Planned (M3) | S | None | nvpro: sampler 26–32 % faster (NVIDIA) |
| 14 | No `texture_external` source | fiddle | Candidate | M | None | Saves one full-frame copy per frame |
| 15 | Sub-rectangles rewrite exact per-level regions, not 64-texel tiles | SPD | Done (M2.4), tighter by design | — | None | Fewer texels per partial update; bit-identical to a full recompute |
| 16 | sRGB through the buffer output path, not a UNORM alias | SPD | Done (M2.3.5), by necessity | — | Linear-light averages, correctly rounded encode; loads snap to the exact decode (SPD trusts the hardware's) | Extra write and copy of levels 1+; scratch texture measured a tie; render passes faster from 1080p up on M5 Pro |
| 17 | Compute only; no per-device render fallback | `greggman`, `toji` | Risk (M3), reduced | M | None | Windows: with 32×32 exact tiles (2026-11-13) `'exact'` ÷ render is 1.19 / 1.04 in rgba8 (NVIDIA / Intel), the tail kernel 0.98 / 1.02; `'spd'` beats render everywhere |
| 18 | Needs `STORAGE_BINDING` on the texture | gpuweb#5253 (`Kangz`) | Risk, reduced | — | None | No measurable cost on NVIDIA or Intel (D3D12); Apple, AMD unmeasured |
| 19 | Timestamp benchmark only | `greggman`, nvpro | Planned (M3) | S–M | — | Credibility of every claim |
| 20 | API conventions: mip count, levels written, no counter buffer | SPD | Keep | S (docs) | — | — |
| 21 | No size or level cap | SPD | Keep | — | — | — |
| 22 | webgpu-spd 3.0.0, measured side by side ([results](results/compare/webgpu-spd.md)) | webgpu-spd | Keep | — | Its default handoff gave stale levels 7+ in up to 77 % of frames; Hi-Z up to 52 % non-conservative | Measured: our `'spd'` 1.2–2.1× faster (M5), 1.6× / 3.5× faster than its default (RTX / Intel, rgba8); no stale frames on Windows |
| 23 | Power-of-two Hi-Z from a render-size source: area folds, not one 2×2 tap | niagara; SPD (FFX SSSR), Capsaicin | Keep | — | Ours conservative at any ratio; a 2×2 tap misses a texel at 1920 → 1024 | Up to ⌈S/n⌉ + 1 loads per axis; not timed |
| 24 | Integer averages: exact mean, rounded to nearest (ties to even), no overflow; not a truncating `/ 4` | webgpu-spd | Keep | S | Truncation loses 3/8 of a unit per level on average; its 32-bit sums wrap | A few integer ops per texel; odd sizes split values into 16-bit halves; not timed |

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
  - Part of it is the exact kernel's arithmetic: `'spd'` is 0.81× at 1024² rgba8 with the same 2 dispatches, and 0.63× for max.
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
- **Performance (measured, M5 Pro, loaded):** `'spd'` is 1.4–2.1× faster than `'exact'` at render sizes up to 1440p in rgba8, 1.2–1.6× in rgba16f up to 1080p, within 3–9 % at 4K and at powers of two from 2048². `'box'` has `'exact'`'s dispatches; its odd-level batches load up to 9 texels per output instead of one sampler fetch. Not timed yet (`--compare=spd` has `box` and `tailBox`).
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
- **We:** quantize each level exactly as stored. For 8-bit and 16-bit normalized formats, the math runs in exact integer units with ties to even.
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
- **Why binary:** a pairwise reduction composes inside the tile, and normalisation happens once when a level is emitted. A 4-way custom reduction adds nothing that a binary sum plus a weight channel can't express.
- **Still missing:** argmin/argmax with a payload (ssgiDev's normal taken from the min-depth texel), and a non-commutative "first valid" scan (GI-1 probe masks).
- **Verdict:** mostly converged. The payload fold is a candidate.

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
  - WGSL defines quads only in fragment shaders and gives no mapping between lanes and `local_invocation_index`, so those layouts are undefined behaviour.
  - Firefox lacks `subgroups`; Chrome and Safari 27.3 have it.
- **Effort:** L. The portable scheme in SPEC §13 builds a linear id from `subgroup_id` and uses `subgroupShuffleXor`. It ships with the workgroup-memory fallback.
- **Quality:** butterfly order pairs sums as `(a+b)+(c+d)`. That doesn't matter for integer units, but the float oracle must pair the same way.
- **Performance:** unknown. Barriers would fall from 10 to 1–2 and workgroup memory from 16 KiB to about 1 KiB. A gain is plausible at ≤512², where the kernel is latency-bound, and unlikely at 4096², where it is bandwidth-bound.
- **Verdict:** planned for M3, behind an upper-bound test; build it only if that shows ≥5 %.

### 11. No packed fp16 math
- **They:** SPD has a packed half-precision path.
- **We:** compute in f32.
- **Effort:** M. It needs a `shader-f16` kernel variant and oracle pairing.
- **Performance:** unknown. nvpro saw no gain from f16 shared memory on an RTX 3090.
- **Verdict:** planned for M3; keep only if measurably faster.

### 12. Workgroup memory layout
- **They:** SPD stores intermediates in separate per-channel arrays.
- **We:** use one `array<vec4f>`.
- **Effort:** S.
- **Performance:** unknown. Bank behaviour differs by vendor.
- **Verdict:** planned for M3; measure it.

### 13. float32 formats load instead of sampling
- **They:** nvpro measured the bilinear sampler 26–32 % faster than loads on NVIDIA.
- **We:** load float32 formats, because sampling them needs `float32-filterable`.
- **Effort:** S. Use the sample path when the device has the feature.
- **Performance:** for rgba8, our own load path measured 12–50 % slower than sampling (Apple, under load). The same is likely for float32 where filterable.
- **Verdict:** planned for M3.

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
  - zeux's niagara (`567ebe3`) uses a power-of-two pyramid, `previousPow2` of the swapchain size, filled by one min-reduction-sampler tap per texel centre. Its comment calls that conservative because each reduction is "at most by 2x2". At 1920 → 1024 a texel's area spans up to 3 source texels, so the 2×2 tap misses one.
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
- **Performance:** one extra write and copy of the written levels, as for every buffer-path format. `encodePass` can't record the copies. Against render passes (three.js's path for sRGB textures today), M5 Pro at load average 14: ours is 1.19–1.98× faster up to 1280×720 and at 1024², a tie at 1707×960, and slower from 1080p up (render passes take 0.79× our time at 1920×1080, 0.55–0.65× at 2560×1440 and 4096²), where the copies dominate. That bears on #17 for large sRGB textures; re-measure idle and on other vendors.
- **Verdict:** done, by necessity.

## Compute vs render, and measurement

### 17. Compute only, with no per-device render fallback
- **They:** `greggman` found compute generators slower than render passes on NVIDIA/D3D12, up to 10× for SPD-style ports, and `toji` reported a partner seeing the same. The community default is render passes.
- **We:** are compute only. The three.js adapter falls back to render passes only for textures we can't handle.
- **Effort:** M. Collect cross-vendor data (M3), then either choose the path per device in the adapters or add a render-pass backend here (SPEC §3 leaves room for one).
- **Performance (measured):**
  - **Apple M5 Pro:** 1.3–27× faster than render passes.
  - **Windows, WebGPU, same run** (Chrome 148, D3D12; medians at render sizes; results/windows/timings.md): with 8 × 8 exact workgroups, our default `'exact'` ÷ WebGPU render passes was **1.25 (rgba8) / 1.06 (rgba16f) on an RTX 3060 Laptop** and **1.71 / 0.94 on Intel UHD**. Most of Intel's gap was our kernel: with 32 × 32 exact tiles (the default since 2026-11-13) it is **1.19 / 1.06 on the RTX and 1.04 / 0.66 on Intel**. The tail kernel is now 0.98 / 0.90 and 1.02 / 0.63; `'spd'` 0.75 / 0.75 and 0.78 / 0.68.
  - **Render passes for the large levels, then `'exact'`** (same results) cost as much as `'exact'` or more: render passes aren't cheaper per texel (4096² level 1 ties), only per dependent step (~10 µs less than a compute dispatch on the RTX at render sizes).
  - **Native platform paths** (D3D12): on the RTX, AMD SPD beats render passes by 27–31 % and `GenerateMips` by 15–20 %; on Intel, render passes and `GenerateMips` beat every SPD variant by ~27 % in rgba8 (level in rgba16f). nvpro's Vulkan compute is 3–5× slower than `vkCmdBlitImage` on Intel. Nothing like `greggman`'s 10× on NVIDIA.
- **Verdict:** risk, reduced. After the tile fix only RTX rgba8 still favours render passes (by 19 %), and the tail kernel closes that (0.98). The lever is fewer dependent dispatches (the tail kernel, or exact batches across odd sizes), not a render-pass path; a render fallback stays for textures we can't write.

### 18. Textures need `STORAGE_BINDING`
- **They:** render-pass generators need only `RENDER_ATTACHMENT`. `Kangz` flagged that adding storage usage to sampled textures has an unmeasured cost.
- **We:** need `STORAGE_BINDING` on every destination; the three.js integration would add it automatically.
- **Effort:** none to measure the cost (M3). Avoiding it would require the buffer output path for every format.
- **Performance:** some drivers may disable compression on storage textures. Measured natively on Windows (D3D12, results/windows/native-tables.md): the same render-pass chain on a texture that also allows UAV access vs a render-target-only texture is 0.99–1.01× on the RTX 3060 and 0.96–1.08× on Intel UHD (no consistent direction). The flag itself costs nothing measurable there; still unmeasured on Apple and AMD.
- **Verdict:** risk, reduced: no cost on NVIDIA or Intel under D3D12.

### 19. Benchmark method
- **They:**
  - `greggman` measures work per frame, because he found timestamp queries unreliable.
  - nvpro interleaves alternatives, takes the min and median over 256 batches, and compares against a speed-of-light kernel.
- **We:** use timestamps bracketing 16 repeats, with medians of 11 rotated rounds.
- **Effort:** S–M. Add a throughput mode and a speed-of-light copy kernel in M3.
- **Verdict:** planned. Every cross-vendor claim depends on it.

## API conventions

### 20. Mip count, levels written, no counter buffer
- **They:** SPD counts `mips` excluding the source level and always halves, and needs an atomic counter buffer allocated and reset.
- **We:**
  - Count `mipLevelCount` including the source level in place, so SPD's `mips = N` is `N + 1` here.
  - With a separate source, write destination `baseMipLevel` at the same size (a copy) or half.
  - Need no counter buffer.
- **Effort:** S, docs only (M4 notes for SPD users).
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
