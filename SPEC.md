# downsampler — core specification

Status: milestone 2 in progress, refocused on our tools (M2T, §11). M2.1 (reductions, load path), M2.2 (separate source, float32 formats), format steps 1–2, the GI formats (buffer output path, rgb9e5ufloat, rg11b10ufloat), M2.4 (layer ranges and partial updates, `rect`), M2.5 (`oddSizes: 'box'`, resampled separate sources), M2.3.5 (sRGB destinations), rgb10a2unorm (M2.3.3) and the integer formats (M2.3.7) are done. This document is the source of truth for scope, design decisions, correctness requirements and the remaining plan. Update it when a decision changes.

## 1. Purpose

Generate texture pyramids on the GPU: mip chains first, then min/max reduction pyramids (Hi-Z), custom reductions and related downsampling chains. WebGPU first and engine-agnostic. Thin adapters (three.js first) sit on top.

Feature target: parity with AMD FidelityFX SPD, implemented independently in WGSL. SPD is the reference for what a downsampler should offer, not code to port. See §6.

**Primary users (decided 2026-11-11):** our own tools: the upscaler (pmndrs/upscaler, based on FSR 3), the denoiser and GI. three.js mipmaps benefit from the same work but don't drive it. Those tools use SPD for fused passes: a custom per-texel load from several inputs, a few levels reduced inside a tile, and a custom store, all in the tool's own dispatch. So the main product is a WGSL reduction generator that tools embed or dispatch themselves; `Downsampler` (mip chains, Hi-Z) is one user of it. Speed at render resolutions, which are rarely powers of two, is a first-class goal. A single dispatch is not, because WGSL can't make SPD's handoff safe and it measured slower than two dispatches (§2.2). Chains of up to 6 levels already take one dispatch. Full chains take two at powers of two and under `oddSizes: 'spd'`, and 5–9 under `'exact'` at render sizes (§2.1).

## 2. Two decisions to know up front

### 2.1 Odd sizes: `'exact'` (default), `'spd'` (opt-in, measured), `'box'` (opt-in)

A 2:1 reduction is only exact when a dimension halves cleanly. For odd, non-unit sizes (decided 2026-11-10; `'spd'` built and measured 2026-11-11, M2T.4):

- **`'exact'` (default, implemented).** An odd size ends the current batch, and the next level is re-sampled from the stored level with normalized-coordinate bilinear filtering.
  - Results match render-pass and WebGL mipmaps (the CPU reference in §7), so engines can switch without visual change. **"Match" means within 1 step per level**: each level within one step of a render pass's level made from the same source level (8-bit formats; float formats within their rounding and the sampler's weight step). Whole chains made independently can differ by **up to 2 steps** on Apple GPUs, because a one-step difference at one level carries into the next and Metal render passes follow no fixed tie rule: against them, a round-half-up kernel differed at 46 % of exact ties and round-to-even (ours) at 37 % (three.js parity test, `test/webgpu/mipmaps-compute.js` in the `compute-mipmaps` branch, M5 Pro, Chrome 155, 72 of 72 cases; at even sizes every 8-bit difference fell on a tie). So no kernel reproduces render-pass mipmaps bit for bit on every GPU; ours is bit-reproducible against the CPU oracle instead (§7).
  - Min/max use the conservative `2x … 2x+2` footprint, so Hi-Z is correct.
  - Cost: more dispatches at render sizes, 5–9 instead of 2 (1920×1080: 7). The exact tail kernel (M2T.4; **default since 2026-11-13**, `tail: false` turns it off) cuts that to 3–6 with the same rule: 0.82–0.90× of `'exact'` on an RTX 3060, 0.81–1.02× on an M5 Pro, neutral on Intel UHD (decision 3 below).
- **`'box'` (implemented 2026-11-13, M2.5, opt-in).** Area-weighted odd sizes, as in nvpro_pyramid. Output texel x of n covers `[x·S/n, (x+1)·S/n)` of its source level, and each source texel weighs its overlap (`areaReference` in core, `src/core/box.ts`). Per axis that is 1 tap at size 1, 2 taps of 1/2 at even sizes, and at an odd size `2n+1` 3 taps `2x … 2x+2` with weights `(n−x, n, x+1)/(2n+1)`. Every source texel contributes in proportion to its overlap, so every level keeps the base level's mean, and a 31×31 texture's 1×1 is its true mean (`greggman`'s case, gpuweb#5253; core and GPU tests).
  - **Formula checked** against nvpro's source (`nvpro_pyramid.glsl`, `reduceStoreSample_`: `w0 = (n − x)/(2n+1)`, `w1 = n/(2n+1)`, `w2 = 1 − w0 − w1`) and [docs/strategy.md](https://nvpro-samples.github.io/vk_compute_mipmaps/docs/strategy.md.html) ("Kernel Background": 1, 2 or 3 taps per axis, "energy conserving", from NVIDIA's 2005 *Non-Power-of-Two Mipmap Creation* report). Capsaicin's `gpu_mip.comp` (`914b915`) uses the same odd weights.
  - **Batches: those of `'exact'`** (decided 2026-11-13, M2.5 agent). An odd step reads `2x … 2x+2`, so a tile of the previous level lacks one texel at its far edge. Carrying it inside a batch needs a halo that grows 1, 3, 7 … texels per further odd step: nvpro's general pipeline stops at 2 levels for that reason, and found 3+ "not promising". That halo batch is the same lever `'exact'` has (§13 Consequences 3), so it should be built once, for both, and measured. Until then `'box'` ends a batch at an odd size, and the tail kernel (now the default) carries odd steps once a level fits one workgroup, with no halo. Dispatches equal `'exact'`'s: 960×540 7 (tail 4), 1129×635 6 (5), 1477×831 9 (6), 1920×1080 7 (4), powers of two unchanged (core test).
  - **Kernels.** A batch whose source level is odd in a non-unit dimension, and every tail batch, uses the box kernel (`KernelOptions.box`; a resampled source's batch uses `area`, the first output alone): loads, with the weights in the shader (`ds_area_weight`; integer overlaps, exact 1/2 and 1 at halving and same-size axes, because GPU division isn't correctly rounded). Batches that read an even level are the `'exact'` kernels unchanged (an even step is the same 2×2 box), so they still sample by default, and `'box'` at a power of two is `'exact'` bit for bit (GPU-tested). In-tile levels are exact 2×2 steps as before; the tail's odd steps are area-weighted. The extra cost is in the odd-level batches only: up to 9 loads per texel instead of the sampler's one fetch. Not timed (coordinator); `--compare=spd` has `box` and `tailBox` strategies.
  - **Rounding** follows §4 Quantization: each level is quantized as stored, normalized formats in integer units with ties to even, float16 through `ds_round_f16` (`f16Rounding`), packed formats through their encoders. The oracle rounds normalized averages from the exact rational value (Σ overlaps · units / S²); GPU float weights may land a hair off an exact tie, so as for `'exact'`, a batch's first output and a tail's odd steps get the declared tolerance, and every other level must match bit for bit.
  - **Folds** (`'min'`, `'max'`, `'minmax'`, custom): accepted, and identical to `'exact'` (decided 2026-11-13, coordinator; owner can overrule, §12). Their `'exact'` footprint is already every texel the area overlaps.
  - **Composes** with the tail kernel, both output paths and compatibility mode: bit-identical to `encode` through `encodePass`, the buffer path and the compatibility path (GPU-tested, rgba16float 133×75).
  - Recommended for quality-sensitive content: GI, irradiance and roughness chains, UI.
- **`'spd'` (implemented 2026-11-11, opt-in; replaces the dropped `'fast'`).** AMD SPD's semantics. Each batch is the `Reducer`'s `'clamp'` tile kernel emitting every level: clamped loads, `⌊n/2⌋` sizes, the last odd row or column dropped, and no rounding between the levels of a batch. The next batch reads the last stored level. Any size takes `⌈levels / levelsPerDispatch⌉` dispatches: 2 up to 4096² at 6 levels per dispatch, 3 at the default limit of 4. For filterable formats, averages take one bilinear sample at the 2×2 corner, as SPD does. A same-size separate source is copied by a 1-level `'exact'` batch first. Oracle: `spdReference` (core), checked per batch from the GPU's own source level.
  - **`'average'` only (decided 2026-11-11, coordinator; owner can overrule).** Min/max under SPD's semantics are not a conservative Hi-Z: a dropped row or column belongs to no texel, so a max pyramid misses it. `Downsampler` therefore rejects `'min'`, `'max'`, `'minmax'` and custom folds under `'spd'`, and the error points to `'exact'` or to a `Reducer` with `'zero-weight'` edges. The measurements below include max chains timed before that decision.
  - **Composes with the buffer output path and with compatibility mode.** On the buffer path the `'clamp'` tile kernel writes packed texels (`bufferOutputWgsl`, normalized), and the same layout uniform and copies follow each batch. In compatibility mode the batches write scratch levels and the per-level copy dispatches move them, as for `'exact'`. Both are tested: bit-identical to the storage path (rgba16float, where both round with `pack2x16float`) and to the direct path, plus the oracle on the buffer path's subset shapes.
  - **Measured, not made the default.** It is faster at render sizes. The price is quality and Hi-Z correctness, for tens of µs per chain (below).

**M2T.4 measurements.** Apple M5 Pro, Metal, Chrome 148 headless, 2026-11-11; `npm run bench -- --compare=spd`, `results/spd/m5-pro-compare-{1…7}.json`.
- **Load:** other agents' GPU suites ran throughout, with load averages of 8–110 over the seven runs. Absolute times are inflated, so only same-run ratios mean anything.
- **Method:** each cell is the median over the 7 runs of each run's median paired per-round ratio. Order rotates; 11 rounds × 16 repeats; marker cost subtracted.
- **Reading the table:** ratios are to `'exact'` at 6 levels per dispatch, on the same input; under 1 is faster. Dispatches are in brackets. "Atomic" is the single-dispatch probe of §2.2, texture-ready.

| Size | Format | `'exact'` ms | tail | `'spd'` | atomic | render passes |
|---|---|---:|---:|---:|---:|---:|
| 960×540 | rgba8 | 0.024 [7] | 0.85 [4] | **0.47** [2] | 1.02 [2] | 6.7 |
| 1129×635 | rgba8 | 0.031 [6] | 1.07 [5] | **0.52** [2] | 1.05 [3] | 5.7 |
| 1280×720 | rgba8 | 0.051 [5] | 0.94 [3] | **0.50** [2] | 0.97 [3] | 2.5 |
| 1477×831 | rgba8 | 0.043 [9] | 0.97 [6] | **0.50** [2] | 0.88 [3] | 3.6 |
| 1707×960 | rgba8 | 0.071 [8] | 0.90 [4] | **0.60** [2] | 0.90 [3] | 2.5 |
| 1920×1080 | rgba8 | 0.050 [7] | 0.99 [4] | **0.66** [2] | 0.91 [3] | 2.4 |
| 2560×1440 | rgba8 | 0.077 [5] | 1.02 [3] | **0.73** [2] | 0.88 [4] | 2.1 |
| 3840×2160 | rgba8 | 0.193 [7] | 1.00 [4] | 0.91 [2] | 0.96 [4] | 1.3 |
| 1024² | rgba8 | 0.020 [2] | 1.00 [2] | 0.81 [2] | 1.54 [3] | 7.6 |
| 2048² | rgba8 | 0.075 [2] | 0.98 [2] | 0.93 [2] | 1.07 [4] | 2.0 |
| 4096² | rgba8 | 0.384 [2] | 1.00 [2] | 1.00 [2] | 1.02 [5] | 1.4 |
| 960×540 | rgba16f | 0.037 [7] | 0.83 [4] | **0.64** [2] | 1.42 [2] | 3.2 |
| 1129×635 | rgba16f | 0.033 [6] | 1.02 [5] | **0.68** [2] | 1.36 [3] | 3.4 |
| 1280×720 | rgba16f | 0.042 [5] | 0.91 [3] | 0.83 [2] | 1.60 [3] | 2.6 |
| 1477×831 | rgba16f | 0.052 [9] | 0.93 [6] | **0.62** [2] | 1.00 [3] | 2.9 |
| 1707×960 | rgba16f | 0.059 [8] | 0.90 [4] | **0.69** [2] | 1.06 [3] | 2.4 |
| 1920×1080 | rgba16f | 0.100 [7] | 0.97 [4] | 0.84 [2] | 1.05 [3] | 2.0 |
| 2560×1440 | rgba16f | 0.189 [5] | 0.99 [3] | 0.94 [2] | 1.06 [4] | 1.4 |
| 3840×2160 | rgba16f | 0.368 [7] | 0.99 [4] | 0.97 [2] | 1.01 [4] | 1.3 |
| 1024² | rgba16f | 0.038 [2] | 0.98 [2] | 0.97 [2] | 1.64 [3] | 3.1 |
| 2048² | rgba16f | 0.184 [2] | 1.00 [2] | 0.99 [2] | 1.08 [4] | 1.4 |
| 4096² | rgba16f | 0.715 [2] | 1.02 [2] | 0.99 [2] | 1.01 [5] | 1.5 |

- **`'max'`**, as ratios to `'exact'` max at render sizes up to 1080p: `'spd'` 0.30–0.56 for rgba8 (also at 1440p) and 0.37–0.78 for rgba16f; 0.92–0.97 at 4K. The tail is 0.72–1.08.
- **The atomic probe's brackets** count its one dispatch plus a copy per tail level past the 8-storage-texture limit. Without the copies ("kernel only") it is 0.73–0.88 for rgba8 and 0.90–1.50 for rgba16f at render sizes.

What the numbers say:
- **`'spd'` is 1.4–2.1× faster than `'exact'` at render sizes up to 1440p in rgba8, and 1.2–1.6× in rgba16f up to 1080p.** At 4K, and at powers of two from 2048², it is within 3–9 %: those chains are bandwidth-bound and already take 2 dispatches.
- **The saving is the chain of dependent dispatches, not pass overhead.** Recording every batch into one compute pass (`encodePass`) instead of one pass per batch measured 0.94–1.03× (runs 4–7).
- **Part of it is kernel arithmetic.** At 1024² rgba8, with 2 dispatches each, `'spd'` is 0.81× (0.63× for max). The exact kernel's integer tie rounding and odd-footprint branches cost there; that is worth optimizing in `'exact'` by itself.
- **The tail kernel doesn't close the gap.** It removes 1–4 trailing dispatches, but those are cheap. The time goes to the breaks at the large levels: 1477×831 ends a batch at every level down to 46×25, and 1129×635 at level 1.
  - The tail measured 0.83–1.07×: a gain at 960×540 (0.83–0.85), about 0.90–0.97 elsewhere, often inside the noise. These runs used a 32×32 tail tile everywhere. The tile now follows the first output (§4 Tiles), which changes only 1280×720 among these sizes (10×5 → a 16×16 tile); not re-timed.
  - The `'exact'`-preserving fix for the large-level breaks is halo batches (§13). Their upside is bounded by the `'spd'` column.
- **The absolute stakes are small.** On this loaded machine, `'spd'` saves 13–29 µs per render-size chain up to 1080p. On an idle M5 Pro the whole `'exact'` 1080p chain took 0.037 ms (§8).

**Quality of `'spd'` against `'exact'`** (`npm run bench -- --quality`, `results/spd/m5-pro-quality.json`).
- **Images:** two per size. "Scene" has smooth gradients and waves, a bright one-texel line every 97 texels and on the last row and column, and sparse highlights (HDR ×2, highlights of 16 in rgba16f). "Noise" is uniform per-texel noise.
- **Ideal:** the exact area-weighted mean of the base texels under each texel's normalized footprint.
- **Units:** normalized (1.0 is full scale for rgba8). Means are texel-weighted over all levels.

| Size | rgba8 scene: max / mean \|spd − exact\| | rgba8 scene, levels ≥ 4: mean \|· − ideal\|, exact / spd | rgba16f scene: max \|spd − exact\| | Hi-Z: `'spd'` max texels below a base texel in their footprint (scene; noise) | Same for `'exact'` and the tail |
|---|---:|---:|---:|---:|---:|
| 960×540 | 0.25 / 0.0011 | 0.004 / 0.026 | 0.65 | 0.34 %; 1.2 % | 0 |
| 1129×635 | 0.64 / 0.0103 | 0.007 / 0.027 | 10.9 | 15 %; 51 % | 0 |
| 1280×720 | 0.36 / 0.0005 | 0.002 / 0.006 | 0.74 | 0.06 %; 0.02 % | 0 |
| 1477×831 | 0.64 / 0.0097 | 0.007 / 0.024 | 10.0 | 13 %; 52 % | 0 |
| 1707×960 | 0.47 / 0.0046 | 0.005 / 0.006 | 4.0 | 12 %; 32 % | 0 |
| 1920×1080 | 0.24 / 0.0006 | 0.004 / 0.011 | 0.52 | 0.10 %; 0.18 % | 0 |
| 2560×1440 | 0.37 / 0.0004 | 0.001 / 0.002 | 0.73 | 0.01 %; 0 | 0 |
| 3840×2160 | 0.24 / 0.0004 | 0.002 / 0.003 | 0.51 | 0.03 %; 0.02 % | 0 |
| 1024², 2048², 4096² | 0.004 / ≤ 0.0006 | equal | 0.002 | 0 | 0 |

- **Where sizes halve cleanly, the modes agree to one 8-bit step.** The only difference there is that `'spd'` doesn't round between levels. Every odd size changes the result from that level down: at 1920×1080, levels 1–3 agree to 0.004, and levels 4–10 differ by up to 0.24 (mean 0.008–0.047 per level).
- **The first odd size decides how bad it gets.** When level 0 or 1 is odd (1129×635, 1477×831, 1707×960), `'spd'`'s level-1 texels drift by up to one texel from their normalized footprint towards the far edge. Every level is then off: up to 0.64 in rgba8, and 11 on HDR highlights.
- **On the scene image, `'exact'` is closer to the area-weighted ideal:** 1.3–7× at levels ≥ 4, and up to 10× on single levels (1920×1080 level 6: 0.004 against 0.039). On uniform noise `'spd'` is closer, because a 2×2 box averages away more noise than a 2-tap bilinear. So it is not worse everywhere; `'box'` beats both by construction.
- **Hi-Z:** for `'max'`, `'spd'` leaves 0.01–15 % of texels smaller than some base texel inside their footprint on the scene image. On noise it is up to 52 %, and up to 94 % on smooth rgba16f gradients, where the partly covered third texel is always the larger one. `'exact'` and the tail have no violations at any size, which is the property Hi-Z culling needs.
- **The tail kernel matches `'exact'`** within one rgba8 step (0.004), and within 0.008 in rgba16f, where its shader bilinear weights replace the sampler's 8-bit ones.

**Decision (unchanged): `'exact'` stays the default; `'spd'` ships as an opt-in.** The data supports the default. `'spd'`'s speed comes with footprints that drift at odd sizes, deep levels several times further from the area-weighted ideal, and a non-conservative Hi-Z, for tens of µs per chain.

**Decided (coordinator, 2026-11-11; owner can overrule):**
1. `oddSizes: 'spd'` stays a public opt-in, for 'average' only: SPD parity and speed-first average chains that tolerate dropped edges, such as bloom or exposure-style pyramids. Folds are rejected under it.
2. The atomic probe stays test-only (`test/bench/atomic-spd.js`).
3. ~~The tail kernel stays opt-in~~. **Superseded 2026-11-13 (owner rule: default if faster on most devices): `tail: true` is the default.** Tail ÷ `'exact'` at render sizes, idle: RTX 3060 0.82 / 0.90 (rgba8 / rgba16f), Intel UHD 0.97 / 0.97, M5 Pro 0.81–1.02 with 16×16 exact tiles (`results/render/m5-pro-render-tiles-*.json`, `results/windows/timings.md`). The same sweep set `exactTile`'s default to 16 on Apple GPUs: 32 was 1.04–1.28× slower than 8 or 16 there at render sizes, while on the RTX and Intel 32 was the fast one. Tail levels use shader bilinear weights, within one rgba8 step of the sampler's. It composes with both output paths and with compatibility mode, where devices with fewer than 256 invocations per workgroup cap tail tiles at 8×8.

**Still for the owner:**
- If `'exact'` must get closer to `'spd'` at render sizes, the levers are halo batches for the large-level breaks (§13) and a cheaper exact kernel (integer rounding, fold footprint branches). Measure on NVIDIA, AMD and Intel first (§9): the cost of a dependent dispatch there is unknown.

### 2.2 No single-dispatch mode

SPD writes every level in one dispatch. The last workgroup to finish, detected with a global atomic counter, reduces the final levels. We prototyped that handoff in WGSL and measured it. Two independent reasons rule it out:

1. **The handoff isn't guaranteed by WGSL.**
   - All WGSL atomics are relaxed.
   - All barriers (`storageBarrier`, `workgroupBarrier`, `textureBarrier`) have workgroup memory scope.
   - So seeing the counter at N−1 does not guarantee visibility of another workgroup's payload: "No synchronization or ordering guarantees apply … between atomic accesses acting on different memory locations." ([WGSL §memory model](https://gpuweb.github.io/gpuweb/wgsl/#memory-model))
   - A request for acquire/release atomics was closed on 2026-07-14 as not widely implementable, because Metal and D3D lack them ([gpuweb#5572](https://github.com/gpuweb/gpuweb/issues/5572)). Barriers were narrowed to workgroup scope for the same reason ([gpuweb#2297](https://github.com/gpuweb/gpuweb/pull/2297)).
   - Inter-workgroup message-passing violations have been observed through WebGPU on real AMD GPUs (Levine et al., *MC Mutants*, ASPLOS 2023).
   - Dispatch boundaries are fully synchronized by the API, so a second dispatch is the portable equivalent.
2. **It was slower anyway.** Measured on Apple M5 Pro / Metal / Chrome 155, in GPU ms per full chain. "Original" is the prototype as first written: 4 direct levels, the counter cleared every dispatch, 4-word payloads. "Best variant" adds packed payloads, a self-resetting counter and 8 direct levels. "Two dispatches" is this library's kernel. The columns come from the same experiment harness, with medians of 11 rounds.

   | Size | Format | Atomic single dispatch, original | Atomic, best variant | Two dispatches, coalesced |
   |---|---|---:|---:|---:|
   | 256² | rgba8 | 0.0319 | 0.0099 | **0.0068** |
   | 1024² | rgba8 | 0.0466 | 0.0305 | **0.0151** |
   | 4096² | rgba8 | 0.3903 | 0.3857 | **0.3545** |
   | 1024² | rgba16f | 0.0689 | 0.0592 | **0.0246** |

   The atomic tail costs a near-constant ~20 µs. The last workgroup issues dozens of serially dependent atomic loads, and packing the payload reduced but did not remove that. The dispatch boundary it saves costs about 1–2 µs. The prototype lives in the three.js experiment branch (`webgpu-spd-audit`).

3. **Re-measured at render sizes (M2T.4, 2026-11-11): it still loses to two dispatches.** `test/bench/atomic-spd.js` is a measurement-only probe, not an API. It relies on the handoff WGSL doesn't guarantee, and says so in its header.
   - **Construction:** phase 1 is the library's 6-level `'clamp'` tile kernel, so `oddSizes: 'spd'` semantics at any size up to 4096². Each workgroup publishes its level-6 texel with `atomicStore` (packed: 1 word for rgba8, 2 for rgba16f), runs `storageBarrier()` and takes a ticket. The last workgroup resets the self-resetting counter and reduces the remaining ≤ 6 levels from the payloads with the same tile kernel and `atomicLoad`. Tail levels past the 8-storage-texture limit go to a packed buffer and are copied into the texture.
   - **Checks before timing:** it must reproduce `'spd'` on 4 changing inputs, with levels 1–6 bit for bit and the tail within the format tolerance, and leave the counter at 0. It passed at all 22 size/format pairs on the M5 Pro. That is evidence for one device, not a proof.
   - **Result:** against the same kernel split into 2 dispatches (`'spd'`), it is slower at every size measured except 4096², where the two tie within 2 %. Texture-ready, it is 0.88–1.05× of `'exact'` for rgba8 at render sizes, where `'spd'` is 0.47–0.73×, and 1.00–1.60× for rgba16f, where `'spd'` is 0.62–0.94×. At 1024² it is 1.5–1.6×. Without its copies it is still behind `'spd'` (§2.1 table).
   - **Why (likely):** the boundary it removes is cheap, and what it adds isn't. Every workgroup pays a storage barrier and an atomic, and the elected workgroup reads mip 6 through serially dependent atomic loads. A second dispatch reads it with plain texture loads, and the API guarantees their visibility.

If WGSL gains device-scope release/acquire, revisit with a measurement first.

## 3. Architecture

```
src/core/     backend-neutral: batch planning, odd-size policy, formats and
              quantization, CPU reference (the oracle for every backend)
src/webgpu/   WGSL kernel generator + Downsampler (GPUDevice/GPUTexture API)
src/three/    adapter for three.js WebGPURenderer (three is an optional peer)
test/         unit tests (core), GPU suite + device lab (browser), benchmark
```

Rules:
- **Core stays backend-neutral:** no WebGPU types in `src/core`. A future backend (WebGL2 render fallback, Dawn/Node, wgpu-native, Deno) reuses planning and the reference unchanged.
- **Zero runtime dependencies.**
- **Kernels are generated WGSL strings** so they can be inspected, cached and vendored, e.g. into three.js.

## 4. Algorithm

### Batches

A batch reads one source level and writes up to `levelsPerDispatch` consecutive levels.

- **First output:** a hardware bilinear sample at each texel centre, using normalized coordinates. That is an exact 2×2 box for even sizes. For odd sizes it reproduces render-pass and WebGL mipmaps, which is **not** area-correct: 3→1 takes only the centre texel, so a 31×31 texture can reduce to an unrepresentative 1×1 (`greggman`'s example in gpuweb#5253). `oddSizes: 'box'` (§2.1) area-weights odd sizes instead, and a resampled separate source (below) area-weights any ratio.
- **Later levels in the batch:** exact 2:1 reductions in workgroup memory, with clamped reads for unit dimensions and partial tiles.
- **Batch boundaries:** a batch ends at `levelsPerDispatch`, or where the next reduction is inexact under `'exact'` (and `'box'`, which plans the same batches). With `tail: true`, a first output of at most 32×32 (8×8 below 256 invocations per workgroup) starts a tail batch that finishes the chain (up to `levelsPerDispatch` levels) in one workgroup by the same rule. Under `'spd'`, batches end only at `levelsPerDispatch` (§2.1).
- **Layers:** workgroup z selects the array layer, so all layers go in one dispatch. A layer range (`layers`, §5) offsets z by the region uniform below.

### Partial updates (`rect`, `layers`; M2.4, 2026-11-13)

A partial update rewrites only what depends on a changed region: `rect` marks the dirty texels of the first source level (level `baseMipLevel` in place, or the separate source's level), `layers: [ first, count ]` the array layers. The region math is in core (`src/core/region.ts`: `planRegions`, `exactDependents`, `clampDependents`, `areaDependents`).
- **Per-level regions.** Each level's region is the set of texels whose footprint touches the previous level's region, per axis:
  - `'exact'` and tail batches: the fold footprint `2x … 2x+1`, plus `2x+2` along an odd source size (`footprint()`), so output `x` depends on dirty `[a, b]` when `⌊(a−1)/2⌋ ≤ x ≤ ⌊b/2⌋` (odd) or `⌊a/2⌋ ≤ x ≤ ⌊b/2⌋` (even or unit). It covers the bilinear average too, at any sampler weight rounding: a truncating sampler (Intel) moves a weight-0 tap from `2x+1` to `2x`, still inside. A same-size copy maps `[a, b]` to itself. Levels chain one step at a time. `oddSizes: 'box'` reads the same footprint (its odd steps are the three taps `2x … 2x+2`), so its regions are these.
  - A resampled separate source (`resample: 'area'`, §4 Separate source): output `x` depends on dirty `[a, b]` when its area span (the source texels of `[x·S/n, (x+1)·S/n)`) meets it, so `⌈(a·n + 1)/S⌉ − 1 ≤ x ≤ ⌈(b + 1)·n/S⌉ − 1`, clamped (`areaDependents`; at 2:1 and for a copy it equals the rule above). Each texel reads exactly its span, so this is conservative and bit-identical (M2.5, 2026-11-13; core test against brute force over spans, GPU cases in `validate-region.js`).
  - `'spd'` (`'clamp'` batches): level `l` of a batch reduces the clamped block `[x·2^l, (x+1)·2^l)` of the batch source, so `x ∈ [⌊a/2^l⌋, min(⌊b/2^l⌋, n−1)]`, computed from the batch source directly: a dropped last row or column belongs to no level-1 texel, but can reach a deeper one through the clamp (3 → 1 → 1), so in-batch levels can't be chained. A level whose region is empty writes nothing; a batch with no texels to write is left out.
- **Bit-identical.** A texel in a region is computed by the same kernel code from the same inputs as in a full dispatch, so when the source changed only inside the rect and the old chain was valid (made by the same plan: a chain from another plan, such as another Downsampler's at another number of levels per dispatch, differs in the last bit where one stores and rounds a level the other keeps in the tile), the result equals a full recompute bit for bit, and every texel outside the regions (and outside the layer range) is untouched. Updates compose: several rect updates in a row (or in one encoder) give the chain of the final source.
- **Kernels.** A `region` variant of the exact, tail and `'clamp'` kernels adds a uniform (binding 11, 112 bytes): the first workgroup (in tiles) and the first layer, and per level the region's bounds. The dispatch covers the workgroups whose tiles hold any region texel (a tail batch is one workgroup); each computes its whole tile as usual and stores only inside the region. Full updates keep the old kernels: WGSL and layouts unchanged (4420 option sets compared byte for byte).
- **Everything composes.** Every odd-size policy (`'box'` included), reduction, kernel width, the tail kernel (the default), separate sources (the rect is in source texels, also when resampled), the buffer output path (the shared full-level buffer layout; each level's copy takes the region of the layers in range), compatibility mode (scratch levels get the regions, the copy dispatches copy the regions, and batches read the destination, which the copies have brought up to date) and `encodePass`. Nothing is rejected except invalid values: a rect that isn't integer, non-negative and inside its level (the error names the level), and a layer range outside the texture. **An empty rect** (width or height 0, still inside the level: `x ≤ width`, `y ≤ height`) is a no-op, as WebGPU's zero-size copies are (owner, 2026-11-13): once the other options are validated, `prepare` returns no batches and `encode`/`encodePass` record nothing, and no plan, buffer or bind group is created (`planRegions` in core gives every batch null regions and 0 workgroups).
- **Cost and caching.** Per distinct rect and layer range: one uniform buffer (a 256-byte slot per dispatch) and bind groups, written once. The 16 most recent are cached per texture. An evicted plan's uniform buffer is destroyed after `queue.onSubmittedWorkDone()` (owner, 2026-11-13), as the Reducer's and ScalarReduction's evictions are (§4 Caching), so many partial updates may still be recorded before one submit in the same task; so is a plan replaced by a new source texture. `release( texture )` destroys the rest at once. GPU-tested with a buffer tracker (`region/options`). The buffer path's scratch buffer is shared by a texture's partial plans; a larger one replaces it and the old one is left to the plans that still hold it (GC). A ring buffer of region uniforms with dynamic offsets, for rects that change every frame, is a later task (PROGRESS, Later).
- **Against SPD's rectInfo:** SPD snaps the rect to 64-texel source tiles and recomputes levels below 64×64 whole; we write only the exact dependent texels of each level (DIVERGES #15). Not timed.

### Reductions

- **`'average'`** is the bilinear sample above. It matches render-pass mipmaps.
- **Folds** (`'min'`, `'max'`, `'minmax'`, custom) combine every source texel an output texel covers with a binary `reduce`.
  - **Footprint:** texels `2x` and `2x+1`, plus `2x+2` along an odd source dimension, clamped (`footprint()` in core). With an odd size `2n+1`, output texel `x` spans `[x(2+1/n), (x+1)(2+1/n))`, which touches `2x … 2x+2`. So this is the smallest conservative footprint, the rule Hi-Z needs. Under `'exact'` an odd size only occurs at a batch's first output, which reads it with `textureLoad`. Inside a batch every reduction is an exact 2×2.
  - **Contract:** `reduce` must be associative, commutative and idempotent (a semilattice). Idempotence makes clamped duplicate reads and the overlap between neighbouring odd footprints harmless. A sum or count is therefore not a valid fold.
  - **`'minmax'`** takes the min of even channels and the max of odd channels, so `rg` holds (min, max) and `ba` a second pair. In-texture levels are read as already-reduced pairs. Expanding a single-channel source to `(d, d)` belongs to the separate source (M2.2).
  - **Units:** custom folds see normalized values. For unorm8 the kernel converts to and from its 0–255 working units around the call.
  - **Tolerance:** folds select existing values, so the GPU must match the oracle exactly.

### Separate source

- **Use:** a destination that can't be its own source, mainly a depth buffer feeding a Hi-Z pyramid (depth formats can't be storage textures), or a format conversion.
- **Size rule:** destination level `baseMipLevel` must be the same size as the source level (a converting copy) or half of it (one reduction step), or any size with `source.resample: 'area'` (below). Every destination level in range is written, including `baseMipLevel`. `mipLevelCount` counts destination levels in all modes.
- **Resampled source (`resample: 'area'`, M2.5, 2026-11-13):** for a destination whose first level is another size, mainly a power-of-two Hi-Z (1024×512 or 2048×1024 from 1920×1080) whose later levels halve cleanly.
  - **Semantics of the first step.** Destination texel x of n covers `[x·S/n, (x+1)·S/n)` of the source per axis, at any ratio up or down. `'average'` weights each source texel by its overlap (the `'box'` weights, of which nvpro's are the 2:1 case). Folds combine every source texel with a positive overlap, so min/max stay conservative: every source texel a Hi-Z texel's normalized area touches is in its fold, at every level (GPU-checked with `maxViolations` against the source, 1920×1080 → 1024×512 and → 2048×1024, 1129×635 → 512×256, strips, layers). The kernel is the box kernel (`ds_area_first` … `ds_area_last`, ⌈S/n⌉ + 1 taps per axis at most), so it costs loads, not a sampler fetch.
  - **Later levels** follow `oddSizes`. With a power-of-two destination they halve exactly, so a 12-level 2048×1024 Hi-Z takes 2 dispatches at 6 levels per dispatch (the first batch reads the source and writes 6 levels). Under `'spd'` the resampled level is its own 1-level batch, as a copy is.
  - **Validation:** without `resample`, another size is rejected, and the reason names the option; unknown values are rejected; so are sources more than `maxResampleRatio` times larger than the destination along an axis. The cap is a constructor option (`new Downsampler( device, { maxResampleRatio } )`, any number ≥ 1, default the exported `maxResampleRatio`, 8: at most 9 taps per axis), owner 2026-11-13. It costs nothing: the kernel's tap loops run over each texel's span at run time (`ds_area_first` … `ds_area_last`), so only validation reads it and no pipeline key changes; a raised cap only makes the first level's taps grow with its square (GPU-checked at 16.25:1 against `areaReference`). `resample: 'area'` with a same-size source is a copy.
  - **Others.** Capsaicin's HZB (`914b915`, `visibility_buffer.cpp`) copies depth into a render-size level 0, then `GPUMip` DepthMin (`2x … 2x+2` at odd sizes, SPD once a level is a power of two). FidelityFX SSSR (SDK 1.1.4, `ffx_sssr_depth_downsample.h`) also keeps level 0 at the depth's size and runs SPD's min, which drops odd rows and columns. The power-of-two layout is zeux's niagara (`567ebe3`, `niagara.cpp`): `previousPow2` of the swapchain size, filled by one 2×2 min-reduction-sampler tap at each texel centre. Its comment says this keeps every reduction "at most by 2x2", so conservative; at 1920 → 1024 a texel's area spans up to 3 source texels, and the 2×2 tap misses one (a core test counts over 100 such texels on a 1920-texel row). WebGPU has no reduction sampler anyway, so we fold the overlap.
- **Planning:** the first batch reads the source and every later batch reads the destination (`Batch.sourceMipLevel === null` marks the source batch). The copy is a runtime branch on equal sizes, uniform across the dispatch, so it needs no pipeline variant.
- **Source formats:** anything float-sampleable, including depth (through the depth aspect, bound as `unfilterable-float`). Depth and 32-bit float sources always load.
- **Splat:** single-channel sources (depth, `r*`) are read as `(v, v, v, v)`, so `'minmax'` of depth gives `(d, d)` with no special case.
- **unorm8 rounding on load** applies only when the source is unorm8 itself. Rounding a 16- or 32-bit source to 8 bits before reducing would round twice.
- **Caching:** a plan is cached per destination and option set, holding only the most recent source texture. Recreating a depth buffer on resize therefore replaces the entry instead of growing the cache.

### Source access

- **`'sample'`:** one `textureSampleLevel`. `'average'` only, filterable formats only.
- **`'load'`:** `textureLoad` of the footprint, bound as `unfilterable-float`. Required for folds, depth, float32 formats without `float32-filterable`, and the box kernel (`'box'` batches that read an odd level, tails under `'box'`, a resampled source). For `'average'` it computes the bilinear weights in the shader, without the sampler's fixed-point weight rounding.
  - **Integer weights:** the position `(x + 0.5)·S/n − 0.5` is computed as `((2x+1)·S − n) / 2n`, giving an exact base texel and a weight of `remainder / 2n`. GPU float division isn't correctly rounded (WGSL allows 2.5 ulp). A float version computed 2.9999998 for an exact 3.0, so `floor` picked the wrong pair.
  - **unorm8:** texels are rounded back to exact integers on load, so even-size averages are exact.
- **sRGB sources** (M2.3.5, §6.1 sRGB) are read through their sRGB view, so the hardware decodes every texel to linear, and the sampler filters in linear. Hardware decoding isn't exact everywhere: the M5 Pro returns 12-bit linear values (k/4095; 254 of 256 codes differ from the exact decode, by up to 0.39 of a code step; each still re-encodes to its own code; `results/probes/srgb-probe.mjs`). So `'load'` snaps every loaded texel to the exact decode table (re-encode, then look up), and its results are the same on every GPU; `'sample'` keeps the hardware's decode and filtering, as render passes do, within about a code of the exact result (the main matrix's tolerance is two codes).
- Bind group layouts are explicit, not `'auto'`, so layouts can be shared across reductions and the load path can declare `unfilterable-float`.
- **`'average'` keeps the sampler by default.** Load against sample, same run with rotated order (Apple M5 Pro, Chrome 155, 2026-11-10, measured while the machine had a load average of ~18, so only the ratios are usable; `results/m5-pro-m2.1-under-load.json`):

  | Format | 256² | 512² | 1024² | 2048² | 4096² | 1920×1080 | 512²×6 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | rgba8 | +12 % | +19 % | +50 % | +26 % | −3 % | +16 % | +13 % |
  | rgba16f | +11 % | +4 % | +2 % | +3 % | +2 % | +1 % | +2 % |

  The unorm8 load path does four loads plus rounding where the sampler does one fetch. A `'max'` fold costs about the same as load-average. `averageAccess: 'load'` stays as an option: it is the path for unfilterable formats, and the GPU suite uses it. Re-measure on an idle machine and on other vendors (M3).

### Tiles

| Levels in batch | Workgroup | First-output tile | Source tile | Workgroup memory |
|---|---|---|---|---|
| ≤ 4 | 8×8 (`'exact'`: 16×16 by default, below) | 8×8 (`'exact'`: 32×32) | 16×16 (64×64) | 1 KiB (16 KiB) |
| 5 | 16×16 | 16×16 | 32×32 | 4 KiB |
| 6 | 16×16 | 32×32 (2×2 per invocation) | 64×64 (as in SPD) | 16 KiB (default limit) |
| tail (any count) | 8×8 or 16×16 | 8, 16 or 32, the first output's larger side rounded up | — (one workgroup) | 1–16 KiB |

**`'exact'` batches use a 32×32 tile at any level count** (`exactTile`, default 32 where a workgroup can have 256 invocations; 8 in compatibility mode and on fallback adapters). Same results; 8×8 workgroups were 1.2–1.6× slower on Intel UHD and up to 1.3× on an RTX 3060 at render sizes (results/windows/timings.md, 2026-11-13). Level sizes inside a batch are derived from the first output, not queried per level with `textureDimensions` on a storage texture (5–25 % on Intel, rgba8). Not yet re-measured on Apple. Tile kernels (`Reducer`, `'spd'` batches under 5 levels) still follow the level count.

Tail tiles follow the first output, not the level count: SwiftShader took 30 s or more to compile each 32×32-tile pipeline, and an 8×8 tail of a small image doesn't need one.

**Layout matters.** Adjacent invocations sample and store adjacent texels, and each processes four 16×16 sub-tiles. Giving each invocation its own 2×2 block measured 30–45 % slower.

### Levels per dispatch

- `min(6, maxStorageTexturesPerShaderStage)`. The default WebGPU limit is 4.
- `requiredLimits( adapter )` (from `downsampler/webgpu`; also `Downsampler.requiredLimits`) gives apps the limits to request, for `Reducer` and `ScalarReduction` too.
- Around 94 % of devices report at least 8 (web3dsurvey).
- Storage texture views must cover exactly one mip level ([WebGPU spec](https://gpuweb.github.io/gpuweb/)), so each output level needs its own binding.

### Quantization

Every intermediate level is quantized exactly as stored, so a chain equals level-by-level generation.

- **rgba8:** values stay in integer 0–255 units in workgroup memory. Sums are exact and ties round to even. With `k/255` floats, float error turned ties into round-ups. A 1×128 tail drifted 3 steps by mip 5: half of all 1D texels are ties.
- **float16 formats:** values are rounded to nearest even in the shader with integer math (`float16Wgsl`, `ds_round_f16`; mirrors core `toHalf` bit for bit), then stored as exact halves, so no device conversion rounds anything. `f16Rounding: 'device'` (Downsampler and Reducer options) uses `pack2x16float` / `unpack2x16float` instead: cheaper, but D3D12 truncates there (§ Findings, 2026-11-12).
- **sRGB (rgba8unorm-srgb, bgra8unorm-srgb; `srgb8`, `src/core/srgb.ts`):** colour is linear f32 in workgroup memory, alpha in integer 0–255 units (ties to even, as rgba8unorm). Each level is quantized as stored, `decode(encode(v))` per colour channel: the code of v is the one whose rounding interval holds it, between the thresholds `t(k) = decode((k + ½)/255)`, i.e. `round(255 · decode⁻¹(v))` with the exact IEC 61966-2-1 curve, and its value is read back from the exact decode table (the f32 nearest each code's exact decode). The shader's `ds_srgb_code` is a `pow` estimate corrected by one step against the thresholds, stored as the smallest f32 above each (`srgbBounds`): for an f32 v, v > t(k) exactly when v ≥ bound(k), so the encoder is correctly rounded for its f32 input, bit for bit with core `srgbCode` (GPU-checked at every threshold ± 3 ulps, every table value, specials and 40000 random values). **Ties:** no f32 lies on a threshold (thresholds are not binary fractions; core test), so the rule is plain round-to-nearest of the f32 working value. Real-number ties exist only in the linear segment (codes ≤ 10), where an average of codes can end in exactly .5 (e.g. 0, 0, 0, 2); the f32 decode table puts the working value an ulp to one side, so they resolve deterministically, the same in the shader and in the oracle's float32 pairing, but not by a declared tie rule. NaN and negatives encode to 0, values above 1 to 255. Stores pack the codes; encoding a table value returns its code, so stored codes are what the next level reads.
- **rgb9e5ufloat, rg11b10ufloat:** whole texels go through our encoder and back (`ds_pack9e5` / `ds_pack11_11_10` in WGSL, `quantizeTexel` in core; rules in §6.1); alpha reads 1. The rg11b10 storage path stores already-quantized values, so the GPU's own f32 → 11/10-bit conversion never rounds anything.

### Caching

- Pipelines are cached per `Downsampler` (per device) and per reduction; bind group layouts per format, view dimension, width and access.
- Bind groups are cached per texture and options in a `WeakMap`.
- `release(texture)` drops a texture's entries early and destroys the buffers and scratch textures they own.
- **Buffer lifetime (all three classes, 2026-11-13).** A class destroys only buffers it created (size and layout uniforms, region uniforms, buffer-output buffers, scalar params and partials), never a caller's. `release` destroys them at once, so call it after submitting the last work that uses them. Entries a bounded cache evicts or replaces (a `Reducer`'s 16 bind groups, a `ScalarReduction`'s 4 plans per resource, a `Downsampler`'s 16 partial-update plans per texture, and a `Downsampler` plan whose source texture was replaced) are destroyed after `queue.onSubmittedWorkDone()`, not at once: an evicted entry can still be recorded in an encoder the caller hasn't submitted (more than 16 Reducer dispatches in one encoder), and destroying it then makes that submit fail ("used in submit while destroyed", GPU-tested). The promise resolves in a later task, after the usual encode-and-submit of a frame.

### Compatibility mode (2026-11-11, reflections D4100)

WebGPU compatibility mode is a device without `'core-features-and-limits'` (`requestAdapter( { featureLevel: 'compatibility' } )`; three r187 falls back to it). The rules that touch us, from the WebGPU spec (checked against the editor's draft, 2026-11-11):
- **One view per texture per dispatch.** "All bindings referring to the same GPUTexture must have compatible GPUTextureViews": equal `baseMipLevel`, `mipLevelCount`, `aspect` and swizzle, across all bind groups of a draw or dispatch (dispatch validation). An in-place chain reads level n and writes n+1 … in one dispatch, so it fails there. That is D4100.
- **View dimension is fixed at creation.** Texture and storage views must have the texture's `textureBindingViewDimension` (bind group creation). A `'cube'` texture can't be written at all (WGSL has no cube storage); a 6-layer `'2d-array'` one can.
- **Whole textures only:** bound views have `baseArrayLayer` 0 and every layer.
- **Limits:** 4 storage textures, 128 invocations per workgroup and 128 along x by default (the 5- and 6-level kernels and `ScalarReduction` need 256). `rg32float` has no storage (probed on Chrome 148 compat; every other format in §6.1 has it).

**What `Downsampler` does.** With `compatibility: 'auto'` (default) a texture takes the compatibility path when its device lacks `'core-features-and-limits'` and the texture has a `textureBindingViewDimension` (that attribute exists only on compatibility-mode devices, so older implementations that don't know the feature name stay on the direct path). `true` and `false` force it.
- Each batch writes its levels to scratch textures, one per level at that level's size (created on first use, kept per destination, destroyed by `release( texture )`). The batch reads the destination only at its source level, or a scratch level, or the separate source. Then one dispatch per level copies the scratch into the destination: the 1-level fold kernel on a same-size source, which reads one texel per output, so the copy is exact (an average with weight 0 would turn an Inf neighbour into NaN). Every dispatch binds one view of each texture.
- Batches, odd-size planning and results are unchanged: bit-identical to the direct path (GPU-tested on a core device with the path forced, in place, sub-range, depth source and layers, through `encodePass`). `prepare()` lists the copies with `copy: true`.
- Cost: one extra write and read of every written level (about a third of the base level's bytes) and a dispatch per level. It needs only `STORAGE_BINDING` and `TEXTURE_BINDING`, no copy usages, and it works in a caller-owned pass, which reflections' bake uses.
- `unsupportedReason` names `rg32float` in compatibility mode. `requiredLimits( adapter )` also asks for 256 invocations and 256 along x where the adapter has them (compat adapters on Chrome report 1024), and the default `levelsPerDispatch` stays at 4 when a device allows fewer than 256 invocations.
- **`Reducer`** writes only the outputs it's given: in compatibility mode it rejects a source or outputs that share a texture (e.g. FSR-style mips of one texture), with that reason; separate textures or a store hook work. Tile and scalar kernels needing 256 invocations name the limit if the device lacks it.
- **Verified** on a real compatibility-mode device: headless Chrome 148 on M5 Pro, `featureLevel: 'compatibility'`, device without `'core-features-and-limits'`, with `requiredLimits` (`npm run test:gpu -- --compat`). The full matrix passes there (rg32float reported as skipped). Before the change (the direct path forced), the first case failed with "must not have different views in a single draw/dispatch command" (D4100).

### Scalar reductions

`ScalarReduction` (`src/webgpu/scalar.ts`, oracle in `src/core/scalar.ts`) reduces an image to one `vec4f` and a weight per layer in two dispatches, as §14.1 asks. Built 2026-11-11 (M2T.2).

- **Input:** a source texture read with `textureLoad` (any float-sampleable 2D format at one mip level; depth through its depth aspect; single-channel formats splatted to `(v, v, v, v)` as in §4 Separate source), or a `load` hook alone with an explicit `size`, reading the tool's `@group(1)`. The hook form lets a tool reduce something computed from several of its own textures. A hook and a source texture together are rejected; a hook replaces the source read.
- **Layers (decided):** each layer in `layers: [ first, count ]` is reduced separately, to its own result. Workgroup z selects the layer in both dispatches, so all layers still take two dispatches. Array slices and cube faces are independent images; a value across layers is a sum of results, or one hook that loops over layers.
- **Ops** (`reduction`, as on `Downsampler` and `Reducer`): per channel `'sum' | 'average' | 'min' | 'max' | 'weight' | 'weighted'` (§14.1), plus payload folds; one op for every channel, or `'minmax'` (even channels min, odd max, the depth-range shorthand). At most one `'weight'` channel; `'weighted'` needs one. Emitted: sums, averages (÷ texel count), weighted means (÷ Σ w, and 0 when Σ w = 0), Σ w, min, max. The weight passed on is Σ w with a weight channel, else the texel count. Custom `{ wgsl, kind }` reductions aren't offered yet.
- **Output:** `finish( value: vec4f, weight: f32 )`, run once per layer by invocation 0 of the combine workgroup, with an optional third parameter `layer: u32` (an extension of §14.1 for multi-layer runs). Without `finish`, results go to `output: { buffer, offset }` as `struct { value: vec4f, weight: f32 }`, 32 bytes per layer, at an offset aligned to the device's `minStorageBufferOffsetAlignment` (256 by default), as `partials.offset` is.
- **Every texel counted once.** Pass 1 covers the image with `⌈w/64⌉ × ⌈h/64⌉` tiles. Off-image reads in partial tiles are clamped to the edge, then zeroed for sum-like channels; min and max see a duplicate in-image texel, which is harmless. No edge row or column is dropped (SPD), and nothing is limited to a top-left `2ⁿ` square (FSR 3 exposure covers 1024×1024 of 1080p, §14).
- **Pass 1 (partials):** 16×16 invocations. Each reads 4×4 texels 16 apart, so adjacent invocations read adjacent texels, and combines them as a tree. The workgroup combines its 256 values in workgroup memory with a stride-halving tree (`slot[i] ∘= slot[i + s]`, s = 128 … 1) and writes one `vec4f` per tile and layer to a storage buffer the reduction owns.
- **Pass 2 (combine):** one 256-invocation workgroup per layer. Invocation t folds partials t, t + 256, … in sequence, then the same tree. It normalises and calls `finish` or writes the output.
- **Tile size (measured):** 4×4 texels per invocation (64×64 tile). Ratios to it, rgba16float `'average'`, Apple M5 Pro, Chrome, compute-pass timestamps, 32 repeats, medians of 11 rotated rounds, 2026-11-11. Other agents were running GPU suites at the same time, so only the ratios mean anything:

  | Texels per invocation (tile) | 960×540 | 1920×1080 | 3840×2160 |
  |---|---:|---:|---:|
  | 1×1 (16²) | 2.67 | 2.51 | 1.52 |
  | 2×2 (32²) | 1.22 | 1.02 | 1.03 |
  | **4×4 (64²)** | **1.00** | **1.00** | **1.00** |
  | 8×8 (128²) | 3.10 | 2.84 | 1.62 |

  Small tiles pay for more partials and barriers. At 8×8, 64 loads unrolled per invocation cost occupancy. `block` (1, 2, 4, 8) stays an option for re-measuring on other vendors.
- **Precision.** Each term passes through at most D roundings: log₂16 in the invocation, 8 in the workgroup, ⌈N/256⌉ − 1 in the sequential fold over N partials, and 8 in the final tree. That gives D = 20 up to 1024² (N ≤ 256), 21 at 1920×1080 and 27 at 3840×2160. The standard bound for a summation tree (Higham §4.2) is |ŝ − s| ≤ γ_D·Σ|xᵢ|, with γ_D = D·u/(1 − D·u) and u = 2⁻²⁴. At 4K that is 1.6·10⁻⁶ of Σ|x|, so an average is good to 1.6·10⁻⁶ of the mean magnitude. A serial f32 sum over 1080p has a bound of n·u ≈ 0.12 and in practice misses it (core test). Averages add one rounding for the division; weighted means add one for the product, plus the error of Σ w. `scalarReference` documents the bound per op and returns it.
- **Oracle (§7 style).** `scalarReference` combines in exactly the GPU's order with float32 rounding after every operation, and also returns the exact float64 result and the bound above. GPU tests require |GPU − exact| ≤ bound (plus a declared input error: 2⁻²¹ relative for unorm sources, and for the exposure hook's `log`, 2⁻²⁰ absolute plus 2⁻²¹ relative). Bit-for-bit pairing isn't required: FMA contraction of `w·v + …` and WGSL's 2.5-ulp division are allowed. On M5 Pro, 889 of 1044 float results matched the pairing bit for bit, and the worst error was 15 % of the bound. Integer-valued cases (procedural hooks returning counts, hash bits and coordinates below 2²⁴) and min/max must match exactly.
- **Recording:** `encode( encoder, run )` (two passes), `encodePass( pass, run )`, and `prepare( run )` → pipelines, our `@group(0)` bind group, the tool's group 1 and both workgroup counts, for tools that dispatch it themselves. `kernel( { dimension, splat } )` returns the WGSL, layout and pipelines; `createScalarKernel` and `scalarLayoutEntries` are the pure generator. Generator identifiers start with `ds_`.
- **Caching:** pipelines per source view dimension and splat. Per run shape, a plan holds a uniform buffer (size, first layer, partial count) and the partials buffer, cached per source texture, or else per output buffer or reduction, with at most 4 shapes per owner (resizes evict the oldest, and its buffers are destroyed once the queue is done, § Caching). `release( resource )` drops one owner's plans and destroys their buffers; `release()` drops every plan.
- **Combine-only runs (2026-11-11).** A run with `partials: { buffer, offset?, count }` records the combine dispatch alone, over `count` accumulations per layer that the tool wrote (one `vec4f` each, layer after layer). The usual source is a `Reducer`'s deepest level through its store hook, so "Reducer levels + a scalar of the same source" takes 2 dispatches with one read of the source (3 before).
  - **Partials are accumulations:** sum-like channels hold sums (a `'weighted'` channel Σ w·v, the `'weight'` channel Σ w), min/max channels the min/max. A Reducer emits these with `'sum'`, `'min'`, `'max'` channels; its `'average'` and `'weighted'` channels are normalised, so a tool feeding a scalar uses `'sum'` and forms w·v in its load hook.
  - **`texels`** is the texel count behind the partials: the divisor of `'average'` channels and the weight passed on without a `'weight'` channel. A run needs it in those cases.
  - **Group 0** of the combine-only pipeline has no source texture (`combineKernel()`). The tool's buffer is bound read-write at binding 1, so the run's group 1 must not bind it too (aliased writable bindings fail validation). In the FSR fixture the Reducer's group 1 holds the partials and the scalar's is FSR's own.
  - **Offset** must be a multiple of `minStorageBufferOffsetAlignment`. The WGSL is unchanged: the combine entry point of the same module.
  - **Oracle:** `scalarCombineReference( partials, count, layers, reduction, { texels, partialDepth, magnitude } )` pairs in the combine pass's order; the bound uses depth `partialDepth` + ⌈N/256⌉ − 1 + 8 (a Reducer's level l adds 2l roundings).
- **Shared tool plumbing (2026-11-11):** `src/webgpu/tool.ts` holds `ToolBindings`, hook validation (`checkHook`, `hookArity`), the tool's group-1 layout, the `[ group 0, group 1 ]` pipeline layout and the bind-group checks, for `Reducer` and `ScalarReduction` both. WGSL and layouts were unchanged by the move (5620 option sets compared byte for byte).

## 5. API

### Implemented

```ts
import { Downsampler, requiredLimits } from 'downsampler/webgpu';

const device = await adapter.requestDevice( { requiredLimits: requiredLimits( adapter ) } ); // also Downsampler.requiredLimits; covers Reducer and ScalarReduction
const downsampler = new Downsampler( device );          // { levelsPerDispatch?, output?: 'auto' | 'storage' | 'buffer' (§6.1), averageAccess?: 'sample' | 'load', compatibility?: 'auto' | boolean (§4), tail? (§2.1), f16Rounding?: 'nearest' | 'device' (below), exactTile?: 8 | 16 | 32, maxResampleRatio?: number (§4 Separate source) }
                                                         // fields hold the resolved values: downsampler.averageAccess === 'sample', .tail, .exactTile, …
downsampler.supports( texture );                         // boolean
downsampler.unsupportedReason( texture );                // string | null
downsampler.outputPath( texture );                       // 'storage' | 'buffer' | null (§6.1 Buffer output path)
downsampler.encode( encoder, texture, {
	baseMipLevel, mipLevelCount, oddSizes: 'exact' | 'box' | 'spd', // 'box': area-weighted odd sizes; 'spd': SPD semantics, ⌈levels/6⌉ dispatches (§2.1)
	reduction: 'average' | 'min' | 'max' | 'minmax' | { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { … }' },
	source,                                              // a texture or { texture, mipLevel, resample } (Sources below)
	layers: [ first, count ], rect: { x, y, width, height }, // partial updates (§4); an empty rect is a no-op
} );
downsampler.generate( texture );                         // encode + submit
downsampler.encodePass( pass, texture, options );        // into a compute pass you own (one timestamp pair; storage path only)
downsampler.prepare( texture );                          // what encode records: PreparedBatch[] (pipeline, bindGroup, groups, copies), for self-dispatch
downsampler.release( texture );                          // drop its plans, destroy its buffers and scratch textures
downsampler.scratchSize( texture, options );             // bytes a shared buffer-path scratch needs (0 on the storage path)
downsampler.encode( encoder, texture, { scratch } );     // buffer path: write through a caller's buffer (STORAGE | COPY_SRC), shared across textures

import { generateMipmaps, getDownsampler, requiredLimits, unsupportedReason } from 'downsampler/three';
generateMipmaps( renderer, texture, { reduction, encoder, downsampler: { averageAccess: 'load' } } ); // false if it declines; no submit with `encoder`
unsupportedReason( renderer, texture, options );          // why it declines: not allocated, one level, or the Downsampler's reason
getDownsampler( renderer, downsamplerOptions );          // shared per device and option set

import { planBatches, reduceReference, footprint, spdReference, maxViolations, boxReference, areaReference } from 'downsampler'; // backend-neutral core
```

```ts
// Separate source: depth → Hi-Z. Level 0 the same size (copy) or half (reduce).
downsampler.encode( encoder, hiZ /* rg32float */, { source: depthTexture, reduction: 'minmax' } ); // or { texture, mipLevel }
downsampler.sourceUnsupportedReason( destination, source, baseMipLevel ); // string | null
// Any destination size by area (§4 Separate source): a power-of-two Hi-Z, 2 dispatches for 12 levels.
downsampler.encode( encoder, hiZ /* 2048×1024 r32float */, { source: { texture: depth1080p, resample: 'area' }, reduction: 'min' } );
```

```ts
// Scalar reductions (§4 Scalar reductions): one vec4f + weight per layer, 2 dispatches.
import { ScalarReduction } from 'downsampler/webgpu';
const range = new ScalarReduction( device, { reduction: 'minmax' } ); // = [ 'min', 'max', 'min', 'max' ]; or 'average', [ 'weighted', …, 'weight' ]
range.encode( encoder, { source: depth, output: { buffer, offset: 0 } } ); // { value: vec4f, weight: f32 } per layer; offset aligned to minStorageBufferOffsetAlignment
const exposure = new ScalarReduction( device, {
	bindings: { wgsl: '@group(1) @binding(0) var colour: texture_2d<f32>; …', entries }, // the tool's group 1
	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f { … log( max( luma, 6.1e-5 ) ) … }',
	finish: 'fn finish( value: vec4f, weight: f32 ) { … temporal smoothing, store … }',
} );
exposure.encodePass( pass, { width, height, bindGroup } ); // also { layers: [ first, count ] }; without a load hook { source: texture | { texture, mipLevel } }
exposure.bindGroupLayouts[ 1 ];  // the hooks' group-1 layout, as on a Reducer (index 0: group 0 of a '2d' source)
exposure.prepare( run );        // what encodePass records: pipelines, group 0 and 1 bind groups, workgroup counts (dispatch it yourself)
exposure.release( resource );   // drop a texture's or buffer's plans, destroy their params/partials (release(): all)
exposure.kernel();              // { wgsl, layout, partials, combine }
exposure.unsupportedReason( run );

// Combine only: partials a tool wrote (e.g. a Reducer's level 6 via its store hook), one dispatch.
const mean = new ScalarReduction( device, { reduction: [ 'weighted', 'weighted', 'max', 'weight' ], finish, bindings } );
reducer.encodePass( pass, dispatch );                                         // stores Σ terms per 64² tile
mean.encodePass( pass, { partials: { buffer, count: tiles }, bindGroup } );   // also `texels` for 'average'
mean.combineKernel();           // { wgsl, layout, combine } for self-dispatch

import { scalarReference, scalarCombineReference } from 'downsampler'; // oracles: GPU pairing in f32, exact f64 result, error bound
```

```ts
// Tile reductions for tools (§14.1): FSR 3.1 shading-change pyramid shape.
import { Reducer } from 'downsampler/webgpu';

const shadingChange = new Reducer( device, {
	reduction: [ 'average', 'average', 'sum', 'max' ],   // per channel; 'weight'/'weighted' for weighted means
	levels: [ 1, 2, 3 ],                                  // emitted levels (1 = half); unemitted ones get no binding
	edges: 'clamp',                                       // SPD/FSR edges, ⌊n/2⌋ sizes; 'zero-weight' (default): ⌈n/2⌉, every texel once
	format: 'rg16float',                                  // output textures; or a `store` hook
	bindings: {                                           // the tool's @group(1)
		wgsl: '@group(1) @binding(0) var current: texture_2d<f32>; @group(1) @binding(1) var previous: texture_2d<f32>; …',
		entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, … ]
	},
	load: `fn load( texel: vec2i, layer: u32 ) -> vec4f {
		let diff = textureLoad( current, texel, 0 ).r * exposure - textureLoad( previous, texel, 0 ).r;
		return vec4f( diff, sign( diff ), 1.0, abs( diff ) );
	}`,
	// store: 'fn store( level: u32, texel: vec2i, layer: u32, value: vec4f ) { … }',
	// size: 'fn size() -> vec2u { … }',               // with a load hook; default: a uniform the Reducer manages
} );
shadingChange.encodePass( pass, {                        // or encode( encoder, … )
	width: renderWidth, height: renderHeight,            // the load hook's source size
	bindGroup: toolGroup1,                               // created from shadingChange.bindGroupLayouts[ 1 ]
	outputs: [ 0, 1, 2 ].map( mipLevel => ( { texture: spdMips, mipLevel } ) ),
} );
// Self-dispatch: shadingChange.prepare( dispatch ) → { pipeline, bindGroup, toolBindGroup, workgroups, buffer?, copies? }
// (what encodePass records), or .wgsl, .entryPoint, .layout, .pipeline, .bindGroup( dispatch ), .workgroups( w, h, layers ).
// shadingChange.release( texture? ): drop cached bind groups, destroy the size uniforms / output buffers it created.
// 8 channels (§14.1 Wide accumulators): reduction: [ 'sum', 'sum', 'sum', 'sum', 'max', … ] (or channels: 8);
// hooks then use DsWide { lo: vec4f, hi: vec4f }: fn load( … ) -> DsWide, fn store( …, value: DsWide ).
// Payload folds (§14.1): [ 'argmin', 'payload', 'payload', 'payload' ] = min depth + the normal of that texel;
// ties go to the smaller payload (put a texel index first for row-major ties). Also in ScalarReduction.
// Without a load hook: { source: texture | { texture, mipLevel } }. Output textures must be at least each level's size.
// Array layers: { layerCount } from layer 0 (a count: the tile kernel has no layer offset, and hooks see the workgroup's layer).

import { reducePyramid, levelSize } from 'downsampler';  // the CPU oracle and level sizes
```

**Sources: one shape (2026-11-13, ship prep).** Every class takes `source: GPUTexture | TextureLevel`, with `TextureLevel = { texture, mipLevel? }` (`src/webgpu/source.ts`): `Downsampler`'s `EncodeOptions.source` (a `SourceOptions`, which adds `resample`), the `Reducer` dispatch's `source` and `outputs`, and the `ScalarRun`'s `source` (it replaced `texture` + `mipLevel`). One module holds the checks, with one wording: dimension `2d`, `textureBindingViewDimension` (where defined) `2d` or `2d-array`, sample count 1, `TEXTURE_BINDING`, the sample kind wanted (float-sampleable by default, depth through its depth aspect; an integer `Downsampler` destination takes an integer source of its signedness, M2.3.7), `mipLevel` in range; and the source view (`depth-only` aspect for depth-stencil formats). Field names follow two rules: sizes are `width`/`height` (`ScalarRun.size: [ w, h ]` became `width`, `height`; with a source they default to its level's size and must match it), and a `layers` field is always `[ first, count ]` (`EncodeOptions.layers`, `ScalarRun.layers`, `PreparedBatch.layers`, `BufferCopy.layers`, which replaced `layer` + `layers`), while a bare count is `layerCount` (the `Reducer` dispatch, `ReducerCopy`). A first layer for the `Reducer` was declined: its tile kernel has no layer offset (a uniform and a kernel change), hook kernels pass the workgroup's layer to the hooks, and compatibility mode can't bind views of part of the layers. The keys used before 0.1 (`ScalarRun.texture`, `mipLevel`, `size`; the Reducer's `layers`) are rejected with a message naming the new ones, not silently ignored.

**`f16Rounding`** (Downsampler and Reducer): `'nearest'` (default) rounds float16 levels to nearest even in the shader, the same bits on every GPU (§4 Quantization). `'device'` uses `pack2x16float`, the device's conversion: it truncates toward zero on D3D12 (NVIDIA, Intel), so each level is up to one f16 step low and the bias carries down the chain. Use it only where the in-shader rounding measurably costs and the bias is acceptable; the default stays `'nearest'` (owner, 2026-11-13; re-measured on Windows in round 2). Guide: docs/guide.md, float16 rounding.

```ts
// Vendoring (§10): in-place 'average' chains in one self-contained file, dist/mipmap.js (no imports).
import { createMipmapKernel, planMipmapBatches, kernelDefaults, mipmapFormats, mipmapAccess } from 'downsampler/mipmap';

const defaults = kernelDefaults( device.limits, device.adapterInfo ); // { levelsPerDispatch, bufferLevelsPerDispatch, exactTile, tailMaxTile, tail }: Downsampler's defaults
const batches = planMipmapBatches( width, height, { mipLevelCount, ...defaults } ); // = planBatches in place under 'exact' (tail on by default)
createMipmapKernel( { format, dimension, count, access: mipmapAccess( format, device.features ), tail, tile, f16Rounding } ); // = createKernel( { ...options, f16Pack: false } )
```

The destination must have `TEXTURE_BINDING` usage, plus `STORAGE_BINDING` (storage path) or `COPY_DST` (buffer path, §6.1), and a format from `formats`: `rgba8unorm`, `rgba16float`, `rgba8snorm`, `r32float`, `rg32float`, `rgba32float`, `rgb9e5ufloat`, `rgba8unorm-srgb`, `bgra8unorm-srgb` (buffer path only), plus `bgra8unorm` (storage needs `bgra8unorm-storage`), `rg11b10ufloat` (storage needs `texture-formats-tier1`; else the buffer path) and `r8unorm`, `rg8unorm`, `r8snorm`, `rg8snorm`, `r16float`, `rg16float` and the 16-bit normalized formats (need `texture-formats-tier1`), `rgb10a2unorm` (storage needs tier1; else the buffer path), and the integer formats (§6.1 Integer formats): `rgba8uint`/`sint`, `rgba16uint`/`sint`, `r32uint`/`sint`, `rg32uint`/`sint`, `rgba32uint`/`sint` (core), `r8`, `rg8`, `r16`, `rg16` `uint`/`sint` and `rgb10a2uint` (tier1; the 4-byte ones take the buffer path without it). `unsupportedReason` names a missing feature or usage. Its dimension must be `2d`, with `textureBindingViewDimension` either `2d` or `2d-array`.

```ts
// GI formats (§6.1): rgb9e5ufloat is never a storage texture, so it takes the buffer path.
const irradiance = device.createTexture( { format: 'rgb9e5ufloat', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, … } );
downsampler.generate( irradiance );                      // 6 levels per dispatch at any storage texture limit
new Reducer( device, { levels: [ 1, 2, 3 ], format: 'rgb9e5ufloat', … } ); // output 'buffer' by default for it
reducer.encodePass( pass, dispatch ); pass.end(); reducer.encodeCopies( encoder, dispatch ); // buffer outputs
```

```ts
// Partial updates (§4 Partial updates): rewrite only what depends on a changed region.
downsampler.encode( encoder, texture, { rect: { x, y, width, height } } );   // dirty texels of level baseMipLevel
downsampler.encode( encoder, cubeAs2dArray, { layers: [ 2, 1 ] } );          // only layer 2; others untouched
downsampler.encode( encoder, hiZ, { source: { texture: depth }, rect, reduction: 'minmax' } ); // rect in source texels
downsampler.prepare( texture, { rect } );   // batches carry regions (per level), groupX/groupY and layers

import { planRegions, exactDependents, clampDependents } from 'downsampler'; // the region math (core)
```

**Scratch (M2.3.5, 2026-11-13).** sRGB destinations take the buffer output path (§6.1 sRGB), so the planned `scratch: GPUTexture` became `scratch: GPUBuffer`: by default each texture's plan keeps its own buffer, sized for its largest batch (about a third of the base level), until `release( texture )`. Many textures (three.js) can share one: commands run in order, and each batch's copies are recorded before the next dispatch writes the buffer, also across chains in one encoder (GPU-tested: two chains through one buffer in one encoder, bit-identical). `scratchSize( texture, options )` is every written level in the buffer layout, which holds any batch; a smaller buffer is rejected with the size the chain needs, a buffer without STORAGE | COPY_SRC with the usage. Ignored on the storage path (so one option set can serve every texture); never destroyed by the library. Plans are cached per scratch buffer, as per source texture.

### Package (0.1.0, 2026-11-13)

- **Entry points** (`exports`, each with its types): `downsampler` (core), `downsampler/webgpu`, `downsampler/three`, `downsampler/mipmap` (the self-contained vendoring file, §10), plus `./package.json`. ESM only (`"type": "module"`), `sideEffects: false`. `typesVersions` maps the subpaths for TypeScript's old `node10` resolution; `bundler` and `node16` resolve through `exports`.
- **Contents** (`files`): `dist` without source maps (they point at `src/`, which isn't shipped), `README.md`, `LICENSE`, `CHANGELOG.md`. SPEC.md doesn't ship: it is the design record, with process notes and measurements, and the README links to it in the repository. `prepack` cleans and rebuilds `dist`, so no stale module ships.
- **Types:** the declarations use the WebGPU types of `@webgpu/types` without referencing them (a consumer without them would get an unresolvable reference); the README asks TypeScript users to add it, as any WebGPU project does. Checked from a consumer project against the packed tarball with `moduleResolution` `bundler` and `node16`.
- **No runtime dependencies;** `three` is an optional peer (`>=0.170.0`), typed structurally. No `engines`: the package runs in browsers, and the core runs in any ES2022 runtime.
- **Release:** `.github/workflows/release.yml`, manual (`workflow_dispatch`, a version guard and a dry-run default), typecheck + build + core tests, then `npm publish --provenance`. The full GPU matrix runs on real GPUs before it (CI has none).

### Planned

**Transform hooks around the average (deferred, 2026-11-13, M2.3.5 agent).** A WGSL `forward`/`inverse` pair applied around `'average'` (premultiplied alpha, log-luminance): every average of a chain would become `inverse(mean(forward(x)))`, quantized as stored. The sRGB path has that structure with the hardware decode as `forward`, so it is the template. Not built, because it doesn't fit the kernels cleanly yet:
- The sampler filters before any hook, so a transform forces `'load'` at every first output.
- It touches every averaging site of the exact kernel (bilinear first output, in-tile 2×2, tail steps, box weights), and of the `'clamp'` kernel under `'spd'`, which doesn't round between levels, so it would apply once around a whole batch: different semantics, to be defined.
- The oracle needs a JS twin of each WGSL pair, and in-tile levels lose bit-exact checks (the hook's f32 math against f64), unless hooks come with declared error bounds.
- The two motivating uses have homes already: premultiplied alpha is an alpha-weighted mean, which `Reducer` computes with a `'weight'` channel (`'weighted'` colour); log-luminance is a `Reducer` or `ScalarReduction` `load` hook.
A minimal version, if a tool needs it: `transform: { wgsl }` (`fn forward(v: vec4f) -> vec4f`, `fn inverse(v: vec4f) -> vec4f`), `'average'` under `'exact'` and `'box'` only (rejected under `'spd'` and for folds), loads only, its own oracle option (`ReferenceOptions.transform`) and mutation checks.

## 6. SPD feature parity

| SPD feature | Status | Plan / notes |
|---|---|---|
| Up to 12 levels from ≤4096² | ✅ | any size, by chaining (2 dispatches up to 4096²) |
| Single dispatch | ❌ by design | §2.2; a measurement-only atomic probe (`test/bench/atomic-spd.js`) lost to two dispatches at render sizes and 1024², tied at 4096² |
| Custom reduction (`SpdReduce4`) | ✅ | average/min/max/minmax + custom WGSL `reduce(a, b)` fold (§4 Reductions). A weighted custom reduction (SPD's `SpdReduce4` used for averages) isn't offered under `'exact'`, where it has no defined odd-size meaning; tile kernels (`Reducer`, §14.1) add sums, weighted means and custom sums under `'zero-weight'` and `'clamp'`. sRGB is built in (§6.1); other transforms around an average (premultiply, log) are deferred (§5 Planned). |
| Load vs. linear-sampler source | ✅ | `textureLoad` path (§4 Source access). `textureGather` for single-channel Hi-Z is an M3 candidate: one gather replaces four loads. |
| Separate source/destination | ✅ | `source: { texture, mipLevel }`, same size (copy) or half (§4 Separate source). Depth sources supported. |
| Texture arrays | ✅ all layers, or a range | `layers: [ first, count ]` (M2.4); cube maps as 6 layers (2d-array view; not available in compatibility mode) |
| Sub-rectangle (`rectInfo`) | ✅ tighter | `rect` (M2.4, §4 Partial updates): each level rewrites exactly the texels that depend on the rect, bit-identical to a full recompute, under `'exact'` and `'spd'`. SPD snaps to 64-texel tiles and recomputes levels below 64×64 whole (DIVERGES #15) |
| Custom load/store (format conversion) | the formats of §6.1, sRGB included; any float/depth source | see §6.1; `Reducer` and `ScalarReduction` take WGSL load/store hooks; `Downsampler` transform hooks deferred (§5 Planned) |
| Global atomic counter, `globallycoherent` mip 6 | ❌ by design | No counter buffer to allocate, reset or alias; the dispatch boundary replaces it (§2.2) |
| Mip count (`mips`, max 12) | ✅ no cap | **Off by one for SPD users:** in place, `mipLevelCount` includes the source level, so SPD `mips = N` is `mipLevelCount: N + 1` |
| Levels written | ✅ | In place, as SPD: `baseMipLevel` is read and later levels written. With `source`, destination `baseMipLevel` is written at the source size (copy) or half; SPD always halves |
| sRGB (sample sRGB view, write UNORM UAV) | ✅ differently | WebGPU forbids sRGB storage even through a view (and a UNORM view needs `viewFormats`), so we read the sRGB view, encode in the shader and write through the buffer path (§6.1 sRGB, DIVERGES #16). Loads snap to the exact decode; SPD trusts the hardware's |
| Intermediate precision | differs by design | SPD keeps unrounded values in LDS between levels; we round each level as stored, so a chain equals level-by-level generation and 8-bit results won't match SPD bit for bit (§4 Quantization) |
| LDS layout (one float4 array vs per channel) | ❌ | M3 candidate; measure first |
| Packed fp16 math | ❌ | optional `shader-f16`; keep only if measurably faster |
| Wave/subgroup ops | ❌ | optional `subgroups` (Chrome, Safari 27.3; workgroup-memory fallback for Firefox). WGSL defines no quad↔`local_invocation_index` mapping in compute, so derive positions from `subgroup_id`/`subgroup_invocation_id` and use `subgroupShuffleXor`. Upper-bound test first (§13 Subgroups); current kernels are near bandwidth-bound at ≥2048². |
| Odd sizes | ✅ `'exact'`, `'spd'`, `'box'` | SPD drops the last row/column; `'exact'` (default) matches render passes. `oddSizes: 'spd'` reproduces SPD's semantics in ⌈levels/6⌉ dispatches (M2T.4, §2.1). `'box'`: nvpro's area weights (M2.5) |

Beyond SPD: sizes over 4096², render-pass-matching odd sizes, conservative min/max at odd edges (SPD drops the last row/column, and unclamped loads read zeros past the edge), and automatic fallback to the 4-storage-texture limit.

### 6.1 Formats

**Storage support, probed** (Chrome 148, Apple M5 Pro, 2026-11-10; storage + sampling usage, with and without the optional features):

| Group | Formats | Storage | Work needed |
|---|---|---|---|
| Done | rgba8unorm, rgba16float, r32float, rg32float, rgba32float | core | ✅ |
| Same quantization, fewer channels | r8unorm, rg8unorm, r16float, rg16float | `texture-formats-tier1` | ✅ (no WGSL `enable` needed on Chrome 148) |
| Canvas format | bgra8unorm | `bgra8unorm-storage` | ✅ hardware swizzles on read and write, so kernels and minmax channel parity are unaffected |
| snorm8 | rgba8snorm (core), r8snorm, rg8snorm (tier1) | mixed | ✅ `snorm8`: integer units ×127, clamped to [−127, 127] (−128 and −127 both decode to −1); tie parity in `i32` (negative float → `u32` is undefined) |
| 16-bit norm | r16unorm, rg16unorm, rgba16unorm, and the snorm versions | tier1 | ✅ `unorm16`/`snorm16`: integer units ×65535 / ×32767; four-texel sums stay exact in f32 (< 2²⁴). They sample as `unfilterable-float` on Dawn/Apple, so they always load |
| Packed | rgb10a2unorm | tier1 for storage; the buffer path without it | ✅ (M2.3.3) `unorm10a2`: integer units per channel, scale `(1023, 1023, 1023, 3)` (`channelDomain`; a vector `SCALE` in the kernel), sums exact, ties to even, as the other normalized formats. Filterable, so averages sample by default. The buffer path packs `r | g << 10 | b << 20 | a << 30` |
| Packed float | rg11b10ufloat | tier1 for storage; the buffer path without it | ✅ (M2.3.6) `rg11b10ufloat`: unsigned floats, 5-bit exponent, 6/6/5-bit mantissas; round to nearest even, negatives and −Inf → 0, finite overflow → the largest finite value (65024 / 64512), +Inf and NaN kept. No WGSL pack builtin, so `ds_ufloat` encodes in the shader and `toUfloat` in core. Storage path with tier1, buffer path without it (or without STORAGE_BINDING usage); both quantize with our encoder, so they write identical bits |
| Integer | rgba8, rgba16, r32, rg32, rgba32 uint/sint (core; rg32 not in compatibility mode); r8, rg8, r16, rg16 uint/sint and rgb10a2uint (tier1) | mixed (probed: `results/probes/int-format-caps.mjs`) | ✅ (M2.3.7) a separate kernel (`createIntKernel`, `vec4u`), exact integer arithmetic; below |
| sRGB | rgba8unorm-srgb, bgra8unorm-srgb | **never** (even with a non-sRGB view format) | ✅ (M2.3.5) read the sRGB view (linear), reduce in linear, encode in the shader with correct rounding, write through the **buffer output path**; quantize-as-stored is `decode(encode(v))` with the exact table (§4 Quantization, "sRGB" below) |
| Shared exponent | rgb9e5ufloat (32-bit HDR, common for GI and irradiance) | **never** storage or renderable, so render-pass mipmaps can't produce it either | ✅ (M2.3.4) **buffer output path** (below): pack in the shader, `copyBufferToTexture` into the level. Filterable, so later batches sample it as usual |
| Out of scope | BC/ETC2/ASTC (generate before compression); depth as destination (use r32float, §4 Separate source) | — | — |

**sRGB** (built 2026-11-13, M2.3.5; `src/core/srgb.ts`, `srgbWgsl` in `src/webgpu/kernel.ts`, DIVERGES #16).
- **Problem.** WebGPU has no sRGB storage textures, not even through a view (a texture with STORAGE_BINDING can't have an sRGB format, and WGSL has no sRGB texel format; probed, `results/probes/srgb-probe.mjs`). SPD samples the sRGB view and writes a UNORM alias; in WebGPU an alias view needs `viewFormats` at creation, which three.js doesn't set, and wouldn't help: it is the storage of the sRGB texture that's forbidden.
- **What we do.** Each batch reads its source level through the sRGB view: the hardware decodes to linear, and the sampler filters in linear (probe: halfway between codes 0 and 255 samples 0.5001, not the encoded-space 0.214). Levels are reduced in linear f32, alpha in unorm8 units, and each is encoded to codes in the shader (correctly rounded, §4 Quantization) and written through the buffer path below: packed words (`ds_srgb_code` per colour channel, rgba or bgra byte order), then `copyBufferToTexture`, which may write sRGB formats (probed: bytes kept). Loads snap the hardware's decode to the exact table (§4 Source access).
- **Path chosen by measurement and simplicity.** The alternative is the same kernel writing storage textures of the linear alias format (rgba8unorm, or bgra8unorm, which needs `bgra8unorm-storage`), then `copyTextureToTexture` into each sRGB level, which WebGPU allows between formats that differ only in sRGB-ness (probed). Both write the same bits (GPU-tested). `npm run bench -- --compare=srgb` (`test/bench/srgb.js`) times both with the library's plan, rgba8unorm-srgb, against `buffer` (paired per-round ratios, medians of 11 rounds × 16 repeats; Apple M5 Pro, Chrome, 2026-11-13; `results/srgb/m5-pro-compare-{1,2,3}.json`). Runs 1 and 2 ran beside other agents' suites (load average 400–660; run 1's times were inflated 7–30×), run 3 on a quieter machine (load average 14, still not idle). Only same-run ratios mean anything:

  | Size | `scratch` run 1 | run 2 | run 3 | `bufferLoad` run 3 | render passes run 3 | `buffer` ms run 3 [dispatches] |
  |---|---:|---:|---:|---:|---:|---:|
  | 960×540 | 1.20 | 1.11 | 1.04 | 1.66 | 1.25 | 0.089 [4] |
  | 1129×635 | 0.81 | 0.90 | 1.08 | 1.95 | 1.19 | 0.110 [5] |
  | 1280×720 | 1.22 | 0.98 | 0.96 | 1.65 | 1.25 | 0.102 [3] |
  | 1707×960 | 0.71 | 0.97 | 1.07 | 2.07 | 1.01 | 0.124 [4] |
  | 1920×1080 | 0.85 | 1.01 | 0.99 | 1.80 | 0.79 | 0.156 [4] |
  | 2560×1440 | 0.98 | 0.99 | 0.98 | 1.78 | 0.55 | 0.248 [3] |
  | 1024² | 1.11 | 1.04 | 1.01 | 1.86 | 1.98 | 0.067 [2] |
  | 4096² | 0.92 | 1.07 | 0.99 | 2.11 | 0.65 | 0.824 [2] |

  - **Scratch texture vs buffer: a tie.** Medians 0.95, 1.00 and 1.00; run 3 within 0.96–1.08. Re-measure on an idle machine and on other vendors before revisiting.
  - **Render passes** (three.js's path for these textures today) take 1.19–1.98× our time up to 1280×720 and at 1024², tie at 1707×960, and are faster from 1080p up: 0.79× at 1920×1080, 0.55–0.65× at 2560×1440 and 4096² (run 2: 0.55–0.59× at the two largest). There the buffer path's extra write and copy of every level dominates (§6.1 Buffer output path, Cost), and in rgba8 render passes are already close to `'exact'` at those sizes (§8). So for large sRGB textures render passes are the faster path on this GPU; see DIVERGES #17. To re-measure idle and on other vendors.
  - **`averageAccess: 'load'`** costs 1.7–2.1×: four loads per output texel, each snapped through the encoder and the table.
  The buffer path won on simplicity: it existed, composes with compatibility mode (one view per dispatch; a multi-level scratch texture would need per-level textures there), `rect`, `layers` and separate sources, writes 6 levels per dispatch at the default storage texture limit, and needs no `bgra8unorm-storage`. The scratch variant stays a kernel option (`output: 'storage'` on an sRGB format writes `srgbScratchFormat( format )`) for tools that dispatch it themselves.
- **Folds** (`'min'`, `'max'`, `'minmax'`, custom) fold the decoded values and store their codes. Decoding is strictly monotonic and `ds_srgb_code` inverts the table exactly, so they select stored codes: the same as folding the encoded bytes, with no extra decode (the view decodes anyway). Custom folds see linear colour and normalized alpha.
- **Composes** with every odd-size policy, the tail kernel, `rect` and `layers`, separate sources (into or out of sRGB; HDR clamps to [0, 1]), compatibility mode (the buffer path skips it; compatibility mode has no bgra8unorm-srgb textures at all, so the suite reports that format skipped there) and a shared `scratch` buffer (§5). Rejected: `encodePass` (the buffer path's copies; use `encode`), and a texture without COPY_DST (`unsupportedReason` names it). `Reducer` takes sRGB sources and outputs (its tile kernel's buffer output encodes; GPU-tested).
- **Cost:** that of the buffer path, plus the encoder: one `pow` and two table reads per colour channel per stored texel, and on `'load'` a re-encode and table read per loaded texel. `averageAccess: 'load'` measured 1.7–2.1× the default sample path (table above). The tables are read from a 2 KiB storage buffer (binding 12, `srgbTable( device )`), not declared in the shader: as `const` (or `var<private>`) arrays indexed at run time they sit in per-invocation memory, which on an RTX 3060 (D3D12) made the chain 3.8–4× slower and each pipeline 0.5–5 s to compile (2026-11-14, `results/windows/round2/srgb.md`). There, render passes are 2.6–3.3× faster than our sRGB path even after the fix.

**Integer formats** (M2.3.7, 2026-11-13). r32uint, rgba8sint, rgb10a2uint and the rest are read as `texture_2d<u32>` / `<i32>` (never filtered, so every batch loads) and reduced by their own kernel generator (`src/webgpu/integer.ts`), the integer form of the `'exact'` kernel: same batches, tiles, tail kernel, area-weighted steps, partial updates and output paths. Oracle: `intReference` (core, `src/core/integer.ts`), exact; the GPU must match every level bit for bit.
- **Working values** are `vec4u`. Signed formats carry an offset of 2³¹ (`v ^ 0x80000000`) from load to store: it maps i32 order onto u32 order, so min and max are unchanged, and it is even, so means and their tie parity are unchanged too.
- **Folds** (`'min'`, `'max'`, `'minmax'`, custom) are as for every format: the `2x … 2x+2` footprint, the area span for a resampled source. A custom fold is `fn reduce(a: vec4u, b: vec4u) -> vec4u` for uint formats and `vec4i` for sint formats (decided 2026-11-13: the format's own integers, not normalized values), e.g. a bitwise or for occupancy pyramids; it must be a semilattice as usual.
- **`'average'` is the exact weighted mean, rounded to nearest with ties to even** (decided 2026-11-13; owner can overrule). Under `'exact'` the weights are the render-pass bilinear weights at the texel centre, as exact rationals: per axis the position `((2x+1)·S − n) / 2n` gives a base texel and a remainder r, and the two taps weigh `(2n − r)/2n` and `r/2n` (an even step: the 2×2 mean; a same size: the texel). Under `'box'` and for a resampled source, each texel weighs its overlap / S (§2.1). Nothing is rounded but the result; nothing overflows at any width, 32-bit included.
  - **How, in u32:** each value splits into 16-bit halves. Per row of taps, Σ w·half stays below 2³¹ (weights up to 2¹⁵, any texture up to 32768 texels per axis) and is kept as quotient and remainder by the row's denominator; the column step and a 16-step long division (2 bits at a time) give the quotient and remainder of the full sum by dx·dy ≤ 2³⁰, and the remainder decides the rounding (`ds_mean`, `intMeanWgsl`). The in-tile 2×2 step is `Σ ⌊v/4⌋ + ⌊Σ (v mod 4) / 4⌋` plus the rounding of the last remainder (`ds_mean4`). A GPU case checks both against core's `roundedMean` on 170 000 values across the u32 range.
  - Why not truncation (webgpu-spd's `(a + b + c + d) / 4`): it loses 3/8 of a unit per level on average and its 32-bit sums wrap (DIVERGES #24). Why not f32: it can't hold 32-bit integers, and a mean that is exact at 8 and 16 bits only would give the formats different rules.
- **`'box'`** differs from `'exact'` only at odd sizes; at powers of two they are identical (GPU-tested), as are `tail: true` and `tail: false` (integer means don't depend on who computes the weights).
- **Separate sources:** an integer destination reads an integer source of the same signedness (uint into uint, sint into sint; others are rejected with the reason), copy, half or resampled; values are clamped to the destination's range (rgba16uint → rgba8uint saturates). Float destinations still reject integer sources.
- **Rejected:** `oddSizes: 'spd'` (its batches carry unrounded f32 sums between levels, which can't hold 32-bit integers; the error points to `'exact'` and `'box'`), `Reducer` integer outputs and sources, `ScalarReduction` integer sources (tile and scalar kernels accumulate floats; a load hook can read integers itself). Not built: integer tile reductions; cheap if a tool needs them (sums would need 64-bit accumulators).
- **Buffer path:** 4-byte-multiple texels (all but r8, rg8, r16): stored bits packed per word (rgba8: bytes, rg16/rgba16: halves, 32-bit: one word per channel, rgb10a2uint: 10/10/10/2). It is the path without tier1 for rg16uint/sint and rgb10a2uint, and for rg32uint/sint in compatibility mode.

**Buffer output path** (built 2026-11-11, M2.3.4). WebGPU allows `copyBufferToTexture` into formats that are neither storage nor renderable, so a kernel can write texels to a storage buffer and copy them into each level.
- **Layout** (`outputBufferLayout` in core): each level of a batch, and each layer of a level, at its own offset of one buffer, with rows padded to 256 bytes and level offsets on 256 bytes. The kernel finds it in a 112-byte uniform (binding 9: first-output size, then per level its offset and row stride in u32 words and its rows per layer) and writes packed words to binding 10 with `ds_put( level, texel, layer, value )` (`bufferOutputWgsl`). That is a single binding, so this path isn't bound by `maxStorageTexturesPerShaderStage`: 6 levels per dispatch even at the default limit of 4 (tested on a default-limit device).
- **Recording:** after each batch's pass, `encode` records one `copyBufferToTexture` per level (all layers in one copy). The next batch reads the destination texture as usual. One scratch buffer, sized for the largest batch, serves every batch of a plan: commands in an encoder are ordered, so each batch's copies finish before the next dispatch writes the buffer. `release( texture )` destroys it. `encodePass` throws on this path: a compute pass can't record copies, and later batches need them.
- **Which path (`DownsamplerOptions.output`):** `'auto'` (default) writes storage textures whenever the device can write the format there (the texture then needs STORAGE_BINDING; decided 2026-11-12), and takes the buffer path only for formats it can't: rgb9e5ufloat, rg11b10ufloat, rgb10a2unorm and the 4-byte tier1 formats (rg16float, rg16uint, …) without tier1, rg32float/uint/sint in compatibility mode. `'storage'` or `'buffer'` forces one; `outputPath( texture )` says which applies. On the buffer path `levelsPerDispatch` defaults to 6 at any storage texture limit (4 where workgroups are limited to 128 invocations, as in compatibility mode by default).
- **Compatibility mode (§4):** the buffer path needs none of its scratch textures or copy dispatches: a batch binds one level of the texture (its source) and writes a buffer, and copies, which aren't dispatches, write the levels. So on the buffer path the compatibility path is skipped (`usesCompatibility` still reports the device), and it is how rg32float gets mips in compatibility mode. The `Reducer`'s compatibility check (one view of a texture per dispatch) ignores buffer outputs, which aren't bound. Formats: any whose texels are whole 32-bit words (not the 1- and 2-byte r8/rg8/r16 formats). Normalized formats pack their integer units, bgra8unorm in its byte order. `unsupportedReason` names both paths' requirements.
- **Reducer:** the same buffer, uniform and `ds_put` in tile kernels (`ReducerOptions.output`; default buffer when the format can't be a storage texture on the device). Emitted levels are packed once (WGSL `pack4x8unorm`, `pack2x16float`, … or our encoders) and copied into the output textures, which need COPY_DST. `encode` records the copies; after `encodePass` or a self-dispatch, call `encodeCopies( encoder, dispatch )` once the pass has ended. 8-channel accumulators (`DsWide`) need a store hook, so they never meet output textures or the buffer output (rejected with an error); payload folds store their `vec4f` like any reduction.
- **Limits:** the buffer of the largest batch must fit `maxStorageBufferBindingSize` (128 MiB by default), else `prepare` throws and names the limit. 4096² rgba32float fits (85 MiB); larger first batches need raised limits (or a split first batch, not built).
- **Cost:** one extra write and read of the written levels (the levels below 0, about a third of the base texture's bytes), plus the copies. In bytes that predicts about 1.5× the storage path where bandwidth-bound. Measured (`npm run bench`, `buffer6` against `compute6`, same batches; Apple M5 Pro, Chrome 148, 11 rotated rounds × 16 repeats, 2026-11-11; two runs, load average 65–94 then ~14 from other agents' suites, so only same-run ratios count; `results/m5-pro-gi-buffer-path-loaded.json`, `results/m5-pro-gi-buffer-path.json`):

  | buffer ÷ storage | 256² | 1024² | 2048² | 4096² | 1920×1080 | 512²×6 |
  |---|---:|---:|---:|---:|---:|---:|
  | rg11b10ufloat | 2.3 / 4.1 | 2.0 / 3.5 | 1.7 / 2.4 | 1.9 / 1.9 | 2.3 / 2.1 | 1.6 / 2.1 |
  | rgba16float | 2.1 / 6.4 | 2.2 / 1.7 | 1.4 / 1.6 | 1.7 / 1.6 | 2.8 / 1.8 | 2.6 / 2.3 |

  - Small textures pay a fixed ~0.03–0.06 ms, about 4 µs per `copyBufferToTexture` (one per level; Metal blits). Large ones pay 1.6–1.9×, a little above the byte estimate.
  - So the storage path stays the default wherever it exists. The buffer path is for formats and devices without it, and against render-pass mipmaps (impossible for rgb9e5 anyway) it wins only where dispatch count dominates: 1.1–4.8× faster at 256²–2048², 10–13× at 512²×6, a tie at 1920×1080 for rgba16float (1.03–1.04×) and 4096² (0.94–1.01×), and slower for rg11b10ufloat at 1920×1080 (0.66–0.74×) and 4096² (0.73–0.84×). So `'auto'` doesn't send storable formats to the buffer path (a texture without STORAGE_BINDING is rejected; callers with a render fallback, like the three adapter, keep it); `output: 'buffer'` opts in.
  - rg11b10ufloat's own storage path is ALU-heavy: the integer encoder runs twice per texel and level (store, then quantize for the next level), and at 1024² it measured 1.2–3.4× rgba16float's storage time (noisy under load). A cheaper encoder (e.g. via `pack2x16float` and a mantissa round) is an M3 candidate; measure on an idle machine first.
  - Not done: copy fewer, larger regions (a level's layers already share one copy), or skip the buffer for the last batch's tiny levels.
- **Uses:**
  - rgb9e5ufloat, the only GPU route to its mips in WebGPU.
  - rg11b10ufloat without tier1 (SwiftShader has none).
  - Six levels per dispatch at the default storage texture limit, for any 32-bit-word format.
  - sRGB destinations (§6.1 sRGB, M2.3.5): `copyBufferToTexture` into rgba8unorm-srgb and bgra8unorm-srgb, chosen over a scratch texture.
  - A fallback for bgra8unorm without `bgra8unorm-storage` (works; untested without the feature).
- **rgb9e5 packing** (`packRgb9e5` in core, `ds_pack9e5` in WGSL), per EXT_texture_shared_exponent (Vulkan's E5B9G9R9):
  - Clamp each channel to [0, 65408] (511/512 · 2¹⁶); NaN and negatives → 0, +Inf → 65408.
  - Shared exponent `e = max(−16, ⌊log2 max(r,g,b)⌋) + 16`, taking ⌊log2⌋ from the float's exponent bits, not `log2()`, which isn't exact near powers of two. Scales are powers of two built from bits too (`exp2` isn't exact either).
  - Mantissas `⌊c / 2^(e−24) + 0.5⌋`: **round half up**, as the extension specifies, not to even (WGSL `round()` would be). Bump `e` when the largest mantissa rounds to 512.
  - Quantize-as-stored is pack → unpack (`quantizeTexel`), per texel: the shared exponent can flush a small channel next to a large one (1000 and 0.01 store as 1000 and 0), so min/max of rgb9e5 is not conservative per channel.
  - Probe kernel: `results/probes/rgb9e5-buffer-copy.mjs` (its `log2`/`round` packing predates this and is not the shipped one).
- **Mutation checks** (M5 Pro, `--only=packed --quick`, 2026-11-11): 12 of 14 caught. Caught: no 512 bump, rgb9e5 rounding half to even (`round()`), rg11b10 ties rounding up, rg11b10 truncation, rg11b10 overflow to Inf and a wrong subnormal decode (both by the codec check only: fixtures never overflow), no intermediate quantization, unpadded rows and unaligned level offsets (validation errors), a kernel row stride that ignores the padding, layers sharing one region, and copying only a batch's first level. Survived, as equivalent on this GPU: **⌊log2⌋ from `log2()`** (an exponent off by one is absorbed by the bump: one too low rounds the largest mantissa to exactly 512 and bumps; one too high needs a `log2` error above ~2⁻⁹ to change a mantissa) and **NaN not cleared** before `clamp` (Metal's `clamp` already returns 0 for NaN; WGSL leaves it undefined, so the guard stays). The bit-based exponent stays: it costs nothing and doesn't depend on `log2` accuracy.
- **Finding (M5 Pro, Chrome 148):** `textureStore` to rgba16float truncates f32 → f16, where `pack2x16float` rounds to nearest even (`results/probes/f16-store-rounding.mjs`). `Downsampler` is unaffected: it quantizes with `pack2x16float` before every store. `Reducer` float16 texture outputs are now quantized with `pack2x16float` before `textureStore` too (2026-11-12), so both its paths round to nearest even; a GPU case with values ¼, ½ and ¾ of a step above a half catches truncation. Packed floats are quantized by our encoder on both paths.
- **Finding (RTX 3060 Laptop and Intel UHD, D3D12, Chrome 148, 2026-11-12):** `pack2x16float` truncates too, as does `textureStore` (`results/windows/f16-store-rounding-*.json`), so quantizing with it gave truncated halves on Windows (about 211 of 287 full-matrix failures on the RTX). Owner decision: float16 outputs round to nearest even in the shader by default (`f16Rounding: 'nearest'`, integer math on the float's bits); `'device'` keeps `pack2x16float` for comparison. A GPU case checks `ds_half_bits` against `toHalf` at every half's rounding boundaries.

Optimisation: with `float32-filterable`, float32 formats may take the sample path; measure first (nvpro measured the sampler 26–32 % faster on NVIDIA, §13).

**What every new format needs:**
1. A row in `formats` (quantization family, channels, per-channel scale, required feature). `unsupportedReason` then names a missing feature.
2. Quantization in core, with exhaustive unit tests for 8/10/11/16-bit encodings, plus the matching WGSL `quantize`. Integer families keep the integer working domain, so sums stay exact.
3. GPU suite entries:
   - Fixtures and readback decode, including CPU pack/unpack for rgb10a2 and rg11b10.
   - Gated on the feature, and reported as skipped, not passed, when it's missing. SwiftShader may lack tier1.
   - Plus the three mutation checks.
4. Docs and the format table.

**Suggested order:**
1. fewer-channel unorm8/float16 + bgra8unorm (table only)
2. snorm8, unorm16, snorm16
3. rgb10a2unorm ✅
4. buffer output path + rgb9e5ufloat ✅
5. sRGB ✅ (M2.3.5: buffer output, chosen over a scratch texture by measurement and simplicity); transform hooks deferred (§5 Planned)
6. rg11b10ufloat ✅
7. integers ✅

## 7. Correctness specification

**Oracle.** `reduceReference` is a normalized-coordinate bilinear sample at each output texel centre with clamped edges, quantized to the storage format. Each GPU level is checked against the reference computed from the GPU's own previous level, so permitted rounding does not accumulate.

**Render passes are not the oracle.** "Matches render-pass mipmaps" (§2.1) is a looser, separate claim: each level within 1 step of a render pass's level from the same source level, and independent chains within 2 steps on Apple, where Metal render passes round exact ties with no fixed rule (three.js's parity test). Ties round to even here, against the CPU oracle, bit for bit where the tolerance below says exact.

**Tolerance.** Declared test tolerances for these fixtures, not a formal error bound:
- unorm8: 2.01 steps.
- float16: `max(0.002, 0.3 % of the reference value)`.
- float32: `max(1e-6, 1e-5 relative)`.
- Plus a filter allowance where bilinear weights aren't exactly representable with 8 fractional bits.
- **Normalized formats, levels reduced in workgroup memory:** exact. Those levels use integer math, so this is what tests round-half-to-even. The oracle computes averages in integer units for the same reason: readback values like 8/255 as f32 average to a hair off an exact tie and round the wrong way. That flaw was in the oracle from M1 until 2026-11-10, hidden by the 2-step tolerance. Only a batch's first output (sampler, or float weights) gets the tolerance.
- Folds: exact. The one exception is a level read straight from a unorm source, which allows 2⁻²¹ relative, because unorm → float conversion needn't be correctly rounded (SwiftShader's depth16unorm differs from k/65535 by an ulp).

**Required matrix.** Every change must pass all of it.

- **Shapes:** 1×1, 1D tails (1×128, 128×1, 128×8), odd and prime sizes, partial workgroups, rectangles, NPOT (192×108, 255×257, 1920×1080), layers (×3), both 2D view variants, and multi-batch sizes (2048², 96×4096).
- **Kernel widths:** 1, 4, 5 and 6 levels per dispatch, filtered by the device limit.
- **Formats:** all supported destination formats. One 4-channel format per quantization family (rgba8unorm, rgba8snorm, rgba16unorm, rgba16snorm, rgba16float, rgba32float, rgb10a2unorm) runs every shape (rgb10a2unorm without tier1: the buffer path's subset). The rest of each family runs a representative subset: unit and odd sizes, 1D tails, partial tiles, layers, both view dimensions, and 512×32 (multi-batch at every width). The untrimmed matrix passed once with every format on every shape (10187 cases, 2026-11-10). float32 formats skip the 'average (load)' variant, since they always load. Formats whose feature the device lacks are reported as skipped (`validation.skipped`), never silently passed. The suite requests `texture-formats-tier1` and `bgra8unorm-storage` when available, but not `float32-filterable`, so float32 keeps exercising the load path.
- **Patterns:**
  - constant (weights)
  - random, including HDR values in [−4, 12] for float formats
  - last row/column (edges)
  - per-64×64-tile values (tile boundaries, stale data)
- **Reuse:** one texture reused across patterns, exercising the cached plans.
- **Poisoning:** every level is overwritten with a poison colour before generation, so skipped writes fail.
- **Options:** a sub-range writes only its levels and doesn't modify its source; unsupported textures report a reason and `generate` throws.
- **GPU validation errors and device loss fail the run.**

- **Reductions:** every shape × width × format case runs `'average'` through the sampler and through loads, plus `'min'`, `'max'`, `'minmax'` and a custom fold. The custom fold depends on the value domain, so it catches unit mix-ups. Folds skip the constant pattern and must match exactly. The `edge` pattern on odd shapes exercises the `2x+2` footprint.
- **Separate source:** 8 source → destination pairs (depth32float/depth16unorm/r32float → r32float/rg32float Hi-Z, rgba8 ↔ rgba16f conversions), each in copy or half relation. 13 shapes (odd, 1D, NPOT, layers, 512² for multi-batch, 1920×1080), all kernel widths, random/edge/tiles patterns, and the destination poisoned from level 0. The oracle's source values are read back from the source texture, not taken from the fill data. Options cover rejected sources, `source.mipLevel`, and replacing the source texture with the same options.
- **Odd-size modes (M2T.4, `--only=downsampler`):** every shape × width × format case, on the storage path and on the buffer path's shapes, also runs `oddSizes: 'spd'` (average through the sampler and through loads; folds are rejected), each batch checked against `spdReference` from the GPU's own batch source level; float sums get `pyramidTolerance` on top of the format tolerance, folds are exact. The exact tail kernel (`tail: true`) runs average (sample, load), max, minmax and the custom fold wherever the plan has a tail batch, against the same per-level `'exact'` oracle; normalized formats must match bit for bit except at a batch's first output and the tail's odd steps. Separate sources run `'exact'`, `'spd'` (averages; a same-size source is a 1-level exact copy batch, then clamp batches) and the tail. `encodePass`, the buffer output path and the compatibility-mode path give the same bits as `encode` for both modes; unknown policies and folds under `'spd'` are rejected; tail tiles are 8×8 below 256 invocations per workgroup. `--quick` (the SwiftShader CI run) trims these to the family formats for `'spd'` and to rgba8unorm and rgba16float for the tail, with two patterns, skips 32×32-tile tails (SwiftShader takes minutes to compile each), and leaves the tail out of the separate-source cases; the full matrix runs them all. A core test shows `'spd'` max is not a conservative Hi-Z while `'exact'` is (`maxViolations`).
- **Tile reductions (`Reducer`, §14.1):** both edge policies × kernel widths 1, 4, 5, 6 (all on SwiftShader too: the hook path needs no storage textures) × 17 shapes × four reductions: sum/average/min/max channels, weighted with a weight channel (a third of the texels weigh 0), a custom fold and a custom sum that counts its combines. Each runs through the texture path (source texture → output textures at exact level sizes) and the hook path (load from a storage buffer, store into an atlas), with complementary level selections, so every level of every width is checked and unemitted levels must stay poisoned. Tolerance: `4e-6 × Σ|terms|` (exact for min/max and folds). Plus the FSR 3.1 shading-change shape (clamp, levels 1–3 into half-resolution rgba16float mips, 1920×1080 and 1917×1077), the denoiser weighted mean (zero-weight, sky weight 0, atlas, sky-only blocks emit 0), and the API: encode, encodePass, self-dispatch and size hook produce identical bits; two sizes in one pass; rejected outputs and options.
- **8-channel accumulators (`validate-wide.js`):** 13 shapes × both edges × 4 reductions (channel ops with weights, minmax, a custom fold and a counting custom sum on `DsWide`) × 5 level sets through hooks, unemitted levels poisoned; scalar image and combine-only runs; rejections (no hooks, 6 levels over the workgroup-memory limit).
- **Payload folds (`validate-payload.js`):** argmin/argmax variants with tie-heavy keys × 13 shapes × both edges × 4 level sets (texture path), ssgiDev's G-buffer downsample against its own loop, scalar image and combine-only runs; selections exact.
- **Combine-only scalar runs (`validate-scalar.js`):** tool-written partials at 1 … 8161 per layer, layers at an offset; a Reducer's level 4 or 6 combined in the same pass against the whole image's exact result.
- **Compatibility mode:** the whole matrix also runs on a compatibility-mode device (`--compat`), and the compatibility path forced on a core device must match the direct path bit for bit.
- **Oracle independence:** a core test checks that max chains are conservative: every texel bounds each base texel its normalized area touches, without using `footprint()`.

- **Packed float formats and the buffer path (§6.1):** rgb9e5ufloat and rg11b10ufloat are families of their own and run every shape: rgb9e5 on the buffer path, rg11b10 on the storage path with tier1 and on the buffer path (subset shapes; every shape without tier1). The buffer path runs widths 1, 4, 5 and 6 on any device, and the subset shapes for each family's 4-channel format plus bgra8unorm, rg16float, r32float and rg32float (not with `--quick`, where only the packed formats take the buffer path, on the subset shapes at widths 1, 4 and 6, and the extra checks below use smaller shapes: the packed encoders are slow on SwiftShader, and quick runs with the full additions hit the runner's 30-minute protocol timeout before CI split the suite). In the `--only` partition, the matrix, codec and Downsampler buffer-path checks run in `downsampler`, the Reducer buffer-output and float16-rounding checks in `reducer`. In compatibility mode rg32float has no storage, so it runs the buffer path's subset instead. Fixtures pack r, g, b with the core encoder: HDR values in [−4, 12) mixed with channels spread over 2⁻¹⁴ … 2¹⁰, so texels mix small and large channels. Tolerance on a batch's first output: two steps of the stored value (rgb9e5: of the texel's shared exponent). In-tile levels are exact, against `reduceReference( …, { pairing: 'float32' } )`, which sums the 2×2 block in float32 in the kernel's order (packed sums aren't always exact in f32). Plus: the WGSL codecs against core bit for bit (every 11- and 10-bit code decoded; every rounding midpoint and its neighbouring floats, specials and 20000 random values encoded; rgb9e5 decoded for every exponent × mantissa and encoded at bump, tie, power-of-two and clamp boundaries plus 30000 random texels); buffer path bit-identical to storage path (rgba16float, rg11b10ufloat, rgba8unorm, rgba32float; average and max) for the same plan: both take the storage path's levels per dispatch and the test checks their batches match, since below 6 storage textures (SwiftShader) the default plans differ (6 levels per buffer batch, 4 per storage batch), and a level that is a sampled first output in one and in-tile or a tail step in the other may differ by a step (fixed 2026-11-13: SwiftShader's 67×45 rgba8unorm level 6, since the tail default in `198728a`); a default-limit device without features (6-level batches, rg11b10 falls back to the buffer path); separate sources into both formats; `Reducer` buffer outputs against `reducePyramid` and bit-identical to its storage outputs (rgba32float; rg11b10ufloat), `encodePass` + `encodeCopies` equal to `encode`; API reasons. Core tests: decode → encode identity for every 11-/10-bit code, value identity for every rgb9e5 (exponent, mantissa, mantissa) and 300000 random codes, every 11-/10-bit rounding boundary, and random values against independent oracles (nearest-value search; a log2-free restatement of the extension).

- **Integer formats (M2.3.7, `validate-integer.js`; `--only=integer`, part of `downsampler`):** every integer format, on the storage path and (subset shapes) the buffer path, or the buffer path alone where the device has no storage for it; rgba8uint, rgba32uint and rgba32sint run every shape, the rest the subset. Variants: average, min, max, minmax, a custom fold (bitwise or for uint; for sint a sign-sensitive order that breaks on offset values), the tail (average, max, custom), `'box'` average and max, tail `'box'` average. Patterns: random over each channel's range with a quarter of the values at or next to its ends, `small` (values within 3 of the ends: ties next to the extremes), edge, constant and tiles. Every level against `intReference` from the GPU's previous level, bit for bit. Plus: `ds_mean` and `ds_mean4` against `roundedMean` (120 000 weighted and 50 000 2×2 means over the u32 range, weights up to 2¹⁵ per axis); separate sources (r32uint half, rgba16uint → rgba8uint copy clamped, rgba32sint → rgba16sint half, rgba32uint → rgb10a2uint, r32sint and rgba8uint resampled) under `'exact'`, `'box'` and the tail; `encodePass`, the buffer path, the compatibility path forced and `tail: false` bit-identical to `encode`; `'box'` = `'exact'` at a power of two; a rect update equal to a full recompute; rejections (`'spd'`, mixed signedness, float sources, Reducer outputs and sources, ScalarReduction sources). Core: `roundedMean` against BigInt on 20 000 cases, `intReference` against a brute-force definition (bilinear and area, average and max, signed and unsigned, 8 to 32 bits, 9 shapes), ties, clamping, rgb10a2unorm codes and alpha units. Mutation checks (M5 Pro, `results/probes/mutate-integer.mjs --quick`, 2026-11-13), 16 of 16 caught: the 2×2 mean rounding ties up, truncating, and as a plain u32 sum; the weighted mean rounding ties up, dropping the low half's remainder, and a long division one step short (all by the mean check); bilinear weights swapped, an area overlap off by one, the fold footprint without `2x+2`, the sign offset not removed on store, not applied on load, and not removed before a custom `reduce`, no clamp for a narrower destination, the buffer path masking 8-bit integers to 7 bits; rgb10a2unorm's alpha in 10-bit units and its buffer word with alpha at bit 28 (`--only=format:rgb10a2unorm`). Verified 2026-11-13 on the M5 Pro (Chrome 148, tier1): full matrix 35 517 cases after merging sRGB (33 289 before; 9 shards, none skipped; 971 integer case groups, 9097 integer cases), `--only=format:rgb10a2unorm --quick` (1287 cases), 21 example pages; SwiftShader (no tier1) `--only=integer --quick` 2886 cases, the tier1 formats reported skipped or on the buffer path.

- **`'box'` and resampled sources (M2.5, `validate-box.js`; `--only=box` runs them alone):** the main matrix runs `'box'` average (sample, load), max and the tail on the box formats (each family's 4-channel format, rgb9e5ufloat, rg11b10ufloat, r32float), each level against `boxReference` from the GPU's previous level, normalized in-tile levels bit for bit. Separate sources add `'box'` and tail `'box'` modes, folds included. Render sizes (960×540, 1129×635, 1477×831, 1920×1080), 31×31 (the 1×1 must be the mean of every texel), 1×300, 300×1 and layers (33×17×3, 31×31×6 2d-array) in rgba8unorm, rgba16float and r32float. Resampled sources: 7 source → destination pairs (depth32float, depth16unorm, r32float, rgba8unorm, rgba16float into r32float, rg32float, rgba8unorm, rgba16float, rg11b10ufloat) × 16 size pairs (1920×1080 → 1024×512 and → 2048×1024, 1129×635 → 512×256 and → 1024×512, 960×540 → 512×256, 1477×831 → 1024×512, 31×31 → 16² and → 32², strips, layers, a ratio of 8, a same-size and a half step) × `'exact'`, `'box'`, `'spd'` and both tails (render-size shapes at the widest kernel only); level 0 against `areaReference` from the source, later levels against each mode's oracle, and float32 folds conservative against every source texel at every level. API: `'box'` plans `'exact'`'s batches, equals it at powers of two and for folds, and is bit-identical through `encodePass`, the buffer path and the compatibility path; resample rejections, 2 dispatches for a 1024×512 Hi-Z, a resized source, a same-size copy. Core: nvpro's weights for n ≤ 64, overlaps at 10 ratios, the 31×31 mean, the mean at every level of render-size chains, exact ties, delegation to the 2×2 box, conservative area folds (and a non-conservative 2×2 tap), plans.

- **Partial updates (`validate-region.js`, `--only=region`, part of `downsampler`):** sequences of updates on one texture, each changing the source inside a rect (and a layer range) with `writeTexture` of that rect alone, from the state the last update left. After each: (1) bit-identical to a full GPU recompute of the changed source, and every written level against the CPU oracle from the GPU's own previous level (`spdReference` per batch under `'spd'`); (2) outside each level's region and layer range, bit-identical to the previous chain; (3) on a poisoned chain, everything outside the regions still poisoned (no stray writes). Rects: both corners 1×1, the last column and row, an interior block, the whole level and seeded random ones. Modes: `'exact'`, the tail (default), 1 level per dispatch, 4 (where the limit is above 4), `'spd'`, the buffer path, the compatibility path forced, `encodePass`; rgba8unorm average (sample, load), max, minmax and the custom fold, rgba16float average and max, r32float min; 9 small shapes (1×1, odd, prime, 1D, 133×75, 255×257) plus 1129×635, and 1920×1080 and 1477×831 for the main policies. Layer ranges alone and with rects on 3- and 6-layer arrays and a 1-layer 2d-array, on the tail, exact, buffer and compatibility paths; separate sources (r32float → rg32float copy, depth32float → r32float, rgba16float → rgba8unorm half, and copy and half under `'spd'`) with source rects and layer ranges; rejections (rect size, bounds, integers, the level named; layer ranges), a rect valid for a larger source, encodePass on the buffer path, 1×1, and 20 updates in one encoder (more than the 16 cached) against one submit each and a full recompute. Core tests check the regions against brute-force propagation of the footprint over 10 shapes × 11 plans × 3 source relations × 9 rects, and conservativeness against the CPU oracles (exact average and max, `'spd'` at 1, 4 and 6 levels per dispatch). First chains come from a second Downsampler with the same settings, so an update never leans on state its own instance left (the compatibility path's scratch levels); the two must plan alike (checked per sequence: below 6 storage textures, copying the resolved `levelsPerDispatch` pinned sRGB's buffer path to 4 levels and the first chain's untouched texels differed from a full recompute by one code, CI's first run on SwiftShader). On devices above 4 storage textures, `region/default-limits` repeats sRGB average under the tail, `'exact'` and `'spd'` on a second device with WebGPU's default limits (4 storage textures, 256 invocations: storage path 4 levels, buffer path 6), SwiftShader's split on real hardware. Mutation checks (M5 Pro, `--only=region --quick`, 2026-11-13), 11 of 11 caught: the odd footprint not expanded (`⌊a/2⌋` at odd sizes), a region one texel short at the end, a `'clamp'` region one texel late, `'clamp'` regions chained per level (misses 3 → 1 → 1), the exact kernel ignoring the workgroup origin, the exact, tail and tile kernels storing outside the region (three mutations), the layer offset ignored, buffer copies ignoring the first layer, and compatibility updates reading their scratch levels instead of the destination. The last one survived until first chains came from another instance.

- **sRGB (M2.3.5; `--only=srgb`, part of `downsampler`):** rgba8unorm-srgb is a family of its own and runs every shape, bgra8unorm-srgb the subset, both on the buffer path at widths 1, 4, 5 and 6 and every variant (averages through the sampler and loads, folds, `'spd'`, the tail, `'box'`), against the core oracles with quantization `'srgb8'` (colour reduced in linear float32 and encoded, alpha as unorm8): two codes of tolerance at a batch's first output and a tail's odd steps, in-tile levels bit for bit against the float32 pairing. Separate sources: sRGB → sRGB (copy, half, rgba and bgra), rgba16float HDR → sRGB, sRGB → rgba16float. Partial updates: rgba8unorm-srgb average and max under `'exact'`, the tail, `'spd'` and `'box'`. `validate-srgb.js`: the WGSL encoder and decode table against core bit for bit (every threshold ± 3 ulps, every table value, specials, 40000 random values); the hardware's decode of every code reported and required to re-encode to its code, alpha linear within 10⁻⁶ (a compatibility-mode SwiftShader device reads alpha 3 one ulp high); a black and white checkerboard reducing to code 188 (and alpha 128, a tie to even); load-path chains (`'exact'`, tail, `'box'`; 7 shapes, all codes and dark codes 0–24) code for code against the oracle, one code apart only where the exact value lies within 10⁻⁶ of a threshold; the compatibility path, `output: 'buffer'` and a shared scratch buffer (two chains in one encoder) bit-identical; bgra8unorm-srgb byte for byte with rgba8unorm-srgb; the kernel's storage output into an rgba8unorm scratch plus `copyTextureToTexture` bit-identical to the buffer path (on the buffer path's plan, or below 6 storage textures on both paths' plan at the storage limit: the scratch binds a storage texture per level); API (encodePass, reasons without COPY_DST and with `output: 'storage'`, scratch size and usage). Core: the decode table, the encoder against the standard's encode formula (200000 values and every threshold), no f32 on any threshold (and each at least 64 f64 ulps from the f32 grid), the bounds, linear-light averages, alpha ties, the box, area, spd and copy oracles. Mutation checks (M5 Pro, `results/probes/mutate-srgb.mjs --quick`, 2026-11-13), 10 of 10 caught: no decode (codes averaged as UNORM, SPD's alias in WebGPU terms), in-tile averages in encoded space, loads not snapped to the exact decode (caught only by the near-exact chains: the two-code tolerance hides it), the encoder truncating, the encoder by `pow` alone without the threshold correction (in-tile exact checks, linear-segment values), gamma 2.2 instead of the sRGB curve, no intermediate quantization, alpha ties rounding up, alpha encoded as sRGB, and bgra8unorm-srgb packed in rgba byte order.

New features add their own oracle first: each new format family (rect, layer ranges, `'box'`, sRGB and the integer formats have theirs).

## 8. Performance methodology

- **Timestamps** come from two one-pixel render marker passes bracketing 16 repeats. The markers depend on the texture under test: the first writes level 1, the last samples the final level. Empty or independent compute markers gave invalid timestamps on Metal.
- **Marker cost** is measured as an empty batch and subtracted (about 0.0006 ms per chain on M5 Pro).
- **Results:** medians of 11 rounds, rotating order across strategies, warmed, with a parity check before timing.
- **Baseline:** one render pass per level and layer, linear sampler. This is the common approach and what three.js does today.

**Odd-size strategies** (`npm run bench -- --compare=spd --skip-validation`, M2T.4) time `'exact'`, the tail kernel, `'spd'`, the atomic probe (texture-ready and kernel only) and render passes, for average and max, at the render sizes plus 1024²/2048²/4096². `encodePass` variants separate pass overhead from dispatch structure. The device requests the adapter's storage texture limit for the probe. Before timing, the probe must reproduce `'spd'`, and `'spd'` its oracle. `--quality` measures `'spd'` against `'exact'` and the area-weighted ideal per level, and Hi-Z violations (§2.1). `run.js` records load averages before and after.

**Tool fixtures** (`npm run tools`, `test/tools/`) time compute passes with their own `timestampWrites` around 16 repeats (never empty), 11 rotated rounds, median.

**M3 measurement modes** (2026-11-13; details, flags and JSON in [test/bench/README.md](test/bench/README.md)). Each takes `--strategies=a,b,c` (as do `spd` and `render`); the reference a mode divides by is always kept. Bench `exact*` strategies pin `tail: false`, since the library default is now the tail kernel.
- **Throughput** (`--compare=throughput`): greggman's work-per-frame method, which needs no timestamp-query. N chains per submit (calibrated to about 8 ms), submits kept in flight for a 1 s window (or one per `requestAnimationFrame`), wall ms per chain, rotated, median of 5. Strategies: render, `'exact'`, `'spd'`, tail. Rows where JS encoding takes over 80 % of the window are flagged as CPU-bound. Where timestamps exist, the same cases are then timed with timestamps, and the report counts strategy pairs that both methods order alike (ties under 5 %).
- **`STORAGE_BINDING` sampling cost** (`--compare=storage-usage`): two otherwise identical textures, with and without the usage. Each is read by a fullscreen pass (1 tap) and a material pass (9 trilinear taps), and also rendered into. Content is either rendered (compressible) or uploaded noise (the control). Formats rgba8unorm/rgba16float/r32float, at render sizes and 4096². Timed by `timestampWrites` on the passes; storage ÷ plain paired per round.
- **Subgroup upper bound** (`--compare=subgroups`): the library's chain with each multi-level `'exact'` batch's `main()` replaced:
  - `slot`: one barrier per level, identical results.
  - `shuffle`: the first reduction by `subgroupShuffleXor`, then `slot`. It checks itself against the library.
  - `nosync`: no barriers or workgroup memory for levels 2+; timing only. (lds − nosync) ÷ lds bounds any subgroup kernel's gain. The §13 rule applies: build one only if the bound is at least 5 %.
- **fp16** (`--compare=f16`): the same harness with levels 2+ averaged in f16 (`f16`), and with quarters held in workgroup memory so sums can't overflow (`f16safe`). Precision is reported against the library's f32 chain, in steps and ulps, including non-finite values.

**Baseline results.** Apple M5 Pro, Metal 3, Chrome 155, 2026-11-10, GPU ms per chain:

| Size | Format | render | compute (4 levels) | compute (6 levels) | 6 vs render |
|---|---|---:|---:|---:|---:|
| 256² | rgba8 | 0.0801 | 0.0092 | 0.0079 | 10.1× |
| 512² | rgba8 | 0.0882 | 0.0149 | 0.0098 | 9.0× |
| 1024² | rgba8 | 0.1159 | 0.0199 | 0.0161 | 7.2× |
| 2048² | rgba8 | 0.1369 | 0.0619 | 0.0604 | 2.3× |
| 4096² | rgba8 | 0.4501 | 0.3361 | 0.3371 | 1.3× |
| 1920×1080 | rgba8 | 0.1176 | 0.0365 | 0.0374 | 3.1× |
| 512²×6 | rgba8 | 0.6337 | 0.0288 | 0.0236 | 26.8× |
| 1024² | rgba16f | 0.1178 | 0.0273 | 0.0263 | 4.5× |
| 4096² | rgba16f | 0.8341 | 0.6477 | 0.6508 | 1.3× |

**Reading the results:**
- From 2048² up, every strategy is near this chip's memory bandwidth (about 250 GB/s), so gains there must come from bytes, not dispatches.
- Differences under about 5 % are within run-to-run noise. 1920×1080 runs identical dispatches in both compute columns, and the gap there is noise.

## 9. Device coverage

`npm run lab` serves the device lab at `/test/browser/`. It runs validation and the benchmark on whatever device opens it and offers "Copy JSON report". WebGPU needs a secure context: localhost, or HTTPS for other devices (deploy the repo statically, or tunnel).

Target matrix before recommending a default: Apple (macOS, iOS), NVIDIA, AMD and Intel on Windows (D3D12) and Linux (Vulkan), Android (Mali, Adreno, PowerVR), plus Chrome, Edge, Safari and Firefox (Firefox once WebGPU compute + storage textures ship). Record each report under `results/`.

CI runs the unit tests and the quick GPU suite on SwiftShader, split into disjoint shards of case groups balanced by recorded durations (test/ci/README.md); the full tier (main, weekly, manual) adds the same suite on a compatibility-mode device. SwiftShader exposes 4 storage textures, so CI covers the 1- and 4-level kernels; 5- and 6-level kernels and all timing need real GPUs.

## 10. three.js integration

three.js does not take runtime dependencies for core features. The plan:

1. **`downsampler/three` (this repo).** `generateMipmaps(renderer, texture)` for textures three allocates with storage usage (`StorageTexture`). Useful today, opt-in.
2. **In-tree PR to three.js.** Vendor the kernel generator into `WebGPUTextureMipmapUtils`, behind `new WebGPURenderer({ mipmapGeneration: 'compute' })`.
   - The renderer adds `STORAGE_BINDING` to eligible textures and requests the storage texture limit unless `requiredLimits` sets it.
   - It falls back to render passes for unsupported textures.
   - A prototype with these semantics passes three's integration tests in the `webgpu-spd-audit` branch.
   - The first PR adds sRGB (most three.js color textures) and cube maps. Hi-Z follows separately. sRGB works in `downsampler/three` since M2.3.5: three's sRGB colour textures are `rgba8unorm-srgb` with COPY_DST, so they take the buffer path with no STORAGE_BINDING (which an sRGB texture can't have anyway). Many textures can share one `scratch` buffer (§5).
   - Keep this repo's GPU suite as the shared oracle, so the vendored copy can't drift.
   - **Done (2026-11-13):** `WebGPUTextureMipmapUtils` (611 lines, branch `compute-mipmaps`) vendors the in-place `'average'` subset by hand from commit 488eff6: layers and cube maps as 2d-array views, but no sRGB (the buffer path) in this first PR; textures it can't write keep render passes, as does every texture in compatibility mode.

### Vendoring (2026-11-13)

What three.js's integration needed, built so that a copy is one file and its drift is a test failure:

- **One file.** `downsampler/mipmap` (`src/mipmap.ts` → `dist/mipmap.js`, plain ES2022, no imports, ~19 KB with comments): `createMipmapKernel`, `planMipmapBatches`, `kernelDefaults` and the format subset (`mipmapFormats`, `mipmapAccess`). It is a pruned copy of the generator, not a bundle: the full `kernel.ts` with its imports is several thousand lines, nearly all for paths an engine's in-place chain never takes. A core test (`test/core/mipmap.test.js`) requires it to equal the full generator on its whole option space (every format × dimension × count × tile × tail × access × `f16Rounding`, ~34 000 kernels) and `planBatches` on 16 shapes × every plan option, so a change to the kernel bodies fails there until the copy follows.
- **Defaults as a pure function.** `kernelDefaults( limits, adapterInfo )`; `Downsampler`'s constructor resolves its defaults from it, so an engine using the function gets the library's levels per dispatch, tiles and tail.
- **float16.** `float16Wgsl( rounding, false )` and `KernelOptions.f16Pack: false` give `ds_round_f16` alone: `ds_half_bits` and `ds_pack_f16` serve only the buffer output path. The default stays true, so every existing option set's WGSL is unchanged byte for byte (compared before and after over 723 240 option sets); the mipmap entry uses false.
- **Goldens.** `test/golden/mipmap/` holds the WGSL of 56 fixed option sets (every format at a 4-level batch and a 6-level tail with its usual access; every batch shape for rgba8unorm; both accesses, both dimensions and `'device'` rounding) and `cases.json`. `test/core/golden.test.js` fails when either generator's output changes; `npm run golden:update` (`node test/core/golden.test.js --update`) rewrites them, and their diff is what a vendored copy must follow.
- **Drift check.** `node scripts/check-vendored.mjs <file>` imports a vendored copy (relative imports resolved against it), calls its generator (an exported `createMipmapKernel`, or a module-level `createKernel` taking options or three's positional arguments) on the golden cases and compares byte for byte: the matching count, the differing cases with line diffs, and the changed lines across them. Cases the copy rejects are reported as not covered. Exit 1 on any difference.
- **GPU.** `validate-mipmap.js` (`--only=mipmap`, part of `downsampler`) records chains from the entry alone, with the bindings its documentation names, and requires them bit-identical to a `Downsampler`'s, and its plan and defaults equal: 11 formats × 9 shapes (layers, 2d-array, odd, 1D, 1×1, 1920×1080).
- **Three's copy today** (check-vendored against `compute-mipmaps` @ dc4bd5e5c1): 18 of 56 goldens match, 36 differ, 2 not covered (its positional generator has no `f16Rounding`). Three kinds of difference, none changing results: float16's `ds_round_f16` is emitted after the storage bindings instead of before them; single-channel formats don't splat (`.xxxx`) the sampled or loaded value; the load path leaves out the same-size copy line (`if (all(sourceSize == size))`), which an in-place chain never takes. Adopting the library's text for these three makes the copy pass.

## 11. Milestones

Build first, benchmark once coverage is broad (decided 2026-11-10). Live status is in [PROGRESS.md](PROGRESS.md).

1. **M1 (done):** port the kernel, Downsampler, three adapter, core unit tests, GPU suite, device lab, benchmark and CI.
2. **M2 (features).** Done: load path and reductions; separate source (Hi-Z) with the float32 formats; formats steps 1–2 (fewer-channel unorm8/float16, bgra8unorm, snorm8, unorm16, snorm16).
   **M2T (tools first, 2026-11-11)**, in this order (details in §14 once the survey lands):
   1. WGSL reduction generator: tool-supplied `load` with its own bindings, per-channel reductions including sums, choice of emitted levels, outputs (texture, buffer, WGSL `store` hook), edge policy. Done: `Reducer` (§14.1), with buffer output into textures (§6.1, `output: 'buffer'`).
   2. Reduce-to-scalar in 2 dispatches with an epilogue hook.
   3. Proof on the upscaler's FSR 3.1 passes as bench fixtures, against an oracle, timed on M5 and by the owner on RTX 3060 / Intel UHD. Done on M5 (§14, M2T.3); RTX 3060 / Intel UHD pending.
   4. An SPD-style mode, built at least as a measured comparison that justifies, or not, its not being the primary method. Done (2026-11-11): `oddSizes: 'spd'` (opt-in, 'average' only), the exact tail kernel (opt-in) and an atomic single-dispatch probe (test-only), measured at render sizes; `'exact'` stays the default (§2.1, §2.2).
   Then: buffer output + rgb9e5ufloat and rg11b10ufloat (GI formats; done 2026-11-11, §6.1); layer range + rect (done 2026-11-13, M2.4, §4 Partial updates); `'box'` and a power-of-two destination from a separate source (done 2026-11-13, M2.5: §2.1, §4 Separate source); the tail kernel's default (on since 2026-11-13).
   Deferred, three.js-driven: none left. Done 2026-11-13: sRGB (M2.3.5, §6.1 sRGB; transform hooks deferred, §5 Planned), rgb10a2unorm (M2.3.3) and integer formats (M2.3.7), §6.1.
3. **M3 (dial in: coverage and tuning):**
   - per-frame throughput benchmark alongside timestamps (§13)
   - device lab runs across vendors (§9)
   - `STORAGE_BINDING` sampling cost
   - sampler vs load for float32 with `float32-filterable`
   - subgroup kernel with the workgroup-memory fallback, only if the upper-bound test shows ≥5 %
   - fp16
4. **M4 (examples and docs):** usage docs, notes for SPD users (§13), examples (mip chain, Hi-Z from depth, custom reduction, three.js).
5. **M5:** npm publish, the upscaler and other tools as consumers, three.js in-tree PR (§10), and the data-backed gpuweb#5253 comment.

## 12. Open questions

- ~~Publish as bare `downsampler` or `@pmndrs/downsampler`?~~ Decided (owner, 2026-11-12): `downsampler`, unscoped while it is free (still free on 2026-11-13: `npm view downsampler` is a 404), else `@pmndrs/downsampler`. Package shape: §5 Package.
- ~~Buffer output path (§6.1): should `output: 'auto'` take the buffer path for textures without STORAGE_BINDING?~~ Decided (owner, 2026-11-12): no. `'auto'` takes it only for formats the device can't store; a missing STORAGE_BINDING usage is an error, and the buffer path for storable formats is opt-in (`output: 'buffer'`). ~~Should `Reducer` round rgba16float outputs before the store?~~ Yes: it quantizes with `pack2x16float` first, like `Downsampler`.
- Cube maps in compatibility mode: they can't be viewed as 2d-array, so either per-face views or the render fallback. (2026-11-11: per-face views don't help either; compatibility mode binds whole textures only, and a `'cube'` texture has no storage view. A 6-layer `'2d-array'` texture works, which is what reflections bakes into, then copies to a cube; §4 Compatibility mode.)
- ~~`'box'` for `'minmax'` and other folds is meaningless (they already use the full footprint). Should `'box'` simply be ignored there, or rejected?~~ Decided (coordinator, 2026-11-13; owner can overrule): accepted, and identical to `'exact'`. A pipeline can then set `oddSizes: 'box'` once for its average and Hi-Z chains.
- ~~`source.resample: 'area'` is an explicit opt-in, so a destination of the wrong size (say, stale after a resize) still fails loudly. Should a power-of-two destination take it without the option? And is 8× per axis (`maxResampleRatio`) the right cap?~~ Decided (owner, 2026-11-13): it stays explicit; 8× stays the default, as the `maxResampleRatio` constructor option (free: validation only), raised later if a tool needs it.

## 13. Research (2026-11-10)

Reference material, mostly for after M2–M3: prior art and the standards discussion, reviewed against this spec. Consult it when a problem matches something solved here. Sources are linked; claims attributed to people come from their comments.

### Community prior art (WebGPU)

- [webgpu-spd](https://github.com/JolifantoBambla/webgpu-spd) (`JolifantoBambla`, 2024): a WGSL SPD port, single dispatch with the atomic handoff; see the #7 note below.
  - 3.0.0 measured side by side on the M5 Pro (2026-11-12): its handoff returned stale levels, and our `'spd'` was 1.2–2.1× faster at render sizes ([results/compare/webgpu-spd.md](results/compare/webgpu-spd.md), DIVERGES #22).
- `davidar`'s SPD port (2022, gpuweb#386): 0.3–0.4 ms against about 2 ms for repeated blits.
- `greggman`'s compute generator (gpuweb#5253, 2025): 4 levels per dispatch, faster on Mac and slower on NVIDIA/D3D12. It also covers his render-pass approach and his per-frame throughput method.
- [`Volcomix`'s fiddle](https://jsfiddle.net/1w9L6u07/) (2026): video frame → pyramid with quad/subgroup ops (see below).
- Outside WebGPU: [nvpro_pyramid](https://github.com/nvpro-samples/vk_compute_mipmaps) (Vulkan) and AMD FidelityFX SPD (D3D12/Vulkan).

### gpuweb: mipmaps in core ([#386](https://github.com/gpuweb/gpuweb/issues/386), [#5253](https://github.com/gpuweb/gpuweb/discussions/5253))

- **History:** proposed in 2019 as `generateMipmaps(texture)`. It is still open on "Milestone 4+", with no spec PR. The group leans toward a userland or "official helpers" library over a core API. No implementer has re-engaged since 2022.
- **Cross-vendor performance is the open question.** `greggman` found a compute generator slower than render passes on NVIDIA/D3D12, up to 10× for SPD-style ports. `toji` reported an internal partner seeing the same and suggested drivers may fast-path the render-pass pattern. `greggman` also found timestamp-query unreliable and measured work per frame instead. `aleino-nv` (NVIDIA) asked for NVIDIA numbers. **Every number in §8 is Apple-only and timestamp-based, so no general speed claim can be made yet.**
- **Other requirements raised:** wrap vs clamp edges, alpha-coverage preservation, out-of-bounds texels contributing nothing, cube maps (compute needs a 2d-array view; compatibility mode can't provide one), and the unmeasured cost of adding `STORAGE_BINDING` to sampled textures (`Kangz`). The last matters for §10, where three.js would add the usage.
- **Evidence for §2.2:** [webgpu-spd#7](https://github.com/JolifantoBambla/webgpu-spd/issues/7) fixed missing mips on M1 by adding a `storageBarrier`. In WGSL that barrier is workgroup-scoped, so the fix isn't guaranteed.
- **Video:** there's no zero-copy way to put an external texture into level 0 (`kainino0x`). A separate source plus `copyExternalImageToTexture` is the current path. [This jsfiddle](https://jsfiddle.net/1w9L6u07/) samples level 1 from a `texture_external` and uses quad/subgroup ops in compute. That layout is undefined behaviour in WGSL (§6 subgroup row), and its odd sizes drop the last row/column (SPD-style, the dropped `'fast'` of §2.1).

### nvpro_pyramid ([vk_compute_mipmaps](https://github.com/nvpro-samples/vk_compute_mipmaps), [strategy](https://nvpro-samples.github.io/vk_compute_mipmaps/docs/strategy.md.html))

- **Odd sizes are area-weighted**, not render-pass parity. Per axis the kernel is 1, 2 or 3 taps, with weights `(n−x, n, x+1) / (2n+1)` for size `2n+1`: each source texel's overlap with the output. 3→1 averages all three texels. Its support (2x … 2x+2) is exactly our fold footprint.
- **At most 2 levels per dispatch through odd sizes.** Each tile recomputes a 1-texel halo of the intermediate level. The authors found 3+ levels unpromising, because the halo grows 1, 3, 7… texels per level.
- **No single dispatch**, by design. A fast pipeline (power-of-two tiles, up to 6 levels) handles even sizes.
- **Their measurements (RTX 3090 only):**
  - Dropping the bilinear sampler cost 26–32 %, consistent with §4 Source access.
  - 1024-thread workgroups were worse everywhere, and 1-level dispatches worse than multi-level.
  - Packed sRGB or f16 shared memory gave no gain.
  - On even sizes it ran near their speed-of-light baseline; at 1080p it took twice the baseline's time.
- **Subgroups:** `subgroupShuffleXor` across quads and teams of 16 reduces shared memory to 64 B. It assumes a lane = local index mapping that GLSL/WGSL don't guarantee, and has a TODO for subgroup sizes other than 32. The gain isn't isolated in their data.
- **Benchmark method:** interleaved alternatives, min and median over 256 batches, a speed-of-light baseline kernel, and a CPU reference check.

### SPD manual ([single-pass downsampler](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/))

The rows are folded into §6. Differences SPD users must be told about: mip count off by one, odd-size results, per-level rounding, no zero reads at edges, the `reduce(a, b)` contract, and that a same-size source writes destination level 0 at full size.

### Subgroups ([WGSL §15.4–15.5, §17.12–17.13](https://gpuweb.github.io/gpuweb/wgsl/), [Chrome 134](https://developer.chrome.com/blog/new-in-webgpu-134))

- **Availability:**
  - `subgroups` shipped in Chrome 134 and Safari 27.3; Firefox doesn't have it. (caniuse still shows Safari without it; this is corrected from the project owner's testing.)
  - Chrome added `subgroup_id` in 144 (emulated on D3D), `subgroup_uniformity` in 145 and `subgroup-size-control` in 151–152.
  - Sizes are powers of two from 4 to 128 and may vary on Vulkan.
  - Requirement: any subgroup kernel ships alongside the workgroup-memory kernel as the fallback, chosen by `device.features`.
- **Constraints:**
  - Quads are defined only for fragment shaders, and there is no defined relationship between lanes and `local_invocation_index`. SPD's `ARmpRed8x8` + `WaveReadLaneAt` and nvpro's shuffles both assume one.
  - Without `subgroup_uniformity`, subgroup calls must sit in workgroup-uniform control flow, so not inside our `local.x < extent` guards.
- **Portable scheme (not built):**
  - Linear id `subgroup_id · subgroup_size + subgroup_invocation_id`, mapped to an `ARmpRed8x8`-style layout.
  - `subgroupShuffleXor` with masks 1/2 per level, then 4/8, 16/32. Shuffles are hoisted out of the guards, with clamped neighbours swapped for the lane's own value.
  - A uniform `num_subgroups · subgroup_size == 256` test selects it over the LDS path.
- **Payoff:** barriers fall from 10 to 1–2 in the 6-level kernel, and LDS from 16 KiB to about 1 KiB. Plausible gain at ≤512² (latency-bound); about nothing at 4096² (bandwidth-bound).
- **Oracle:** the butterfly sums `(a+b)+(c+d)`. That doesn't matter for unorm8 integers, but float averages need the oracle to pair the same way.
- **Cheaper first:**
  - Without subgroups, writing each level's result to slot (2x, 2y), or ping-ponging two buffers, removes one of the two barriers per level.
  - Decide with an upper-bound kernel (barriers and LDS for levels 2+ removed, timing only) and a layout-only control. Abandon if the upper bound is under ~5 %.

### Consequences for the plan

1. **Cross-vendor throughput data before any default or speed claim** (M3 moves ahead of the three.js PR):
   - Add a throughput benchmark mode (work per frame, as in `greggman`'s method) alongside timestamps.
   - Get NVIDIA/D3D12, AMD and Intel numbers from the device lab.
   - Measure `STORAGE_BINDING` sampling cost.
   - If compute loses on a vendor, adapters must choose the path per device.
2. **Exact tail kernel** (built and measured, M2T.4; opt-in `tail: true`). Once a first output fits one 32×32 workgroup, that workgroup reduces every remaining level through any parity with exact `'exact'` weights (no halo needed). 1920×1080 drops from 7 dispatches to 4, and render sizes from 5–9 to 3–6. The time saved is small, 0.83–1.07× on a loaded M5 Pro, because the trailing dispatches it removes are cheap (§2.1).
3. **Halo batches under `'exact'`** (candidate). Our odd bilinear taps also fall in 2x … 2x+2, so nvpro's halo structure carries one odd step per dispatch without changing results. This is now the lever for the large-level breaks that the tail can't reach. Its upside is bounded by `'spd'`: 0.47–0.73× of `'exact'` at render sizes in rgba8 on the M5 Pro.
4. **Area-weighted odd sizes** (`oddSizes: 'box'`, done 2026-11-13, §2.1). The quality fix for the 31×31 case, with its own oracle (`areaReference`). A defined meaning for weighted reductions at odd sizes (`Reducer` edges `'box'`) is not built.
5. **Engagement:** one data-backed comment in #5253 (method, named devices, device-lab link), and a comment on webgpu-spd#7 citing the memory model. Both only after the cross-vendor data exists.

## 14. Tools survey (2026-11-11)

What our tools and their references actually build, to decide the M2T generator (§11). Sources: AMD FidelityFX SDK at `7539ae8` (FSR3UPSCALER 3.1.6, `Kits/FidelityFX/upscalers/fsr3/`) and the local tool repos; claims spot-checked against the code.

### FSR 3.1 upscaler (the reference for pmndrs/upscaler)

Two SPD passes over the full render resolution, with the same setup: groups `⌈W/64⌉×⌈H/64⌉`, `mips = min(⌊log2 max(W,H)⌋, 12)`, the atomic tail, and a counter cleared every frame.

| Pass | Custom load (per render texel) | Reduce | Stored | Read downstream |
|---|---|---|---|---|
| Luma pyramid | `(max(6.1e-5, log luma), luma, farthest depth m)` from R16F current luma and farthest depth | average, all channels | level 1 depth → `FarthestDepthMip1` (R16F, half res); level 6 `.xy` (SPD tail only); last level `(0,0)` → `FrameInfo` (RGBA32F 1×1) with temporal smoothing → exposure `1/(9.6·exp(Lavg))` | depth mip 1 bilinear (accumulate, luma instability); exposure by load in six passes |
| Shading-change pyramid | signed min-difference of 5-tap sorted crosses: current luma × exposure vs previous luma reprojected by dilated motion; `(diff, sign, 1)` | average | `.xy` levels 1–6 → `SpdMips` (RG16F, half res, full chain) | levels 1–3 only, bilinear; max of `|x·y|` → `ShadingChange` (R8, half res) |

Behaviour we'd diverge from, deliberately or not (decide per case, record in DIVERGES):
- **Exposure covers only `[0, 2^mips)²`.** Only `(0,0)` of the last level is kept, so at 1920×1080 the scene average covers the top-left 1024×1024. Inside that window, tiles past the mip-6 resource (`(⌊w/2⌋ >> 5) × (⌊h/2⌋ >> 5)`) read as 0: their stores are dropped and D3D out-of-bounds loads return 0.
  - Against every texel, log luma comes out −28 % at 1280×720, −38 % at 1129×635 and −8.7 % at 1707×960, so exposure is +13.5 %, +19 % and +4 % (`npm run tools`).
  - `max(ε, log L)` makes every texel darker than luma 1 count as 0.
  - The reset value `(−1, 1, 0, 0)` makes the first frame ease from log luma 1.
- **Partial tiles repeat the edge texel** (clamped loads), so edge texels are over-weighted.
- **The last odd row or column is dropped.**
- **The fp16 paths are stubs** that return 0.

What it needs from us:
- **Luma pyramid:** a full chain to a scalar, with an epilogue hook for the smoothing. That is 2 dispatches. Only level 1 and the scalar are stored.
- **Shading change:** 3 levels of a 2-channel average with a heavy custom load. That is 1 dispatch with no tail.
- **Single dispatch:** neither pass needs it. Both need custom load/store and level selection.

pmndrs/upscaler replaced both passes because no downsampler existed:
- **Exposure:** one thread takes 1024 serial samples (`luminancePyramid.ts:87-99`). It costs 0.06 ms on an RTX 3060 laptop and 0.17 ms on Intel UHD, both in the full pipeline.
- **Shading change:** a fused 16×16-tile block-statistics pass (sums of cur, prev, cur², prev², plus a max flag) that uses 2 levels. It costs 0.044 ms on Apple at 960×540.
- **Bench:** its SPD-style bench candidate re-read the source per level with no shared memory. It measured 5× slower, so that comparison says nothing about a tiled kernel.
- **Where the time goes:** accumulate is 67–78 % of its compute.
- **Fixture ratios** (M5 Pro, one run, load average ~63 on 18 CPUs, so same-run ratios only):
  - FSR's SPD luma pyramid takes 0.32–0.68× the upscaler's single-thread exposure, a 1.5–3× win that grows as size shrinks.
  - FSR's shading-change pyramid plus its consumer takes 2.0–2.4× the upscaler's fused shading-change pass. The upscaler's design is faster for that pass, so the generator must beat the fused pass, not FSR's.
- **Generator versions (M2T.3, 2026-11-11)**, built from `Reducer` and `ScalarReduction` with only tool code in `test/tools/generator.js`. All 7 pass their oracles at all 5 sizes, and all fail under `--mutate`. Ratios are paired per round over six runs on the M5 Pro (load average 14–30, other agents' GPU work running), 960×540 … 1920×1080 (`test/tools/README.md`):
  - **Upscaler shading change** as a `Reducer` (its load, levels [2, 3], block memory and response as the store hook): **0.95–1.02×** the hand-fused pass, a tie within noise, and 0.39–0.52× FSR's pyramid + consumer. Four profiling attempts changed nothing beyond noise: exposure texels hoisted, 16×16 workgroups, and one barrier fewer (a kernel patch). The pass is bound by its loads.
  - **Upscaler exposure** as a `ScalarReduction`: 0.33–0.60× the single-invocation loop over every texel, and 0.22–0.32× over the same 1024 taps (same values).
  - **FSR shading-change pyramid** (FSR's hooks, `'clamp'` [1, 2, 3]): 0.74–0.86× the SPD port.
  - **FSR luma pyramid:** every texel with one source read (`Reducer` [1, 6] sums → `ScalarReduction` of the tile partials), 0.89–1.15× FSR's every-texel pass (3 dispatches; 2 since combine-only runs, below). Every texel with two reads, ~1.4×. AMD parity, 1.09–1.31× AMD's pass at 3 dispatches. AMD's top-left `2^mips` window, its zero reads and its f16 mip 6 are reproduced by a `ScalarReduction` over the window of mip 6, to ≤ 1.1e-7. That needs no option, and isn't worth one (§14 above).
- **API gaps found (M2T.3)**, none blocking:
  - **Five values per texel.** The upscaler's block statistics are 4 sums plus a max flag, and the kernel carries one `vec4f`. The flag rides in the sign of a non-negative sum, through a custom `'sum'` reduction. Wider accumulators are the clean fix if a second tool needs them. **Done (2026-11-11):** 8-channel accumulators (§14.1 Wide accumulators); the fixture uses them, at 0.86–1.01× the sign-bit version's speed.
  - **An epilogue that needs two levels at once** (signal = max(4×4, 8×8) at the 2×2 resolution). Expressible: store hooks keep workgroup state across levels. That is now a documented guarantee with a GPU test (§14.1 Hooks); no kernel change.
  - **Clamped loads with ⌈n/2^l⌉ blocks.** Expressed with `'clamp'` at a padded size (§14.1 Edges); no option added.
  - **Per-texel side outputs from the load hook** (the upscaler's luma history). This is safe when every texel is loaded exactly once: `'clamp'` at whole source tiles (§14.1 Hooks).
  - **A Reducer feeding a scalar.** ~~Here it goes through a tool partials buffer and a `ScalarReduction` with a load hook, 3 dispatches.~~ **Done (2026-11-11):** combine-only runs (§4 Scalar reductions) take the Reducer's tile partials directly, 2 dispatches. `generator/luma-pyramid-fused` uses them: the 3-dispatch version takes 1.11–1.23× its time at 4 of 5 sizes in each of two same-run comparisons (M5 Pro, load average ~6 on 18 CPUs; 1920×1080 was noisy, 0.92× and 1.11×). Against `fsr/luma-pyramid-correct` it is now 0.74–0.92×.
  - **No per-invocation prologue hook.** Hoisting the exposure texels measured 0.98–0.99×, so it isn't needed on Apple.

### Our other tools

| Tool | Pyramid / reduction | Shape | Built today |
|---|---|---|---|
| reflections (SSSR port) | Hi-Z from depth, min (max for reverse-Z), r32float, ≤7 levels; `textureLoad` traversal | full chain, separate source | **on downsampler** via `encodePass` (branch `reflections-integration`): 1080p 0.23 → 0.15 ms (M5 Pro) |
| reflections | probe prefilter source chain, rgba16float cube (6 layers), average; `textureSampleLevel` | full chain | on downsampler; ~~fails in compatibility mode~~ compatibility path since 2026-11-11 (§4 Compatibility mode; reflections' own compat check not re-run here) |
| ssgiDev | depth min pyramid, 5 levels, one dispatch per level brute force; optional max pyramid; light pyramid (average rgba16f); G-buffer downsample (min depth, normal of the min texel, mean light) | Hi-Z, chain, fold with payload | hand-written |
| Denoiser | OIDN auto-exposure: log-mean luma → scalar (f32 buffer); FFX reflections 8×8 luma-weighted radiance mean fused into reproject; ReBLUR-style history-fix mips 1–3 in one LDS dispatch, sum ÷ non-sky count, off-image weight 0, ceil sizes, atlas layout | scalar, fused block stats | hand-written |
| webgpu-ddgi | material store: alpha-weighted 2×2 average, 9 levels of 256² layers, sRGB via buffer + `copyBufferToTexture`; sky SH 19 sums → scalar (subgroups); radiance cache: r32uint "first valid of 2×2" probe mask (~8 levels), hash-grid tile sums `(rgb·n, n)` 4 levels | chain, scalar, uint fold, block stats | hand-written, one dispatch per level |
| Capsaicin (reference) | `GPUMip`: area-weighted odd sizes (nvpro weights), depth min/max `2x…2x+2`, sRGB re-encode, then SPD once the size is a power of two; HZB with a min-reduction sampler (WebGPU has none); `GPUReduce` 1D sum/min/max/product. (Checked at `914b915`, 2026-11-13: `gpu_mip.comp` applies `(n−x)/S` and `n/S` at even sizes too, which sum to less than 1 for x > 0, so its non-power-of-two even levels look darkened towards the far edge. Ours use 1/2, 1/2 there.) | chain, Hi-Z, scalar | D3D12 |

### Shapes that recur, and what the generator must cover

1. **Hi-Z from depth:** min, max or a min+max pair, full chain, separate source, read with `textureLoad`. This is covered today.
2. **Average colour chains:** cubes and arrays, sometimes alpha-weighted, sometimes with sRGB output. Mostly covered; sRGB and weighting are missing.
3. **Fused block statistics:** 2–4 levels in one tile, from a custom load. This is the most common shape and the biggest gap.
   - The reductions are sums or weighted means with a carried weight channel, and off-image texels get weight 0.
   - Some carry a payload, such as the normal of the min-depth texel (payload folds, §14.1, 2026-11-11).
   - None of them are semilattice folds.
4. **Scalar 1×1:** exposure, SH and statistics, usually into a buffer, with an epilogue.
5. **uint folds:** "first valid", or max over a packed key.

**Generator requirements (M2T.1–2).**
- **Hooks:**
  - **load:** WGSL with the tool's own bindings, returning `vec4f` (later also `vec4u`).
  - **reduce:** per channel, one of sum, average, weighted average (a channel holds the weight), min, max, or a custom fold. Plus argmin/argmax with a payload (done: §14.1 Payload folds).
  - **store or epilogue:** WGSL per emitted level.
- **Level selection:** which levels to emit, and where: texture mips, an atlas, a buffer, or nothing.
- **Edge policy:** `exact`, `box` (area), `zero-weight` (off-image texels contribute nothing; ceil sizes), and `clamp` (FSR/SPD-compatible, for parity tests).
- **Recording:**
  - Into a caller-owned pass (`encodePass`), so one timestamp pair covers the chain.
  - Dispatches only, so a tool can embed the kernel in its own pass order.
- **Scalar:** partial results per workgroup, then a final dispatch with the epilogue. It writes to a buffer or a texel.
- **Compatibility mode:** split the read and write views of one texture across dispatches, or use the buffer path. **Done (2026-11-11)** for `Downsampler` with scratch levels and copy dispatches (§4 Compatibility mode).

The fold contract in §4 (idempotent only) stays for folds. Sums and weighted means become reductions in their own right, each with its own oracle. They are defined at odd sizes only under `box` and `zero-weight`.

### 14.1 Generator contract (M2T; agents build against this)

The fold rule is relaxed (owner, 2026-11-11). **Sums, counts and weighted means are reductions** alongside the idempotent folds of §4, each with its own oracle.

**Implemented (M2T.1, 2026-11-11)** as `Reducer` (`src/webgpu/reducer.ts`), on the tile family of `createKernel` (`src/webgpu/kernel.ts`), with the oracle `reducePyramid` (`src/core/pyramid.ts`). The API is in §5. One generator now has two kernel families:
- **`'exact'`:** mip chains, used by `Downsampler`. Unchanged: its WGSL is byte-identical to before M2T.1 for every format, width, reduction, access and source format (12540 combinations compared).
- **Tile kernels (`'zero-weight'`, `'clamp'`):** used by `Reducer`. One dispatch reduces every level up to the deepest emitted one inside one workgroup tile, at any size.

**Tool bindings live in `@group(1)`.** The generator owns `@group(0)`. A tool passes WGSL declarations for `@group(1)` together with their `GPUBindGroupLayoutEntry[]` (`bindings: { wgsl, entries }`), and supplies the group-1 bind group at dispatch (`bindGroup`). Tool code never collides with ours, and a tool's bind group works with any of our kernels, because layouts with equal entries are compatible.

Group 0 of a tile kernel holds only what it needs:

| Binding | Resource | Present when |
|---|---|---|
| 1 | source texture, `unfilterable-float`, `textureLoad` | no `load` hook |
| 2 … 2+k−1 | write-only storage texture per emitted level, in `levels` order | no `store` hook |
| 8 | uniform `{ size: vec2u }` | a `load` hook and no `size` hook |
| 9, 10 | buffer output: layout uniform (112 bytes) and `array<u32>` storage buffer, instead of 2… | `output: 'buffer'` (§6.1) |

**Hooks** are WGSL strings. Each must define the named function and may define helpers. Identifiers starting with `ds_`, the entry point `main` and the hook names are reserved.
- `fn load( texel: vec2i, layer: u32 ) -> vec4f`: the value of one source texel. Coordinates are always in bounds: the kernel clamps them, and under `'zero-weight'` it then leaves the clamped duplicates out. So a hook can be called more than once for an edge texel, and must not have side effects. Without a hook, the generator reads the source texture, splatting single-channel formats (`sourceFormat`).
  - Exception (M2T.3): under `'clamp'`, when the size is a whole number of source tiles (`max(16, 2^deepest)` per axis), every texel is loaded exactly once, so a hook may write a per-texel output there. The upscaler's shading change writes its luma history this way, at a padded size (Edges below).
- `fn store( level: u32, texel: vec2i, layer: u32, value: vec4f )`: called once per emitted texel inside its level, with `level` counted from the source (1 = half) and `value` normalised. Without a hook, the generator writes storage textures.
  - **Cross-level state (guaranteed, M2T.3).** Levels are emitted shallow to deep, with a workgroup barrier between levels. The workgroup that emits a texel has emitted all of its in-level descendants, and its texels of any level below 1 form an aligned block of at most 16 × 16. So a store hook may keep `var<workgroup>` state from one level for a deeper one, in slot `texel % 16`. The upscaler's shading change combines its 4×4 and 8×8 responses this way. Descendants past their level are not emitted; under `'zero-weight'` they count 0. Tool workgroup memory adds to the tile's: a 6-level kernel already uses the default 16 KiB.
- `fn size() -> vec2u` (addition): the source size, for load-hook kernels. Use it when the size comes from the tool's own uniforms, e.g. with dynamic resolution. Without it, the Reducer keeps a 16-byte uniform per size, cached with the bind group, so two sizes in one submit stay correct.
- `fn finish( value: vec4f, weight: f32 )`: scalar reduction only, run by one invocation. `weight` is the total weight, which is the texel count under `'average'`.

**Per-channel reduction**, `reduction: Reduction | [ op, op, op, op ]`. It is named `reduction`, not `reduce`, to match `Downsampler`. Each `op` is one of:
- `'sum'`, `'average'`, `'min'`, `'max'`
- `'weight'`: this channel carries a weight. Under `'weighted'`, the other channels are `Σ wᵢ·vᵢ / Σ wᵢ` when emitted, and the weight channel is emitted as `Σ wᵢ`.
- `'weighted'`: `Σ wᵢ·vᵢ / Σ wᵢ`, and `0` where `Σ wᵢ = 0` (an all-sky block).
- `'argmin'`, `'argmax'`, `'payload'`: a payload fold (Payload folds below).

Validation: at most one `'weight'` channel, and `'weighted'` needs one. Builtins expand per channel: `'average'`, `'min'` and `'max'` to four of the same, `'minmax'` to min, max, min, max, and the shorthand `'sum'` to four sums. A custom `{ wgsl, kind }` applies to the whole `vec4`:
- `kind: 'fold'` (default) must be idempotent.
- `kind: 'sum'` is combined exactly once per counted texel and emitted as is.

**Wide accumulators (8 channels, 2026-11-11).** `channels: 8` (or eight channel ops) carries two `vec4f` per texel, in `Reducer` and `ScalarReduction` alike.
- **WGSL contract:** the generator declares `struct DsWide { lo: vec4f, hi: vec4f }`, channels 0–3 in `lo` and 4–7 in `hi`. With 8 channels the hooks use it in place of `vec4f`: `fn load( texel, layer ) -> DsWide`, `fn store( level, texel, layer, value: DsWide )`, `fn finish( value: DsWide, weight: f32 )`, and a custom `fn reduce( a: DsWide, b: DsWide ) -> DsWide`. Chosen over two out-parameters or an `array<f32, 8>`: a struct is one value through every hook, the halves stay `vec4f` (so per-half `select`, `min`, `max` and arithmetic stay vector ops), and the shape matches two `rgba` texels or 32 bytes of buffer.
- **Ops:** any per-channel op on any of the 8 channels, still at most one `'weight'`. Builtins and `'minmax'` expand to 8 with `channels: 8`.
- **Sources and outputs:** a texture has at most 4 channels, so an 8-channel `Reducer` needs both a load and a store hook. An 8-channel `ScalarReduction` needs a load hook to read an image; without one it only takes combine-only runs (partials of 32 bytes). Output results are `{ value: DsWide, weight: f32 }`, 48 bytes per layer (`resultBytes`, `scalarResultSize( 8 )`).
- **Cost when unused: none.** Every 4-channel kernel is byte-identical to before (5620 option sets compared).
- **Workgroup memory doubles:** a 5-level tile takes 8 KiB, a 6-level tile 32 KiB, above the default 16 KiB `maxComputeWorkgroupStorageSize`. The Reducer rejects a tile the device can't hold, naming the limit. The scalar's 256 slots take 8 KiB.
- **Measured on the upscaler's shading change** (5 values: 4 sums and a max flag, which used to ride in a sign bit through a custom sum). `generator/upscaler-shading-change` now uses `[ 'sum' ×4, 'max' ×4 ]`; the sign-bit version is `experiment/shading-signbit`. Sign-bit ÷ wide, two same-run comparisons on M5 Pro (load average ~9): 0.93 / 0.99 at 960×540, 1.01 / 1.01 at 1280×720, 0.86 / 0.92 at 1129×635, 1.01 / 1.00 at 1707×960, 1.00 / 1.01 at 1920×1080. So the wide kernel ties from 1280×720 up and costs up to ~8–14 % at the small odd size, where its doubled workgroup traffic is not hidden behind the loads. Against the upscaler's hand-fused pass it is 0.89–1.03×.
- **Oracles:** `reducePyramid( …, { channels: 8 } )`, `scalarReference` and `scalarCombineReference` take 8-channel data (8 floats per texel or partial); core tests check that an 8-channel result equals the two 4-channel halves.

**Payload folds (argmin / argmax, 2026-11-11).** Channel ops `'argmin'` and `'argmax'` mark a key; the `'payload'` channels right after it carry the values of the texel whose key wins. ssgiDev's G-buffer downsample (min depth, the normal of the min-depth texel, mean light) is `[ 'argmin', 'payload' ×4, 'average' ×3 ]` with 8 channels.
- **Rule (decided):** the fold is the lexicographic min of (key, payload₁, payload₂, …) for `'argmin'`; for `'argmax'` the largest key, then the smallest payload. Ties on the key go to the smaller first payload, then the next. That is a total order, so the fold is associative, commutative and idempotent: the result is the same at every tree shape, kernel width, edge policy and in the scalar's different pairing, and the oracles need no model of the GPU's order. Position-based ties are the tool's choice: a texel index as the first payload breaks ties by row-major position, which reproduces ssgiDev's strict-`<` scan exactly (GPU-tested, including its `minD = 1.0` start for all-sky blocks).
- **Validation:** a `'payload'` must follow a key or another payload of it. Several folds can share one value (`[ 'argmin', 'payload', 'argmax', 'payload' ]`), next to sums, averages and weights. Keys and payloads must not be NaN (comparisons with NaN are false, so the winner would depend on the order); −0 and +0 tie, and either may be returned.
- **WGSL:** `ds_winsN( a, b ) -> bool` per fold (key compare, then each payload), and `ds_combine` selects the fold's channels from the winner. Payload and key channels are selections: not zeroed off-image in the scalar, not normalised on emission. Kernels without payload ops are byte-identical.
- **Where:** `Reducer` (texture or hook paths, 4 or 8 channels) and `ScalarReduction` (image and combine-only runs). `Downsampler` rejects them, like other channel ops.
- **Oracles:** `reducePyramid`, `scalarReference` and `scalarCombineReference` apply the same rule (`payloadWins`, `payloadGroups` in core); a core test checks them against a sort of each block. GPU: 4 variants (argmin and argmax with 3 payloads, two folds in one value, a fold next to an average and a min) × 13 shapes × both edges × 4 level sets, keys in {0…3} and the first payload in {0, 1, 2}, so most texels are decided by ties; the ssgiDev G-buffer shape against its own loop; scalar image and combine-only runs. Everything selected matches exactly.

**Accumulation.** Workgroup memory holds unnormalised accumulations: sums, weighted sums `Σ w·v`, min/max, or the custom combine. A level is normalised only when it is emitted.
- The count that divides an `'average'` is analytic, not stored: the in-bounds part of the block under `'zero-weight'`, `4^l` under `'clamp'`.
- Nothing is rounded between levels. So every emitted level is the reduction of its own source block, which is what block statistics need, and texture outputs are rounded once, by the store.
- This differs on purpose from `Downsampler`'s quantize-as-stored (§4 Quantization). The unorm8 integer-unit guarantees therefore apply to `'exact'` chains only.

**Levels:** `levels: number[]`, the source-relative levels to emit, e.g. `[ 2, 3 ]` for 4×4 and 8×8 blocks. Levels must be strictly increasing, from 1 to 6. Levels that aren't emitted are still reduced, but never stored, and get no binding. The tile, and so the workgroup size, follows the deepest emitted level (§4 Tiles).
- **One dispatch, at most 6 levels.** Chaining beyond 6 isn't implemented for tile kernels. A second batch would have to re-read emitted (normalised) values and recover their counts and weights. The full-chain tool shapes are scalars (M2T.2) or `'exact'` chains (`Downsampler`).

**Edges:** `edges` takes one of these values. Output texel x of level l covers the source block `[x·2^l, (x+1)·2^l)` on each axis.
- `'exact'`: the current behaviour, `Downsampler` only. A Reducer rejects it, because it ends a batch at every odd size.
- `'box'` (area weights): built for `Downsampler` (`oddSizes: 'box'`, §2.1), not as a `Reducer` edge policy. `edges` is a string union that `createKernel` dispatches on; a tile `'box'` would need the halo of §2.1 for odd sizes inside a tile, and sums over fractional overlaps (`areaReference` is the oracle).
- `'zero-weight'` (the Reducer's default): out-of-bounds texels contribute nothing; sizes are `⌈n/2^l⌉`; averages divide by the in-bounds count. Every source texel is counted exactly once per level. min/max of it is a conservative Hi-Z with ⌈n/2⌉ sizes.
- `'clamp'`: SPD/FSR-compatible. Source coordinates are clamped to the edge; sizes are `⌊n/2^l⌋`, at least 1, dropping the last odd row or column. Blocks that reach past the image (unit dimensions, deep levels of small images) count the edge texel more than once, exactly as SPD's clamped loads do inside a 64×64 tile. Used for parity fixtures.
  - AMD still writes its 1×1 `FrameInfo` from texel (0,0) of a level that is 0 texels tall at ⌊n/2^l⌋ (1280×720, level 10). Parity there needs the scalar's `finish`, not `store`. M2T.3 does this with a `ScalarReduction` over the top-left `2^(mips−6)` square of the stored mip 6, reading 0 past it (§14).
  - **Clamped loads with ⌈n/2^l⌉ blocks** (pmndrs/upscaler, most hand-written block passes): run `'clamp'` on the size rounded up to whole source tiles, clamp to the real size in the load hook, and bound the store hook to the real outputs. No option was added for it: it costs nothing measurable, and one tool needs it so far.

`'zero-weight'` and `'clamp'` keep every level in one tile, so batches never end at odd sizes.

**Where `'zero-weight'` outputs go (decided).** ⌈n/2⌉ sizes aren't WebGPU mip sizes (⌊n/2⌋).
- Output textures must be at least the level's size, and the Reducer checks this. A mip of the source-sized texture is one texel short at odd sizes, so it is rejected with that explanation, rather than silently losing the last row or column.
- Use separate textures, or an atlas or buffer through the `store` hook (the denoiser's layout).
- `'clamp'` sizes equal mip sizes, so FSR-style outputs can be the mips of one half-resolution texture.
- Built-in buffer output (§6.1): `output: 'buffer'` packs emitted levels into a buffer the Reducer owns and copies them into output textures (any 32-bit-word format, including rgb9e5ufloat). Writing into a tool's own buffer is still the store hook's job.

**Recording:**
- `encode( encoder, dispatch )` opens its own pass.
- `encodePass( pass, dispatch )` records into a caller-owned pass.
- For tools that dispatch the kernel themselves: `prepare( dispatch )` (pipeline, group 0, the dispatch's group 1, workgroups, and with output `'buffer'` the copies), or the pieces: `wgsl`, `entryPoint`, `layout` (the `GPUPipelineLayout`), `bindGroupLayouts`, `pipeline`, `bindGroup( dispatch )` (group 0) and `workgroups( width, height, layerCount )`.
- **Dispatch shape (2026-11-13):** `{ source?, outputs?, width?, height?, layerCount?, bindGroup? }`. `source` and each output are a texture or `{ texture, mipLevel }` with the shared source checks (§5 Sources); `layerCount` counts layers from 0 (it was `layers`, which now always means `[ first, count ]`).
- **`prepare` means the same on all three classes (2026-11-13):** what `encodePass` (or `encode`) would record, validated (it throws with the same reason) and cached, for a tool that dispatches it itself. `Downsampler.prepare( texture, options )` returns one `PreparedBatch` per dispatch, `Reducer.prepare( dispatch )` a `PreparedReducer`, `ScalarReduction.prepare( run )` a `PreparedScalar` (two pipelines).
- Pipelines are cached per device by WGSL and layouts, so equal Reducers share one. Group-0 bind groups are cached per size and textures (the 16 most recent). `release( texture? )` drops entries (all, or those using a texture) and destroys the buffers they own (§4 Caching).

**Verification** (§7): every emitted level of every kernel width (1, 4, 5, 6) is checked against `reducePyramid`, which computes each level straight from the source in f64. The matrix covers both edge policies; channel ops including weights with sky texels; a custom fold and a custom sum; texture and hook paths; 17 shapes (unit, odd, 1D, partial tiles, layers, `2d-array`, 1920×1080); and unemitted levels left poisoned. Two tool shapes (FSR shading change, denoiser weighted mean), the cross-level store state above (levels [ 2, 3 ], [ 1, 4 ], [ 2, 5 ], both edge policies, 4 shapes) and API checks run alongside. A core test checks the oracle against iterated 2×2 reductions of the padded image.

**Scalar** (`reduceToScalar`): one dispatch writes per-workgroup partials into a storage buffer; a second, single-workgroup dispatch combines them and calls `finish`. That is 2 dispatches at any size, and every texel is counted. A combine-only run (`partials`) takes partials a tool wrote, e.g. a Reducer's deepest level from its store hook, and records the second dispatch alone (§4 Scalar reductions).
