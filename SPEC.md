# downsampler — core specification

Status: milestone 2 in progress, refocused on our tools (M2T, §11). M2.1 (reductions, load path), M2.2 (separate source, float32 formats), format steps 1–2 and the GI formats (buffer output path, rgb9e5ufloat, rg11b10ufloat) are done. This document is the source of truth for scope, design decisions, correctness requirements and the remaining plan. Update it when a decision changes.

## 1. Purpose

Generate texture pyramids on the GPU: mip chains first, then min/max reduction pyramids (Hi-Z), custom reductions and related downsampling chains. WebGPU first and engine-agnostic. Thin adapters (three.js first) sit on top.

Feature target: parity with AMD FidelityFX SPD, implemented independently in WGSL. SPD is the reference for what a downsampler should offer, not code to port. See §6.

**Primary users (decided 2026-11-11):** our own tools: the upscaler (pmndrs/upscaler, based on FSR 3), the denoiser and GI. three.js mipmaps benefit from the same work but don't drive it. Those tools use SPD for fused passes: a custom per-texel load from several inputs, a few levels reduced inside a tile, and a custom store, all in the tool's own dispatch. So the main product is a WGSL reduction generator that tools embed or dispatch themselves; `Downsampler` (mip chains, Hi-Z) is one user of it. Speed at render resolutions, which are rarely powers of two, is a first-class goal. A single dispatch is not, because WGSL can't make SPD's handoff safe and it measured slower than two dispatches (§2.2). Chains of up to 6 levels already take one dispatch. Full chains take two at powers of two and under `oddSizes: 'spd'`, and 5–9 under `'exact'` at render sizes (§2.1).

## 2. Two decisions to know up front

### 2.1 Odd sizes: `'exact'` (default), `'spd'` (opt-in, measured), `'box'` (planned)

A 2:1 reduction is only exact when a dimension halves cleanly. For odd, non-unit sizes (decided 2026-11-10; `'spd'` built and measured 2026-11-11, M2T.4):

- **`'exact'` (default, implemented).** An odd size ends the current batch, and the next level is re-sampled from the stored level with normalized-coordinate bilinear filtering.
  - Results match render-pass and WebGL mipmaps (the CPU reference in §7), so engines can switch without visual change.
  - Min/max use the conservative `2x … 2x+2` footprint, so Hi-Z is correct.
  - Cost: more dispatches at render sizes, 5–9 instead of 2 (1920×1080: 7). The exact tail kernel (`new Downsampler( device, { tail: true } )`, opt-in, M2T.4) cuts that to 3–6 with the same rule, but recovers little time (measurements below).
- **`'box'` (planned, opt-in).** Area-weighted odd sizes, as in nvpro_pyramid: per axis, 3 taps with weights `(n−x, n, x+1)/(2n+1)` for size `2n+1`. Every source texel contributes in proportion to its overlap, which fixes the unrepresentative 1×1 of render-pass mipmaps on sizes like 31×31.
  - Same dispatch structure and footprint as the folds, so no extra cost.
  - Recommended for quality-sensitive content: GI, irradiance and roughness chains, UI.
  - Needs its own oracle.
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
3. The tail kernel stays opt-in (`tail: false` by default). Make it the default only if an idle-machine run shows ≥ 5 % at render sizes; on this loaded machine it measured 0.83–1.07×. It composes with both output paths and with compatibility mode, where devices with fewer than 256 invocations per workgroup cap tail tiles at 8×8.

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

- **First output:** a hardware bilinear sample at each texel centre, using normalized coordinates. That is an exact 2×2 box for even sizes. For odd sizes it reproduces render-pass and WebGL mipmaps, which is **not** area-correct: 3→1 takes only the centre texel, so a 31×31 texture can reduce to an unrepresentative 1×1 (`greggman`'s example in gpuweb#5253). Area-weighted odd sizes are a candidate opt-in (§13).
- **Later levels in the batch:** exact 2:1 reductions in workgroup memory, with clamped reads for unit dimensions and partial tiles.
- **Batch boundaries:** a batch ends at `levelsPerDispatch`, or where the next reduction is inexact under `'exact'`. With `tail: true`, a first output of at most 32×32 (8×8 below 256 invocations per workgroup) starts a tail batch that finishes the chain (up to `levelsPerDispatch` levels) in one workgroup by the same rule. Under `'spd'`, batches end only at `levelsPerDispatch` (§2.1).
- **Layers:** workgroup z selects the array layer, so all layers go in one dispatch.

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
- **Size rule:** destination level `baseMipLevel` must be the same size as the source level (a converting copy) or half of it (one reduction step). Every destination level in range is written, including `baseMipLevel`. `mipLevelCount` counts destination levels in both modes.
- **Planning:** the first batch reads the source and every later batch reads the destination (`Batch.sourceMipLevel === null` marks the source batch). The copy is a runtime branch on equal sizes, uniform across the dispatch, so it needs no pipeline variant.
- **Source formats:** anything float-sampleable, including depth (through the depth aspect, bound as `unfilterable-float`). Depth and 32-bit float sources always load.
- **Splat:** single-channel sources (depth, `r*`) are read as `(v, v, v, v)`, so `'minmax'` of depth gives `(d, d)` with no special case.
- **unorm8 rounding on load** applies only when the source is unorm8 itself. Rounding a 16- or 32-bit source to 8 bits before reducing would round twice.
- **Caching:** a plan is cached per destination and option set, holding only the most recent source texture. Recreating a depth buffer on resize therefore replaces the entry instead of growing the cache.

### Source access

- **`'sample'`:** one `textureSampleLevel`. `'average'` only, filterable formats only.
- **`'load'`:** `textureLoad` of the footprint, bound as `unfilterable-float`. Required for folds, depth, and float32 formats without `float32-filterable`. For `'average'` it computes the bilinear weights in the shader, without the sampler's fixed-point weight rounding.
  - **Integer weights:** the position `(x + 0.5)·S/n − 0.5` is computed as `((2x+1)·S − n) / 2n`, giving an exact base texel and a weight of `remainder / 2n`. GPU float division isn't correctly rounded (WGSL allows 2.5 ulp). A float version computed 2.9999998 for an exact 3.0, so `floor` picked the wrong pair.
  - **unorm8:** texels are rounded back to exact integers on load, so even-size averages are exact.
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
- `Downsampler.requiredLimits(adapter)` gives apps the limits to request.
- Around 94 % of devices report at least 8 (web3dsurvey).
- Storage texture views must cover exactly one mip level ([WebGPU spec](https://gpuweb.github.io/gpuweb/)), so each output level needs its own binding.

### Quantization

Every intermediate level is quantized exactly as stored, so a chain equals level-by-level generation.

- **rgba8:** values stay in integer 0–255 units in workgroup memory. Sums are exact and ties round to even. With `k/255` floats, float error turned ties into round-ups. A 1×128 tail drifted 3 steps by mip 5: half of all 1D texels are ties.
- **float16 formats:** values are rounded to nearest even in the shader with integer math (`float16Wgsl`, `ds_round_f16`; mirrors core `toHalf` bit for bit), then stored as exact halves, so no device conversion rounds anything. `f16Rounding: 'device'` (Downsampler and Reducer options) uses `pack2x16float` / `unpack2x16float` instead: cheaper, but D3D12 truncates there (§ Findings, 2026-11-12).
- **rgb9e5ufloat, rg11b10ufloat:** whole texels go through our encoder and back (`ds_pack9e5` / `ds_pack11_11_10` in WGSL, `quantizeTexel` in core; rules in §6.1); alpha reads 1. The rg11b10 storage path stores already-quantized values, so the GPU's own f32 → 11/10-bit conversion never rounds anything.

### Caching

- Pipelines are cached per `Downsampler` (per device) and per reduction; bind group layouts per format, view dimension, width and access.
- Bind groups are cached per texture and options in a `WeakMap`.
- `release(texture)` drops a texture's entries early.

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
- **Ops:** per channel `'sum' | 'average' | 'min' | 'max' | 'weight' | 'weighted'` (§14.1). At most one `'weight'` channel; `'weighted'` needs one. Emitted: sums, averages (÷ texel count), weighted means (÷ Σ w, and 0 when Σ w = 0), Σ w, min, max. The weight passed on is Σ w with a weight channel, else the texel count. Custom `{ wgsl, kind }` reductions aren't offered yet.
- **Output:** `finish( value: vec4f, weight: f32 )`, run once per layer by invocation 0 of the combine workgroup, with an optional third parameter `layer: u32` (an extension of §14.1 for multi-layer runs). Without `finish`, results go to `output: { buffer, offset }` as `struct { value: vec4f, weight: f32 }`, 32 bytes per layer, at a 256-byte-aligned offset.
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
- **Caching:** pipelines per source view dimension and splat. Per run shape, a plan holds a uniform buffer (size, first layer, partial count) and the partials buffer, cached per source texture, or else per output buffer or reduction, with at most 4 shapes per owner (resizes evict the oldest). `release( resource )` drops them.
- **Combine-only runs (2026-11-11).** A run with `partials: { buffer, offset?, count }` records the combine dispatch alone, over `count` accumulations per layer that the tool wrote (one `vec4f` each, layer after layer). The usual source is a `Reducer`'s deepest level through its store hook, so "Reducer levels + a scalar of the same source" takes 2 dispatches with one read of the source (3 before).
  - **Partials are accumulations:** sum-like channels hold sums (a `'weighted'` channel Σ w·v, the `'weight'` channel Σ w), min/max channels the min/max. A Reducer emits these with `'sum'`, `'min'`, `'max'` channels; its `'average'` and `'weighted'` channels are normalised, so a tool feeding a scalar uses `'sum'` and forms w·v in its load hook.
  - **`texels`** is the texel count behind the partials: the divisor of `'average'` channels and the weight passed on without a `'weight'` channel. A run needs it in those cases.
  - **Group 0** of the combine-only pipeline has no source texture (`combineKernel()`). The tool's buffer is bound read-write at binding 1, so the run's group 1 must not bind it too (aliased writable bindings fail validation). In the FSR fixture the Reducer's group 1 holds the partials and the scalar's is FSR's own.
  - **Offset** must be a multiple of `minStorageBufferOffsetAlignment`. The WGSL is unchanged: the combine entry point of the same module.
  - **Oracle:** `scalarCombineReference( partials, count, layers, reduce, { texels, partialDepth, magnitude } )` pairs in the combine pass's order; the bound uses depth `partialDepth` + ⌈N/256⌉ − 1 + 8 (a Reducer's level l adds 2l roundings).
- **Shared tool plumbing (2026-11-11):** `src/webgpu/tool.ts` holds `ToolBindings`, hook validation (`checkHook`, `hookArity`), the tool's group-1 layout, the `[ group 0, group 1 ]` pipeline layout and the bind-group checks, for `Reducer` and `ScalarReduction` both. WGSL and layouts were unchanged by the move (5620 option sets compared byte for byte).

## 5. API

### Implemented

```ts
import { Downsampler } from 'downsampler/webgpu';

const device = await adapter.requestDevice( { requiredLimits: Downsampler.requiredLimits( adapter ) } );
const downsampler = new Downsampler( device );          // { levelsPerDispatch?, averageAccess?, output?: 'auto' | 'storage' | 'buffer' (§6.1), compatibility?: 'auto' | boolean (§4), tail? (§2.1) }
downsampler.supports( texture );                         // boolean
downsampler.unsupportedReason( texture );                // string | null
downsampler.outputPath( texture );                       // 'storage' | 'buffer' | null (§6.1 Buffer output path)
downsampler.encode( encoder, texture, {
	baseMipLevel, mipLevelCount, oddSizes: 'exact' | 'spd',   // 'spd': SPD semantics, ⌈levels/6⌉ dispatches (§2.1)
	reduction: 'average' | 'min' | 'max' | 'minmax' | { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { … }' },
} );
downsampler.generate( texture );                         // encode + submit
downsampler.encodePass( pass, texture, options );        // into a compute pass you own (one timestamp pair; storage path only)
downsampler.prepare( texture );                          // planned dispatches (inspection/testing)

import { generateMipmaps, requiredLimits } from 'downsampler/three';
generateMipmaps( renderer, texture );                    // false if not allocated/supported

import { planBatches, reduceReference, footprint, spdReference, maxViolations } from 'downsampler'; // backend-neutral core
```

```ts
// Separate source: depth → Hi-Z. Level 0 the same size (copy) or half (reduce).
downsampler.encode( encoder, hiZ /* rg32float */, { source: { texture: depthTexture, mipLevel: 0 }, reduction: 'minmax' } );
downsampler.sourceUnsupportedReason( destination, source, baseMipLevel ); // string | null
```

```ts
// Scalar reductions (§4 Scalar reductions): one vec4f + weight per layer, 2 dispatches.
import { ScalarReduction } from 'downsampler/webgpu';
const range = new ScalarReduction( device, { reduce: [ 'min', 'max', 'min', 'max' ] } ); // or 'average', [ 'weighted', …, 'weight' ]
range.encode( encoder, { texture: depth, output: { buffer, offset: 0 } } ); // { value: vec4f, weight: f32 } per layer
const exposure = new ScalarReduction( device, {
	bindings: { wgsl: '@group(1) @binding(0) var colour: texture_2d<f32>; …', entries }, // the tool's group 1
	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f { … log( max( luma, 6.1e-5 ) ) … }',
	finish: 'fn finish( value: vec4f, weight: f32 ) { … temporal smoothing, store … }',
} );
exposure.encodePass( pass, { size: [ width, height ], bindGroup } ); // also { layers: [ first, count ], mipLevel }
exposure.prepare( run );        // pipelines, group 0 bind group, workgroup counts (dispatch it yourself)
exposure.kernel();              // { wgsl, layout, partials, combine }
exposure.unsupportedReason( run );

// Combine only: partials a tool wrote (e.g. a Reducer's level 6 via its store hook), one dispatch.
const mean = new ScalarReduction( device, { reduce: [ 'weighted', 'weighted', 'max', 'weight' ], finish, bindings } );
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
// Self-dispatch: shadingChange.wgsl, .entryPoint, .layout, .pipeline, .bindGroup( dispatch ), .workgroups( w, h, layers ).
// 8 channels (§14.1 Wide accumulators): reduction: [ 'sum', 'sum', 'sum', 'sum', 'max', … ] (or channels: 8);
// hooks then use DsWide { lo: vec4f, hi: vec4f }: fn load( … ) -> DsWide, fn store( …, value: DsWide ).
// Payload folds (§14.1): [ 'argmin', 'payload', 'payload', 'payload' ] = min depth + the normal of that texel;
// ties go to the smaller payload (put a texel index first for row-major ties). Also in ScalarReduction.
// Without a load hook: { source: texture | { texture, mipLevel } }. Output textures must be at least each level's size.

import { reducePyramid, levelSize } from 'downsampler';  // the CPU oracle and level sizes
```

The destination must have `TEXTURE_BINDING` usage, plus `STORAGE_BINDING` (storage path) or `COPY_DST` (buffer path, §6.1), and a format from `formats`: `rgba8unorm`, `rgba16float`, `rgba8snorm`, `r32float`, `rg32float`, `rgba32float`, `rgb9e5ufloat` (buffer path only), plus `bgra8unorm` (storage needs `bgra8unorm-storage`), `rg11b10ufloat` (storage needs `texture-formats-tier1`; else the buffer path) and `r8unorm`, `rg8unorm`, `r8snorm`, `rg8snorm`, `r16float`, `rg16float` and the 16-bit normalized formats (need `texture-formats-tier1`). `unsupportedReason` names a missing feature or usage. Its dimension must be `2d`, with `textureBindingViewDimension` either `2d` or `2d-array`.

```ts
// GI formats (§6.1): rgb9e5ufloat is never a storage texture, so it takes the buffer path.
const irradiance = device.createTexture( { format: 'rgb9e5ufloat', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, … } );
downsampler.generate( irradiance );                      // 6 levels per dispatch at any storage texture limit
new Reducer( device, { levels: [ 1, 2, 3 ], format: 'rgb9e5ufloat', … } ); // output 'buffer' by default for it
reducer.encodePass( pass, dispatch ); pass.end(); reducer.encodeCopies( encoder, dispatch ); // buffer outputs
```

### Planned

```ts
downsampler.encode( encoder, destination, {
	layers: [ first, count ],
	rect: { x, y, width, height },      // SPD rectInfo
	oddSizes: 'exact' | 'spd' | 'box',
	scratch: GPUTexture,                 // optional reusable scratch for sRGB destinations
} );
```

## 6. SPD feature parity

| SPD feature | Status | Plan / notes |
|---|---|---|
| Up to 12 levels from ≤4096² | ✅ | any size, by chaining (2 dispatches up to 4096²) |
| Single dispatch | ❌ by design | §2.2; a measurement-only atomic probe (`test/bench/atomic-spd.js`) lost to two dispatches at render sizes and 1024², tied at 4096² |
| Custom reduction (`SpdReduce4`) | ✅ | average/min/max/minmax + custom WGSL `reduce(a, b)` fold (§4 Reductions). A weighted custom reduction (SPD's `SpdReduce4` used for averages) isn't offered under `'exact'`, where it has no defined odd-size meaning; tile kernels (`Reducer`, §14.1) add sums, weighted means and custom sums under `'zero-weight'` and `'clamp'`. Transforms around an average (premultiply, sRGB) are planned as load/store hooks with §6.1. |
| Load vs. linear-sampler source | ✅ | `textureLoad` path (§4 Source access). `textureGather` for single-channel Hi-Z is an M3 candidate: one gather replaces four loads. |
| Separate source/destination | ✅ | `source: { texture, mipLevel }`, same size (copy) or half (§4 Separate source). Depth sources supported. |
| Texture arrays | ✅ all layers | add a layer range; cube maps as 6 layers (2d-array view; not available in compatibility mode) |
| Sub-rectangle (`rectInfo`) | ❌ | offset/size uniform; output tile offset. SPD snaps the rect to 64-texel source tiles and recomputes levels below 64×64 for the whole texture: decide whether to match |
| Custom load/store (format conversion) | 5 destination formats; any float/depth source | see §6.1; load/store hooks (premultiply, sRGB) planned with M2.3 |
| Global atomic counter, `globallycoherent` mip 6 | ❌ by design | No counter buffer to allocate, reset or alias; the dispatch boundary replaces it (§2.2) |
| Mip count (`mips`, max 12) | ✅ no cap | **Off by one for SPD users:** in place, `mipLevelCount` includes the source level, so SPD `mips = N` is `mipLevelCount: N + 1` |
| Levels written | ✅ | In place, as SPD: `baseMipLevel` is read and later levels written. With `source`, destination `baseMipLevel` is written at the source size (copy) or half; SPD always halves |
| sRGB (sample sRGB view, write UNORM UAV) | planned | WebGPU forbids sRGB storage even through a view, so: scratch copy (§6.1) |
| Intermediate precision | differs by design | SPD keeps unrounded values in LDS between levels; we round each level as stored, so a chain equals level-by-level generation and 8-bit results won't match SPD bit for bit (§4 Quantization) |
| LDS layout (one float4 array vs per channel) | ❌ | M3 candidate; measure first |
| Packed fp16 math | ❌ | optional `shader-f16`; keep only if measurably faster |
| Wave/subgroup ops | ❌ | optional `subgroups` (Chrome, Safari 27.3; workgroup-memory fallback for Firefox). WGSL defines no quad↔`local_invocation_index` mapping in compute, so derive positions from `subgroup_id`/`subgroup_invocation_id` and use `subgroupShuffleXor`. Upper-bound test first (§13 Subgroups); current kernels are near bandwidth-bound at ≥2048². |
| Odd sizes | ✅ `'exact'`, `'spd'` | SPD drops the last row/column; `'exact'` (default) matches render passes. `oddSizes: 'spd'` reproduces SPD's semantics in ⌈levels/6⌉ dispatches (M2T.4, §2.1). Area-weighted `'box'` planned |

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
| Packed | rgb10a2unorm | tier1 | per-channel scale `(1023, 1023, 1023, 3)` |
| Packed float | rg11b10ufloat | tier1 for storage; the buffer path without it | ✅ (M2.3.6) `rg11b10ufloat`: unsigned floats, 5-bit exponent, 6/6/5-bit mantissas; round to nearest even, negatives and −Inf → 0, finite overflow → the largest finite value (65024 / 64512), +Inf and NaN kept. No WGSL pack builtin, so `ds_ufloat` encodes in the shader and `toUfloat` in core. Storage path with tier1, buffer path without it (or without STORAGE_BINDING usage); both quantize with our encoder, so they write identical bits |
| Integer | r32uint, rg32uint, rgba32uint, rgba8uint, rgba16uint, r32sint, rgba8sint (core); r8/r16/rg8/rg16 uint/sint (tier1) | mixed | separate `u32`/`i32` kernel (`texture_2d<u32>`, `vec4u`). Folds only at first (min, max, bitwise custom: ID, visibility and occupancy pyramids); `'average'` would need a declared rounding rule |
| sRGB | rgba8unorm-srgb, bgra8unorm-srgb | **never** (even with a non-sRGB view format) | reduce linear values, then write sRGB-encoded values to an rgba8unorm (or bgra8unorm, which needs `bgra8unorm-storage`) scratch and `copyTextureToTexture` each level (verified allowed: formats differing only in sRGB-ness). Needs load/store transform hooks; quantize-as-stored is `decode(round(encode(v)·255)/255)` |
| Shared exponent | rgb9e5ufloat (32-bit HDR, common for GI and irradiance) | **never** storage or renderable, so render-pass mipmaps can't produce it either | ✅ (M2.3.4) **buffer output path** (below): pack in the shader, `copyBufferToTexture` into the level. Filterable, so later batches sample it as usual |
| Out of scope | BC/ETC2/ASTC (generate before compression); depth as destination (use r32float, §4 Separate source) | — | — |

**Buffer output path** (built 2026-11-11, M2.3.4). WebGPU allows `copyBufferToTexture` into formats that are neither storage nor renderable, so a kernel can write texels to a storage buffer and copy them into each level.
- **Layout** (`outputBufferLayout` in core): each level of a batch, and each layer of a level, at its own offset of one buffer, with rows padded to 256 bytes and level offsets on 256 bytes. The kernel finds it in a 112-byte uniform (binding 9: first-output size, then per level its offset and row stride in u32 words and its rows per layer) and writes packed words to binding 10 with `ds_put( level, texel, layer, value )` (`bufferOutputWgsl`). That is a single binding, so this path isn't bound by `maxStorageTexturesPerShaderStage`: 6 levels per dispatch even at the default limit of 4 (tested on a default-limit device).
- **Recording:** after each batch's pass, `encode` records one `copyBufferToTexture` per level (all layers in one copy). The next batch reads the destination texture as usual. One scratch buffer, sized for the largest batch, serves every batch of a plan: commands in an encoder are ordered, so each batch's copies finish before the next dispatch writes the buffer. `release( texture )` destroys it. `encodePass` throws on this path: a compute pass can't record copies, and later batches need them.
- **Which path (`DownsamplerOptions.output`):** `'auto'` (default) writes storage textures whenever the device can write the format there (the texture then needs STORAGE_BINDING; decided 2026-11-12), and takes the buffer path only for formats it can't: rgb9e5ufloat, rg11b10ufloat without tier1, rg32float in compatibility mode. `'storage'` or `'buffer'` forces one; `outputPath( texture )` says which applies. On the buffer path `levelsPerDispatch` defaults to 6 at any storage texture limit (4 where workgroups are limited to 128 invocations, as in compatibility mode by default).
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
  - An alternative to the sRGB scratch texture: `copyBufferToTexture` into rgba8unorm-srgb is allowed (not built).
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
3. rgb10a2unorm
4. buffer output path + rgb9e5ufloat ✅
5. sRGB + transform hooks (the three.js PR needs this), choosing between scratch texture and buffer output by measurement
6. rg11b10ufloat ✅
7. integers

## 7. Correctness specification

**Oracle.** `reduceReference` is a normalized-coordinate bilinear sample at each output texel centre with clamped edges, quantized to the storage format. Each GPU level is checked against the reference computed from the GPU's own previous level, so permitted rounding does not accumulate.

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
- **Formats:** all supported destination formats. One 4-channel format per quantization family (rgba8unorm, rgba8snorm, rgba16unorm, rgba16snorm, rgba16float, rgba32float) runs every shape. The rest of each family runs a representative subset: unit and odd sizes, 1D tails, partial tiles, layers, both view dimensions, and 512×32 (multi-batch at every width). The untrimmed matrix passed once with every format on every shape (10187 cases, 2026-11-10). float32 formats skip the 'average (load)' variant, since they always load. Formats whose feature the device lacks are reported as skipped (`validation.skipped`), never silently passed. The suite requests `texture-formats-tier1` and `bgra8unorm-storage` when available, but not `float32-filterable`, so float32 keeps exercising the load path.
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

- **Packed float formats and the buffer path (§6.1):** rgb9e5ufloat and rg11b10ufloat are families of their own and run every shape: rgb9e5 on the buffer path, rg11b10 on the storage path with tier1 and on the buffer path (subset shapes; every shape without tier1). The buffer path runs widths 1, 4, 5 and 6 on any device, and the subset shapes for each family's 4-channel format plus bgra8unorm, rg16float, r32float and rg32float (not with `--quick`, where only the packed formats take the buffer path, on the subset shapes at widths 1, 4 and 6, and the extra checks below use smaller shapes: the packed encoders are slow on SwiftShader, and quick runs with the full additions hit the runner's 30-minute protocol timeout before CI split the suite). In the `--only` partition, the matrix, codec and Downsampler buffer-path checks run in `downsampler`, the Reducer buffer-output and float16-rounding checks in `reducer`. In compatibility mode rg32float has no storage, so it runs the buffer path's subset instead. Fixtures pack r, g, b with the core encoder: HDR values in [−4, 12) mixed with channels spread over 2⁻¹⁴ … 2¹⁰, so texels mix small and large channels. Tolerance on a batch's first output: two steps of the stored value (rgb9e5: of the texel's shared exponent). In-tile levels are exact, against `reduceReference( …, { pairing: 'float32' } )`, which sums the 2×2 block in float32 in the kernel's order (packed sums aren't always exact in f32). Plus: the WGSL codecs against core bit for bit (every 11- and 10-bit code decoded; every rounding midpoint and its neighbouring floats, specials and 20000 random values encoded; rgb9e5 decoded for every exponent × mantissa and encoded at bump, tie, power-of-two and clamp boundaries plus 30000 random texels); buffer path bit-identical to storage path (rgba16float, rg11b10ufloat, rgba8unorm, rgba32float; average and max); a default-limit device without features (6-level batches, rg11b10 falls back to the buffer path); separate sources into both formats; `Reducer` buffer outputs against `reducePyramid` and bit-identical to its storage outputs (rgba32float; rg11b10ufloat), `encodePass` + `encodeCopies` equal to `encode`; API reasons. Core tests: decode → encode identity for every 11-/10-bit code, value identity for every rgb9e5 (exponent, mantissa, mantissa) and 300000 random codes, every 11-/10-bit rounding boundary, and random values against independent oracles (nearest-value search; a log2-free restatement of the extension).

New features add their own oracle first: each new format family, sRGB, rect, layer range, and `'box'`.

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
   - The first PR adds sRGB (most three.js color textures) and cube maps. Hi-Z follows separately.
   - Keep this repo's GPU suite as the shared oracle, so the vendored copy can't drift.

## 11. Milestones

Build first, benchmark once coverage is broad (decided 2026-11-10). Live status is in [PROGRESS.md](PROGRESS.md).

1. **M1 (done):** port the kernel, Downsampler, three adapter, core unit tests, GPU suite, device lab, benchmark and CI.
2. **M2 (features).** Done: load path and reductions; separate source (Hi-Z) with the float32 formats; formats steps 1–2 (fewer-channel unorm8/float16, bgra8unorm, snorm8, unorm16, snorm16).
   **M2T (tools first, 2026-11-11)**, in this order (details in §14 once the survey lands):
   1. WGSL reduction generator: tool-supplied `load` with its own bindings, per-channel reductions including sums, choice of emitted levels, outputs (texture, buffer, WGSL `store` hook), edge policy. Done: `Reducer` (§14.1), with buffer output into textures (§6.1, `output: 'buffer'`).
   2. Reduce-to-scalar in 2 dispatches with an epilogue hook.
   3. Proof on the upscaler's FSR 3.1 passes as bench fixtures, against an oracle, timed on M5 and by the owner on RTX 3060 / Intel UHD. Done on M5 (§14, M2T.3); RTX 3060 / Intel UHD pending.
   4. An SPD-style mode, built at least as a measured comparison that justifies, or not, its not being the primary method. Done (2026-11-11): `oddSizes: 'spd'` (opt-in, 'average' only), the exact tail kernel (opt-in) and an atomic single-dispatch probe (test-only), measured at render sizes; `'exact'` stays the default (§2.1, §2.2).
   Then: buffer output + rgb9e5ufloat and rg11b10ufloat (GI formats; done 2026-11-11, §6.1); layer range + rect; `'box'`, power-of-two destination from a separate source, and the tail kernel's default (idle re-measure, §2.1).
   Deferred, three.js-driven: rgb10a2unorm, sRGB, integer formats.
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

- Publish as bare `downsampler` or `@pmndrs/downsampler`? (`package.json` currently uses `downsampler`.)
- ~~Buffer output path (§6.1): should `output: 'auto'` take the buffer path for textures without STORAGE_BINDING?~~ Decided (owner, 2026-11-12): no. `'auto'` takes it only for formats the device can't store; a missing STORAGE_BINDING usage is an error, and the buffer path for storable formats is opt-in (`output: 'buffer'`). ~~Should `Reducer` round rgba16float outputs before the store?~~ Yes: it quantizes with `pack2x16float` first, like `Downsampler`.
- Cube maps in compatibility mode: they can't be viewed as 2d-array, so either per-face views or the render fallback. (2026-11-11: per-face views don't help either; compatibility mode binds whole textures only, and a `'cube'` texture has no storage view. A 6-layer `'2d-array'` texture works, which is what reflections bakes into, then copies to a cube; §4 Compatibility mode.)
- `'box'` for `'minmax'` and other folds is meaningless (they already use the full footprint). Should `'box'` simply be ignored there, or rejected?

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
4. **Area-weighted odd sizes** (candidate opt-in, e.g. `oddSizes: 'box'`). This is the quality fix for the 31×31 case, and a defined meaning for weighted reductions at odd sizes. It needs its own oracle.
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
| Capsaicin (reference) | `GPUMip`: area-weighted odd sizes (nvpro weights), depth min/max `2x…2x+2`, sRGB re-encode, then SPD once the size is a power of two; HZB with a min-reduction sampler (WebGPU has none); `GPUReduce` 1D sum/min/max/product | chain, Hi-Z, scalar | D3D12 |

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
- `'box'` (area weights, §2.1): not implemented, and not blocked. `edges` is a string union that `createKernel` dispatches on; `'box'` would add its own taps and oracle.
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
- For tools that dispatch the kernel themselves: `wgsl`, `entryPoint`, `layout` (the `GPUPipelineLayout`), `bindGroupLayouts`, `pipeline`, `bindGroup( dispatch )` (group 0) and `workgroups( width, height, layers )`.
- Pipelines are cached per device by WGSL and layouts, so equal Reducers share one. Group-0 bind groups are cached per size and textures (the 16 most recent).

**Verification** (§7): every emitted level of every kernel width (1, 4, 5, 6) is checked against `reducePyramid`, which computes each level straight from the source in f64. The matrix covers both edge policies; channel ops including weights with sky texels; a custom fold and a custom sum; texture and hook paths; 17 shapes (unit, odd, 1D, partial tiles, layers, `2d-array`, 1920×1080); and unemitted levels left poisoned. Two tool shapes (FSR shading change, denoiser weighted mean), the cross-level store state above (levels [ 2, 3 ], [ 1, 4 ], [ 2, 5 ], both edge policies, 4 shapes) and API checks run alongside. A core test checks the oracle against iterated 2×2 reductions of the padded image.

**Scalar** (`reduceToScalar`): one dispatch writes per-workgroup partials into a storage buffer; a second, single-workgroup dispatch combines them and calls `finish`. That is 2 dispatches at any size, and every texel is counted. A combine-only run (`partials`) takes partials a tool wrote, e.g. a Reducer's deepest level from its store hook, and records the second dispatch alone (§4 Scalar reductions).
