# downsampler — core specification

Status: milestone 2 in progress, refocused on our tools (M2T, §11). M2.1 (reductions, load path), M2.2 (separate source, float32 formats) and format steps 1–2 are done. This document is the source of truth for scope, design decisions, correctness requirements and the remaining plan. Update it when a decision changes.

## 1. Purpose

Generate texture pyramids on the GPU: mip chains first, then min/max reduction pyramids (Hi-Z), custom reductions and related downsampling chains. WebGPU first and engine-agnostic. Thin adapters (three.js first) sit on top.

Feature target: parity with AMD FidelityFX SPD, implemented independently in WGSL. SPD is the reference for what a downsampler should offer, not code to port. See §6.

**Primary users (decided 2026-11-11):** our own tools: the upscaler (pmndrs/upscaler, based on FSR 3), the denoiser and GI. three.js mipmaps benefit from the same work but don't drive it. Those tools use SPD for fused passes: a custom per-texel load from several inputs, a few levels reduced inside a tile, and a custom store, all in the tool's own dispatch. So the main product is a WGSL reduction generator that tools embed or dispatch themselves; `Downsampler` (mip chains, Hi-Z) is one user of it. Speed at render resolutions, which are rarely powers of two, is a first-class goal. A single dispatch is not, because WGSL can't make SPD's handoff safe (§2.2): chains of up to 6 levels already take one dispatch, and full chains take two.

## 2. Two decisions to know up front

### 2.1 Odd sizes: `'exact'` (default), `'box'` (planned), no `'fast'`

A 2:1 reduction is only exact when a dimension halves cleanly. For odd, non-unit sizes (decided 2026-11-10):

- **`'exact'` (default, implemented).** An odd size ends the current batch, and the next level is re-sampled from the stored level with normalized-coordinate bilinear filtering.
  - Results match render-pass and WebGL mipmaps (the CPU reference in §7), so engines can switch without visual change.
  - Min/max use the conservative `2x … 2x+2` footprint, so Hi-Z is correct.
  - Cost: more dispatches on sizes like 1920×1080 (7 today). The planned tail kernel (§13) targets about 3–4 with unchanged results.
- **`'box'` (planned, opt-in).** Area-weighted odd sizes, as in nvpro_pyramid: per axis, 3 taps with weights `(n−x, n, x+1)/(2n+1)` for size `2n+1`. Every source texel contributes in proportion to its overlap, which fixes the unrepresentative 1×1 of render-pass mipmaps on sizes like 31×31.
  - Same dispatch structure and footprint as the folds, so no extra cost.
  - Recommended for quality-sensitive content: GI, irradiance and roughness chains, UI.
  - Needs its own oracle.
- **No `'fast'` (dropped).** SPD-style clamped reduction through odd sizes drops the last row or column. That is lower quality, and it makes Hi-Z non-conservative. Its only benefit is fewer dispatches, which the tail kernel delivers for every mode without changing results. Revisit only if the tail kernel fails its measurement.

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
- **Batch boundaries:** a batch ends at `levelsPerDispatch`, or where the next reduction is inexact under `'exact'`.
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
| ≤ 4 | 8×8 | 8×8 | 16×16 | 1 KiB |
| 5 | 16×16 | 16×16 | 32×32 | 4 KiB |
| 6 | 16×16 | 32×32 (2×2 per invocation) | 64×64 (as in SPD) | 16 KiB (default limit) |

**Layout matters.** Adjacent invocations sample and store adjacent texels, and each processes four 16×16 sub-tiles. Giving each invocation its own 2×2 block measured 30–45 % slower.

### Levels per dispatch

- `min(6, maxStorageTexturesPerShaderStage)`. The default WebGPU limit is 4.
- `Downsampler.requiredLimits(adapter)` gives apps the limits to request.
- Around 94 % of devices report at least 8 (web3dsurvey).
- Storage texture views must cover exactly one mip level ([WebGPU spec](https://gpuweb.github.io/gpuweb/)), so each output level needs its own binding.

### Quantization

Every intermediate level is quantized exactly as stored, so a chain equals level-by-level generation.

- **rgba8:** values stay in integer 0–255 units in workgroup memory. Sums are exact and ties round to even. With `k/255` floats, float error turned ties into round-ups. A 1×128 tail drifted 3 steps by mip 5: half of all 1D texels are ties.
- **rgba16float:** values go through `pack2x16float` / `unpack2x16float`.

### Caching

- Pipelines are cached per `Downsampler` (per device) and per reduction; bind group layouts per format, view dimension, width and access.
- Bind groups are cached per texture and options in a `WeakMap`.
- `release(texture)` drops a texture's entries early.

## 5. API

### Implemented

```ts
import { Downsampler } from 'downsampler/webgpu';

const device = await adapter.requestDevice( { requiredLimits: Downsampler.requiredLimits( adapter ) } );
const downsampler = new Downsampler( device );          // { levelsPerDispatch?, averageAccess? }
downsampler.supports( texture );                         // boolean
downsampler.unsupportedReason( texture );                // string | null
downsampler.encode( encoder, texture, {
	baseMipLevel, mipLevelCount, oddSizes: 'exact',
	reduction: 'average' | 'min' | 'max' | 'minmax' | { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { … }' },
} );
downsampler.generate( texture );                         // encode + submit
downsampler.prepare( texture );                          // planned dispatches (inspection/testing)

import { generateMipmaps, requiredLimits } from 'downsampler/three';
generateMipmaps( renderer, texture );                    // false if not allocated/supported

import { planBatches, reduceReference, footprint } from 'downsampler'; // backend-neutral core
```

```ts
// Separate source: depth → Hi-Z. Level 0 the same size (copy) or half (reduce).
downsampler.encode( encoder, hiZ /* rg32float */, { source: { texture: depthTexture, mipLevel: 0 }, reduction: 'minmax' } );
downsampler.sourceUnsupportedReason( destination, source, baseMipLevel ); // string | null
```

The destination must have `TEXTURE_BINDING | STORAGE_BINDING` usage and a format from `formats`: `rgba8unorm`, `rgba16float`, `r32float`, `rg32float`, `rgba32float`, plus `bgra8unorm` (needs `bgra8unorm-storage`) and `r8unorm`, `rg8unorm`, `r16float`, `rg16float` (need `texture-formats-tier1`). `unsupportedReason` names a missing feature. Its dimension must be `2d`, with `textureBindingViewDimension` either `2d` or `2d-array`.

### Planned

```ts
downsampler.encode( encoder, destination, {
	layers: [ first, count ],
	rect: { x, y, width, height },      // SPD rectInfo
	oddSizes: 'exact' | 'box',
	scratch: GPUTexture,                 // optional reusable scratch for sRGB destinations
} );
```

## 6. SPD feature parity

| SPD feature | Status | Plan / notes |
|---|---|---|
| Up to 12 levels from ≤4096² | ✅ | any size, by chaining (2 dispatches up to 4096²) |
| Single dispatch | ❌ by design | §2.2 |
| Custom reduction (`SpdReduce4`) | ✅ | average/min/max/minmax + custom WGSL `reduce(a, b)` fold (§4 Reductions). A weighted custom reduction (SPD's `SpdReduce4` used for averages) isn't offered: under `'exact'` it has no defined odd-size meaning. Transforms around an average (premultiply, sRGB) are planned as load/store hooks with §6.1. |
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
| Odd sizes | ✅ `'exact'` | SPD drops the last row/column; we match render passes. Area-weighted `'box'` planned; SPD-style `'fast'` dropped (§2.1) |

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
| Packed float | rg11b10ufloat | tier1 | `ufloat11/10` quantization (no sign, 6/5-bit mantissa, negatives clamp to 0, round to nearest even); no WGSL pack builtin, so encode in the shader and in core |
| Integer | r32uint, rg32uint, rgba32uint, rgba8uint, rgba16uint, r32sint, rgba8sint (core); r8/r16/rg8/rg16 uint/sint (tier1) | mixed | separate `u32`/`i32` kernel (`texture_2d<u32>`, `vec4u`). Folds only at first (min, max, bitwise custom: ID, visibility and occupancy pyramids); `'average'` would need a declared rounding rule |
| sRGB | rgba8unorm-srgb, bgra8unorm-srgb | **never** (even with a non-sRGB view format) | reduce linear values, then write sRGB-encoded values to an rgba8unorm (or bgra8unorm, which needs `bgra8unorm-storage`) scratch and `copyTextureToTexture` each level (verified allowed: formats differing only in sRGB-ness). Needs load/store transform hooks; quantize-as-stored is `decode(round(encode(v)·255)/255)` |
| Shared exponent | rgb9e5ufloat (32-bit HDR, common for GI and irradiance) | **never** storage or renderable, so render-pass mipmaps can't produce it either | **buffer output path** (below): pack in the shader, `copyBufferToTexture` into the level. Verified on Chrome 148 / M5 Pro: packed texels round-trip exactly through the copy and a sampled read |
| Out of scope | BC/ETC2/ASTC (generate before compression); depth as destination (use r32float, §4 Separate source) | — | — |

**Buffer output path.** WebGPU allows `copyBufferToTexture` into formats that are neither storage nor renderable, so a kernel can write texels to a storage buffer and copy them into each level.
- **Layout:** one buffer per batch, with each level and layer at its own offset and rows padded to 256 bytes. That is a single binding, so this path isn't bound by `maxStorageTexturesPerShaderStage`: 6 levels per dispatch even at the default limit of 4.
- **Cost:** one extra write and read of the written levels (the levels below 0, about a third of the base texture's bytes), plus the copies.
- **Uses:**
  - rgb9e5ufloat, the only GPU route to its mips in WebGPU.
  - An alternative to the sRGB scratch texture: `copyBufferToTexture` into rgba8unorm-srgb is allowed.
  - A fallback for bgra8unorm and rg11b10ufloat on devices without `bgra8unorm-storage` or tier1.
- **rgb9e5 packing:**
  - Shared exponent `e = max(−16, ⌊log2 max(r,g,b)⌋) + 16`; mantissas `round(c / 2^(e−24))`, bumping `e` when the largest rounds to 512 (EXT_texture_shared_exponent).
  - Take ⌊log2⌋ from the float's exponent bits, not `log2()`, which isn't exact near powers of two.
  - Quantize-as-stored is pack → unpack, mirrored in core with exhaustive tests.
  - Probe kernel: `results/probes/rgb9e5-buffer-copy.mjs`.

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
4. buffer output path + rgb9e5ufloat
5. sRGB + transform hooks (the three.js PR needs this), choosing between scratch texture and buffer output by measurement
6. rg11b10ufloat
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
- **Oracle independence:** a core test checks that max chains are conservative: every texel bounds each base texel its normalized area touches, without using `footprint()`.

New features add their own oracle first: each new format family, sRGB, rect, layer range, and `'box'`.

## 8. Performance methodology

- **Timestamps** come from two one-pixel render marker passes bracketing 16 repeats. The markers depend on the texture under test: the first writes level 1, the last samples the final level. Empty or independent compute markers gave invalid timestamps on Metal.
- **Marker cost** is measured as an empty batch and subtracted (about 0.0006 ms per chain on M5 Pro).
- **Results:** medians of 11 rounds, rotating order across strategies, warmed, with a parity check before timing.
- **Baseline:** one render pass per level and layer, linear sampler. This is the common approach and what three.js does today.

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

CI runs the unit tests and the quick GPU suite on SwiftShader. SwiftShader exposes 4 storage textures, so CI covers the 1- and 4-level kernels; 5- and 6-level kernels and all timing need real GPUs.

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
   1. WGSL reduction generator: tool-supplied `load` with its own bindings, per-channel reductions including sums, choice of emitted levels, outputs (texture, buffer, WGSL `store` hook), edge policy.
   2. Reduce-to-scalar in 2 dispatches with an epilogue hook.
   3. Proof on the upscaler's FSR 3.1 passes as bench fixtures, against an oracle, timed on M5 and by the owner on RTX 3060 / Intel UHD.
   4. Deferred until 1–3 clear: an SPD-style mode, built at least as a measured comparison that justifies, or not, its not being the primary method.
   Then: buffer output + rgb9e5ufloat and rg11b10ufloat (GI formats); layer range + rect; `'box'`, power-of-two destination from a separate source, and the exact tail kernel.
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
- Cube maps in compatibility mode: they can't be viewed as 2d-array, so either per-face views or the render fallback.
- `'box'` for `'minmax'` and other folds is meaningless (they already use the full footprint). Should `'box'` simply be ignored there, or rejected?

## 13. Research (2026-11-10)

Reference material, mostly for after M2–M3: prior art and the standards discussion, reviewed against this spec. Consult it when a problem matches something solved here. Sources are linked; claims attributed to people come from their comments.

### Community prior art (WebGPU)

- [webgpu-spd](https://github.com/JolifantoBambla/webgpu-spd) (`JolifantoBambla`, 2024): a WGSL SPD port, single dispatch with the atomic handoff; see the #7 note below.
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
2. **Exact tail kernel** (candidate). Once a level fits one workgroup's memory, reduce every remaining level there through any parity, with exact `'exact'` weights (no halo needed). 1920×1080 would drop from 7 dispatches to about 3–4. Measure it: a boundary costs ~1–2 µs (§2.2).
3. **Halo batches under `'exact'`** (candidate). Our odd bilinear taps also fall in 2x … 2x+2, so nvpro's halo structure carries one odd step per dispatch without changing results.
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
- **Exposure covers only `[0, 2^mips)²`.** Only `(0,0)` of the last level is kept, so at 1920×1080 the scene average covers the top-left 1024×1024.
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

### Our other tools

| Tool | Pyramid / reduction | Shape | Built today |
|---|---|---|---|
| reflections (SSSR port) | Hi-Z from depth, min (max for reverse-Z), r32float, ≤7 levels; `textureLoad` traversal | full chain, separate source | **on downsampler** via `encodePass` (branch `reflections-integration`): 1080p 0.23 → 0.15 ms (M5 Pro) |
| reflections | probe prefilter source chain, rgba16float cube (6 layers), average; `textureSampleLevel` | full chain | on downsampler; **fails in compatibility mode** (one texture read and written in one dispatch) |
| ssgiDev | depth min pyramid, 5 levels, one dispatch per level brute force; optional max pyramid; light pyramid (average rgba16f); G-buffer downsample (min depth, normal of the min texel, mean light) | Hi-Z, chain, fold with payload | hand-written |
| Denoiser | OIDN auto-exposure: log-mean luma → scalar (f32 buffer); FFX reflections 8×8 luma-weighted radiance mean fused into reproject; ReBLUR-style history-fix mips 1–3 in one LDS dispatch, sum ÷ non-sky count, off-image weight 0, ceil sizes, atlas layout | scalar, fused block stats | hand-written |
| webgpu-ddgi | material store: alpha-weighted 2×2 average, 9 levels of 256² layers, sRGB via buffer + `copyBufferToTexture`; sky SH 19 sums → scalar (subgroups); radiance cache: r32uint "first valid of 2×2" probe mask (~8 levels), hash-grid tile sums `(rgb·n, n)` 4 levels | chain, scalar, uint fold, block stats | hand-written, one dispatch per level |
| Capsaicin (reference) | `GPUMip`: area-weighted odd sizes (nvpro weights), depth min/max `2x…2x+2`, sRGB re-encode, then SPD once the size is a power of two; HZB with a min-reduction sampler (WebGPU has none); `GPUReduce` 1D sum/min/max/product | chain, Hi-Z, scalar | D3D12 |

### Shapes that recur, and what the generator must cover

1. **Hi-Z from depth:** min, max or a min+max pair, full chain, separate source, read with `textureLoad`. This is covered today.
2. **Average colour chains:** cubes and arrays, sometimes alpha-weighted, sometimes with sRGB output. Mostly covered; sRGB and weighting are missing.
3. **Fused block statistics:** 2–4 levels in one tile, from a custom load. This is the most common shape and the biggest gap.
   - The reductions are sums or weighted means with a carried weight channel, and off-image texels get weight 0.
   - Some carry a payload, such as the normal of the min-depth texel.
   - None of them are semilattice folds.
4. **Scalar 1×1:** exposure, SH and statistics, usually into a buffer, with an epilogue.
5. **uint folds:** "first valid", or max over a packed key.

**Generator requirements (M2T.1–2).**
- **Hooks:**
  - **load:** WGSL with the tool's own bindings, returning `vec4f` (later also `vec4u`).
  - **reduce:** per channel, one of sum, average, weighted average (a channel holds the weight), min, max, or a custom fold. Plus argmin/argmax with a payload.
  - **store or epilogue:** WGSL per emitted level.
- **Level selection:** which levels to emit, and where: texture mips, an atlas, a buffer, or nothing.
- **Edge policy:** `exact`, `box` (area), `zero-weight` (off-image texels contribute nothing; ceil sizes), and `clamp` (FSR/SPD-compatible, for parity tests).
- **Recording:**
  - Into a caller-owned pass (`encodePass`), so one timestamp pair covers the chain.
  - Dispatches only, so a tool can embed the kernel in its own pass order.
- **Scalar:** partial results per workgroup, then a final dispatch with the epilogue. It writes to a buffer or a texel.
- **Compatibility mode:** split the read and write views of one texture across dispatches, or use the buffer path.

The fold contract in §4 (idempotent only) stays for folds. Sums and weighted means become reductions in their own right, each with its own oracle. They are defined at odd sizes only under `box` and `zero-weight`.
