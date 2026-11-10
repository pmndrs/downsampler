# downsampler — core specification

Status: milestone 1 (port of the validated three.js prototype). This document is the source of truth for scope, design decisions, correctness requirements and the remaining plan. Update it when a decision changes.

## 1. Purpose

Generate texture pyramids on the GPU: mip chains first, then min/max reduction pyramids (Hi-Z), custom reductions and related downsampling chains. WebGPU first and engine-agnostic. Thin adapters (three.js first) sit on top.

Feature target: parity with AMD FidelityFX SPD, implemented independently in WGSL. SPD is the reference for what a downsampler should offer, not code to port. See §6.

## 2. Two decisions to know up front

### 2.1 Odd sizes: `'exact'` (default) vs `'fast'` (planned)

A 2:1 reduction is only exact when a dimension halves cleanly. For odd, non-unit sizes there are two reasonable behaviours:

- **`'exact'` (default, implemented).** An odd size ends the current batch. The next level is re-sampled from the stored level with normalized-coordinate bilinear filtering. Results match render-pass and WebGL mipmaps (the CPU reference in §7), so engines can switch to it without visual change. Cost: more dispatches on sizes like 1920×1080 (7 instead of 2).
- **`'fast'` (planned, opt-in).** Keep reducing in workgroup memory through odd sizes with SPD-style clamped 2×2 reads. That drops the last row or column at each odd step. Fewer dispatches, but edge weights differ from render-pass mipmaps, so outputs are not interchangeable with the reference. Its tests must use a separate reference that models the clamping.

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

- **First output:** a hardware bilinear sample at each texel centre, using normalized coordinates. That is an exact 2×2 box for even sizes, and correct weights for odd ones.
- **Later levels in the batch:** exact 2:1 reductions in workgroup memory, with clamped reads for unit dimensions and partial tiles.
- **Batch boundaries:** a batch ends at `levelsPerDispatch`, or where the next reduction is inexact under `'exact'`.
- **Layers:** workgroup z selects the array layer, so all layers go in one dispatch.

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

- Pipelines are cached per `Downsampler` (per device).
- Bind groups are cached per texture and options in a `WeakMap`.
- `release(texture)` drops a texture's entries early.

## 5. API

### Implemented (M1)

```ts
import { Downsampler } from 'downsampler/webgpu';

const device = await adapter.requestDevice( { requiredLimits: Downsampler.requiredLimits( adapter ) } );
const downsampler = new Downsampler( device );          // { levelsPerDispatch? }
downsampler.supports( texture );                         // boolean
downsampler.unsupportedReason( texture );                // string | null
downsampler.encode( encoder, texture, { baseMipLevel, mipLevelCount, oddSizes: 'exact' } );
downsampler.generate( texture );                         // encode + submit
downsampler.prepare( texture );                          // planned dispatches (inspection/testing)

import { generateMipmaps, requiredLimits } from 'downsampler/three';
generateMipmaps( renderer, texture );                    // false if not allocated/supported

import { planBatches, reduceReference } from 'downsampler'; // backend-neutral core
```

The texture must have `TEXTURE_BINDING | STORAGE_BINDING` usage and format `rgba8unorm` or `rgba16float`. Its dimension must be `2d`, with `textureBindingViewDimension` either `2d` or `2d-array`.

### Planned

```ts
downsampler.encode( encoder, destination, {
	source: { texture, mipLevel },      // separate source, e.g. depth32float → r32float Hi-Z
	reduction: 'average' | 'min' | 'max' | 'minmax' | { wgsl: 'fn reduce4(a: vec4f, b: vec4f, c: vec4f, d: vec4f) -> vec4f' },
	layers: [ first, count ],
	rect: { x, y, width, height },      // SPD rectInfo
	oddSizes: 'exact' | 'fast',
	scratch: GPUTexture,                 // optional reusable scratch for sRGB destinations
} );
```

## 6. SPD feature parity

| SPD feature | Status | Plan / notes |
|---|---|---|
| Up to 12 levels from ≤4096² | ✅ | any size, by chaining (2 dispatches up to 4096²) |
| Single dispatch | ❌ by design | §2.2 |
| Custom reduction (`SpdReduce4`) | ❌ | average/min/max/minmax + custom WGSL hook. Min/max must not use the bilinear sampler. |
| Load vs. linear-sampler source | sampler only | add a `textureLoad`/`textureGather` path: needed for min/max, depth sources, and r32float/rgba32float without `float32-filterable` |
| Separate source/destination | ❌ | enables depth → Hi-Z (depth formats cannot be storage textures) |
| Texture arrays | ✅ all layers | add a layer range; cube maps as 6 layers (2d-array view; not available in compatibility mode) |
| Sub-rectangle (`rectInfo`) | ❌ | offset/size uniform; output tile offset |
| Custom load/store (format conversion) | 2 formats | see §6.1 |
| Packed fp16 math | ❌ | optional `shader-f16`; keep only if measurably faster |
| Wave/subgroup ops | ❌ | optional `subgroups`; WGSL defines no quad↔`local_invocation_index` mapping in compute, so derive positions from `subgroup_id`/`subgroup_invocation_id`, or use shuffles. Measure first; current kernels are near bandwidth-bound at ≥2048². |
| Odd sizes | ✅ `'exact'` | add `'fast'` (§2.1) |

Beyond SPD: sizes over 4096², exact odd-size results, and automatic fallback to the 4-storage-texture limit.

### 6.1 Formats

Probed in Chrome 155 on Apple/Metal:

| Format | Storage | Status / plan |
|---|---|---|
| rgba8unorm, rgba16float | core | ✅ |
| rgba32float, r32float, rg32float | core | add quantization `float32`. A filtering sampler needs `float32-filterable`; otherwise use the load path. |
| r8unorm, rg8unorm, r16float, rg16float, rgb10a2unorm, rg11b10ufloat, 16-bit norm formats | `texture-formats-tier1` | add per-format quantization and channel count |
| rgba8snorm | core | add `snorm8` quantization |
| rgba8unorm-srgb, bgra8unorm-srgb | **never storage**: Dawn rejects STORAGE_BINDING even with a non-sRGB view format | reduce in linear space, write sRGB-encoded values to an `rgba8unorm` scratch, then `copyTextureToTexture` (copies between formats that differ only in sRGB-ness are allowed: verified) |
| bgra8unorm | `bgra8unorm-storage` (write-only) | optional |

## 7. Correctness specification

**Oracle.** `reduceReference` is a normalized-coordinate bilinear sample at each output texel centre with clamped edges, quantized to the storage format. Each GPU level is checked against the reference computed from the GPU's own previous level, so permitted rounding does not accumulate.

**Tolerance.** Declared test tolerances for these fixtures, not a formal error bound:
- unorm8: 2.01 steps.
- float16: `max(0.002, 0.3 % of the reference value)`.
- Plus a filter allowance where bilinear weights aren't exactly representable with 8 fractional bits.

**Required matrix.** Every change must pass all of it.

- **Shapes:** 1×1, 1D tails (1×128, 128×1, 128×8), odd and prime sizes, partial workgroups, rectangles, NPOT (192×108, 255×257, 1920×1080), layers (×3), both 2D view variants, and multi-batch sizes (2048², 96×4096).
- **Kernel widths:** 1, 4, 5 and 6 levels per dispatch, filtered by the device limit.
- **Formats:** all supported formats.
- **Patterns:**
  - constant (weights)
  - random, including HDR values in [−4, 12] for float formats
  - last row/column (edges)
  - per-64×64-tile values (tile boundaries, stale data)
- **Reuse:** one texture reused across patterns, exercising the cached plans.
- **Poisoning:** every level is overwritten with a poison colour before generation, so skipped writes fail.
- **Options:** a sub-range writes only its levels and doesn't modify its source; unsupported textures report a reason and `generate` throws.
- **GPU validation errors and device loss fail the run.**

New features add their own oracle first: min/max, sRGB, the source/destination split, rect, layer range, and `'fast'` with its clamped reference.

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

1. **M1 (done):** port the kernel, Downsampler, three adapter, core unit tests, GPU suite, device lab, benchmark and CI.
2. **M2 (parity), in this order:**
   1. load path + reductions (min/max/minmax/custom)
   2. separate source/destination (Hi-Z)
   3. format table + sRGB via scratch
   4. layer range + rect
   5. `'fast'` odd sizes
3. **M3 (coverage and tuning):** device lab runs (§9). Try fp16 and subgroup paths only where the data shows headroom.
4. **M4:** three.js in-tree PR (§10). Publish `downsampler` to npm.

## 12. Open questions

- Publish as bare `downsampler` or `@pmndrs/downsampler`? (`package.json` currently uses `downsampler`.)
- Cube maps in compatibility mode: they can't be viewed as 2d-array, so either per-face views or the render fallback.
- Whether `'fast'` should be the default for Hi-Z, where conservative min/max coverage of odd edges matters more than matching render-pass weights. Hi-Z also needs an odd-edge rule: include the dropped row/column in min/max.
