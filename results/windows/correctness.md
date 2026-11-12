# Windows correctness, 2026-11-12

Machine: laptop, Windows 11 Home 10.0.26200, i9-11900H (8 cores), 40 GB, on AC power, power plan "Turbo".
Browser: puppeteer's Chrome 148.0.7778.97, headless, D3D12 backend; GPU pinned by LUID (`test/gpu-choice.js`).

| GPU | Driver | WebGPU adapter |
|---|---|---|
| NVIDIA GeForce RTX 3060 Laptop (`--power=high`) | 32.0.15.9200 (NVIDIA 592.00) | nvidia ampere, `texture-formats-tier1`, 8 storage textures |
| Intel UHD Graphics (`--power=low`) | 31.0.101.5333 | intel gen-12lp |

## Runs

All with `node test/parallel.js` (sharded, `--keep-going`, per-shard cap and whole-run budget; one summary per run).

| Run | Cases | Groups | Wall | Failures | Summary |
|---|---|---|---|---|---|
| RTX 3060, full matrix | 18289 | 1674 / 1674 | 17 min 12 s (36 shards, 6 jobs) | 287 | [parallel-high-full](parallel-high-full/summary.md) |
| RTX 3060, `--quick` | 10444 | 731 / 731 | 2 min 3 s (18 shards, 6 jobs) | 189 | [parallel-high-quick](parallel-high-quick/summary.md) |
| RTX 3060, `--quick --compat` | 10355 | 721 / 721 | 2 min 1 s | 187 | [parallel-high-compat-quick](parallel-high-compat-quick/summary.md) |
| Intel UHD, `--quick` | 10437 | 731 / 731 | 2 min 22 s (12 shards, 4 jobs) | 629 | [parallel-low-quick](parallel-low-quick/summary.md) |

The same suites run one page at a time took 4.6 min (`--quick`, RTX; [baseline/](baseline/)) and over 35 min (full, RTX; the
first attempts in [logs/](logs/) hit the 30 min protocol timeout and an uncollected assertion before finishing). The suite is
CPU-bound: each case's oracle runs in JavaScript on the page, and the GPU sat at idle clocks (median 285 MHz SM under load).

## Failures, by cause

No structural failures: every level of every case was written, layouts, layers and batch plans match, and min / max /
custom folds pass everywhere except where an f16 conversion of the source is involved. What fails is precision against
oracles that assume round-to-nearest-even and the sampler-weight model in `src/core/reference.ts`.

### 1. f32 → f16 truncates toward zero (NVIDIA and Intel)

`results/probes/f16-store-rounding.mjs --power=high|low` ([rtx3060](f16-store-rounding-rtx3060.json), [intel-uhd](f16-store-rounding-intel-uhd.json)):
for 60 values between two halves, `textureStore` to rgba16float **and** WGSL `pack2x16float` both give the half toward
zero in all 30 cases where round-to-nearest-even rounds away (3.00146484375 → 3, not 3.001953125). On the M5 Pro only
`textureStore` truncated; `pack2x16float` rounded, which is why the library quantizes before storing. On D3D12 that
quantize step truncates too.

Failures it explains (RTX full: about 211 of 287):
- every `… -> rgba16float … max … exact mip 0` check of the separate-source path (one f16 ulp low, exact check);
- `packed/reducer-half-rounding` (the Reducer's f16 rounding test);
- the luma-pyramid fixtures, `fsr/luma-pyramid` and `generator/luma-pyramid-clamp` at all five sizes: FrameInfo
  2–5×10⁻⁴ relative off against a 1×10⁻⁴ tolerance, both low ([detail](baseline/high-fixtures-luma.json)); AMD's pass and
  ours agree with each other. `fsr/luma-pyramid-correct` passes.

### 2. NVIDIA bilinear filtering is outside the oracle's tolerance at odd sizes

Only 65×129 and 255×257, `average` only, only the first output of a batch (sampler-filtered), all formats with float
storage: e.g. 65×129 rgba16float mip 1, 4.82421875 against 4.8046875 (tolerance 0.0144), 2356 of 8192 values outside.
The native D3D12 render-pass chain, `render-rtonly` and the Vulkan blit show the same against `'exact'` in rgba16float at
1129×635 (18 values, max 12.25 ulp; D3D11 `GenerateMips` 1036 values, max 16.5 ulp;
[native-semantics-rtx3060.json](native-semantics-rtx3060.json)), so it is the sampler, not our kernel.

### 3. Intel: the same mechanism, a different sampler

Intel fails the checks of 1 (132) and many more filtered ones: odd and even shapes (3×7 … 1920×1080 in the probe; 15×17,
31×33, 192×108, 255×257 in the suite), levels 1–4, `'exact'`, `'spd'` and the tail, rg11b10ufloat up to 17 % (values span
2^−14 … 2^10, so a weight step times a large neighbour is large), and a same-size `'spd'` copy (8/255 where 0 is due).

### Why 2 and 3: how each sampler weights (probes, 2026-11-12)

| | RTX 3060 | Intel UHD |
|---|---|---|
| Axis weights ([sampler-weights](sampler-weights-rtx3060.json)) | 8-bit, rounded to nearest | 8-bit, **truncated** ([intel](sampler-weights-intel-uhd.json)) |
| 2D tap weights ([sampler-2d-weights](sampler-2d-weights-rtx3060.json)) | **each of the four products rounded to 1/256** (512 of 1089 grid points off) | exact products ([intel](sampler-2d-weights-intel-uhd.json)) |
| Filtered f16 result | rounded to f16 (≤ 0.5 ulp) | exact |
| The library's own taps ([kernel-ramp](kernel-ramp-rtx3060.json)) | exact at 65, 129, 192, 108; ≤ 0.57/256 elsewhere | 1/256 low almost everywhere, **even sizes too** (at 1920 → 960 all 960 weights are 127/256: `uv = (x + 0.5) / n` lands a float ulp under the texel edge and truncation takes the step below; [intel](kernel-ramp-intel-uhd.json)) |

`reduceReference`'s `filterTolerance` allows a 1/256 step per axis only where the axis weight isn't representable in 8
bits. NVIDIA's per-tap rounding breaks that at sizes with representable axis weights (65, 129: weights k/64, k/128),
and with random data spanning [−4, 12) the four rounding errors scale with the taps, up to about 0.05; reproduced
on 65×9, 33×129, 129×129 too. Intel's truncation breaks it everywhere, even sizes included.

**`averageAccess: 'load'` passes on both GPUs in every case** (no failure among the `(load)` variants, 18289 cases on
the RTX, 10437 on Intel): it applies the weights in the shader. Our kernel arithmetic is right; the hardware samplers
differ, and the native platforms' own mip paths differ with them.

Still open on Intel: `packed/buffer-path`, "rgb9e5 from a separate source: level 0 is not the quantized copy" (a
same-size converting copy, so likely the truncated weight at a texel centre, not yet confirmed). The first full-suite
attempt on Intel also lost the device after 580 s (log: [logs/correctness-low.log](logs/correctness-low.log), first run);
the quick runs did not.

## Not changed

No library code (`src/`) was changed on this branch, per the handoff. Owner decisions (2026-11-12): 1 gets a
shader-side round-to-nearest-even as a configurable default (branch `f16-rounding`); 2 and 3 wait for the timings of
`averageAccess: 'sample'` against `'load'` on both GPUs.
