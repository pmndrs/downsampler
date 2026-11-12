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
The error 5/256 is a 1/256 weight step times a texel difference of about 5, on weights the oracle treats as exact.
The native D3D12 render-pass chain, `render-rtonly` and the Vulkan blit show the same against `'exact'` in rgba16float at
1129×635 (18 values, max 12.25 ulp; D3D11 `GenerateMips` 1036 values, max 16.5 ulp;
[native-semantics-rtx3060.json](native-semantics-rtx3060.json)), so it is the sampler, not our kernel.

### 3. Intel only: larger filtered deviations, not yet explained

On top of 1 (132 failures, the same checks as NVIDIA):
- filtered levels off at many odd shapes (3×7, 7×4, 15×17, 17×9, 31×33, 65×129, 192×108, 255×257), at levels 1–4, in
  `'exact'`, `'spd'` and the tail: e.g. 3×7 rgba16float `'spd'` mip 1, 2.8828 against 2.9219 (tolerance 0.0088);
- rg11b10ufloat up to 17 % off (15×17 mip 1: 34 against 29);
- a same-size rgba16float → rgba8unorm `'spd'` copy giving 8/255 where the oracle has 0;
- `packed/buffer-path`: "rgb9e5 from a separate source: level 0 is not the quantized copy".

These are bigger than a weight-precision step and need a targeted look (sampler coordinates on Intel, the packed
encoders, or the oracle) before anything is concluded. The first full-suite attempt on Intel also lost the device after
580 s (log: [logs/correctness-low.log](logs/correctness-low.log), first run); the quick runs did not.

## Not changed

No library code (`src/`) was changed on this branch, per the handoff. Fixing 1 needs a shader-side round-to-nearest-even
(or an oracle that accepts truncation); 2 needs the tolerance model widened for NVIDIA's sampler. Both are owner calls
(PROGRESS.md, *Waiting on owner*).
