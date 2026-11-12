# Windows timings, 2026-11-12

Library (WebGPU, Chrome 148, D3D12): `npm run bench -- --compare=spd --skip-validation` (11 rounds × 16 repeats, rotated; branch
`f16-rounding`, f16 outputs rounded to nearest even). Native: `native/d3d12-mips` (D3D12 AMD FidelityFX SPD, D3D12 render passes,
D3D11 `GenerateMips`) and `native/vk-mips` (nvpro_pyramid, `vkCmdBlitImage`), 128 rounds × 16 repeats, timer overhead subtracted.
Every job ran alone under the shared GPU lock (`results/windows/run-benches.sh`, logs and clock samples in `logs/`). Full tables:
[native-tables.md](native-tables.md) (`node native/summarize.mjs`).

| GPU | Driver | Power |
|---|---|---|
| RTX 3060 Laptop | 32.0.15.9200 (NVIDIA 592.00) | AC, plan "Turbo" |
| Intel UHD (gen-12lp) | 31.0.101.5333 | AC; no clock telemetry (nvidia-smi only) |

## RTX clocks: unlocked vs locked (measured)

Unlocked, the bench lock sampled the RTX at a median 1460–1630 MHz in the native jobs (throttling in 41–74 % of samples) but
only 620–920 MHz in the WebGPU jobs: the browser's short bursts don't ramp this laptop GPU. Rerun with the clocks locked
(`nvidia-smi -lgc 1200,1200`, all three jobs VALID at a steady 1200 MHz, no throttling or power limit; `*-locked.json`):

| RTX, medians over 960×540 … 2560×1440 (rgba8 / rgba16f) | unlocked | locked 1200 MHz |
|---|---|---|
| ours `'spd'` ÷ AMD SPD best variant | 1.70 / 2.21 | **1.26 / 1.27** |
| ours `'exact'` ÷ nvpro_pyramid (Vulkan) | 1.41 / 2.18 | **1.20 / 1.06** |
| ours `'exact'` ÷ D3D12 render passes | 2.64 / 2.53 | **1.78 / 1.25** |
| WebGPU render passes ÷ the same D3D12 render passes | 2.01 / 2.29 | **1.45 / 1.34** |
| `'load'` ÷ `'sample'` (same run) | 1.13 / 1.10 | 1.14 / 1.04 |
| D3D12 render ÷ AMD SPD best (native only) | 1.27 / 1.27 | 1.31 / 1.32 |

Same-run and native-only ratios don't move; **cross-run WebGPU ÷ native ratios were overstated 1.2–2×**. Rule: lock the clocks
(admin; reset with `nvidia-smi -rgc`) for any WebGPU-vs-native comparison on this laptop; same-run ratios don't need it.

## Decision data (medians over 960×540 … 2560×1440; range in brackets)

| Ratio | RTX rgba8 | RTX rgba16f | Intel rgba8 | Intel rgba16f |
|---|---|---|---|---|
| `averageAccess: 'load'` ÷ `'sample'` (exact) | 1.13 (1.08–1.30) | 1.02 (1.01–1.51) | 1.31 (1.20–1.60) | 1.22 (1.05–1.26) |
| `f16Rounding: 'nearest'` (integer, first version) ÷ `'device'` | – | 1.04 (1.01–1.23) | – | **1.29** (1.04–1.34) |
| `f16Rounding: 'nearest'` (f32 add, current) ÷ `'device'` | – | 1.04 (1.00–1.10) | – | **1.14** (1.01–1.17) |

- Shader weights (`'load'`, identical results on every GPU) cost 13 % (rgba8) and 2 % (rgba16f) on the RTX, 22–31 % on Intel.
- Rounding f16 to nearest even with integer math cost 4 % on the RTX but 29 % on Intel at render sizes. The current
  version rounds with one f32 add and subtract of 2^(exponent + 13) (the adder rounds to nearest even, as WGSL requires):
  bit-exact against core `toHalf` on both GPUs, 14 % on Intel, noise on the RTX ([rtx](rtx3060-spd-f16fast.json),
  [intel](intel-uhd-spd-f16fast.json)); RTX full matrix unchanged (85 failures, all sampler or AMD hooks).

## Native platform paths (same-API comparisons, unaffected by the caveat)

| Ratio | RTX rgba8 | RTX rgba16f | Intel rgba8 | Intel rgba16f |
|---|---|---|---|---|
| D3D12 render passes ÷ AMD SPD (best variant) | 1.27 | 1.27 | **0.73** | 1.05 |
| D3D11 `GenerateMips` ÷ AMD SPD (best) | 1.15 | 1.20 | **0.74** | 1.02 |

- NVIDIA: compute (SPD) beats the driver's render-pass and `GenerateMips` paths by 15–27 %; all eight SPD variants (atomic
  or split, quad ops or LDS, load or sampler) are within noise of each other, so the atomic single dispatch buys nothing there
  (DIVERGES #1).
- Intel: render passes and `GenerateMips` beat every compute variant for rgba8 by ~27 %; SPD's quad-op path (`wave`, as AMD
  ships it) is 2× its LDS path; nvpro's compute is 3–5× slower than `vkCmdBlitImage`. Evidence for a per-device render
  fallback (DIVERGES #17), on Intel rather than NVIDIA.

## Library against native (Intel, where clocks aren't in question)

| Ratio (Intel) | rgba8 | rgba16f |
|---|---|---|
| ours `'spd'` ÷ AMD SPD as shipped (atomic, wave, load) | **0.50** | 1.33 |
| ours `'spd'` ÷ AMD SPD best (split, LDS, linear) | 1.15 | 2.58 |
| ours `'exact'` ÷ D3D12 render passes | 3.04 | 2.10 |
| WebGPU render passes ÷ the same D3D12 render passes | 2.04 | 2.59 |

The same render-pass chain costs about 2× in WebGPU what it costs in native D3D12 here, so part of every WebGPU ÷ native gap
is the platform (Dawn's translation, barriers, per-pass overhead), not the kernel.
