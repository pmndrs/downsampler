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

## webgpu-spd 3.0.0 (`--compare=webgpu-spd`, `--webgpu-spd-check --frames=256`)

Same-run ratios against our modes (no clock question); against native AMD SPD through the locked runs on the RTX
(webgpu-spd ÷ ours, same run × ours ÷ native, locked) and directly on Intel. Medians over 960×540 … 2560×1440 (range).
`wspd` is its default (8 mips per pass: single dispatch with the atomic handoff, subgroups on), `wspd6` its handoff-free
6 mips per pass, `wspdNoSg` the default without subgroups.

| Ratio | RTX rgba8 | RTX rgba16f | Intel rgba8 | Intel rgba16f |
|---|---|---|---|---|
| webgpu-spd default ÷ ours `'spd'` | 1.64 (1.19–2.08) | 1.07 (1.04–1.47) | **3.52** (2.43–3.76) | **2.68** (1.04–2.81) |
| webgpu-spd 6 per pass ÷ ours `'spd'` | 1.24 (0.93–1.80) | 0.98 (0.90–1.08) | 3.34 | 2.58 |
| webgpu-spd without subgroups ÷ ours `'spd'` | 1.53 | 1.05 | 1.78 | 1.35 |
| webgpu-spd default ÷ ours `'exact'` | 0.83 | 0.91 | 1.89 | 1.85 |
| webgpu-spd default ÷ AMD SPD best (native) | 2.02 | 1.36 | 4.04 | 4.12 |
| ours `'spd'` ÷ AMD SPD best (native) | **1.26** | **1.27** | **1.28** | **1.50** |

| Hi-Z (r32float max) | RTX | Intel |
|---|---|---|
| webgpu-spd default ÷ ours `'exact'` max | 0.95 (0.58–1.48) | 2.26 |
| ours `'exact'` max ÷ AMD SPD max (native, best) | 2.19 | 1.94 |

- On the RTX, our `'spd'` is 1.6× faster than webgpu-spd's default in rgba8 and level in rgba16f; webgpu-spd's handoff-free
  mode is close to ours. Against native, ours is 1.26× AMD's best SPD build, webgpu-spd's default 2.0×.
- On Intel, webgpu-spd is 2.7–3.5× slower than ours, mostly from subgroups: without them it is 1.4–1.8×. The same as
  AMD's own quad-op path natively (2× its LDS path on Intel): subgroup/quad reductions are a loss on this GPU.
- Hi-Z: ours `'exact'` max is conservative (every odd edge covered) and costs a batch per odd size; webgpu-spd's max and
  AMD SPD's drop odd rows and columns. Not the same work, so these ratios aren't a like-for-like speed comparison.
- **Correctness:** 0 stale frames of 41,472 on each GPU (27 size/format rows × 3 variants × 256 frames, one encoder and one
  submit per frame; isolated runs all agree). On the M5 Pro its default handoff returned stale levels 7+ in up to 77 % of
  frames (DIVERGES #22): the race is real on Apple, not observed on NVIDIA or Intel here. Its min/max are non-conservative
  on both (dropped edges), as on the M5. One probe variant (`enable f16`) fails to compile without `shader-f16`.

## Render passes vs `'exact'`, and the exact kernel's tile (2026-11-13, `--compare=render`)

DIVERGES #17 asked whether a render-pass path should replace `'exact'` on Windows. `npm run bench -- --compare=render
--skip-validation` times render passes, `'exact'`, `'spd'`, the tail kernel, the buffer output path, render passes for the
first K levels then `'exact'` (`hybrid1`–`3`, same results), and level 1 alone (`renderL1`, `exactL1`). Clocks unlocked
(RTX median 1425–1867 MHz under load; Intel has no telemetry), idle machine, 11 rounds × 16 repeats, paired per-round ratios.
Files: `*-render.json` (8×8 tiles, before the changes below), `*-render-unsafe.json` (Dawn toggles), `*-render-exact32.json`
(final, new default).

**Render passes for the large levels don't help.** `hybrid1`–`3` cost the same as `'exact'` or more on the RTX (1.24–1.30×
render in rgba8 vs 1.29×). At 4096² level 1 alone costs the same either way (`renderL1` ÷ `exactL1` 1.00); at render sizes
a compute dispatch carries ~10 µs more fixed cost than a render pass, and `'exact'`'s first batch already writes several levels.

**Dawn's checks are not the gap.** `--dawn-features=disable_robustness,skip_validation` (new `test/run.js` flag) moved every
strategy together within noise: RTX 1.00 ± 0.02, Intel ± 10 % (its run-to-run spread). Chrome doesn't report whether a toggle
took effect, so this is "no change seen", not a confirmed ablation.

**The cause on Intel was the exact kernel's workgroup.** Batches of up to four levels ran 8 × 8 workgroups (tile 8). With a
32 × 32 tile (16 × 16 invocations, 2 × 2 texels each, the 6-level kernel's shape) and level sizes derived instead of queried
with `textureDimensions` on storage textures, `'exact'` (medians over 960×540 … 2560×1440, range in brackets):

| | RTX rgba8 | RTX rgba16f | Intel rgba8 | Intel rgba16f |
|---|---|---|---|---|
| new ÷ old (`exact` ÷ `exact8`, same run) | **0.95** (0.78–0.99) | 1.03 (0.97–1.07) | **0.81** (0.61–0.97) | **0.94** (0.65–1.00) |
| `'exact'` ÷ render, old (8 × 8) | 1.34 | 1.04 | 1.62 | 0.98 |
| `'exact'` ÷ render, new | 1.19 | 1.06 | **1.04** | **0.66** |
| tail kernel ÷ render (new tiles) | **0.98** | **0.90** | 1.02 | 0.63 |
| tail ÷ `'exact'` | 0.82 (0.71–0.93) | 0.90 (0.81–0.96) | 0.97 | 0.97 |
| `'spd'` ÷ render | 0.75 | 0.75 | 0.78 | 0.68 |

- Powers of two plan the same 6-level batches either way (1.00–1.06 at 1024² and 4096²: noise).
- Intel's largest gains are at 1280×720, 1920×1080 and 2560×1440 (0.61–0.74). The size queries alone were worth 5–25 % there
  (rgba8, measured before the tile change). RTX rgba16f is 1.03, inside the 5 % noise line but leaning slower.
- With the tail kernel, `'exact'` now ties or beats render passes on the RTX (0.98 / 0.90). On Intel the tail adds little
  (0.97). That re-opens the tail default (owner decided opt-in on M5 data; PROGRESS *Waiting on owner*).
- Default now: `exactTile` 32 where a workgroup can have 256 invocations, on hardware adapters; 8 in compatibility mode and on
  fallback adapters (SwiftShader compiles 32 × 32-tile pipelines slowly). Verified: RTX full matrix 18,292 cases, RTX and Intel
  quick, SwiftShader CI shard 2/14 (default stays 8, 32 s); only the known FSR fixtures fail. Not yet measured on the M5 Pro.
