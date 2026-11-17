# M5 Pro: sRGB render path, `colorSpace: 'srgb'`, float32 sampling, halo (2026-11-17)

Branch `srgb-store` (`d8368d0` + the bench fix below), Apple M5 Pro, Chrome 148 headless (Metal). Task list:
[docs/M5-HANDOFF-SRGB.md](../../docs/M5-HANDOFF-SRGB.md). Benches ran alone on the GPU after the suite, nothing else
busy (top process WindowServer at 8 % CPU); load averages (1 / 5 / 15 min) are in each JSON's `loadAverage`: 5.2 / 7.1 /
6.7 falling to 2.0 / 4.7 / 5.8 over the runs (the suite had just finished).

## Correctness

- `npm run test:core`: 129 / 129.
- `node test/run.js --only=srgb --keep-going`: 2149 cases, 7344 pattern runs, 132 groups, all pass (8 min).
- Full GPU matrix in two parts (`test/parallel.js --shards=24 --part=i/2 --jobs=6 --cap=14 --budget=28`):
  [part 1](srgb-full-p1/summary.md) 17,709 cases, 1608 groups, 12 min 54 s; [part 2](srgb-full-p2/summary.md) 18,199
  cases, 1630 groups, 9 min 39 s. Together **35,908 cases, 3238 / 3238 groups, no failures, nothing skipped**, including
  `srgb/render`, `srgb/render-api`, `srgb/alias` and `downsampler/float32-sample` (72 chains, 495 levels, on a device with
  `float32-filterable`). Metal's fast math changed nothing the tolerances see.

## A bench fix first: the timing markers must bracket the texture that is written

The first `--compare=srgb` run ([unbracketed JSON](m5-pro-srgb-renderpath-unbracketed.json)) timed `colorSpace` at
0.0000 ms (some rows −0.0000), although its check found the same bytes as the buffer path. `timeGpu`'s two marker
passes depend on the bench texture (the first renders into its level 1, the last samples its last level). `colorSpace`
writes an aliased twin instead, so nothing ordered the end marker after its dispatches, and on Metal it ran before they
finished. The `alias` prototype and `--compare=srgb-cost`'s `linearBuffer` / `linearStorage` had the same flaw.

Fix (`test/bench/bench.js`, `test/bench/srgb.js`): a strategy that writes another texture names it (`target`), and the
markers bracket that one. The twins get `RENDER_ATTACHMENT` for the start marker, so the `colorSpace` strategy now asks
for `output: 'storage'` (otherwise `'auto'` would take the render path, a different strategy). The other strategies
write the bench texture, so the fix doesn't touch them (between runs they moved by up to a third at 960×540, buffer
0.105 → 0.070 ms with the first run at load 5, and a few percent elsewhere). D3D12 orders passes for its own reasons, which is presumably why the Windows
`colorSpace` numbers looked plausible; re-running there would confirm them.

## sRGB paths

`npm run bench -- --compare=srgb --skip-validation --strategies=render,buffer,colorSpace,renderPath,renderDevice
--rounds=11 --repeats=32`, twice: [run 1](m5-pro-srgb-renderpath.json) ([log](m5-pro-srgb-renderpath.txt)),
[run 2](m5-pro-srgb-renderpath-2.json). Ratios paired per round, run 2 (run 1 within ±0.15 at the small sizes, ±0.04
from 1080p up); under 1 is faster.

| Size | render passes ÷ buffer | `colorSpace` ÷ buffer | **render path ÷ buffer** | `'device'` ÷ buffer | `colorSpace` ÷ render path |
|---|---:|---:|---:|---:|---:|
| 960×540 | 1.27 | 0.44 | **1.30** | 1.39 | 0.30 |
| 1129×635 | 1.48 | 0.40 | **1.43** | 1.43 | 0.28 |
| 1280×720 | 1.58 | 0.49 | **1.58** | 1.54 | 0.31 |
| 1707×960 | 1.27 | 0.50 | **1.25** | 1.26 | 0.39 |
| 1920×1080 | 1.00 | 0.57 | **1.01** | 1.00 | 0.56 |
| 2560×1440 | 0.77 | 0.68 | **0.79** | 0.75 | 0.87 |
| 1024² | 2.18 | 0.63 | **2.18** | 2.16 | 0.29 |
| 4096² | 0.81 | 0.66 | **0.83** | 0.81 | 0.79 |

1920×1080 in ms (run 2): render passes 0.120, buffer 0.125, `colorSpace` 0.072, render path 0.128, `'device'` 0.125.

- **The render path loses to the buffer path on the M5 at every size up to 1707×960 and at 1024² (1.25–2.18×)**, ties
  at 1080p and wins only at 2560×1440 and 4096² (0.79–0.83×). On the RTX and Intel it wins everywhere (0.63–0.92×).
- The cause is the render passes, not the fragment shader: the render path runs at plain render-pass speed
  (0.99–1.05× of them over both runs), and render passes cost about 0.01 ms each on the M5 almost regardless of size
  (0.10–0.14 ms for 9–11 passes from 960×540 to 2560×1440), where the compute paths scale with texels.
- **Exact rounding is almost free here**: `'device'` takes 0.96–1.02× of the render path (median 0.99), against up to
  1.6× on the RTX and about 5× on Intel. It still changes bytes: 0.3–1.0 % of them, by up to 2 codes, from level 1 on
  (RTX about 16 %, Intel about 6 %).
- **`colorSpace: 'srgb'` is the fastest sRGB path on the M5 at every size**: 0.40–0.68× of the buffer path,
  0.26–0.91× of render passes and of the render path (median 0.35–0.38). On the RTX it is 0.73–0.83× of the buffer path,
  a little behind the render path there.

## Sampling an aliased texture

`npm run bench -- --compare=storage-usage --formats=rgba8unorm-srgb --skip-validation`
([JSON](m5-pro-srgb-alias-sampling.json), [log](m5-pro-srgb-alias-sampling.txt)): an `rgba8unorm` texture with
`viewFormats: ['rgba8unorm-srgb']` and `STORAGE_BINDING`, sampled through its sRGB view, against a plain
`rgba8unorm-srgb` texture. Medians over 8 sizes: 1.005 (rendered content, fullscreen), 1.000 (material), 1.000 (write),
1.002 (noise, fullscreen), 0.999 (noise, material); every case 0.970–1.042. **No cost**: Metal keeps compression on the
reinterpretable texture, or it doesn't matter for these passes.

## float32: sampled vs loaded

`npm run bench -- --compare=float32 --skip-validation` ([JSON](m5-pro-float32.json), [log](m5-pro-float32.txt)).
Sample ÷ load, paired:

| Size | r32float | rg32float | rgba32float |
|---|---:|---:|---:|
| 960×540 | 0.73 | 0.76 | 0.84 |
| 1129×635 | 0.83 | 0.82 | 0.85 |
| 1280×720 | 0.71 | 0.92 | 0.97 |
| 1707×960 | 0.74 | 0.87 | 0.93 |
| 1920×1080 | 0.81 | 1.01 | 0.96 |
| 2560×1440 | 0.82 | 0.98 | 1.01 |
| 1024² | 0.75 | 0.87 | 0.97 |
| 4096² | 1.00 | 1.00 | 1.01 |

Sampling takes 0.71–1.01× of loading, as on the RTX (0.74–1.00×) and Intel (0.69–1.00×): most for r32float, about even
for rgba32float and at 4096². Never slower beyond noise. DIVERGES #13 stands on all three GPUs.

## Halo batches

`npm run bench -- --compare=render --skip-validation --strategies=render,tail,halo,haloAny`, 16 repeats
([run 1](m5-pro-halo.json)) and 32 ([run 2](m5-pro-halo-2.json)). halo ÷ tail, paired:

| Size | rgba8 run 1 | rgba8 run 2 | rgba16float run 1 | rgba16float run 2 |
|---|---:|---:|---:|---:|
| 960×540 | 0.90 | 0.89 | 0.93 | 0.94 |
| 1129×635 | 0.92 | 0.93 | 1.01 | 1.01 |
| 1280×720 | 0.97 | 1.01 | 0.99 | 1.01 |
| 1707×960 | 0.99 | 1.00 | 0.99 | 1.00 |
| 1920×1080 | 0.96 | 0.93 | 0.98 | 0.93 |
| 2560×1440 | 0.97 | 0.98 | 0.97 | 0.99 |
| 1024² | 1.03 | 1.04 | 1.00 | 0.99 |
| 4096² | 1.01 | 1.00 | 1.00 | 1.00 |

Median 0.98–0.99 over both runs. A gain of 5 % or more at 960×540 (both formats, both runs), 1129×635 rgba8 and, in
run 2 only, 1080p; a reproducible 3–4 % loss at 1024² rgba8. `haloAny` is the same (0.89–1.08, median 0.99). **It doesn't clear the bar**
(≥ 5 % and no losses): halo stays opt-in.

## What the numbers decide

- **Render path as the sRGB default: the rule fails on the M5.** It beats the buffer path only from 1440p up. Keeping
  it costs the M5 1.25–2.2× at common render sizes; going back to the buffer path costs the RTX and Intel their
  0.63–0.92×. Choosing by vendor is the split we avoid. **The owner chose a size rule (2026-11-17):** under
  `'auto'` the render path from 1920 × 1080 texels per layer (`renderPathMinTexels`, tunable), the buffer path below:
  no GPU is slower than with the buffer path everywhere, the M5 gains from 1440p, the RTX and Intel from 1080p.
- **`srgbRounding`** stays `'exact'`: the M5 makes the case stronger (exact rounding costs about nothing here).
- **Halo** stays opt-in (above).
- **`colorSpace: 'srgb'`: the rule holds.** It beats the render path at every size on the M5 (0.26–0.91×), and
  sampling the aliased texture costs 1.00× (as on the RTX). Recommended in the guide for apps that create their own
  8-bit color textures (core devices, not `'spd'`). Whether three.js should create its textures this way is a separate
  question for the owner.
