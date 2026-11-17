# Downsampler performance and code review

November 17, 2026 · Baseline `d8368d0` · Handoff branch `codex/web-performance-review` includes `windows-3` at `0210cd1` and the M5 work through `1ff808b`. [Review Page](https://chatgpt.com/space/page_31c5c354d25c81918908d3903f124634) · [Mac checkout and M5 tests](docs/M5-HANDOFF-SRGB.md)

## DS1 · Keep a working plan after a failed replacement

On a 65×47 RGBA8 texture, replacing a working plan with a four-byte scratch buffer raised the expected error. Retrying the original options after editing the image then produced one WebGPU validation error and six stale mip levels. Full and partial caches now publish a replacement before retiring the previous buffers. On the RTX 3060 the retry preserved both plans, with zero errors and all six levels matching a fresh recompute. Implemented in `5c59385`. [Browser regression](test/browser/validate.js)

## DS2 · Reuse bindings as the brush moves

Moving a 32×32 brush 64 times across a warmed 1920×1080 texture previously created 64 uniform buffers and 256 bind groups. The same run on NVIDIA now creates two buffers and eight bind groups. A fixed brush creates zero of either. Storage encoding shares bindings and views while giving each recorded update its own immutable uniform data. Public `prepare()` keeps its existing batches. Implemented in `db4c381`. [Paired benchmark](results/probes/ds2-region.mjs)

With default device limits and 320 moving strokes, CPU encode p95 fell from 70 to 20 µs on the RTX and from 75 to 20–25 µs on Intel. Moving strokes allocated 319 → 1 buffers, 1,595 → 5 bind groups and 4,785 → 0 views on each adapter. Intel’s fixed-brush GPU median rose by 0.60–1.50 µs across two runs. The storage regression records 514 updates over two layers and submits the two command buffers in reverse order. All mip contents matched a full recompute. [Region regression](test/browser/validate-region.js)

## DS3 · Compile pipelines before interactive use

Call `await downsampler.warmup(texture, options)` before the first interactive `encode()` or `generate()` for that texture and path. The API shares in-flight native compiles, validates pipelines created synchronously and allows a retry after rejection. It uses the actual storage, buffer-copy, compatibility or render descriptors and leaves command submission to the caller. Implemented in `5702f5b` and `a7f3060`. Commits `7f96b7f` and `ff785d9` also verify size-based selection and render-only aliases through both WebGPU and three.js. [API guide](docs/guide.md) · [Probe and reproduction](results/probes/ds3-warmup.md)

On fresh Chrome 157 profiles, the time from the first submission to queue completion fell on all three devices. Each value below is the median of three alternating A/B pairs for a 512×288 texture, in milliseconds. The last column is the time spent in `await warmup()` before submission. The operating system and driver caches were retained.

| Device | Storage, sync → warm | Buffer, sync → warm | sRGB render, sync → warm | Prior warmup, storage / buffer / render |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA RTX 3060 Laptop | 128.0 → 3.2 | 174.5 → 10.6 | 48.9 → 2.6 | 70.5 / 120.4 / 42.8 |
| Intel UHD | 128.1 → 4.1 | 173.5 → 4.0 | 42.3 → 4.7 | 73.7 / 93.1 / 41.5 |
| SwiftShader | 414.8 → 19.2 | 488.9 → 30.8 | 58.2 → 44.7 | 365.4 / 480.2 / 13.6 |

All 108 page runs used one build under the shared GPU lock. Across 972 generated-mip readbacks, the A/B hashes matched and warmed first encode created zero pipelines. Both hardware arms showed two shader-compiler events per fresh-profile path and zero on reload. The main benefit is control over when the application pays the startup cost. [Full medians, frame gaps and compiler traces](results/windows/ds3/summary.md)

## DS4 · Capped halo stays opt-in

With the cap at its current default, the NVIDIA RTX 3060 Laptop benefited most on the odd-size box reductions. Intel UHD showed gains and losses across the same inputs. Each cell is capped-halo GPU time divided by plain GPU time. The median and range summarize 16 paired size/format ratios, from 960×540 through 3840×2160 with RGBA8 and RGBA16F. Each timing is the median of 11 rotating rounds with 16 repetitions in Chrome 148.0.7778.97. Drivers were NVIDIA 592.00 and Intel 31.0.101.5333. The ratios compare strategies within the same run. This pass did not record GPU clocks.

| Windows GPU | Policy | Median | Range | 1080p RGBA8 / RGBA16F |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 Laptop | Exact | 0.956 | 0.866–1.002 | 0.940 / 0.935 |
| RTX 3060 Laptop | Box | 0.915 | 0.731–1.002 | 0.834 / 0.880 |
| Intel UHD | Exact | 0.997 | 0.920–1.082 | 1.022 / 0.941 |
| Intel UHD | Box | 1.011 | 0.981–1.113 | 0.983 / 1.055 |

On Apple M5 Pro, two independent exact-policy runs gave a median halo/plain ratio of 0.98–0.99. At 1024² RGBA8 the capped halo took 3–4% longer both times. It misses the owner’s requirement of a 5% gain and no regressions. The default decision is closed. Apple box timings would add policy-specific data if that workload becomes important. [M5 measurements](results/m5/srgb-m5.md#halo-batches) · [NVIDIA exact](results/windows/round2/handoff-halo-high-exact.json) · [NVIDIA box](results/windows/round2/handoff-halo-high-box.json) · [Intel exact](results/windows/round2/handoff-halo-low-exact.json) · [Intel box](results/windows/round2/handoff-halo-low-box.json)

## DS5 · Show what partial updates and Hi-Z buy

The [brush example](examples/partial-update.html) replays the same 40 strokes in both modes, alternates measurement order and compares every edited mip with a full recompute outside the timing window. It reports CPU encode time and GPU timestamps separately and works on devices without timestamps. The [Hi-Z example](examples/hi-z.html) tests five objects against a thin occluder and a depth block. It compares the hierarchy’s decisions with full-resolution depth so a false cull is visible. Implemented in `f80f5a9`. [Setup from checkout](docs/getting-started.md)

## M5 and Windows sRGB selection

The merged `renderPathMinTexels` option defaults to 1920 × 1080 texels per layer. On M5, exact render took 1.25–2.18× the buffer path below 1080p, reached parity at 1080p and won at larger sizes. Automatic selection uses the size of the source level or dirty rectangle, including for pipeline warmup. `renderPathMinTexels: 0` enables rendering at any size that supports it. [M5 report](results/m5/srgb-m5.md) · [Selection rules](docs/guide.md)

Windows is remeasuring `colorSpace` against buffer output with corrected timing markers under the shared GPU lock. The Apple measurements are complete.

## Windows verification

`ff785d9` built, typechecked and passed all **161 core tests**. Chrome 148 passed 21 targeted cases on each of the RTX 3060 and Intel UHD, including native deferred validation, threshold selection and nonconstant mip readback. SwiftShader passed eight cases and RTX compatibility mode passed seven. At the 1080p sRGB path boundary the RTX had two one-code differences among 84 near ties and Intel had one among 2,371. Full RTX and cross-adapter quick suites on the combined revision are still running. The 24 example pages and presets previously passed on the RTX.
