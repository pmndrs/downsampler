# float32 averages: sample or load (2026-11-17, DIVERGES #13)

The `Downsampler` samples `'average'` chains of r32float, rg32float and rgba32float when the device has
`float32-filterable`, like every other filterable format, and loads them (four `textureLoad`s, shader weights)
when it doesn't. Until now neither the timing nor the sampled path's correctness had been checked: the test
device leaves the feature off, and most apps don't request it.

## Timing

`--compare=float32` (new, `test/bench/float32.js`) requests `float32-filterable` and times the default
Downsampler (`sample`) against `averageAccess: 'load'`, paired per round, 11 rounds × 32 repeats, at the usual 8 sizes.

- RTX 3060 Laptop: [rtx3060-float32.json](rtx3060-float32.json), under the GPU lock: VALID (started at 49 °C, no throttling).
- Intel UHD: [intel-float32.json](intel-float32.json), not clock-locked; ratios are paired per round.

sample ÷ load (under 1 is faster):

| Size | RTX r32float | RTX rg32float | RTX rgba32float | Intel r32float | Intel rg32float | Intel rgba32float |
|---|---:|---:|---:|---:|---:|---:|
| 960×540 | 0.84 | 0.74 | 0.93 | 0.75 | 0.78 | 0.91 |
| 1129×635 | 0.83 | 0.84 | 0.96 | 0.71 | 0.86 | 0.93 |
| 1280×720 | 0.86 | 0.93 | 0.99 | 0.85 | 0.85 | 0.81 |
| 1707×960 | 0.88 | 0.92 | 0.98 | 0.69 | 0.80 | 0.92 |
| 1920×1080 | 0.90 | 0.90 | 0.98 | 0.81 | 0.82 | 0.93 |
| 2560×1440 | 0.94 | 0.95 | 0.99 | 0.82 | 0.98 | 0.93 |
| 1024² | 0.86 | 0.92 | 1.00 | 0.88 | 0.92 | 0.70 |
| 4096² | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 0.98 |

- **Sampling is never slower** (0.69–1.00×), and it helps most for one- and two-channel formats at render sizes,
  where the loads' extra fetches and arithmetic matter more than bandwidth. rgba32float and 4096² are bandwidth-bound and about even.
- This matches nvpro's finding (sampler 26–32 % faster on NVIDIA) for the smaller formats.

## Correctness

New GPU group `downsampler/float32-sample`: a device made with `requiredFeatures( adapter )`, so with
`float32-filterable`, and r32float, rg32float and rgba32float chains under the tail kernel, without it, and
under `'box'`, at 8 shapes (6 with `--quick`). Each level is checked against the oracle from the GPU's level
above: a batch's first output with the sampler's weight tolerance, other levels with shader weights. It also
checks that the chains really sample. RTX 3060: 72 chains, 495 levels, pass.

## What changed

- `requiredFeatures( adapter )` (new, next to `requiredLimits`, also in `downsampler/three`): the optional features the
  library uses that the adapter has: `float32-filterable`, `texture-formats-tier1`, `bgra8unorm-storage`. The
  README, getting-started and the guide's Setup use it.
- No kernel or planning change: the library already sampled float32 with the feature.

## Open

- The M5 Pro: `npm run bench -- --compare=float32 --skip-validation`.

three.js needs nothing: its WebGPU backend requests every feature the adapter supports (`WebGPUBackend.js`, upstream dev), `float32-filterable` included, so `downsampler/three` already samples float32 there.
