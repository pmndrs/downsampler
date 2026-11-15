# What one dependent dispatch costs (2026-11-15)

`npm run bench -- --compare=dispatch --skip-validation --rounds=21` (test/bench/dispatch.js): the same rgba8unorm
chain, same levels and arithmetic, planned at 1–6 levels per dispatch with the tail kernel off, so only the number of
dependent dispatches changes. A least-squares fit of time against dispatches gives the cost of one more dispatch.
`pass` is the 1-level plan recorded in one compute pass (`encodePass`).

**RTX 3060 Laptop** ([rtx3060-dispatch.json](rtx3060-dispatch.json); 3 loaded clock samples, so no verdict):

| Size | lpd6 | lpd5 | lpd4 | lpd3 | lpd2 | lpd1 | Per dispatch |
|---|---|---|---|---|---|---|---:|
| 64² | 5.3 µs [1] | 8.3 [2] | 8.3 [2] | 8.4 [2] | 11.3 [3] | 19.6 [6] | **2.9 µs** |
| 256² | 9.2 [2] | 9.2 [2] | 9.2 [2] | 12.4 [3] | 15.4 [4] | 27.4 [8] | **3.0 µs** |
| 1024² | 32.4 [2] | 32.3 [2] | 35.9 [3] | 39.2 [4] | 42.0 [5] | 59.8 [10] | **3.4 µs** |
| 1920×1080 | 55.5 [7] | 55.5 [7] | 55.7 [7] | 55.5 [7] | 57.8 [8] | 67.5 [10] | **3.9 µs** |
| 4096² | 283 [2] | 286 [3] | 286 [3] | 292 [4] | 308 [6] | 379 [12] | 10 µs (also re-reads a large level from memory) |

[n] = dependent dispatches. 1920×1080 is 7 dispatches at 3–6 levels per dispatch because the chain breaks at its
odd sizes anyway (tail off here).

**Intel UHD** ([intel-uhd-dispatch.json](intel-uhd-dispatch.json)): 3.2 µs (64²) and 3.5 µs (256²). At 1024² and
1920×1080 more dispatches were *faster* (1024²: 139 µs at 6 levels per dispatch, 122 µs at 2): the deep in-tile
levels, where most of a 32 × 32 workgroup waits at barriers, cost Intel more than a dispatch does.

**One compute pass vs a pass per dispatch:** identical on both GPUs (0.999–1.000×), as on the M5 before. The cost
is the dependency (launch, plus waiting for the previous dispatch to finish), not the pass.

## What it means

- A dependent dispatch costs **about 3–4 µs** through Chrome and D3D12 on both Windows GPUs. Round 1's "a render
  step is ~10 µs cheaper than a compute dispatch" (DIVERGES #17) predates the 32 × 32 tiles and overstated it.
- At 1920×1080 the default plan has 4 dispatches, so about 15 µs of a ~50 µs chain on the RTX is dispatch overhead.
  The rest is the work itself, at the low clocks a laptop GPU runs for bursts this short (SM median 1.1–1.5 GHz
  under load in these runs against ~1.6–2 GHz natively).
- The lever left is small: halo batches save one dispatch at 1080p (~4 µs, matching their 0.85–1.00×), and the
  `'spd'` structure's 2 dispatches would save ~8 µs. Below that needs a single dispatch, which WGSL can't make safe
  (DIVERGES #1). Recording into one pass doesn't help.
- Not run on the M5 yet. Running the same mode there would show how much of the M5's faster mid-size chains is a
  cheaper dispatch on Metal.
