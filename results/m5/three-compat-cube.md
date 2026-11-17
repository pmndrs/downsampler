# three.js compat cube fix on Metal (M5 Pro, 2026-11-17)

The Windows finding ([three-compat-cube.md](../windows/round2/three-compat-cube.md)) checked on the Apple M5 Pro
(Chrome headless, Metal; every adapter, including the compatibility one, reported `apple metal-3`).

- Repro: `compat-cube-mips.html` from the Windows run, copied into a detached three.js worktree.
  Runner: [three-compat-cube/run-cube-mac.mjs](three-compat-cube/run-cube-mac.mjs), the Windows `run-cube.mjs` with no
  adapter LUID, puppeteer from a local three.js checkout, and an Apple adapter check.
- Before: fork `origin/compute-mipmaps` `bbe7ddb65e` ([log](three-compat-cube/m5-cube-before.txt)).
  After: fork `origin/compat-cube-mips` `332ba74c4c`, `magFilter: GPUFilterMode.Linear` on the mipmap sampler
  ([log](three-compat-cube/m5-cube-after.txt)).
- Runs: `?compat`, core (`-`), `?compat&format=rgba16float`; after the fix also `?compat&sizes=256,512,96`.

## Before the fix

Only the 1×1 level differs, and the cube's 1×1 equals texel (1, 1) of the 2×2 level in 24 of 24 values at every size
that ends 2² → 1². The numbers are the Intel UHD's, value for value (same seeded data):

| Size | Levels that differ | 1×1: cube vs 2d-array | cube vs reference | 2d-array vs reference | cube 1×1 = texel (1,1) |
|---|---|---:|---:|---:|---:|
| 33² | L5 only | 9 steps (22 of 24 values) | 9.25 | 0.5 | 24 / 24 |
| 32² | L5 only | 8 (23 / 24) | 8 | 0.5 | 24 / 24 |
| 17² | L4 only | 21 (24 / 24) | 20.75 | 0.5 | 24 / 24 |
| 65² | L6 only | 6 (20 / 24) | 6 | 0.5 | 24 / 24 |
| 64² | L6 only | 5 (19 / 24) | 5 | 0.5 | 24 / 24 |
| 31², 63² | none (they end 3² → 1²) | 0 | 0 | 0 | – |

The same in compatibility mode and on a core device with a cube binding view (the repro forces `main_cube`), and in
rgba16float (17²: 21.2 steps; cube 1×1 = texel (1,1) 24 / 24).

## After the fix

Cube and 2d-array are bit-identical at **every level of every size** (128 level comparisons: 33², 32², 31², 17², 65²,
64², 63² in compat, core and compat rgba16float, plus 256², 512², 96² in compat). The 1×1 levels are 0.5 steps from the
reference (rgba16float: 0.062), like the 2d-array.

Unlike the Intel UHD, 96² is bit-identical too: Metal's face projection rounds the reconstructed coordinate the same way
as the 2d-array path, so the up-to-1-step sampler difference seen on D3D12 doesn't appear here.

Verdict: the one-line fix holds on Metal; the upstream issue draft in the Windows write-up stays as is (add "also Apple
M5 Pro, Metal" to its device line when it is filed).
