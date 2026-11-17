# three.js compute mipmaps on Windows (round 2, 2026-11-15)

The three.js branch `compute-mipmaps` on the owner's fork (DennisSmolek/three.js, `2ee2404442`, base upstream dev
`3f7edc3d75`), which vendors `dist/mipmap.js` @ `da3e609`. Until now it had run only on the M5 Pro (72/72).

RTX 3060 Laptop (nvidia ampere, 592.00) and Intel UHD (gen-12lp, 31.0.101.5333), Chrome (puppeteer 25) headless,
D3D12, each GPU pinned with `--use-adapter-luid` (`test/gpu-choice.js`). Fresh `npm ci` + `npm run build` of the
branch in a scratch clone. Two local edits, not committed anywhere: the test takes extra Chrome args from
`GPU_ARGS`, and `KEEP_GOING=1` turns the two tolerance asserts into reports, so all three device configurations
run (by default the first miss stops the run). Logs: `three-parity-{rtx,intel}.txt` (`--verbose`, per level).

## Parity test (`npm run test-webgpu-mipmaps`)

24 cases × 3 configurations (default limits, `?limit=4`, `?compat`) per GPU.

**Hard checks pass everywhere (72/72 on each GPU):** expected path (compute, or the render fallback for sRGB and
everything in compatibility mode), `STORAGE_BINDING` only on compute textures, identical level 0, no GPU
validation errors, no page errors.

**Tolerances (the PR's "≤ 1 step per level, ≤ 2 over a chain") miss in 8 of 144 case runs (4 cases):**

| GPU | Config | Case | Per level | Chain |
|---|---|---|---:|---:|
| RTX | default | rgba8unorm 1920×1080 | 1.000 | **3** |
| RTX | default | rgba8unorm 255×257 | 1.000 | **3** |
| RTX | default, `limit=4` | rgba16float 1129×635 | **1.058** | **1.058** |
| RTX | `limit=4` | rgba8unorm 1920×1080 | 1.000 | **3** |
| RTX | `limit=4` | rgba8unorm 255×257 | 1.000 | **4** |
| Intel | default, `limit=4` | rgba16float 1920×1080 | **1.058** | **1.058** |

(The rgba16float rows are one case each in two configurations.) Compatibility mode passes on both: every texture
falls back to render passes there.

- **rgba8 chains (RTX).** Every level stays within one step of a render pass from the same input; the
  differences accumulate over the chain to 3–4 steps. At 1920×1080 the even levels differ only at exact ties
  (377,538 at L1, all ties: NVIDIA's render passes round 2×2 ties differently from our round-to-even); at
  255×257 the odd levels differ off ties (sampler weights). The M5's chain bound of 2 was empirical; per level
  is the contract that holds on all three GPUs. Intel rgba8 chains stay at 2.
- **rgba16float, RTX 1129×635.** Level 1 only (an odd 1129 → 564 step, sampled by both paths): 7,861 values
  1.058 steps apart, none at ties; later levels ≤ 0.125. Most likely NVIDIA's coarse bilinear weights at slightly
  different sample positions (interpolated UVs against computed texel centers), the same precision limit as the
  sampled-sRGB failure in our own suite.
- **rgba16float, Intel 1920×1080.** Level 1 is an even 2:1 step, and 948,331 values differ by up to 1.058 steps
  (then 0.87, 0.50 … down the chain). Our compute even levels are exact 2×2 averages with round-to-nearest-even,
  checked against the CPU oracle by the library's suite on this GPU, so the difference is most likely Intel's
  render-pass path filtering rgba16float at reduced precision. Not verified against an oracle in three's test.

Neither kind is a kernel bug, but the PR text's tolerances and "Results on Apple M5 Pro: 72 of 72" need Windows
numbers before the PR opens: either widen the chain bound (3–4 steps on NVIDIA) and the float per-level bound
(slightly above 1 step), or compare against a CPU reference instead of render passes. Owner's call (PROGRESS,
*Waiting on owner*).

## Example (`examples/webgpu_textures_mipmaps_compute.html`)

Loaded on each GPU for 5 s at 1280×720 (`three-example-{rtx,intel}.png`): renders on both, left (render) and
right (compute) halves look the same, no WebGPU or page errors. The only console error is a 404 for
`/favicon.ico` from the test server.

## Reproduce

```sh
git clone https://github.com/DennisSmolek/three.js -b compute-mipmaps && cd three.js && npm ci && npm run build
# local edit: append ...process.env.GPU_ARGS.split(' ') to the launch args in test/webgpu/mipmaps-compute.js
GPU_ARGS=--use-adapter-luid=0,82986 KEEP_GOING=1 node test/webgpu/mipmaps-compute.js --verbose   # LUIDs: test/gpu-choice.js
```
