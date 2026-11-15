# Where an sRGB chain's time goes on the RTX 3060 (2026-11-15)

`node test/run.js --bench --compare=srgb-cost --skip-validation --power=high --rounds=21 --repeats=64` under the GPU
lock ([rtx3060-srgb-cost.json](rtx3060-srgb-cost.json), log [rtx3060-srgb-cost.txt](rtx3060-srgb-cost.txt); VALID, no
throttling, SM median 1365 MHz under load). test/bench/srgb.js: the library's buffer path on the same
rgba8unorm-srgb texture with one part of the shader's sRGB work edited out of the WGSL. Timing only: every variant
but `buffer` and `fastStore` writes wrong values. Ratios are paired per round against `buffer`.
An earlier run without the lock ([rtx3060-srgb-cost-unlocked.json](rtx3060-srgb-cost-unlocked.json)) had the GPU near
idle clocks; its absolute times are 2–4× off, but its ratios agree within a few percent.

| Size | buffer (ms) | noCorrect | noPow | noSnap | noEncode | linearBuffer | linearStorage | render |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 960×540 | 0.092 | 0.69 | 0.89 | 0.75 | 0.54 | 0.47 | 0.26 | 0.31 |
| 1280×720 | 0.090 | 0.71 | 0.89 | 0.78 | 0.57 | 0.48 | 0.29 | 0.43 |
| 1920×1080 | 0.118 | 0.74 | 0.91 | 0.82 | 0.68 | 0.51 | 0.26 | 0.48 |
| 2560×1440 | 0.173 | 0.73 | 0.89 | 0.81 | 0.69 | 0.49 | 0.24 | 0.54 |
| 1024² | 0.071 | 0.71 | 0.90 | 0.80 | 0.59 | 0.44 | 0.28 | 0.50 |
| 4096² | 0.728 | 0.68 | 0.94 | 0.82 | 0.61 | 0.45 | 0.21 | 0.51 |

(1129×635 and 1707×960 in the JSON, in line with these.)

- `noCorrect`: `ds_srgb_code` without its ±1 correction against the exact thresholds (a storage-buffer read per channel).
- `noPow`: `sqrt` in place of `pow(c, 1/2.4)`, correction kept.
- `noSnap`: in-tile values not snapped through encode/decode; stores still encode.
- `noEncode`: no sRGB maths in the shader at all.
- `linearBuffer` / `linearStorage`: an rgba8unorm texture of the same size, on the buffer path / the storage path.

## What costs what (1920×1080)

(The parts overlap: the snap includes one of the correction steps.)

| Part | Share of our time |
|---|---:|
| The correction steps (two per stored texel: one in the snap, one in the store) | 26 % |
| In-tile snap (encode + decode of every intermediate value) | 18 % |
| `pow` | 9 % |
| All shader sRGB maths | 32 % |
| Reading an sRGB source rather than rgba8unorm (`noEncode` vs `linearBuffer`) | ~17 % |
| The buffer path's write and copy of each level (`linearBuffer` vs `linearStorage`) | 25 % |
| A linear rgba8 chain on the storage path | 26 % |

Render passes take 0.48× our time here. Even with no sRGB maths at all, our chain is 1.4× render passes: the copies
and the sRGB source read alone cost more than the render path's whole chain. **No shader change gets sRGB to
render-pass speed on NVIDIA**; the buffer path (forced by WebGPU, DIVERGES #16) is a floor.

## A same-bits saving: `fastStore`

Every value a kernel stores has been through `quantize`, so it is exactly `decode(k)` for some code k. The `pow`
formula alone rounds those values back to k: over all 256 codes in f32 on the CPU, the largest miss is 0.000016 of a
code step (against 0.5 to be wrong), far beyond any GPU `pow` error. So the store's correction step can go.

[rtx3060-srgb-faststore.json](rtx3060-srgb-faststore.json) (VALID, SM median 1312 MHz):

| Size | buffer (ms) | fastStore | noCorrect (both steps) | render |
|---|---:|---:|---:|---:|
| 960×540 | 0.081 | 0.81 | 0.68 | 0.31 |
| 1280×720 | 0.087 | 0.83 | 0.71 | 0.44 |
| 1920×1080 | 0.120 | 0.84 | 0.74 | 0.48 |
| 2560×1440 | 0.190 | 0.81 | 0.69 | 0.51 |
| 1024² | 0.071 | 0.83 | 0.70 | 0.50 |
| 4096² | 0.752 | 0.84 | 0.64 | 0.49 |

16–21 % less time, and the bench's parity check (last level against `buffer`) passed. Render passes now take
0.37–0.61× our time instead of 0.31–0.51×.

**Built** (`ds_srgb_stored_code` in `srgbWgsl`, used by the exact kernels' sRGB stores on the buffer and storage paths;
tile kernels, which store unrounded values, keep the correction). Same bits: RTX full matrix (three parts, 35,829
cases, all 3234 groups), RTX compat quick and Intel quick fail only the known groups (the FSR fixtures and the
sampled `rgba8unorm-srgb → rgba16float` source on NVIDIA, the same 27 checks before and after the change).

The built library against the old store (`--strategies=slowStore,render`,
[rtx3060-srgb-built.json](rtx3060-srgb-built.json), VALID): the old store takes 1.19–1.38× the time of the new one at
all 8 sizes, and render passes take 0.38–0.63× ours. The bench's `buffer` is now this version; `slowStore` is the old one.

## What it means for three.js

- On NVIDIA, sRGB textures stay on render passes; `fastStore` narrows the gap but can't close it.
- The plan for three.js stays per format: compute for linear formats (at or below render passes on NVIDIA and Intel,
  much faster on the M5), render passes for sRGB colour textures on NVIDIA-class GPUs.
