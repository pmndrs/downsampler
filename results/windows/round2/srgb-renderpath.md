# sRGB through render passes: the render path (2026-11-16)

A third way to write sRGB levels, next to the buffer path and `colorSpace: 'srgb'`
([srgb-alias.md](srgb-alias.md)): one render pass per level and layer, as three.js does, but the
fragment shader is the exact compute kernel's first output (`firstWgsl`: the same 'exact' or 'box'
rule, reads and rounding). It returns each level's value as stored (for sRGB, the exact decode of the
rounded code), so the attachment's encode lands on that code and the bits equal the compute path's.

`output: 'auto'` now takes it for sRGB textures with RENDER_ATTACHMENT usage (and `colorSpace: 'srgb'`
textures that have it), for 'average' chains in place under 'exact' or 'box'. Other chains (folds,
'spd', a separate source) keep the buffer path. `output: 'render'` asks for it on any render format.

## Same bits

`srgb/render` (GPU suite): each level equals the compute path's with one-level batches and no tail
(the same rule per level), value for value: rgba8unorm-srgb at 8 shapes, bgra8unorm-srgb,
`colorSpace: 'srgb'`, and rgba8unorm, bgra8unorm, rg8unorm, r8unorm, rgba16float, r16float,
rgb10a2unorm, r32float and rgba32float; modes sample, 'load', 'box', sub-range, rect, layers,
f16Rounding 'device'. float32 formats agree within a few ulp of the inputs: with no rounding step,
a fragment and a compute shader may contract the bilinear arithmetic differently (Intel UHD: 1e-7).
The load path's codes match the oracle (55 near-ties). RTX 3060: 151 chains; compatibility mode: 132
(the layer index needs `@interpolate(flat, either)` there); Intel UHD: pass.

## Speed

`--compare=srgb --strategies=render,buffer,colorSpace,renderPath,renderDevice`, 11 rounds × 32.
RTX: [rtx3060-srgb-renderpath.json](rtx3060-srgb-renderpath.json), log
[rtx3060-srgb-renderpath.txt](rtx3060-srgb-renderpath.txt), under the GPU lock: VALID, started at
46 °C, no throttling. Intel UHD: [intel-srgb-renderpath.json](intel-srgb-renderpath.json) (not
clock-locked; ratios are paired per round). Two earlier RTX runs were throttled while another
project's GPU tests ran ([rtx3060-srgb-renderpath-throttled.json](rtx3060-srgb-renderpath-throttled.json)):
same ordering, times not usable.

Ratios to the buffer path (the library's sRGB path until now); under 1 is faster.

| Size | RTX render passes | RTX `colorSpace` | **RTX render path** | RTX `'device'` | Intel render passes | Intel `colorSpace` | **Intel render path** | Intel `'device'` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 960×540 | 0.38 | 0.73 | **0.63** | 0.41 | 0.20 | 0.85 | **0.84** | 0.22 |
| 1129×635 | 0.43 | 0.75 | **0.63** | 0.45 | 0.19 | 0.87 | **0.89** | 0.20 |
| 1280×720 | 0.53 | 0.79 | **0.75** | 0.56 | 0.17 | 0.91 | **0.83** | 0.19 |
| 1707×960 | 0.55 | 0.82 | **0.71** | 0.57 | 0.16 | 0.88 | **0.92** | 0.18 |
| 1920×1080 | 0.53 | 0.76 | **0.66** | 0.55 | 0.16 | 0.88 | **0.92** | 0.18 |
| 2560×1440 | 0.57 | 0.80 | **0.68** | 0.58 | 0.16 | 0.90 | **0.85** | 0.18 |
| 1024² | 0.58 | 0.83 | **0.79** | 0.60 | 0.17 | 0.93 | **0.85** | 0.19 |
| 4096² | 0.73 | 0.74 | **0.76** | 0.72 | 0.39 | 0.83 | **0.74** | 0.40 |

RTX 1920×1080 in ms: render passes 0.063, buffer 0.120, `colorSpace` 0.090, render path 0.079, `'device'` 0.065.

- **The render path is faster than the buffer path on both GPUs**: 0.63–0.79× (RTX), 0.74–0.92× (Intel). As fast as `colorSpace: 'srgb'` or faster, and it needs no change to how textures are created.
- **Plain render passes are still faster**: 1.05–1.64× on the RTX, about 5× on Intel at render sizes.
- **The difference is the exact sRGB rounding, not how levels are written.** `srgbRounding: 'device'` (the attachment encodes, the sampler decodes, no tables) is the same render path without it: within 0–7 % of plain render passes on the RTX and 3–12 % on Intel. On Intel, the exact rounding is most of an sRGB chain's time in every path (`pow` plus table lookups per texel).
- `averageAccess: 'load'` on the render path is slow on the RTX (1.5–2.3× the buffer path, an earlier throttled run): four loads, each snapped through the table, per fragment.

## What `'device'` costs

Its `check` compares every level's bytes with the exact render path ([bench JSON](rtx3060-srgb-renderpath.json), `checks.renderDevice`):

- RTX 3060: about 16 % of the written bytes differ (1920×1080: 431,852 of 2.76 M), by up to 3 codes, from level 1 on.
- Intel UHD: about 6 % (177,180 at 1920×1080), by up to 2 codes.

The differences come from the hardware's encode (not correctly rounded) and decode/filter precision, and they compound down the chain. Each level is still within the declared tolerance of the oracle (`srgb/render`, 30 levels).

## Open

- The M5 Pro, AMD and mobile: the same bench and `srgb/render`.
- Whether `srgbRounding` should default to `'device'` for three.js-style use (PROGRESS, waiting on owner).
