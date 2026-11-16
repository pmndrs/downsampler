# sRGB through an aliased rgba8unorm texture (2026-11-16)

The trick native engines use (D3D12 typeless + UNORM UAV, Vulkan mutable format), in WebGPU: the texture is
**`rgba8unorm` with `viewFormats: ['rgba8unorm-srgb']`**. The kernel reads the level above through an
`rgba8unorm-srgb` view (hardware decode) and writes sRGB codes straight into the rgba8unorm levels as storage. No
buffer, no copies. Materials sample the sRGB view, so filtering still decodes first.

WebGPU rejects the other direction (an sRGB texture with a UNORM storage view: sRGB isn't a storage format). This
direction is valid in Chrome 148. The sRGB view has to name its usage (`usage: TEXTURE_BINDING`), because the
texture's STORAGE_BINDING isn't valid for an sRGB view. Compatibility mode rejects it: "viewFormat must match
format in compatibility mode".

Bench: `aliasPrototype` in test/bench/srgb.js (`--compare=srgb --strategies=render,alias`). It has the library's
plan and kernels (`output: 'storage'`), on an aliased twin of each bench texture.

## Same bits

Every level of the aliased texture equals the buffer path's, byte for byte, at all 8 sizes (the strategy's `check`).

## Speed, RTX 3060 Laptop

[rtx3060-srgb-alias.json](rtx3060-srgb-alias.json), log [rtx3060-srgb-alias.txt](rtx3060-srgb-alias.txt), under the
GPU lock: VALID, no throttling, SM median 1507 MHz under load. 21 rounds × 64 repeats.

| Size | buffer (library) | alias | render | alias ÷ buffer | render ÷ alias |
|---|---:|---:|---:|---:|---:|
| 960×540 | 0.066 ms | 0.048 | 0.025 | 0.72 | 0.52 |
| 1129×635 | 0.081 | 0.061 | 0.035 | 0.75 | 0.58 |
| 1280×720 | 0.074 | 0.057 | 0.039 | 0.78 | 0.68 |
| 1707×960 | 0.094 | 0.076 | 0.052 | 0.81 | 0.69 |
| 1920×1080 | 0.102 | 0.072 | 0.056 | 0.70 | 0.78 |
| 2560×1440 | 0.139 | 0.097 | 0.092 | 0.70 | 0.94 |
| 1024² | 0.057 | 0.046 | 0.034 | 0.81 | 0.73 |
| 4096² | 0.538 | 0.427 | 0.369 | 0.79 | 0.87 |

- The alias path takes 19–30 % less time than the buffer path. That's the copy, as `srgb-cost.md` predicted (25 %).
- Render passes are still faster on NVIDIA, but the gap is about half what it was: at 1080p, 0.016 ms against
  0.046 ms, and nearly a tie at 1440p. What's left is the shader's sRGB maths and the sRGB source read.

## Sampling cost

`--compare=storage-usage --formats=rgba8unorm-srgb` ([rtx3060-srgb-alias-sampling.json](rtx3060-srgb-alias-sampling.json),
VALID). It compares a plain rgba8unorm-srgb texture (what three.js makes today) with the aliased one, sampled through
its sRGB view: post pass, nine trilinear material taps, and rendering into it, on rendered and noise content.

**1.000× in every case** (range 0.994–1.005, all sizes). On the RTX the aliased texture costs nothing to sample or
render into.

## What it needs

- **The library:** a way to say "this rgba8unorm texture holds sRGB color" (an option, since a GPUTexture doesn't
  expose its `viewFormats`), and an sRGB source view with an explicit usage. The kernels exist (`output: 'storage'`).
- **three.js:** create 8-bit sRGB color textures as rgba8unorm with the sRGB view format, and give materials the
  sRGB view. Compatibility-mode devices keep render passes.
- **Unmeasured:** Intel, the M5 Pro (where the copy was the main cost from 1080p up), AMD and mobile; sampling cost
  off NVIDIA (AMD and Apple can lose compression on reinterpretable textures).

## Built: `colorSpace: 'srgb'`

`downsampler.encode( encoder, texture, { colorSpace: 'srgb' } )` on an rgba8unorm or bgra8unorm texture with its sRGB
view format. [rtx3060-srgb-colorspace.json](rtx3060-srgb-colorspace.json) (VALID, SM median 1635 MHz), same bytes as
the buffer path at every size:

| Size | buffer | prototype | `colorSpace` | render |
|---|---:|---:|---:|---:|
| 960×540 | 0.066 ms | 0.048 | 0.048 | 0.025 |
| 1920×1080 | 0.098 | 0.070 | 0.070 | 0.056 |
| 2560×1440 | 0.141 | 0.099 | 0.099 | 0.092 |
| 1024² | 0.057 | 0.046 | 0.046 | 0.033 |
| 4096² | 0.536 | 0.426 | 0.431 | 0.369 |

0.70–0.80× of the buffer path at all 8 sizes, the prototype's speed. Render passes take 0.52–0.93× of it
(1.1–1.9× faster).
