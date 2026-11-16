# How downsampler compares

If you already make mipmaps another way, this is what changes when you switch, and where we chose to do something
different. Each point links to the full record in [DIVERGES.md](../DIVERGES.md), with the measurements behind it.

- [Coming from render-pass mipmaps (three.js, WebGL, most engines)](#coming-from-render-pass-mipmaps)
- [Coming from AMD SPD, FSR or webgpu-spd](#coming-from-amd-spd-fsr-or-webgpu-spd)
- [Coming from nvpro_pyramid](#coming-from-nvpro_pyramid)
- [Building a Hi-Z pyramid](#building-a-hi-z-pyramid)
- [What we don't do (yet)](#what-we-dont-do-yet)

## Coming from render-pass mipmaps

**Your mipmaps look the same.** By default every level is what one render pass per level gives you, including the
odd sizes. You can swap downsampler in without a visual change.

**It's faster, except for sRGB.** A full chain takes:

| GPU | rgba8 | rgba16float | sRGB color |
|---|---|---|---|
| Apple M5 Pro | 3–27× faster | 4.5× faster at 1024² | not measured yet |
| NVIDIA RTX 3060 Laptop | 0.77–0.95× the time | 0.77–1.01× | **1.05–1.6× slower**; the same with `srgbRounding: 'device'` |
| Intel UHD | 0.48–1.05× | 0.62–0.96× | **about 5× slower**; the same with `srgbRounding: 'device'` |

Times are for screen sizes from 960×540 to 2560×1440, plus 1024² and 4096². Cube maps gain the most (3–4× on the RTX,
27× on the M5) because every face goes through the same dispatches. sRGB textures can't be written by compute shaders,
so we draw them with render passes too. The difference is that we round every level exactly, so the results are the
same on every GPU; that rounding is what costs. `srgbRounding: 'device'` lets the GPU round instead: the same speed as
render passes, with codes up to 3 steps apart between GPUs ([guide](guide.md#srgb-color)). ([#16](../DIVERGES.md#16-srgb-through-the-buffer-output-path), [#17](../DIVERGES.md#17-compute-only-with-no-per-device-render-fallback))

**The texture needs `STORAGE_BINDING`.** That's how compute shaders write the levels. It costs nothing when you sample
the texture later: we measured that on Apple, NVIDIA and Intel. ([#18](../DIVERGES.md#18-textures-need-storage_binding))

**Half-float textures come out slightly more accurate.** On Windows and Apple, the GPU's own conversion to half floats
truncates. We round to nearest, so every GPU gives the same result. It costs 3–9 % of an rgba16float chain;
`f16Rounding: 'device'` turns it off. ([#25](../DIVERGES.md#25-float16-levels-are-rounded-to-nearest-not-truncated))

**Results can differ by a step between GPUs.** That's true of render passes too: each vendor's sampler rounds its
filter weights a little differently. If you need identical bits everywhere, use `averageAccess: 'load'`
([getting started](getting-started.md#the-same-results-on-every-gpu)).

## Coming from AMD SPD, FSR or webgpu-spd

The kernel is SPD's idea, a 64 × 64 tile reduced through six levels in workgroup memory, written independently in
WGSL. [Notes for SPD users](spd-users.md) map every SPD option to ours. The differences that matter:

- **Two dispatches, not one.** SPD's last workgroup finishes the chain using an atomic counter. WGSL doesn't
  guarantee that handoff works, and on Apple we saw webgpu-spd return stale levels in up to 77 % of frames. Two
  dispatches are also no slower. ([#1](../DIVERGES.md#1-no-single-dispatch-or-atomic-handoff))
- **Odd sizes match render passes by default.** SPD drops the last row or column. If you want SPD's results (it's
  the fastest), use `oddSizes: 'spd'`. ([#3](../DIVERGES.md#3-odd-sizes-match-render-passes))
- **Min/max stays conservative.** SPD's dropped rows and columns make its Hi-Z miss texels; ours covers every texel
  at every size. ([#4](../DIVERGES.md#4-conservative-minmax-footprint-at-odd-edges))
- **No zero reads past the edge.** SPD reads zeros there unless you pad the texture; AMD's own D3D12 sample darkens
  the 1×1 that way. We clamp. ([#5](../DIVERGES.md#5-clamped-edges-no-zero-reads))
- **Each level is rounded as it's stored**, so a chain equals level-by-level generation. Under `oddSizes: 'spd'` we
  keep SPD's rounding. ([#7](../DIVERGES.md#7-every-level-is-rounded-as-stored))
- **No size cap.** SPD stops at 4096² and 12 levels. ([#21](../DIVERGES.md#21-no-size-or-level-cap))
- **Partial updates are exact.** SPD snaps a sub-rectangle to 64-texel tiles; we rewrite only the texels that
  depend on it. ([#15](../DIVERGES.md#15-sub-rectangles-exact-per-level-regions))
- **Integer formats average exactly.** webgpu-spd's integer averages truncate and overflow. ([#24](../DIVERGES.md#24-integer-averages-are-exact-rounded-to-nearest))

## Coming from nvpro_pyramid

`oddSizes: 'box'` gives nvpro's area weights, so every level keeps the image's mean; it takes 1.2–2× the time of the
default. We don't carry odd sizes inside one dispatch by default: the `halo` option does, as nvpro's halo does, and
it's 0–15 % faster on NVIDIA but not on Intel, so it stays opt-in for now. ([#2](../DIVERGES.md#2-a-batch-ends-at-every-odd-size), [#3](../DIVERGES.md#3-odd-sizes-match-render-passes))

## Building a Hi-Z pyramid

Many engines build a power-of-two pyramid from a screen-size depth buffer with one 2 × 2 sample per texel. At
1920 → 1024 a texel's area covers up to 3 depth texels per axis, so that shortcut misses some and can cull visible
objects. `resample: 'area'` takes every depth texel each pyramid texel touches. ([#23](../DIVERGES.md#23-power-of-two-hi-z-from-a-render-size-source))

## What we don't do (yet)

| Not supported | Why | Workaround |
|---|---|---|
| Wrap edges (tiling textures) | Clamp only; wrap would change odd-size borders and break render-pass parity | A faint seam at odd sizes only ([#6](../DIVERGES.md#6-no-wrap-edge-mode)) |
| Alpha-coverage preservation | Not built; it fits with the transform hooks that are planned | Cutout foliage may thin out in the distance ([#9](../DIVERGES.md#9-no-alpha-coverage-preservation)) |
| Video frames (`texture_external`) as a source | Not built | Copy the frame into a texture first ([#14](../DIVERGES.md#14-no-texture_external-source)) |
| Subgroup or fp16 kernels | Measured: not faster, and fp16 loses precision | — ([#10](../DIVERGES.md#10-no-subgroup--wave-operations), [#11](../DIVERGES.md#11-no-packed-fp16-math)) |

AMD and mobile GPUs haven't been measured yet. If you try it there, `npm run lab` runs the test suite on your device.
