# vk-mips

Vulkan mip chains, timed on one device. Build and options: [../README.md](../README.md).

## Variants

| Name | Dispatches / passes | What it is |
|---|---|---|
| `nvpro-pyramid` | 2–7 | NVIDIA's nvpro_pyramid with its default dispatcher: the fast pipeline (up to 6 levels, subgroup shuffles) while the size divides by 4, the general pipeline (2 levels, 1-texel halo) otherwise. Area-weighted odd sizes: our planned `'box'` |
| `nvpro-general` | ⌈levels / 2⌉ | The general pipeline only |
| `blit` | levels − 1 | `vkCmdBlitImage` per level with `VK_FILTER_LINEAR`, a layout barrier per level: the common Vulkan pattern and the driver's path |
| `empty` | 0 | Two timestamps with nothing between |

r32float runs max (Hi-Z) with nvpro only: `NVPRO_PYRAMID_REDUCE*` is `max` and loads are texel fetches, so the footprint is the same conservative one as our `'exact'` max.

## Shaders

`shaders/preamble.glsl` defines nvpro's load, reduce and store macros, modelled on nvpro's `srgba8_mipmap_preamble.glsl` but storing linear rgba8unorm / rgba16float / r32float directly (nvpro's own is sRGB). Average uses the bilinear `NVPRO_PYRAMID_LOAD_REDUCE4`, which nvpro asks for in any speed comparison. `shaders/pyramid.comp` includes it and `nvpro_pyramid.glsl`; CMake compiles the fast and general pipeline for each format with glslangValidator (Vulkan 1.1 target, for subgroup shuffles).

Upstream: vk_compute_mipmaps `6c248f554e91329875e9860f2258019a32b44ee9`, `nvpro_pyramid/` (`nvpro_pyramid.glsl`, `nvpro_pyramid_dispatch.hpp`). Apache-2.0; nothing copied here.
