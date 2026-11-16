# d3d12-mips

D3D12 and D3D11 mip chains, timed on one adapter. Build and options: [../README.md](../README.md).

## Variants

| Name | Dispatches / passes | What it is |
|---|---|---|
| `spd-atomic-wave-load` | 1 | AMD SPD as AMD ships it: all levels in one dispatch, the last workgroup (global atomic counter) finishes levels 7+, quad intrinsics (`QuadReadAcross*`, SM 6.0), source texels loaded |
| `spd-atomic-wave-linear` | 1 | The same, with `SPD_LINEAR_SAMPLER`: one bilinear sample per 2×2 source quad |
| `spd-atomic-lds-*` | 1 | `FFX_SPD_NO_WAVE_OPERATIONS`: LDS reductions, the structure WGSL can express (no quad ops in WGSL) |
| `spd-split-*` | 2 up to 4096² | The same shader, 6 levels per dispatch: dispatch 2 reads stored mip 6 as its source. No atomic counter. This is the structure of our `oddSizes: 'spd'` |
| `render` | levels − 1 | One render pass per level: full-screen triangle, linear clamp sampler, the UVs of test/bench/render-baseline.js; on a texture that also allows UAV, as WebGPU's test textures do |
| `render-rtonly` | levels − 1 | The same on a render-target-only texture (does the UAV flag cost anything? DIVERGES 18) |
| `d3d11-generatemips` | levels − 1 | `ID3D11DeviceContext::GenerateMips`, the driver's own path |
| `empty` | 0 | Two timestamps with nothing between (timer overhead) |

r32float runs max (Hi-Z) with SPD only: `spd-{atomic,split}-{wave,lds}` (texel loads, one LDS channel).

rgba8unorm-srgb (only with `--formats=rgba8unorm-srgb`), averaged in linear light:

| Name | What it is |
|---|---|
| `spd-*` | The SPD variants above on an `R8G8B8A8_TYPELESS` texture: the source level read through an `_SRGB` SRV (the hardware decodes; the linear sampler filters in linear), LDS and quad reductions in linear, every store encoded in the shader (IEC 61966-2-1, `pow`) to a `UNORM` UAV, and mip 6 decoded in the shader when the last workgroup reads it back (`SPD_SRGB`) |
| `spd-amdsrgb-atomic-wave-{load,linear}` | AMD's own sRGB handling, from its SPD sample (`SPDIntegrationLinearSampler.hlsl`, `SPDCS.cpp`) and FidelityFX SDK 1.1.4's `ffx_spd_callbacks_hlsl.h` (`SPD_SRGB_AMD`): the linear-sampler path encodes the sampled (decoded, filtered) value with `ffxSrgbFromLinear`, SPD then averages encoded values and stores them unchanged through `UNORM` UAVs; the load path reads the codes through a `UNORM` view and never decodes. Only level 1 of the sampler path is a linear-light average |
| `render` | `_SRGB` RTVs and SRVs on the `TYPELESS` texture (also allows UAV): the hardware decodes, filters and encodes |
| `render-rtonly` | The same on an `R8G8B8A8_UNORM_SRGB` render-target-only texture |
| `d3d11-generatemips` | `GenerateMips` on an `R8G8B8A8_UNORM_SRGB` texture |

## SPD callbacks

`shaders/spd.hlsl` compiles AMD's `ffx_spd.h` with callbacks in the style of AMD's SPD sample: the source level through an SRV, mip 6 through a `globallycoherent` UAV that the last workgroup reads back, the other levels through a UAV array, and a `globallycoherent` counter buffer reset by the last workgroup. `mips` and the dispatch size come from AMD's `ffxSpdSetup` (`mips = floor(log2(max(w, h)))`, one 64×64 tile per workgroup). Loads past the edge follow D3D (zero), as in AMD's sample.

Upstream: FidelityFX SDK `7539ae8b6568ad29ed78a753404351daf3b5cca7`, `Kits/Cauldron2/framework/shaders/fidelityfx/` (`ffx_core.h`, `spd/ffx_spd.h`). MIT license; nothing copied here.
