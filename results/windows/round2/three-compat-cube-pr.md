Fixed #xxx.

**Description**

In compatibility mode, cube textures get their render-pass mipmaps from `main_cube` in `WebGPUTexturePassUtils`. The last step (2×2 → 1×1) does not average the 2×2 level. It returns one of its texels.

Cause: `main_cube` samples with `fract( Varys.vTex )`. At a 1×1 target the three helper invocations of the 2×2 quad sit at `vTex` 1.5, which `fract()` maps back to 0.5, the live pixel's own coordinate. The derivatives are zero, so the sampler magnifies, and the mipmap sampler only sets `minFilter`, so magnification uses the default `magFilter: 'nearest'`.

This PR sets `magFilter: GPUFilterMode.Linear` on the mipmap sampler. The source views have a single mip level, so this only changes the magnification case. 2d and 2d-array passes always minify and are unchanged.

Only compatibility devices reach `main_cube` in practice: on core devices Chrome doesn't report `textureBindingViewDimension`, so cubes use the 2d-array path.

**Before / after**

Random rgba8 faces, mipmapped once as a cube and once as a 6-layer 2d-array with `WebGPUTexturePassUtils.generateMipmaps`, compared level by level (differences in 8-bit steps). Intel UHD (gen-12lp), Windows 11, Chrome 156 headless, D3D12, compatibility device.

| Size | 1×1 level, cube vs 2d-array, before | after |
|---|---:|---:|
| 17² | 21 | 0 |
| 32² | 8 | 0 |
| 33² | 9 | 0 |
| 64² | 5 | 0 |
| 65² | 6 | 0 |

Before, the cube's 1×1 value equals texel (1, 1) of its 2×2 level in 24 of 24 channel values at each of these sizes; every other level already matched. After, cube and 2d-array are identical at every level of 17², 31²–33², 63²–65², 256² and 512² (rgba8), and of 17²–65² in rgba16float. Sizes whose last step is 3×3 → 1×1 (31², 63²) were never affected.

`npx eslint` on the file and `npm run test-unit` pass.
