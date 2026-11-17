# three.js: cube render-pass mipmaps in compatibility mode (2026-11-17)

PROGRESS *Next* had: "in compatibility mode its render-pass mips of a 33² rgba8 cube differ by up to 6 steps from
the same faces as a 2d-array (`main_cube` half-texel at odd sizes?)". It is a three.js bug, but not a half-texel
shift and not about odd sizes. **Only the 1×1 level is wrong**: three's cube pass returns one texel of the 2×2 level
instead of their average.

Intel UHD (gen-12lp) only, Chrome headless pinned with `--use-adapter-luid=0,66946`; every run checked that all
adapters (default, high-performance, compatibility) were Intel before starting.

## Cause

`src/renderers/webgpu/utils/WebGPUTexturePassUtils.js` (three.js `origin/dev` and `compute-mipmaps`, same lines):

```js
// line 120, main_cube
return textureSample( imgCube, imgSampler, faceMat[ Varys.vBaseArrayLayer ] * vec3f( fract( Varys.vTex ), 1 ) );

// line 130
this.mipmapSampler = device.createSampler( { minFilter: GPUFilterMode.Linear } );
```

1. A 1×1 render target still runs a 2×2 quad. The three helper invocations sit at `vTex` 1.5, and `fract( 1.5 )` is
   0.5, the live pixel's own coordinate. So the derivatives are zero.
2. Zero derivatives mean LOD < 0, so the sampler magnifies.
3. The sampler sets only `minFilter`; `magFilter` defaults to `'nearest'`. The 2×2 → 1×1 step returns texel (1, 1)
   of the level above.

The 2d and 2d-array passes have no `fract()`, so their derivatives stay non-zero and they always minify. Odd 1×1
steps (3×3 → 1×1) are not affected: nearest and bilinear both land on the center texel. Larger odd targets get large
(not zero) derivatives at their right and bottom edges, which still minify.

Only compatibility mode reaches `main_cube` in practice: three picks the entry point from
`textureGPU.textureBindingViewDimension`, which Chrome reports only on compatibility devices (`undefined` on core,
so core cubes use `main_2d_array`).

## Evidence

Repro: [three-compat-cube/compat-cube-mips.html](three-compat-cube/compat-cube-mips.html) runs three's own
`WebGPUTexturePassUtils.generateMipmaps` on the same random rgba8 faces as a cube and as a 6-layer 2d-array, and
checks each level against a CPU reference of render-pass semantics (bilinear at each destination texel center,
from that path's own previous level). Logs: `three-compat-cube/three-compat-cube-{before,after}.txt`.

Before the fix (compatibility mode; core devices with a cube view give the same numbers):

| Size | Levels that differ | 1×1 level: cube vs 2d-array | cube vs reference | 2d-array vs reference | cube 1×1 equals texel (1,1) |
|---|---|---:|---:|---:|---:|
| 33² | L5 only | 9 steps (20 of 24 values) | 9.25 | 0.5 | 24 / 24 |
| 32² | L5 only | 8 | 7.75 | 0.5 | 24 / 24 |
| 17² | L4 only | 21 | 20.75 | 0.5 | 24 / 24 |
| 65² | L6 only | 6 | 6.25 | 0.5 | 24 / 24 |
| 64² | L6 only | 5 | 5.0 | 0.5 | 24 / 24 |
| 31², 63² | none (end 3² → 1²) | 0 | 0 | 0 | – |

All other levels are bit-identical between the two paths. Random data makes the 1×1 error large; three's parity test
uses smoother content, where the same step showed 2 steps here (and 6 on the owner's run).

In three's own parity test (`test/webgpu/mipmaps-compute.html` with the compatibility per-level check turned on
locally), `cube render target rgba8unorm 33²` fails before the fix: L5 1×1×6 differs by 2 steps (18 values); L1–L4
are 0.

## Fix

Branch `compat-cube-mips` in the worktree `C:\Users\denni\Documents\GitHub\three.js\.claude\worktrees\compat-cube`
(from `origin/compute-mipmaps` `bbe7ddb65e`), commit `332ba74c4c`, not pushed:

```js
this.mipmapSampler = device.createSampler( { minFilter: GPUFilterMode.Linear, magFilter: GPUFilterMode.Linear } );
```

The views have one mip level, so LOD only chooses between the min and mag filter; with both linear the result no
longer depends on derivatives. Downsampling 2d passes always minify, so they don't change.

After the fix (Intel, compatibility and core):

- Repro: cube and 2d-array bit-identical at every level of 33², 32², 31², 17², 65², 64², 63², 256², 512²
  (except 96², below); the 1×1 levels are 0.5 steps from the reference, like the 2d-array.
- Three's parity test with the compatibility per-level check on: the 33² cube is 0 steps at every level.
- Three's parity test as committed (`npm run test-webgpu-mipmaps`, pinned to Intel): 72/72, 1.7 min.

Removing `fract()` instead also fixed it on Intel, but then the helper invocations point into a neighboring face, and
derivatives across faces are up to the implementation. The sampler change doesn't depend on that.

## Not a bug: up to 1 step at sizes like 96²

At 96² (also in rgba16float) the cube and 2d-array paths differ by up to 1 step at every level (e.g. 96² → 48²:
8,030 of 55,296 rgba8 values). 96 → 48 samples exactly halfway between texels at centers like (i + 0.5) / 48, which
aren't exact in binary. The cube path rebuilds the face coordinate from a direction (`1 - 2u`, then the hardware's
face projection), so it rounds differently, and a tiny shift moves an 8-bit filter weight by 1/256. Both paths are
equally close to the reference (rgba8 max 1.25 steps each). 32², 33², 64², 256² and 512² are bit-identical. This is the sampler's precision, not a three.js issue.

## Upstream issue draft (not filed)

> **WebGPURenderer: Cube map mipmaps use nearest filtering for the 1×1 level in compatibility mode**
>
> **Description**
>
> In compatibility mode, cube textures get their mipmaps from `main_cube` in `WebGPUTexturePassUtils`. The last level
> (2×2 → 1×1) is not the average of the 2×2 level: it is one of its texels.
>
> `main_cube` samples with `fract( Varys.vTex )`. At a 1×1 target the helper invocations of the 2×2 quad are at
> `vTex` 1.5, which `fract()` maps back to 0.5, so the derivatives are zero and the sampler magnifies. The mipmap
> sampler only sets `minFilter: 'linear'`, so magnification uses the default `magFilter: 'nearest'`.
>
> 2d and 2d-array textures are not affected (no `fract()`). Neither are core devices in Chrome, which use the
> 2d-array path for cubes.
>
> **Reproduction steps**
>
> 1. Request a compatibility-mode device (`featureLevel: 'compatibility'`).
> 2. Create a 32×32 cube render target with mipmaps and fill it with noise.
> 3. Read back the 2×2 and 1×1 levels.
> 4. The 1×1 value equals texel (1, 1) of the 2×2 level (24 of 24 channel values on Intel UHD, D3D12), up to 21
>    8-bit steps off the average. The same faces as a 2d-array texture give the average.
>
> **Fix**
>
> ```js
> this.mipmapSampler = device.createSampler( { minFilter: GPUFilterMode.Linear, magFilter: GPUFilterMode.Linear } );
> ```
>
> The source views have a single mip level, so this only changes the magnification case. With it, cube and 2d-array
> mipmaps are bit-identical at every level (Intel UHD, compatibility and core devices).
>
> **Version** dev
>
> **Device / OS / Browser** Intel UHD (gen-12lp), Windows 11, Chrome (headless, D3D12)

## Reproduce

Copy `three-compat-cube/compat-cube-mips.html` to `test/webgpu/` of a three.js checkout, then
`node three-compat-cube/run-cube.mjs <three.js checkout> "?compat" "-"` (`?format=rgba16float`, `?sizes=33,32`;
`run-cube.mjs` loads puppeteer from a fixed path, and the Intel LUID changes per boot: set `INTEL_LUID`).

## On three's `dev` (2026-11-17)

The same line cherry-picks cleanly onto `upstream/dev` (`0d2bd48586`): local branch `fix-compat-cube-mipmaps`
(`4bf8797c22`), not pushed. Same repro on the Intel UHD (Chrome 156, D3D12, compatibility device):

- Before: only the 1×1 level differs, 5–21 steps (17², 32², 33², 64², 65²), and it equals texel (1, 1) of the 2×2
  level in 24 of 24 values ([dev-before](three-compat-cube/three-compat-cube-dev-before.txt)).
- After: cube and 2d-array identical at every level: rgba8 at 17², 31²–33², 63²–65², 256², 512², and rgba16float
  ([dev-after](three-compat-cube/three-compat-cube-dev-after.txt)).
- `npx eslint` on the file clean; `npm run test-unit` 1999 pass.
- Not run: the RTX, other browsers, other formats. On core devices Chrome reports no
  `textureBindingViewDimension`, so three sends cubes down the 2d-array path; only compatibility devices reach `main_cube`.

PR text: [three-compat-cube-pr.md](three-compat-cube-pr.md). Nothing filed.
