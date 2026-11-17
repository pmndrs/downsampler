# Native sRGB mip chains on D3D12 (2026-11-16)

Does native compute beat render passes for sRGB too, and how far is our WebGPU render path
([srgb-renderpath.md](srgb-renderpath.md)) from native? `native/d3d12-mips` now takes
`--formats=rgba8unorm-srgb`. Same sizes, input (random 8-bit codes, xorshift32 seed 1337) and
timing as the other native runs: GPU timestamps around 16 chains, median of 128 rounds.

## What was built

- **SPD, sRGB as the format intends.** An `R8G8B8A8_TYPELESS` texture. The source level is read
  through an `_SRGB` SRV, so the hardware decodes (and the linear sampler filters in linear).
  SPD reduces linear values in LDS or with quad ops. Every store encodes in the shader
  (IEC 61966-2-1 with `pow`) and writes a `UNORM` UAV; D3D12 has no sRGB UAVs. When the last
  workgroup reads mip 6 back, the shader decodes it. All eight SPD variants (atomic or split,
  wave or LDS, load or linear sampler).
- **SPD, AMD's own sRGB handling** (`spd-amdsrgb-atomic-wave-{load,linear}`), from AMD's SPD
  sample (`SPDIntegrationLinearSampler.hlsl`; `SPDCS.cpp`: "can't create SRGB UAV") and
  FidelityFX SDK 1.1.4 (`ffx_spd_callbacks_hlsl.h`). `ffx_spd.h` itself has no sRGB code. In the
  linear-sampler path, AMD samples through the sRGB view, then encodes the result with
  `ffxSrgbFromLinear`. From there SPD averages *encoded* values and stores them unchanged. Only
  level 1 is a linear-light average; levels 2 and up are averaged in gamma space. The load path
  reads the codes through a `UNORM` view and never decodes, so the whole chain is a gamma-space
  average. Also, `ffxSrgbFromLinear`'s `clamp` trick returns the curved segment below linear
  0.0031308 (codes 0–10), so dark values come out low: expected code 4, got 2. Timed for
  reference only; it is not the same result.
- **Render passes:** `_SRGB` RTVs and SRVs, so the hardware decodes, filters and encodes. `render`
  uses the typeless texture, which also allows UAV. `render-rtonly` uses an
  `R8G8B8A8_UNORM_SRGB` render-target-only texture. Both take the same time; the tables give
  the faster of the two.
- **D3D11 `GenerateMips`** on an `R8G8B8A8_UNORM_SRGB` texture.

The semantics check (`--dump`, `check-dump.mjs`) now covers sRGB against the library's `'srgb8'`
oracles. Color differences are counted in codes. At 1129×635:
[RTX](native-d3d12-srgb-semantics-rtx3060.json), [Intel](native-d3d12-srgb-semantics-intel-uhd.json).

- SPD sRGB matches `spdChainReference(…, 'srgb8')` as closely as SPD rgba8unorm matches its own:
  1–2 codes at most. The only values outside tolerance are 4 values at level 10, and
  rgba8unorm has the same 4.
- AMD's variants don't match: 125k values on the sampler path and 695k on the load path are
  outside tolerance (of 952k), up to 79 codes.
- Hardware sRGB render passes and `GenerateMips`: Intel is within tolerance (≤ 3 codes). On the
  RTX, 66 values (render) and 155 (`GenerateMips`) are outside tolerance, up to 13 codes, all at
  levels 1–2. They sit where the odd-width source gives uneven bilinear weights, and most come
  out low. This fits NVIDIA's reduced-precision filtered sRGB decode, which the GPU suite
  already allows for (2⁻⁷ of the value).

## RTX 3060 Laptop (D3D12, driver 32.0.15.9200)

[native-d3d12-srgb-rtx3060.json](native-d3d12-srgb-rtx3060.json): four short runs under the GPU
lock, each **VALID** (started at 48–58 °C, no thermal throttling). The first attempt timed
everything in one 38 s run and was INVALID (73 % throttled):
[native-d3d12-srgb-rtx3060-throttled.json](native-d3d12-srgb-rtx3060-throttled.json), not used.
WebGPU columns: [rtx3060-srgb-renderpath.json](rtx3060-srgb-renderpath.json) (VALID, same day).

ms per full chain. SPD is the fastest of the eight variants.

| Size | SPD rgba8 | **SPD sRGB** | render rgba8 | **render sRGB** | GenerateMips rgba8 | **GenerateMips sRGB** | WebGPU render passes | WebGPU render path | WebGPU `'device'` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960×540 | 0.009 | **0.010** | 0.011 | **0.011** | 0.009 | **0.009** | 0.025 | 0.041 | 0.027 |
| 1129×635 | 0.016 | **0.016** | 0.021 | **0.021** | 0.019 | **0.019** | 0.035 | 0.051 | 0.037 |
| 1280×720 | 0.020 | **0.022** | 0.025 | **0.025** | 0.022 | **0.022** | 0.039 | 0.055 | 0.041 |
| 1707×960 | 0.033 | **0.034** | 0.041 | **0.042** | 0.038 | **0.038** | 0.054 | 0.070 | 0.056 |
| 1920×1080 | 0.040 | **0.042** | 0.051 | **0.051** | 0.046 | **0.047** | 0.063 | 0.079 | 0.065 |
| 2560×1440 | 0.068 | **0.069** | 0.085 | **0.086** | 0.080 | **0.081** | 0.125 | 0.142 | 0.129 |
| 1024² | 0.022 | **0.024** | 0.027 | **0.027** | 0.024 | **0.025** | 0.039 | 0.053 | 0.041 |
| 4096² | 0.279 | **0.282** | 0.350 | **0.352** | 0.345 | **0.346** | 0.366 | 0.384 | 0.368 |

| Size | SPD sRGB ÷ render sRGB | (rgba8: SPD ÷ render) | SPD sRGB ÷ SPD rgba8 | WebGPU render passes ÷ native render | `'device'` ÷ native render | **render path ÷ native render** |
|---|---:|---:|---:|---:|---:|---:|
| 960×540 | 0.86 | 0.77 | 1.10 | 2.24 | 2.41 | 3.67 |
| 1129×635 | 0.77 | 0.74 | 1.04 | 1.68 | 1.76 | 2.44 |
| 1280×720 | 0.86 | 0.80 | 1.09 | 1.54 | 1.61 | 2.16 |
| 1707×960 | 0.83 | 0.79 | 1.05 | 1.30 | 1.35 | 1.68 |
| 1920×1080 | 0.82 | 0.80 | 1.03 | 1.25 | 1.29 | 1.55 |
| 2560×1440 | 0.81 | 0.80 | 1.02 | 1.46 | 1.50 | 1.65 |
| 1024² | 0.86 | 0.83 | 1.05 | 1.43 | 1.49 | 1.96 |
| 4096² | 0.80 | 0.80 | 1.01 | 1.04 | 1.05 | 1.09 |

## Intel UHD (D3D12, driver 31.0.101.5333)

[native-d3d12-srgb-intel-uhd.json](native-d3d12-srgb-intel-uhd.json): the same four runs under the
lock. No clock verdict: the lock's clock check reads only the NVIDIA GPU. WebGPU columns:
[intel-srgb-renderpath.json](intel-srgb-renderpath.json) (not clock-checked either).

| Size | SPD rgba8 | **SPD sRGB** | render rgba8 | **render sRGB** | GenerateMips rgba8 | **GenerateMips sRGB** | WebGPU render passes | WebGPU render path | WebGPU `'device'` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960×540 | 0.082 | **0.096** | 0.059 | **0.058** | 0.056 | **0.055** | 0.101 | 0.422 | 0.112 |
| 1129×635 | 0.104 | **0.122** | 0.079 | **0.078** | 0.074 | **0.075** | 0.132 | 0.635 | 0.142 |
| 1280×720 | 0.132 | **0.155** | 0.095 | **0.094** | 0.089 | **0.092** | 0.152 | 0.735 | 0.166 |
| 1707×960 | 0.214 | **0.251** | 0.155 | **0.154** | 0.149 | **0.151** | 0.236 | 1.331 | 0.262 |
| 1920×1080 | 0.266 | **0.315** | 0.190 | **0.188** | 0.183 | **0.182** | 0.278 | 1.585 | 0.311 |
| 2560×1440 | 0.521 | **0.611** | 0.377 | **0.417** | 0.394 | **0.414** | 0.540 | 2.799 | 0.597 |
| 1024² | 0.138 | **0.163** | 0.102 | **0.104** | 0.097 | **0.098** | 0.164 | 0.820 | 0.180 |
| 4096² | 6.185 | **5.994** | 6.510 | **6.515** | 6.638 | **6.786** | 6.472 | 12.428 | 6.674 |

| Size | SPD sRGB ÷ render sRGB | (rgba8: SPD ÷ render) | SPD sRGB ÷ SPD rgba8 | WebGPU render passes ÷ native render | `'device'` ÷ native render | **render path ÷ native render** |
|---|---:|---:|---:|---:|---:|---:|
| 960×540 | 1.65 | 1.39 | 1.17 | 1.72 | 1.91 | 7.22 |
| 1129×635 | 1.56 | 1.31 | 1.18 | 1.68 | 1.82 | 8.13 |
| 1280×720 | 1.64 | 1.39 | 1.17 | 1.61 | 1.76 | 7.80 |
| 1707×960 | 1.64 | 1.38 | 1.17 | 1.54 | 1.71 | 8.67 |
| 1920×1080 | 1.67 | 1.40 | 1.19 | 1.48 | 1.65 | 8.42 |
| 2560×1440 | 1.46 | 1.38 | 1.17 | 1.29 | 1.43 | 6.71 |
| 1024² | 1.57 | 1.35 | 1.19 | 1.57 | 1.73 | 7.88 |
| 4096² | 0.92 | 0.95 | 0.97 | 0.99 | 1.02 | 1.91 |

On Intel, SPD's best variants are split or LDS; the atomic quad-wave ones take about twice as
long.

## Conclusions

- **RTX: native compute beats render passes for sRGB, as it does for rgba8unorm.** SPD with the
  encode in the shader takes 0.77–0.86× the time of sRGB render passes (1920×1080: 0.042 against
  0.051 ms). It is also faster than `GenerateMips`. The in-shader encode costs SPD 1–5 % from
  1707×960 up and 4–10 % at the smaller sizes. Render passes encode in hardware for free.
- **Intel UHD: render passes win, and sRGB widens the gap.** SPD sRGB takes 1.5–1.7× the time of
  sRGB render passes at render sizes (rgba8unorm: 1.3–1.4×). The `pow` encode adds 17–19 % to
  SPD, but nothing to render passes or `GenerateMips`. Only at 4096² is SPD a little faster (0.92×).
- **AMD's own sRGB handling costs about what rgba8unorm costs.** Its load path converts nothing,
  and its sampler path encodes once per level-1 texel. On the same atomic quad-wave variant it is
  up to 16 % faster than the linear-light SPD on the RTX at 960×540 and within 3 % from
  1707×960 up. On Intel it is 3–13 % faster. It averages in gamma space, so it isn't a correct
  sRGB chain.
- **Our WebGPU render path against native, RTX:** 1.55–2.44× native sRGB render passes from
  1129×635 to 2560×1440 (3.7× at 960×540, 1.09× at 4096²). Against native SPD sRGB it is
  1.9–3.2×. At 1920×1080 it takes 0.079 ms, plain WebGPU render passes 0.063 and native render
  passes 0.051. The gap has two parts of about the same size: WebGPU's per-pass cost (native
  0.051 → WebGPU 0.063) and the exact rounding (`'device'` 0.065 → render path 0.079).
- **Our WebGPU render path against native, Intel:** 6.7–8.7× native render passes at render sizes,
  1.9× at 4096². WebGPU's own overhead is the smaller part: plain WebGPU render passes are
  1.3–1.7× native, and `'device'` is 1.4–1.9×. Most of the gap is the exact rounding
  (1920×1080: 1.585 ms against 0.311 for `'device'` and 0.188 native).
- Native render passes use the hardware's encode, as `'device'` does, so `'device'` is the fair
  native comparison. On the RTX it is 1.3–1.8× native (2.4× at 960×540), and that gap is
  WebGPU's per-pass cost, not color work. Native SPD with an in-shader encode is close to exact
  (≤ 2 codes; `pow` can miss a near-tie). The ~1–5 % that SPD pays for it on the RTX is far less
  than the render path's table-exact rounding costs.
