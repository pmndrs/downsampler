# Halo batches (round 2, 2026-11-14)

SPEC §13 item 3, DIVERGES #2. Built as an opt-in `Downsampler` option (`halo: true`); by the
handoff's rule (default only if ≥ 5 % faster at render sizes on both GPUs) it stays opt-in.

RTX 3060 Laptop (nvidia ampere, driver 592.00) and Intel UHD (gen-12lp, 31.0.101.5333), Chrome 148
headless, D3D12, on AC, Turbo plan, GPU idle (45–54 °C), nothing else running on the GPU.

## What it is

An odd step reads the taps `2x … 2x+2` of the previous level, so a tile of that level lacks one texel at
its far edge, and today's `'exact'` and `'box'` end a batch at every odd level. A halo batch has each
workgroup also compute the texels past its tile that the next odd steps read: per level and axis,
`e_i = 2·e_(i+1) + (1 if step i → i+1 is odd)`, so 1, 3, 7 … texels per further odd step (nvpro's
structure). Even steps are the plain kernel's 2×2 combine; odd steps take the 'exact' rule at global
positions (shader bilinear weights, the fold footprint, area weights under `'box'`), the same as a new
batch's first output with `averageAccess: 'load'`.

**Results.** Bit-identical to the plain chain wherever both use shader weights (`'load'`, folds, `'box'`
with `'load'`) in exact formats (normalized, float16, packed, sRGB); within an ulp in float32 where a level
moves between kernels. With the default `'sample'`, an odd step inside a batch uses shader weights where the
plain chain's next batch would sample: within one step, as the tail kernel already is.

**Planner.** At each level the plain batch, or a halo batch through 1 … n further odd steps, while
`(tile + e_x)(tile + e_y)` fits workgroup memory (1024 vec4f at the default 16 KiB), adds at most 25 % to
the tile, and the batch's first output is at most 128² texels (`haloMaxOutput`). Of all plans, the one with
the fewest dispatches wins; ties go to fewer extra texels, then fewer halo batches. So a halo is taken only
where it saves a dispatch, on the smallest level that does. Halo tiles are 16 (16 × 16 invocations).

| Size | plain (tail on) | halo | halo, uncapped (`haloAny`) |
|---|---:|---:|---:|
| 960×540 | 4 | 3 (halo at 120×67) | 3 |
| 1129×635 | 5 | 4 (70×39) | 3 (564×317) |
| 1280×720 | 3 | 3 | 3 |
| 1477×831 | 6 | 5 (92×51) | 3 (738×415) |
| 1707×960 | 4 | 4 | 3 (426×240) |
| 1920×1080 | 4 | 3 (120×67) | 3 |
| 2560×1440 | 3 | 3 | 3 |
| 3840×2160 | 4 | 3 (120×67) | 3 |

## Timings

`npm run bench -- --compare=render --strategies=render,tail,halo,haloAny`, rgba8unorm / rgba16float,
medians of 11 rounds × 16 repeats, ratios to the default chain (`tail`). Two runs per GPU agreed within
0.01 on the RTX and ±0.05 on Intel (`rtx-halo-{1,2}.json`, `intel-halo-{1,2}.json`).

| Size | RTX halo | RTX haloAny | Intel halo | Intel haloAny |
|---|---|---|---|---|
| 960×540 | **0.85 / 0.88** | 0.85 / 0.88 | 0.98–1.00 / 1.01–1.03 | same |
| 1129×635 | **0.88 / 0.90** | 0.96 / 0.89 | 0.93–0.97 / 0.98 | **1.43–1.51 / 1.38–1.52** |
| 1280×720 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | 0.99–1.02 |
| 1477×831 | **0.92 / 0.93** | 0.91 / 0.82 | 0.99–1.01 / 1.00 | **1.38–1.44 / 1.39–1.47** |
| 1707×960 | 1.00 / 1.00 | 1.15 / 1.10 | 0.97–1.00 / 0.86–1.05 | **1.24–1.27 / 1.12–1.29** |
| 1920×1080 | **0.92 / 0.94** | 0.92 / 0.94 | 1.00–1.01 / 1.02–1.11 | 0.98–0.99 / 1.01–1.06 |
| 2560×1440 | 1.00 / 1.00 | 1.00 / 1.00 | 1.01–1.08 / 0.95–0.99 | same |
| 3840×2160 | 0.97 / 0.98 | 0.97 / 0.98 | 1.00–1.01 / 0.96–1.02 | 0.93–0.97 / 1.00–1.03 |

- **RTX:** 0.85–1.00, never slower; 8–15 % where a dispatch goes.
- **Intel:** neutral (0.93–1.11, median 1.00); the dispatch saved costs about what the halo adds.
- **Uncapped, Intel loses 24–52 %** where the halo lands on a large level (564×317, 738×415, 426×240): a
  16-texel tile there, which Intel reduces much slower than 32 (`exact16 ÷ exact` 1.14 median, up to 1.41,
  in `intel-render.json`). The cap removes every loss.
- `'box'` with halo (`haloBox`, `--compare=spd`, uncapped then): RTX 0.91 / 0.96 of `tailBox`, Intel
  1.06 / 1.04.

## Verification

- Core: 125/125 (halo plan tests: invariants, budget, dispatches saved with and without the cap,
  separate sources, `'spd'` rejected). WGSL without a halo byte-identical (golden test).
- GPU (`--only=halo`, `test/browser/validate-halo.js`): RTX full shapes 356 cases, Intel quick 156 — every
  mode bit-identical to the plain chain (float32 within 2⁻²⁰), the sample path against the oracle, rect and
  layer-range updates equal to a full recompute; rgba8unorm, rgba16float (both paths), r32float,
  rgba8snorm, rgba8unorm-srgb, rgb9e5ufloat.
- The first planner (greedy: extend while it fits) took halos that saved nothing and moved levels between
  kernels (129×67: a one-step difference in sampled rgba8 `'box'`, an ulp in r32float); the GPU cases caught
  it, and the dispatch-minimizing planner replaced it.

## Not done

- M5 Pro numbers (Apple's `exactTile` is 16, so halo tiles cost nothing extra there): the deciding run for
  a default. `npm run bench -- --compare=render --strategies=render,tail,halo,haloAny`.
- `src/mipmap.ts` (the vendored copy) has no halo; it would need one only if halo became the default.
- Integer formats ignore `halo` (their kernel has no halo path); `'spd'` rejects it.
