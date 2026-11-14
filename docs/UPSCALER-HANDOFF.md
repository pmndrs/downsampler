# Upscaler handoff (for when pmndrs/upscaler is picked up)

This covers what the library offers the upscaler and where the evidence is. No upscaler code is changed from this repository; the work happens in the upscaler project.

## Consuming the library before npm

Use a local reference until the package is published (the upscaler releases after that):

```
# in downsampler
npm run build
# in upscaler
npm install ../downsampler      # "downsampler": "file:../downsampler"
```

Imports come from `downsampler/webgpu`: `Reducer`, `ScalarReduction`. The library emits plain WGSL and records into the caller's encoder or compute pass (`encode`, `encodePass`), so the upscaler keeps its own pipeline order.

API shape as of 0.1.0 (one source shape, docs/guide.md#sources-one-shape-everywhere): a `ScalarReduction` run takes `{ width, height, bindGroup }` with a load hook (it was `size: [ w, h ]`) and `source: texture | { texture, mipLevel }` without one (it was `texture` + `mipLevel`); a `Reducer` dispatch's layer count is `layerCount` (it was `layers`). The old keys throw with a message naming the new ones.

## Pass by pass

Each row has a working version in `test/tools/generator.js`, checked against the upscaler's own pass as its oracle, with timings in `test/tools/README.md`. Timings are from an M5 Pro; Windows numbers come from the `windows-testing` branch.

| Upscaler pass | Library version | Fixture | Measured (M5 Pro) |
|---|---|---|---|
| Shading change (`shadingChange.ts`, fused 16×16 block statistics) | `Reducer`, `'clamp'` at a padded size, levels [2, 3], 8 channels (`DsWide`: 4 sums + max), the upscaler's load and its store as hooks | `generator/upscaler-shading-change` | 0.95–1.02× the hand-fused pass (a tie) |
| Exposure, the same 1024 taps | `ScalarReduction` with a load hook over the tap grid | `generator/upscaler-exposure-taps` | 0.22–0.32× the serial loop, identical values |
| Exposure, every texel | `ScalarReduction` over the whole image | `generator/upscaler-exposure` | 0.33–0.60× the serial loop; values change |
| FSR-style luma pyramid (if wanted) | `Reducer` [1, 6] sums → combine-only `ScalarReduction`, 2 dispatches | `generator/luma-pyramid-fused` | 0.74–0.92× FSR's every-texel pass |

**Things the fixtures had to work around** (SPEC §14.1):
- **Clamped loads with ⌈n/2^l⌉ blocks:** run `'clamp'` on the size rounded up to whole source tiles, clamp inside the load hook, and bound the store hook to the real outputs.
- **Luma history written from the load hook:** safe only when every texel is loaded exactly once, which is `'clamp'` at whole source tiles.
- **The signal, max(4×4, 8×8):** the store hook keeps `var<workgroup>` state across levels. This is a documented guarantee.

## Upscaler-side items already prepared

- **Branch `perf/exposure-parallel`** (local worktree on the owner's Mac, based on the upscaler's `origin/main`, not pushed): the same 1024 taps on 64 lanes plus a tree sum.
  - 6–10× faster on the M5 Pro.
  - Bit-identical exposure on 1440 bench frames.
  - Windows commands are in its `docs/windows-followups.md`.
  - It doesn't use the library, so it can merge on its own.

## Open decisions

- **Every-texel exposure** (FSR-like, steadier) vs the 1024 taps (identical to today). Judge flicker on the bench scenes.
- **Shading change:** the 8-channel version is decided; the sign-bit variant stays an experiment here.
