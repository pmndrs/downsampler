# Changelog

The package follows [semantic versioning](https://semver.org); before 1.0, a minor version may change the API.

## 0.1.0 (unreleased)

The first release.

### `Downsampler`: mip chains and Hi-Z (`downsampler/webgpu`)

- Mip chains with compute shaders, up to six levels per dispatch (64 × 64 tiles in workgroup memory, SPD's idea in WGSL), dependent batches separated by dispatch boundaries. No single-dispatch mode (WGSL doesn't guarantee its handoff).
- Odd sizes: `'exact'` (default, render-pass and WebGL semantics), `'box'` (area weights, as nvpro_pyramid) and `'spd'` (AMD SPD semantics, averages only). A tail kernel finishes small `'exact'` levels in one dispatch (`tail`, on by default).
- Reductions: `'average'`, `'min'`, `'max'`, `'minmax'` and custom idempotent folds; folds are conservative at odd sizes, so min/max pyramids are a valid Hi-Z.
- Separate sources (depth → Hi-Z, format conversion, any float-sampleable 2D texture): same size, half size, or any size with `resample: 'area'` (power-of-two Hi-Z; ratio cap `maxResampleRatio`, default 8).
- Partial updates: `layers: [ first, count ]` and `rect`, bit-identical to a full recompute; an empty rect is a no-op.
- `encode`, `encodePass` (into your compute pass), `generate`, `prepare` (the planned dispatches, for self-dispatch) and `release`.
- Two output paths (storage textures, or a buffer and copies for formats that can't be storage textures) and a compatibility-mode path, all bit-identical.
- Identical results on every GPU where it matters: `f16Rounding: 'nearest'` (default) rounds float16 levels in the shader; `averageAccess: 'load'` replaces the sampler's vendor-specific weights.
- Formats: `rgba8unorm`, `rgba8snorm`, `rgba16float`, `r32float`, `rg32float`, `rgba32float`, `rgb9e5ufloat`; with `texture-formats-tier1` the 8- and 16-bit one- and two-channel formats, the 16-bit normalized formats, `rg11b10ufloat` and `rgb10a2unorm` storage; `bgra8unorm` with `bgra8unorm-storage`; `rgba8unorm-srgb` and `bgra8unorm-srgb` through the buffer path (an optional shared `scratch` buffer); integer formats with exact averages (round to nearest, ties to even) and bitwise folds.

### Tool kernels (`downsampler/webgpu`)

- `Reducer`: one dispatch reduces up to six levels per tile and stores the levels you choose. Per-channel ops (sums, averages, min, max, weighted means, argmin/argmax with payloads), 4 or 8 channels, `'zero-weight'` or `'clamp'` (SPD/FSR) edges, `load`/`store`/`size` hooks over your own `@group(1)`, output textures or a buffer.
- `ScalarReduction`: any image to one value per layer in two dispatches, every texel counted once, with a `finish` hook on the GPU; combine-only runs over partials a `Reducer` wrote; `bindGroupLayouts[ 1 ]` for the hooks' bind group, as on a `Reducer`.
- One source shape everywhere: `source` is a texture or `{ texture, mipLevel }`, with the same checks; sizes are `width`/`height`; layer ranges are `layers: [ first, count ]` (a bare count is `layerCount`).
- The WGSL generators (`createKernel`, `createScalarKernel`, …) for vendoring, and `requiredLimits( adapter )`. Kernels that read or write sRGB take the exact sRGB tables from a 2 KiB storage buffer at binding 12 (`srgbTable( device )`, `usesSrgbTable( options )`, `srgbTableBinding`); `Downsampler` and `Reducer` bind it. As shader constants the tables made sRGB chains 4× slower and pipelines 8–19× slower to compile on D3D12.

### three.js (`downsampler/three`)

- `generateMipmaps( renderer, texture, options )` for `WebGPURenderer` (storage textures, or `COPY_DST` textures on the buffer path), `unsupportedReason`, `getDownsampler` (shared per device and option set), `getGPUTexture` and `requiredLimits`. `three` is an optional peer, typed structurally.

### Vendoring (`downsampler/mipmap`)

- One self-contained file for in-place `'average'` chains (`createMipmapKernel`, `planMipmapBatches`, `kernelDefaults`, …), the same kernels as `Downsampler`, checked against golden WGSL.

### Core (`downsampler`)

- Backend-neutral planning, region math, format codecs and the CPU oracles every kernel is tested against (`reduceReference`, `boxReference`, `areaReference`, `spdReference`, `reducePyramid`, `scalarReference`, …).
