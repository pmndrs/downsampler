# downsampler

GPU texture pyramids for WebGPU: mip chains, min/max (Hi-Z) pyramids from depth, custom tile reductions and reductions to a single value, as generated WGSL compute kernels. Engine-agnostic, zero runtime dependencies, with a three.js adapter on top.

It implements the ideas of AMD FidelityFX SPD independently in WGSL (a 64×64 tile reduced through six levels in workgroup memory, every level written once) and targets SPD's feature set, with two deliberate differences explained below. Every kernel is checked against a CPU oracle on real GPUs.

> Status: pre-release (milestone 4, docs and examples). Not yet on npm. [PROGRESS.md](PROGRESS.md) has the live status; [SPEC.md](SPEC.md) is the design record.

## Install

```sh
npm install downsampler
```

`three` is an optional peer dependency (`>=0.170.0`), needed only for `downsampler/three`.

| Entry point | What it holds |
|---|---|
| `downsampler/webgpu` | `Downsampler`, `Reducer`, `ScalarReduction` and the WGSL generators |
| `downsampler/three` | `generateMipmaps( renderer, texture )` for `WebGPURenderer` |
| `downsampler` | the backend-neutral core: planning, formats and the CPU reference implementations (oracles) |

## Quick start

```js
import { Downsampler, requiredLimits } from 'downsampler/webgpu';

const adapter = await navigator.gpu.requestAdapter();
// Six levels per dispatch need six storage textures (the default limit is four).
const device = await adapter.requestDevice( { requiredLimits: requiredLimits( adapter ) } );
const downsampler = new Downsampler( device );

// texture: any supported format, TEXTURE_BINDING | STORAGE_BINDING usage, level 0 filled.
downsampler.generate( texture );                                  // levels 1 … n, encoded and submitted
downsampler.encode( encoder, texture, { reduction: 'max' } );     // or into your own encoder

// Partial updates: only the texels that depend on a changed region (bit-identical to a full
// recompute), and only some array layers.
downsampler.encode( encoder, texture, { rect: { x: 128, y: 64, width: 32, height: 32 } } );
downsampler.encode( encoder, cubeAs2dArray, { layers: [ 2, 1 ] } );
```

## Concepts

**Mip chains (`Downsampler`).** A chain is split into batches of up to six levels per dispatch (`min( 6, maxStorageTexturesPerShaderStage )`), and dependent batches are separated by dispatch boundaries. Use `baseMipLevel` and `mipLevelCount` for part of a chain, and `prepare( texture )` to see the planned dispatches.

**Odd sizes: `'exact'` by default, `'box'` and `'spd'` opt-in.** Under `'exact'`, a level with an odd, non-unit dimension is re-sampled with normalized-coordinate bilinear filtering, exactly as render-pass and WebGL mipmaps are, so an engine can switch to it without a visual change. `oddSizes: 'spd'` reproduces SPD instead: clamped loads, the last odd row or column dropped, two dispatches up to 4096², and `'average'` only. It is faster at render sizes, but its levels drift from their footprints at odd sizes ([SPEC §2.1](SPEC.md#21-odd-sizes-exact-default-spd-opt-in-measured-box-opt-in)). `oddSizes: 'box'` weights odd sizes by area, as nvpro_pyramid does: every texel contributes in proportion to its overlap, so a 31×31 texture's 1×1 is its true mean. It takes the same dispatches as `'exact'`; use it for GI, irradiance and roughness chains and UI.

**No single-dispatch mode.** SPD's last-workgroup handoff isn't guaranteed by WGSL's memory model (relaxed atomics, workgroup-scoped barriers), was observed to return stale levels in a WGSL port, and measured slower than two dispatches anyway ([SPEC §2.2](SPEC.md#22-no-single-dispatch-mode)).

**Reductions and Hi-Z.** `reduction: 'average' | 'min' | 'max' | 'minmax' | { wgsl }`. Folds combine every texel an output covers, including the extra row or column at odd sizes, so min/max pyramids are conservative: the property Hi-Z occlusion culling needs. `'minmax'` stores (min, max) pairs.

**Separate source.** `source: { texture, mipLevel }` reads the first level from another texture, such as a depth buffer that can't be a storage texture. Destination level `baseMipLevel` is a converting copy (same size) or one reduction step (half size), or any size with `resample: 'area'`, such as a power-of-two Hi-Z whose later levels halve cleanly. Area-resampled min/max fold every source texel a destination texel overlaps, so they stay conservative.

```js
downsampler.generate( hiZ /* rg32float */, { source: { texture: depth }, reduction: 'minmax' } );
downsampler.generate( hiZPow2 /* 2048×1024 r32float */, { source: { texture: depth1080p, resample: 'area' }, reduction: 'min' } ); // 2 dispatches
```

**`Reducer`** is a tile reduction for tools: a WGSL `load` hook over your own `@group(1)` bindings, per-channel ops (`'sum'`, `'average'`, `'min'`, `'max'`, weighted means, argmin/argmax with a payload), up to six levels in one dispatch with only the levels you choose stored, through output textures or a `store` hook. It is how FSR-style luminance and shading-change pyramids, block statistics and G-buffer downsamples are built.

**`ScalarReduction`** reduces a whole image to one `vec4f` per layer in two dispatches at any size, every texel counted once, with an optional `finish` hook that runs on the GPU (exposure, scene statistics).

**`encode` vs `encodePass`.** `encode( encoder, … )` records its own compute passes. `encodePass( pass, … )` records into a compute pass you own, so it can share the pass and one pair of timestamp writes with your other dispatches. `prepare()` returns what `encodePass` would record (pipelines, bind groups, workgroup counts) on all three classes, so a tool can dispatch the kernels itself. `release()` drops cached bind groups and destroys the buffers the library created for them.

**Compatibility mode** (devices without `'core-features-and-limits'`) is handled: each batch writes scratch levels and a copy dispatch moves them in, with bit-identical results (`compatibility: 'auto'`).

**Output paths.** Levels are written as storage textures, or, for formats that can't be (`rgb9e5ufloat`, the sRGB formats; `rg11b10ufloat` without `texture-formats-tier1`), packed into a buffer and copied in. `outputPath( texture )` says which; `output: 'buffer'` forces it; `scratch` shares one buffer across textures.

**sRGB colour.** `rgba8unorm-srgb` and `bgra8unorm-srgb` are reduced in linear light (the sRGB view decodes, the shader encodes each level with correct rounding, alpha stays linear), as render-pass mipmaps of an sRGB texture are, so regular three.js colour textures work.

**Identical results on every GPU.** Two options: `f16Rounding` (default `'nearest'`) rounds float16 levels to nearest even in the shader, because D3D12 truncates; `averageAccess: 'load'` applies bilinear weights in the shader instead of the hardware sampler, whose 8-bit weights round differently per vendor (2–13 % slower on an RTX 3060, 22–31 % on Intel UHD).

**Formats.** `rgba8unorm`, `rgba8snorm`, `rgba16float`, `r32float`, `rg32float`, `rgba32float`, `rgb9e5ufloat`; with `texture-formats-tier1` also `r8unorm`, `rg8unorm`, `r8snorm`, `rg8snorm`, `r16float`, `rg16float`, the 16-bit normalized formats and `rg11b10ufloat` storage; `bgra8unorm` with `bgra8unorm-storage`; `rgba8unorm-srgb` and `bgra8unorm-srgb` through the buffer path. Sources: any float-sampleable 2D format, including depth.

## three.js

```js
import { generateMipmaps, requiredLimits, unsupportedReason } from 'downsampler/three';

const renderer = new WebGPURenderer( { requiredLimits: requiredLimits( adapter ) } );
await renderer.init();
// A StorageTexture, or an sRGB colour texture (buffer path), with generateMipmaps = true and
// mipmapsAutoUpdate = false, level 0 written.
if ( ! generateMipmaps( renderer, texture ) ) console.warn( unsupportedReason( renderer, texture ) ); // then use three's own path
generateMipmaps( renderer, texture, { encoder, downsampler: { averageAccess: 'load' } } );        // into your encoder, not submitted
```

Native support for every three.js texture is planned as an in-tree option ([SPEC §10](SPEC.md#10-threejs-integration)).

## Performance

GPU time per full chain against one render pass per level (the common approach, and what three.js does). All numbers are medians of timestamped runs; differences under ~5 % are noise.

**Apple M5 Pro** (Metal, Chrome 155, 2026-11-10; [SPEC §8](SPEC.md#8-performance-methodology)):

| Size | Format | Render passes (ms) | downsampler (ms) | Speedup |
|---|---|---:|---:|---:|
| 512² | rgba8unorm | 0.0882 | 0.0098 | 9.0× |
| 1024² | rgba8unorm | 0.1159 | 0.0161 | 7.2× |
| 4096² | rgba8unorm | 0.4501 | 0.3371 | 1.3× |
| 1920×1080 | rgba8unorm | 0.1176 | 0.0374 | 3.1× |
| 512²×6 layers | rgba8unorm | 0.6337 | 0.0236 | 26.8× |
| 1024² | rgba16float | 0.1178 | 0.0263 | 4.5× |

**Windows, D3D12** (Chrome 148, render sizes 960×540 … 2560×1440, same-run ratios to WebGPU render passes, lower is faster; [docs/WINDOWS-REPORT.md](docs/WINDOWS-REPORT.md)):

| ÷ render passes | RTX 3060 Laptop rgba8 / rgba16f | Intel UHD rgba8 / rgba16f |
|---|---|---|
| `'exact'` + tail kernel (the default) | 0.98 / 0.90 | 1.02 / 0.63 |
| `oddSizes: 'spd'` | 0.75 / 0.75 | 0.78 / 0.68 |

- Large textures are memory-bandwidth bound for every method: expect parity with render passes at 4096².
- Our `'spd'` mode is 1.26 / 1.27× AMD's own native D3D12 SPD on the RTX 3060 at render sizes (locked clocks), and 1.00× at 4096² ([WINDOWS-REPORT](docs/WINDOWS-REPORT.md#speed-against-native)).
- Against webgpu-spd 3.0.0 with the same semantics, `'spd'` is 1.2–2.1× faster on the M5 Pro and 1.6× / 3.5× faster than its default on the RTX / Intel UHD in rgba8 ([DIVERGES #22](DIVERGES.md)).
- The M5 Pro table predates the tail kernel and tile defaults; it will be re-measured on an idle machine.

## Browser support

Any browser with WebGPU compute and storage textures. Tested so far: Chrome 148–155 on macOS (Apple M5 Pro, Metal) and Windows (RTX 3060 Laptop and Intel UHD, D3D12), compatibility-mode devices, and SwiftShader in CI. AMD, Android, Safari and Firefox haven't been measured yet; device reports from `npm run lab` are welcome.

## Documentation and examples

- [docs/guide.md](docs/guide.md): usage guide with every option.
- [docs/spd-users.md](docs/spd-users.md): coming from AMD SPD, webgpu-spd or FSR.
- [examples/](examples/): mip chain, Hi-Z from depth, `Reducer` block statistics, `ScalarReduction` exposure, three.js. `npm run examples`, then open http://localhost:8080/examples/.
- [SPEC.md](SPEC.md): design decisions and measurements. [DIVERGES.md](DIVERGES.md): every difference from SPD, nvpro and the WebGPU community, and why.

## Development

```sh
npm install
npm test                  # build, unit tests, GPU suite (headless Chrome, real GPU)
npm run test:gpu -- --quick --adapter=swiftshader   # software WebGPU, no GPU needed
npm run test:examples     # load every example on the GPU and check its readout
npm run bench             # validation + benchmark
npm run lab               # device lab at http://localhost:8080/test/browser/
```

## License

MIT
