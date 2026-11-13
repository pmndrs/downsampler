# downsampler usage guide

Every public API, option and default, checked against the TypeScript source (`src/webgpu/*.ts`, `src/three/index.ts`). The design reasons and measurements live in [SPEC.md](../SPEC.md); this page says how to use the library. Runnable versions of most snippets are in [examples/](../examples/).

- [Setup](#setup)
- [Sources: one shape everywhere](#sources-one-shape-everywhere)
- [Recipes](#recipes)
- [Downsampler: mip chains and Hi-Z](#downsampler-mip-chains-and-hi-z)
- [Reducer: tile reductions for tools](#reducer-tile-reductions-for-tools)
- [ScalarReduction: an image to one value](#scalarreduction-an-image-to-one-value)
- [three.js](#threejs)
- [The core package: oracles and planning](#the-core-package-oracles-and-planning)
- [Coming next](#coming-next)

## Setup

```js
import { Downsampler, Reducer, ScalarReduction, requiredLimits } from 'downsampler/webgpu';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice( {
	requiredFeatures: [ 'texture-formats-tier1' ].filter( feature => adapter.features.has( feature ) ), // optional: more formats
	requiredLimits: requiredLimits( adapter ),
} );
```

`requiredLimits( adapter )` (also `Downsampler.requiredLimits`, and re-exported by `downsampler/three`) returns the limits worth requesting for all three classes:

| Limit | Requested | Why |
|---|---|---|
| `maxStorageTexturesPerShaderStage` | `max( 4, min( 6, adapter's ) )` | one storage texture per level written in a dispatch; 6 gives six levels per dispatch |
| `maxComputeInvocationsPerWorkgroup` | 256, where the adapter has it | 5- and 6-level kernels and `ScalarReduction` use 16 × 16 workgroups; compatibility mode defaults to 128 |
| `maxComputeWorkgroupSizeX` | 256, where the adapter has it | `ScalarReduction`'s combine workgroup is 256 wide |

Without these limits everything still works, with fewer levels per dispatch; only `ScalarReduction` and 5- or 6-level `Reducer`s require 256 invocations, and say so in their error. Not included: a 6-level `Reducer` with 8 channels needs `maxComputeWorkgroupStorageSize: 32768`, which you request yourself.

Generated WGSL identifiers start with `ds_`. Hook code should avoid that prefix, the entry point names (`main`, `partials`, `combine`) and the hook names.

## Sources: one shape everywhere

Every class reads a source the same way: `source` is a `GPUTexture` (read at its level 0) or a `TextureLevel`, `{ texture, mipLevel? }`.

| Where | Field | Extra |
|---|---|---|
| `Downsampler` `encode`/`encodePass`/`generate`/`prepare` options | `source` | `resample: 'area'` (a `SourceOptions`: `{ texture, mipLevel?, resample? }`) |
| `Reducer` dispatch | `source` (and each of `outputs`) | |
| `ScalarReduction` run | `source` | |

Each source passes the same checks, with the same messages: dimension `'2d'`, `textureBindingViewDimension` (where defined) `'2d'` or `'2d-array'`, sample count 1, `TEXTURE_BINDING` usage, a float-sampleable format (depth included, read through its depth aspect; not integer formats or `stencil8`), and `mipLevel` inside the texture.

The other shared names:

- **Sizes** are `width` and `height` fields (Reducer dispatches, ScalarReduction runs). With a source they default to its level's size, and must match it if given.
- **Layer ranges** are `layers: [ first, count ]` (Downsampler options, ScalarReduction runs, `PreparedBatch.layers`, `BufferCopy.layers`). A bare count is `layerCount` (Reducer dispatches, from layer 0; `ReducerCopy.layerCount`).

## Recipes

The lines that matter for each common job, each with a page in [examples/](../examples/) that runs it and checks its own output. `downsampler` is a `new Downsampler( device )`, made once per device; `encoder` is a `GPUCommandEncoder` you submit.

### Mip chain ([mip-chain.html](../examples/mip-chain.html))

```js
const texture = device.createTexture( {
	format: 'rgba8unorm', size: [ width, height ], mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
} );
device.queue.writeTexture( { texture }, pixels, { bytesPerRow: width * 4 }, [ width, height ] );
downsampler.encode( encoder, texture );            // levels 1 … n from level 0, as render-pass mipmaps
downsampler.generate( texture );                   // or: in its own encoder, submitted
```

### Hi-Z from depth, conservative ([hi-z.html](../examples/hi-z.html))

Depth formats can't be storage textures, so the pyramid is a separate `rg32float` (or `r32float`) texture with the depth buffer as its `source`. Odd sizes fold `2x … 2x+2`, so every texel bounds every depth texel under it.

```js
const hiZ = device.createTexture( {
	format: 'rg32float', size: [ width, height ], mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
} );
const pass = encoder.beginComputePass();
downsampler.encodePass( pass, hiZ, { source: depth, reduction: 'minmax' } ); // r = min, g = max
pass.end();
// Conventional depth: occluded if the object's nearest depth is beyond g of the texels it covers (reverse-Z: r).
```

### Power-of-two Hi-Z from a render-size depth buffer ([hi-z-pow2.html](../examples/hi-z-pow2.html))

`resample: 'area'` (with the source as `{ texture, resample }`) lets level 0 be any size (up to 8× smaller per axis by default, `maxResampleRatio`) and folds every depth texel that each Hi-Z texel's area overlaps, so the pyramid stays conservative. Later levels halve cleanly: 12 levels of 2048×1024 take 2 dispatches.

```js
const previousPow2 = n => 2 ** Math.floor( Math.log2( n ) );
const size = [ previousPow2( width ), previousPow2( height ) ];    // 1920×1080 → 1024×1024
const hiZ = device.createTexture( {
	format: 'r32float', size, mipLevelCount: Math.log2( Math.max( ...size ) ) + 1,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
} );
downsampler.encode( encoder, hiZ, { source: { texture: depth, resample: 'area' }, reduction: 'max' } );
```

One 2×2 tap per texel (a min/max reduction sampler at the texel centre) misses depth texels at 1920 → 1024; the page counts them.

### Partial update ([partial-update.html](../examples/partial-update.html))

```js
device.queue.writeTexture( { texture, origin: [ x, y ] }, pixels, { offset: ( y * width + x ) * 4, bytesPerRow: width * 4 }, [ w, h ] );
downsampler.encode( encoder, texture, { rect: { x, y, width: w, height: h } } );   // only the texels that depend on it
downsampler.encode( encoder, faces, { layers: [ 2, 1 ] } );                        // one face of a 6-layer array
downsampler.encode( encoder, faces, { layers: [ 4, 1 ], rect: decal } );           // a decal on face 4
```

Bit-identical to a full recompute when nothing changed outside `rect`. Each distinct rect caches a small uniform buffer; the 16 most recent are kept per texture, and older ones are destroyed once the queue has finished the work submitted so far, so a painter needs no cleanup ([Partial updates](#partial-updates-rect-and-layers)). An empty rect records nothing.

### Area-weighted odd sizes ([odd-sizes.html](../examples/odd-sizes.html))

```js
downsampler.encode( encoder, irradiance, { oddSizes: 'box' } );  // every level keeps the mean: a 31×31's 1×1 is its mean
downsampler.encode( encoder, bloom, { oddSizes: 'spd' } );       // SPD: fewest dispatches, drops the last odd row or column
```

The default, `'exact'`, matches render-pass mipmaps. On the page's 1129×635 test image the mean error against the area-weighted ideal over levels 1–10 is 2.1 (`'exact'`), 5.1 (`'spd'`) and 0.3 (`'box'`) 8-bit steps.

### Block statistics with `Reducer` ([reducer.html](../examples/reducer.html))

```js
const stats = new Reducer( device, {
	reduction: [ 'average', 'average', 'max', 'sum' ],      // mean, mean of squares, max, count
	levels: [ 3, 5 ],                                        // 8×8 and 32×32 blocks, in one dispatch
	format: 'rgba32float',
	bindings: { wgsl: '@group(1) @binding(0) var image: texture_2d<f32>;', entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } } ] },
	load: `fn load( texel: vec2i, layer: u32 ) -> vec4f {
		let l = dot( textureLoad( image, texel, 0 ).rgb, vec3f( 0.2126, 0.7152, 0.0722 ) );
		return vec4f( l, l * l, l, 1.0 );
	}`,
} );
const bindGroup = device.createBindGroup( { layout: stats.bindGroupLayouts[ 1 ], entries: [ { binding: 0, resource: image.createView() } ] } );
const outputs = stats.levels.map( level => device.createTexture( { format: 'rgba32float', size: stats.levelSize( level, width, height ), usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING } ) );
stats.encode( encoder, { width, height, bindGroup, outputs } );
```

### Exposure with `ScalarReduction` ([scalar.html](../examples/scalar.html))

```js
const entries = [
	{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
	{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
];
const metering = new ScalarReduction( device, {
	reduction: [ 'average', 'max', 'min', 'average' ],
	bindings: { wgsl: '@group(1) @binding(0) var hdr: texture_2d<f32>; @group(1) @binding(1) var<storage, read_write> exposure: array<vec4f, 2>;', entries },
	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f { let l = max( dot( textureLoad( hdr, texel, 0 ).rgb, vec3f( 0.2126, 0.7152, 0.0722 ) ), 1e-4 ); return vec4f( log( l ), l, l, l ); }',
	finish: 'fn finish( value: vec4f, weight: f32 ) { exposure[ 0 ] = value; exposure[ 1 ] = vec4f( exp( value.x ), 0.18 / exp( value.x ), weight, 0.0 ); }',
} );
const bindGroup = device.createBindGroup( { layout: metering.bindGroupLayouts[ 1 ], entries: [ { binding: 0, resource: hdr.createView() }, { binding: 1, resource: { buffer: exposureBuffer } } ] } );
metering.encode( encoder, { width, height, bindGroup } );  // two dispatches at any size; the exposure stays on the GPU
```

### three.js ([three.html](../examples/three.html))

```js
import { generateMipmaps, requiredLimits, unsupportedReason } from 'downsampler/three';

const renderer = new THREE.WebGPURenderer( { requiredLimits: requiredLimits( await navigator.gpu.requestAdapter() ) } );
await renderer.init();
const texture = new THREE.StorageTexture( width, height );
texture.generateMipmaps = true;        // allocate the full chain
texture.mipmapsAutoUpdate = false;     // the library writes the mips
texture.minFilter = THREE.LinearMipmapLinearFilter;
// … write level 0 with a compute pass …
if ( ! generateMipmaps( renderer, texture ) ) console.warn( unsupportedReason( renderer, texture ) ); // then three's own path
```

## Downsampler: mip chains and Hi-Z

```js
const downsampler = new Downsampler( device, options );
downsampler.generate( texture );                         // encode + submit
downsampler.encode( encoder, texture, encodeOptions );   // into your encoder, one compute pass per batch
downsampler.encodePass( pass, texture, encodeOptions );  // into a compute pass you own (storage path only)
```

### Textures it takes

- Dimension `'2d'`, sample count 1, `textureBindingViewDimension` (where defined) `'2d'` or `'2d-array'`. Array layers are reduced together, one workgroup z per layer, so all layers take the same dispatches. Cube maps work as 6-layer `'2d-array'` textures (WGSL has no cube storage textures).
- Usage `TEXTURE_BINDING`, plus `STORAGE_BINDING` on the storage path or `COPY_DST` on the buffer path ([Output paths](#output-paths)).
- A supported format:

| Formats | Needs |
|---|---|
| `rgba8unorm`, `rgba8snorm`, `rgba16float`, `r32float`, `rg32float`, `rgba32float` | core |
| `r8unorm`, `rg8unorm`, `r8snorm`, `rg8snorm`, `r16float`, `rg16float`, `r16unorm`, `rg16unorm`, `rgba16unorm`, `r16snorm`, `rg16snorm`, `rgba16snorm` | `texture-formats-tier1` |
| `bgra8unorm` | `bgra8unorm-storage` |
| `rg11b10ufloat` | storage with `texture-formats-tier1`; otherwise the buffer path |
| `rgb9e5ufloat` | always the buffer path (it is never a storage texture) |

Not yet: sRGB formats, `rgb10a2unorm`, integer formats (deferred, SPEC §11). Depth formats can't be destinations: use a separate source into `r32float` or `rg32float`.

`supports( texture )` and `unsupportedReason( texture )` (a string or `null`) say whether it can, and why not: a missing feature, usage or format. `encode`, `generate` and `prepare` throw with the same reason.

### Constructor options (`DownsamplerOptions`)

| Option | Values | Default | What it does |
|---|---|---|---|
| `levelsPerDispatch` | 1 … 6 | storage path: `min( 6, maxStorageTexturesPerShaderStage )`; buffer path: 6; both 4 below 256 invocations per workgroup | Output levels per dispatch. Above the storage texture limit only with `output: 'buffer'`. |
| `output` | `'auto'`, `'storage'`, `'buffer'` | `'auto'` | How levels are written ([Output paths](#output-paths)). |
| `averageAccess` | `'sample'`, `'load'` | `'sample'` | How `'average'` reads each batch's source level: one hardware bilinear sample, or four loads with weights in the shader. `'load'` is identical on every GPU; samplers round their 8-bit weights per vendor. Unfilterable formats and folds always load. |
| `f16Rounding` | `'nearest'`, `'device'` | `'nearest'` | float16 formats: round each level to nearest even in the shader (the same everywhere), or use the device's conversion, which truncates on D3D12 ([float16 rounding](#float16-rounding-f16rounding)). |
| `compatibility` | `'auto'`, `true`, `false` | `'auto'` | The compatibility-mode path ([below](#compatibility-mode)). `'auto'` takes it for textures of a device without `'core-features-and-limits'`. |
| `tail` | boolean | `true` <!-- defaults: tail/exactTile --> | `'exact'` chains: once a level fits one workgroup (32 × 32; 8 × 8 below 256 invocations), finish the chain in one dispatch instead of a batch per odd size. Same rule; tail levels use shader weights (within one rgba8 step of the sampler's). |
| `exactTile` | 8, 16, 32 | 16 on Apple GPUs, 32 on other hardware with 256 invocations per workgroup, 8 on software adapters and in compatibility mode <!-- defaults: tail/exactTile --> | Smallest first-output tile of an `'exact'` batch. Same results; a speed knob measured per vendor. 16 and 32 need 256 invocations. |
| `maxResampleRatio` | a number ≥ 1 | 8 (the exported `maxResampleRatio`) | The largest source ÷ destination size ratio per axis that `source.resample: 'area'` accepts. Only validation reads it: the kernel loops over each texel's span at run time, so it compiles nothing new. The taps per first-level texel grow with its square, (ratio + 1)² at most. |

The constructor validates every option and throws on an unknown value. The resolved values, defaults filled in, are readable as fields (`downsampler.averageAccess` is `'sample'` by default; `.tail`, `.exactTile`, `.levelsPerDispatch`, `.bufferLevelsPerDispatch`, …).

### Encode options (`EncodeOptions`)

| Option | Default | What it does |
|---|---|---|
| `baseMipLevel` | 0 | In place: the level read; outputs start at `baseMipLevel + 1`. With `source`: the first level written. |
| `mipLevelCount` | the rest of the chain | Destination levels involved. In place this **includes** the source level, so `{ baseMipLevel: 2, mipLevelCount: 4 }` reads 2 and writes 3, 4, 5. With `source` it counts written levels. |
| `oddSizes` | `'exact'` | `'exact'`, `'box'` or `'spd'` ([Odd sizes](#odd-sizes)). |
| `reduction` | `'average'` | `'average'`, `'min'`, `'max'`, `'minmax'` or `{ wgsl }` ([Reductions](#reductions)). |
| `source` | none | A texture, or `{ texture, mipLevel = 0, resample? }`: read the first level from another texture; `resample: 'area'` allows any destination size ([Separate source](#separate-source-hi-z-from-depth)). |
| `layers` | every layer | `[ first, count ]`: reduce only these array layers ([Partial updates](#partial-updates-rect-and-layers)). |
| `rect` | the whole level | `{ x, y, width, height }`: the texels that changed; rewrite only what depends on them. Empty (width or height 0): records nothing ([Partial updates](#partial-updates-rect-and-layers)). |

### Odd sizes

A 2:1 reduction is exact only when a dimension halves cleanly. At odd, non-unit sizes:

- **`'exact'` (default).** The odd level ends a batch, and the next level is re-sampled with normalized-coordinate bilinear filtering. Results equal render-pass and WebGL mipmaps (the `reduceReference` oracle), and folds use the conservative `2x … 2x+2` footprint, so Hi-Z is correct. At render sizes this takes more dispatches than SPD; the tail kernel (`tail`, on by default) folds the small trailing levels into one dispatch (1920×1080: 4 dispatches).
- **`'spd'` (opt-in).** AMD SPD's semantics: clamped loads, ⌊n/2⌋ sizes with the last odd row or column dropped, no rounding between the levels of a dispatch, `⌈levels / levelsPerDispatch⌉` dispatches at any size (2 up to 4096²). 1.4–2.1× faster than `'exact'` at render sizes on an M5 Pro, but levels drift from their footprints at odd sizes. **`'average'` only**: min/max under these semantics are not a conservative Hi-Z, so folds are rejected with an error that points to `'exact'` or a `Reducer` with `'zero-weight'` edges.
- **`'box'` (opt-in).** Area-weighted odd sizes, as in nvpro_pyramid: each output texel covers `[x·S/n, (x+1)·S/n)` of the level it reads, and every texel it overlaps contributes in proportion to the overlap. Along an odd size `2n+1` that is three taps `2x … 2x+2` with weights `(n−x, n, x+1)/(2n+1)`; even sizes are the usual 2×2 box. Every level keeps the base level's mean.
  - Same batches and dispatches as `'exact'` (with or without the tail). Batches that read an odd level, and tail batches, load their texels and weight them in the shader; the others are the `'exact'` kernels, so at a power of two `'box'` is `'exact'` bit for bit.
  - Folds (`'min'`, `'max'`, `'minmax'`, custom) are accepted and give exactly the `'exact'` result: their footprint already covers every overlapping texel.
  - Works with the tail kernel, both output paths, compatibility mode, `encodePass` and partial updates (`rect`, `layers`).
  - Recommended for GI, irradiance and roughness chains, and UI.

Like render-pass mipmaps, `'exact'` 3 → 1 takes only the centre texel, so a 31×31 texture can reduce to an unrepresentative 1×1. Under `'box'` its 1×1 is the mean of all 961 texels.

```js
downsampler.generate( irradiance, { oddSizes: 'box' } );
```

### Reductions

| `reduction` | Result | Notes |
|---|---|---|
| `'average'` | bilinear average | matches render-pass mipmaps |
| `'min'`, `'max'` | per channel min / max over every texel the output covers | conservative at odd edges |
| `'minmax'` | even channels min, odd channels max | `rg` = (min, max) and `ba` a second pair; levels read pairs back as already reduced |
| `{ wgsl: 'fn reduce( a: vec4f, b: vec4f ) -> vec4f { … }' }` | a custom fold | must be associative, commutative and **idempotent** (like min, max, bitwise or); values are in normalized units |

Folds select existing values, so they match the CPU oracle exactly. Sums, per-channel op lists and custom `kind: 'sum'` reductions have no meaning under `'exact'` odd sizes and are rejected: use a [`Reducer`](#reducer-tile-reductions-for-tools).

### Separate source: Hi-Z from depth

```js
const hiZ = device.createTexture( {
	format: 'rg32float', size: [ width, height ], mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
} );
downsampler.encode( encoder, hiZ, { source: depth, reduction: 'minmax' } ); // r = min, g = max
```

- The source can be any float-sampleable 2D texture, including depth (read through its depth aspect), with `TEXTURE_BINDING` usage and the destination's layer count. It can't be the destination itself: use `baseMipLevel` to reduce in place.
- Destination level `baseMipLevel` must be the same size as the source level (a converting copy, which loads, so it is exact) or half of it (one reduction step), unless `resample: 'area'` (below). Every destination level in range is written, including `baseMipLevel`.
- **`resample: 'area'`: any destination size**, up to 8× smaller per axis (the `maxResampleRatio` constructor option, default 8), up or down. The usual case is a power-of-two Hi-Z (1024×512 or 2048×1024 from 1920×1080) whose later levels halve cleanly: a 12-level 2048×1024 pyramid takes 2 dispatches. Destination texel x covers `[x·S/n, (x+1)·S/n)` of the source: `'average'` weights every source texel by its overlap, and folds combine every source texel with any overlap, so min/max stay conservative at every level. Later levels follow `oddSizes`. Without the option another size is an error (a stale destination after a resize fails loudly), and the message names it.

```js
// A power-of-two Hi-Z from a 1920×1080 depth buffer: 2048×1024, 12 levels.
downsampler.encode( encoder, hiZ, { source: { texture: depth, resample: 'area' }, reduction: 'max' } );
```
- Single-channel sources (depth, `r*`) are read as `(v, v, v, v)`, so `'minmax'` of depth gives `(d, d)` at level 0.
- Use `'max'` (or the `g` of `'minmax'`) for conventional depth, where an object is occluded if its nearest depth is behind the tile's farthest; `'min'` with reverse-Z.
- `sourceUnsupportedReason( destination, source, baseMipLevel )` checks a pairing without encoding. The source checks are the shared ones ([Sources](#sources-one-shape-everywhere)), plus: not the destination itself, the same layer count, and the size rule.
- Plans are cached per destination and option set, holding only the most recent source texture: recreating a depth buffer on resize replaces the entry.

### Recording: `encode`, `encodePass`, `generate`, `prepare`

| Method | What it does |
|---|---|
| `generate( texture, options? )` | Creates an encoder, encodes, submits. |
| `encode( encoder, texture, options? )` | One compute pass per batch, plus the buffer path's copies after each. |
| `encodePass( pass, texture, options? )` | One dispatch per batch into your compute pass. Dispatches are separate usage scopes, so each batch sees the previous one's writes. Throws on the buffer path (a compute pass can't record copies). |
| `prepare( texture, options? )` | What `encode` records, for dispatching it yourself (or inspecting the plan): `PreparedBatch[]` with `kernel` (`'exact'`, `'tail'` or `'clamp'`), `sourceMipLevel` (`null` for the separate source), `firstMipLevel`, `count`, `outputWidth`/`outputHeight`, `groupsX`/`groupsY`, `layers` (`[ first, count ]`), `pipeline`, `bindGroup`, `output`, and on the buffer path `buffer` and `copies` (`{ mipLevel, offset, bytesPerRow, rowsPerImage, x, y, width, height, layers: [ first, count ] }`). Dispatch them yourself with `dispatchWorkgroups( groupsX, groupsY, layers[ 1 ] )`. Validates and throws like `encode`. |
| `release( texture )` | Drops the texture's cached bind groups and destroys its buffer-path buffers, partial-update uniforms and compatibility scratch textures. Call it after the last submitted use. |

Pipelines are cached per `Downsampler` and compiled synchronously on first use of a format, reduction and batch shape; `prepare()` ahead of time to move that cost.

### Partial updates: `rect` and `layers`

`encode`, `encodePass`, `generate` and `prepare` take:

| Option | Meaning |
|---|---|
| `layers: [ first, count ]` | Reduce only these array layers (2d-array, or a cube viewed as one). Other layers are untouched. |
| `rect: { x, y, width, height }` | The dirty texels of level `baseMipLevel`, or of the source level with a separate `source`. Integers, inside that level. An empty rect (width or height 0) is a no-op, as WebGPU's zero-size copies are: once the other options are validated, it records and creates nothing. |

Each later level rewrites exactly the texels whose footprint touches the previous level's rewritten ones (`2x … 2x+2` under `'exact'`, `'box'` and the tail kernel; each batch's clamped blocks under `'spd'`; a resampled source's first level: every texel whose area span meets the rect). If the source changed only inside `rect` and the old chain was valid, the result is **bit-identical to a full recompute**, and every other texel and layer is untouched. This is tighter than SPD's `rectInfo`, which snaps to 64-texel tiles (DIVERGES #15).

It works on every path: the tail kernel, `'spd'`, the buffer output path, compatibility mode, separate sources and `encodePass`.

**Cost and lifetime.** Each distinct rect and layer range creates one small uniform buffer (a 256-byte slot per dispatch) and its bind groups. The 16 most recent are cached per texture. An evicted one is destroyed once the queue has finished the work submitted so far (`device.queue.onSubmittedWorkDone()`), so many partial updates can be recorded in one encoder, as long as you submit it before awaiting anything; `release( texture )` destroys the rest at once. A rect that changes every frame creates a buffer per frame (a ring buffer with dynamic offsets is a possible later change).

```js
// A decal was painted into level 0 at (128, 64), 32×32: refresh only what depends on it.
downsampler.encode( encoder, texture, { rect: { x: 128, y: 64, width: 32, height: 32 } } );
// One face of a cube baked into a 6-layer 2d-array texture.
downsampler.encode( encoder, faces, { layers: [ 2, 1 ] } );
// Nothing changed this frame: a no-op.
downsampler.encode( encoder, texture, { rect: { x: 0, y: 0, width: 0, height: 0 } } );
```

`prepare( texture, { rect } )` lists what each batch writes: `regions` (per level of the batch, a rect or `null`) and the workgroup offset `groupX`/`groupY`. See [examples/partial-update.html](../examples/partial-update.html), which paints into a texture and checks every update against a full recompute.

### Output paths

| `output` | Levels written | Texture needs |
|---|---|---|
| `'storage'` | one write-only storage texture per level | `STORAGE_BINDING` |
| `'buffer'` | packed texels into one storage buffer, then a `copyBufferToTexture` per level | `COPY_DST`, a format of whole 32-bit words |
| `'auto'` (default) | storage whenever the device can store the format; the buffer path only for formats it can't (`rgb9e5ufloat`, `rg11b10ufloat` without tier1, `rg32float` in compatibility mode) | as above |

`outputPath( texture )` returns `'storage'`, `'buffer'` or `null`. A storable texture without `STORAGE_BINDING` is an error under `'auto'`, not a silent switch to the slower buffer path. The buffer path writes six levels per dispatch at any storage texture limit, and costs one extra write and read of the written levels (1.6–2.3× the storage path where both exist; SPEC §6.1).

### Compatibility mode

In WebGPU compatibility mode (`requestAdapter( { featureLevel: 'compatibility' } )` on a device without `'core-features-and-limits'`) a dispatch may bind only one view of a texture, so an in-place chain can't read level n and write n + 1 together. With `compatibility: 'auto'` the library detects such textures (`usesCompatibility( texture )`), has each batch write per-level scratch textures, and copies each level in with a dispatch. Results are bit-identical, it works with `encodePass`, and `prepare()` lists the copies with `copy: true`. Other compatibility-mode rules: views cover whole textures, view dimensions are fixed at creation (so cube textures can't be written; use 6-layer `'2d-array'`), and `rg32float` has no storage, so it takes the buffer path.

### Identical results on every GPU

By default results may differ between GPUs by a hardware sampler's 8-bit weight step times the local contrast (render passes and `GenerateMips` do the same). For identical bits everywhere:

```js
new Downsampler( device, { averageAccess: 'load' } );  // 2–13 % slower on an RTX 3060, 22–31 % on Intel UHD at render sizes
```

`f16Rounding: 'nearest'` (the default) already makes float16 levels identical; `'device'` opts out ([below](#float16-rounding-f16rounding)).

### float16 rounding: `f16Rounding`

float16 formats (`rgba16float`, `rg16float`, `r16float`) are computed in f32 and rounded to a half once per level, before the level is stored and before the next one reads it. `f16Rounding` (on `Downsampler` and `Reducer`) chooses how:

| Value | How | Result |
|---|---|---|
| `'nearest'` (default) | Round to nearest even in the shader, with integer math on the float's bits (`float16Wgsl`), then store an exact half. | The same bits on every GPU, matching the CPU oracles (`toHalf`). |
| `'device'` | The device's own conversion (`pack2x16float`, and the store). | **Truncates toward zero on D3D12** (NVIDIA and Intel; measured on an RTX 3060 and Intel UHD). On Apple GPUs it rounded to nearest even when measured (M5 Pro, Chrome 148: the suite's tie cases came out bit for bit as under `'nearest'`), but no specification promises either behaviour. |

**What `'device'` costs in accuracy.** Where it truncates, every stored half is up to one f16 step (one unit in the last place, about 0.1 % of the value) closer to zero than the rounded one: positive values come out low by half a step on average, never high. Each level is computed from the previous, already truncated level, so the bias doesn't average out down the chain; the deepest levels of a large texture end up a few steps below the true mean. Images rarely show it. Anything that compares levels numerically does: exposure from a 1×1 mip, a half-precision Hi-Z compared against depth, an oracle-based test suite (it was 211 of 287 full-matrix failures on the RTX 3060 before the default changed).

**When to use it.** Only after measuring the in-shader rounding as a real cost on your target, for chains where a low bias of up to one f16 step per level is acceptable (display-only colour mips, for instance). Measured at render sizes on rgba16float chains, `'nearest'` cost noise on an RTX 3060 and 14–29 % on Intel UHD, depending on the rounding code's version (`results/windows/timings.md`; the current version is being re-measured on Windows). The default stays `'nearest'`; a per-device default would only follow if that cost turns out large.

```js
new Downsampler( device, { f16Rounding: 'device' } );  // the device's conversion: truncation on D3D12
```

## Reducer: tile reductions for tools

A `Reducer` is one compute kernel that reduces up to six levels inside a workgroup tile and stores only the levels you choose. It is the building block for the passes tools fuse around SPD: luminance pyramids, shading-change pyramids, block statistics, G-buffer downsamples. The kernel is compiled in the constructor; pipelines are shared between equal Reducers on a device.

```js
const stats = new Reducer( device, {
	reduction: [ 'average', 'average', 'max', 'sum' ],
	levels: [ 3, 5 ],                                       // 8×8 and 32×32 blocks
	format: 'rgba32float',                                  // output textures (or a store hook)
	bindings: { wgsl: '@group(1) @binding(0) var image: texture_2d<f32>;', entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } } ] },
	load: `fn load( texel: vec2i, layer: u32 ) -> vec4f {
		let l = dot( textureLoad( image, texel, 0 ).rgb, vec3f( 0.2126, 0.7152, 0.0722 ) );
		return vec4f( l, l * l, l, 1.0 );
	}`,
} );
const bindGroup = device.createBindGroup( { layout: stats.bindGroupLayouts[ 1 ], entries: [ { binding: 0, resource: image.createView() } ] } );
const outputs = stats.levels.map( level => device.createTexture( { format: 'rgba32float', size: stats.levelSize( level, width, height ), usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING } ) );
stats.encode( encoder, { width, height, bindGroup, outputs } );
```

### Options (`ReducerOptions`)

| Option | Default | What it does |
|---|---|---|
| `levels` | required | Source-relative levels to emit, strictly increasing, 1 (half size) … 6. Every level up to the last is reduced; only these are stored and get bindings. The deepest sets the tile: 4 levels and fewer use 8 × 8 workgroups, 5 and 6 use 16 × 16 (256 invocations). |
| `reduction` | `'average'` | Four (or eight) per-channel ops, a builtin (`'average'`, `'min'`, `'max'`, `'minmax'`, `'sum'`) expanded to every channel, or a custom `{ wgsl, kind: 'fold' \| 'sum' }`. |
| `edges` | `'zero-weight'` | `'zero-weight'`: off-image texels count nothing, level sizes ⌈n/2ˡ⌉, every texel counted once. `'clamp'`: SPD/FSR, clamped coordinates, ⌊n/2ˡ⌋ sizes, the last odd row or column dropped. |
| `load` | none: read `source` | WGSL `fn load( texel: vec2i, layer: u32 ) -> vec4f`. Coordinates are always in bounds. It may be called more than once for an edge texel, so it must not have side effects (exception: `'clamp'` at a whole number of source tiles). |
| `store` | none: write `outputs` | WGSL `fn store( level: u32, texel: vec2i, layer: u32, value: vec4f )`, called once per emitted texel with `level` counted from the source and `value` normalized. Levels are emitted shallow to deep with a barrier between them, so a store hook may keep `var<workgroup>` state from one level for a deeper one (SPEC §14.1). |
| `size` | a uniform the Reducer manages | WGSL `fn size() -> vec2u`, the source size, with a load hook (e.g. from your own uniforms under dynamic resolution). |
| `format` | none | Output texture format, without a store hook. Any `Downsampler` destination format. |
| `output` | `'storage'` if the device can store `format`, else `'buffer'` | `'buffer'` packs levels into a buffer the Reducer owns and copies them into the outputs (they need `COPY_DST`). Not allowed with a store hook. |
| `f16Rounding` | `'nearest'` | As for `Downsampler`, for float16 output formats. |
| `dimension` | `'2d'` | View dimension of source and outputs; `'2d-array'` reduces every layer. |
| `sourceFormat` | 4 channels | The source texture's format, without a load hook, so single-channel formats (depth, `r*`) are splatted to `(v, v, v, v)`. |
| `bindings` | none | `{ wgsl, entries }`: your `@group(1)` declarations and their `GPUBindGroupLayoutEntry[]`, used by the hooks. |
| `channels` | 4 | 4 (a `vec4f` per texel) or 8 (`DsWide { lo: vec4f, hi: vec4f }`; needs both a load and a store hook). Eight channel ops imply 8. |
| `label` | `downsampler reducer <edges> <levels>` | Debug label. |

### Channel ops

| Op | Emitted value |
|---|---|
| `'sum'` | Σ v |
| `'average'` | Σ v / n, n the texels the edge policy counts (in-image texels under `'zero-weight'`, 4ˡ under `'clamp'`) |
| `'min'`, `'max'` | min / max |
| `'weight'` | Σ w: this channel weighs the `'weighted'` channels (at most one) |
| `'weighted'` | Σ w·v / Σ w, and 0 where Σ w = 0 |
| `'argmin'`, `'argmax'` | the key of a payload fold: the texel with the smallest (largest) key wins |
| `'payload'` | carried from the winning texel; must follow a key or another payload. Ties on the key go to the smaller payload, so put a texel index first for row-major ties. |

Custom reductions: `kind: 'fold'` (default) must be idempotent; `kind: 'sum'` is combined exactly once per counted texel and emitted as is. Nothing is rounded between levels: each emitted level is the reduction of its own source block, rounded once by the store.

### Dispatch (`ReducerDispatch`)

| Field | Default | What it is |
|---|---|---|
| `width`, `height` | the source level's size | Source size in texels. Required with a load hook; with a source, it must match if given. |
| `source` | — | A texture or `{ texture, mipLevel }` ([Sources](#sources-one-shape-everywhere)), without a load hook. |
| `outputs` | — | One texture (or `{ texture, mipLevel }`) per emitted level, in `levels` order, without a store hook. Each must be at least `levelSize( level, width, height )`. Under `'zero-weight'` a mip of a source-sized texture is one texel short at odd sizes, so use separate textures; `'clamp'` sizes equal mip sizes. |
| `layerCount` | the source's (or first output's) layers with `'2d-array'`, else 1 | Array layers to reduce, from layer 0 (workgroup z). A count, not `[ first, count ]`: the tile kernel has no layer offset, and hooks see the workgroup's layer. |
| `bindGroup` | — | Your group 1, created from `reducer.bindGroupLayouts[ 1 ]` or an equal layout. Required when the Reducer has `bindings`. |

### Recording and self-dispatch

| Member | What it does |
|---|---|
| `encode( encoder, dispatch )` | Its own compute pass, then the buffer-output copies. |
| `encodePass( pass, dispatch )` | Into your compute pass. With `output: 'buffer'`, call `encodeCopies( encoder, dispatch )` after the pass ends. |
| `encodeCopies( encoder, dispatch )` | Buffer output only: copy each emitted level into its texture. Does nothing on the storage path. |
| `prepare( dispatch )` | What `encodePass` records, for dispatching it yourself: `{ pipeline, bindGroup, toolBindGroup, workgroups }`, plus `buffer` and `copies` (`{ texture, mipLevel, offset, bytesPerRow, rowsPerImage, width, height, layerCount }`) with `output: 'buffer'` (record those after the pass). Validates and throws like `encode`. |
| `bindGroup( dispatch )` | Group 0 for a dispatch (cached per size and textures, the 16 most recent); `prepare( dispatch ).bindGroup`. |
| `workgroups( width, height, layerCount? )` | `[ x, y, layerCount ]` for `dispatchWorkgroups`. |
| `levelSize( level, width, height )` | Size of an emitted level under this Reducer's edges. |
| `release( texture? )` | Drops cached bind groups (all, or those using `texture`) and destroys the buffers the Reducer created for them: size uniforms and buffer-output buffers. Call it after the last submitted use. Entries evicted from the 16-entry cache are destroyed once the queue has finished the work submitted so far. Your textures and buffers are never destroyed. |
| `wgsl`, `entryPoint`, `pipeline`, `layout`, `bindGroupLayouts` | The kernel, its entry point (`'main'`), compiled pipeline, pipeline layout and `[ group 0, group 1 ]` layouts. |
| `levels`, `edges`, `channels`, `output`, `label` | The resolved options. |

Errors are thrown with a reason: a missing hook or binding, an output that is too small or of the wrong format or usage, more output textures than `maxStorageTexturesPerShaderStage`, a tile larger than the workgroup memory limit (6 levels of 8 channels need 32 KiB), or, in compatibility mode, a source and output sharing a texture.

## ScalarReduction: an image to one value

Reduces an image to one `vec4f` and a weight per layer in two dispatches at any size: 16 × 16 workgroups each reduce a 64 × 64 tile to a partial, then one 256-invocation workgroup per layer combines them. Every texel counts once. Needs 256 invocations per workgroup ([Setup](#setup)).

```js
const range = new ScalarReduction( device, { reduction: 'minmax' } );        // min, max, min, max
range.encode( encoder, { source: depth, output: { buffer, offset: 0 } } );    // { value: vec4f, weight: f32 } per layer

const metering = new ScalarReduction( device, {
	reduction: [ 'average', 'max', 'min', 'average' ],
	bindings: { wgsl, entries },                                           // your @group(1)
	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f { … }',
	finish: 'fn finish( value: vec4f, weight: f32 ) { … store an exposure … }',
} );
metering.encodePass( pass, { width, height, bindGroup } );
```

### Options (`ScalarReductionOptions`)

| Option | Default | What it does |
|---|---|---|
| `reduction` | `'average'` | One op for every channel, `'minmax'` (even channels min, odd max), or one op per channel: `'sum'`, `'average'`, `'min'`, `'max'`, `'weight'`, `'weighted'`, `'argmin'`, `'argmax'`, `'payload'` (as for the Reducer). |
| `load` | none: read `run.source` | WGSL `fn load( texel: vec2i, layer: u32 ) -> vec4f` for in-image texels, reading your `@group(1)`. |
| `finish` | none: write `run.output` | WGSL `fn finish( value: vec4f, weight: f32 )`, optionally with a third `layer: u32`, run once per layer by one invocation. `value` has averages and weighted means divided; `weight` is Σ w with a weight channel, else the texel count. |
| `bindings` | none | `{ wgsl, entries }` of your `@group(1)`. |
| `block` | 4 | Texels per invocation per axis (1, 2, 4, 8): a tile of 16·block. 4 measured fastest on Apple. |
| `channels` | 4 | 4 or 8 (`DsWide`). Without a load hook, an 8-channel reduction only takes combine-only runs. |

### Runs (`ScalarRun`)

| Field | What it is |
|---|---|
| `source` | Without a load hook: a texture or `{ texture, mipLevel }` ([Sources](#sources-one-shape-everywhere)); any float-sampleable 2D format, depth included (single-channel formats splat). |
| `width`, `height` | The image size in texels: required with a load hook; with a source, its level's size (must match if given). |
| `layers` | `[ first, count ]`, each reduced to its own result. Default: all of the source's layers, or `[ 0, 1 ]` with a load hook. |
| `bindGroup` | Your group 1, required with `bindings`. |
| `output` | `{ buffer, offset = 0 }` without a finish hook. The buffer needs `STORAGE` usage; `offset` a multiple of the device's `minStorageBufferOffsetAlignment` (256 unless you requested less); 32 bytes per layer (`{ value: vec4f, weight: f32 }`), 48 with 8 channels (`resultBytes`). |
| `partials` | Combine only: `{ buffer, offset = 0, count }` (`offset` aligned as for `output`), `count` accumulations per layer that your own dispatch wrote (one `vec4f` each), e.g. a Reducer's deepest level through its store hook. Records the combine dispatch alone. Sum-like channels must hold sums. Your group 1 must not bind this buffer too. |
| `texels` | Combine only: the texel count behind the partials, required for `'average'` channels or without a `'weight'` channel. |

`unsupportedReason( run )` explains why a run can't be recorded; `encode`/`encodePass`/`prepare` throw with it. Runs with the keys used before 0.1, `texture`, `mipLevel` or `size`, are rejected with a message naming `source`, `width` and `height`.

### Members

| Member | What it does |
|---|---|
| `encode( encoder, run )` | Each dispatch in its own compute pass. |
| `encodePass( pass, run )` | Both dispatches (or the combine alone) into your pass. |
| `prepare( run )` | What `encodePass` records, for dispatching it yourself: `{ partials, combine, bindGroup, toolBindGroup, workgroups, combineWorkgroups }`; `partials` is `null` for a combine-only run. Validates and throws like `encode`. |
| `kernel( { dimension, splat } )` | `{ wgsl, layout, partials, combine }` for a source shape. |
| `combineKernel()` | `{ wgsl, layout, combine }` for combine-only runs. |
| `workgroups( width, height, layerCount? )`, `partialsBytes( width, height, layerCount? )` | Sizes of the partials dispatch and buffer. |
| `release( resource? )` | Drops the cached plans of a source texture, partials buffer or output buffer (without one: every plan) and destroys the params and partials buffers the reduction created for them, never yours. Call it after the last submitted use. Plans are kept for 4 run shapes per resource; older ones are destroyed once the queue has finished the work submitted so far. |
| `bindGroupLayouts` | `[ group 0, group 1 ]` as on a `Reducer`: create your hooks' bind group from index 1 (with `bindings`). Group 0 is that of a `'2d'` source, a load hook or a combine-only run. |
| `ops`, `block`, `partialBytes`, `resultBytes`, `combinesOnly` | The resolved options. |

`createScalarKernel( options )` and `scalarLayoutEntries( … )` are the pure WGSL generator and its layout, for vendoring.

### Reducer levels and a scalar from one read

A Reducer whose deepest level is 6 emits one texel per 64 × 64 tile. Its store hook can write those as partials (use `'sum'` channels, and form w·v in the load hook), and a combine-only `ScalarReduction` turns them into the scalar: 2 dispatches with one read of the source, which is how the FSR luma pyramid is rebuilt in `test/tools/generator.js` (`generator/luma-pyramid-fused`).

## three.js

`downsampler/three` works with `WebGPURenderer` on the WebGPU backend. `three` is an optional peer and is typed structurally, so the adapter imports nothing from it.

```js
import { generateMipmaps, getDownsampler, getGPUTexture, requiredLimits, unsupportedReason } from 'downsampler/three';

const renderer = new THREE.WebGPURenderer( { requiredLimits: requiredLimits( await navigator.gpu.requestAdapter() ) } );
await renderer.init();

const texture = new THREE.StorageTexture( width, height );
texture.generateMipmaps = true;                   // allocate the full chain
texture.mipmapsAutoUpdate = false;                // stop three regenerating it with render passes after a compute write
texture.minFilter = THREE.LinearMipmapLinearFilter;
renderer.initTexture( texture );                  // or render/compute into it first
// … write level 0 …
if ( ! generateMipmaps( renderer, texture, { reduction: 'max' } ) ) console.warn( unsupportedReason( renderer, texture ) ); // then fall back

// Into your own encoder (not submitted), with a Downsampler of other options:
generateMipmaps( renderer, texture, { encoder, downsampler: { averageAccess: 'load', f16Rounding: 'device' } } );
```

| Export | What it does |
|---|---|
| `generateMipmaps( renderer, texture, options? )` | Generates the chain with a shared `Downsampler`. `options` are `EncodeOptions` plus `encoder` (record into it and don't submit; default: submit at once) and `downsampler` (`DownsamplerOptions` of the shared instance, e.g. `averageAccess`, `f16Rounding`, `output`, `tail`). Returns `false`, doing nothing, if the texture isn't allocated, has one level (in place), isn't supported, or can't take `options.source`. |
| `unsupportedReason( renderer, texture, options? )` | Why `generateMipmaps` would return `false` with these options, or `null`. |
| `getDownsampler( renderer, options? )` | The shared `Downsampler` of the renderer's device for an option set (one per distinct set, any key order). Use it for `encode`, `prepare`, `release`, or Hi-Z with a separate source. |
| `getGPUTexture( renderer, texture )` | The `GPUTexture` behind a three.js texture, or `undefined`. |
| `requiredLimits( adapter )` | The same function as in `downsampler/webgpu`, for `new WebGPURenderer( { requiredLimits } )`. |
| `Downsampler` | Re-exported. |

The texture must have `STORAGE_BINDING` usage (in current three.js a `StorageTexture` or `StorageArrayTexture`), or `COPY_DST` for formats that take the buffer path. Regular textures don't get storage usage from three today; native support for them is planned as an in-tree three.js option (SPEC §10). See [examples/three.html](../examples/three.html).

## The core package: oracles and planning

`import { … } from 'downsampler'` is backend-neutral (no WebGPU types). Useful for testing your own use of the library:

| Export | What it is |
|---|---|
| `reduceReference( values, width, height, layers, quantization, reduction?, options? )` | One `'exact'` step on the CPU: the oracle every backend is tested against. `values` are RGBA floats; it returns the next level and a per-value `filterTolerance`. |
| `tolerance( expected, quantization, filterTolerance?, reduction? )` | The test tolerance for one value (folds: 0). |
| `spdReference(…)`, `spdChainReference(…)` | `oddSizes: 'spd'` batches and chains. |
| `boxReference(…)`, `boxChainReference(…)`, `areaReference( values, width, height, layers, quantization, reduction, outWidth, outHeight, options? )` | `oddSizes: 'box'` steps and chains, and an area step to any size (a resampled source). `areaSpan`, `areaOverlap`: the taps and their integer weights. |
| `reducePyramid( values, width, height, layers, { reduction, levels, edges, channels } )` | The `Reducer` oracle (f64, with each value's term magnitude); `pyramidTolerance( magnitude )`. |
| `scalarReference(…)`, `scalarCombineReference(…)` | `ScalarReduction` oracles: the GPU's pairing in f32, the exact f64 result and an error bound per channel. |
| `maxViolations( base, width, height, level, w, h )` | Counts max-pyramid texels below some base texel in their area: 0 for a conservative Hi-Z. |
| `planBatches( width, height, options )`, `mipSize`, `mipLevelCountFor`, `levelSize`, `footprint` | Planning and sizes. |
| `formats`, `quantize`, `toHalf`, `fromHalf`, `packRgb9e5`, … | Format table and encoders. |

The WGSL generators are exported from `downsampler/webgpu` for vendoring: `createKernel`, `kernelLayoutEntries`, `float16Wgsl`, `packedFloatWgsl`, `bufferOutputWgsl`, `createScalarKernel`.

## Coming next

sRGB destinations, `rgb10a2unorm`, integer formats (deferred, three.js-driven); subgroup kernels and fp16 math only if measured faster (M3). The full list is in [SPEC.md §11](../SPEC.md#11-milestones) and [DIVERGES.md](../DIVERGES.md).
