# downsampler

Fast mipmaps and depth pyramids for WebGPU, built with compute shaders.

```js
const downsampler = new Downsampler( device );
downsampler.generate( texture ); // every mip level from level 0
```

The usual way to make mipmaps in WebGPU is one render pass per level, and another set per layer of a cube map or
array. downsampler writes up to six levels in one compute dispatch and handles every layer in the same dispatch, so a
full chain takes two dispatches at power-of-two sizes and three to six at typical screen sizes.

It also does the jobs render passes can't:

- **Hi-Z depth pyramids** (min, max or both) from a depth buffer, guaranteed conservative for occlusion culling.
- **Partial updates:** repaint a corner of a texture and refresh only the mip texels that depend on it.
- **Statistics:** per-block mean, variance or max (`Reducer`), and a whole image down to one value, like average
  luminance for auto-exposure (`ScalarReduction`).

By default the results match render-pass mipmaps, so you can swap it in without a visual change. Every kernel is
tested against a CPU reference on real GPUs. Zero dependencies; a three.js adapter is included.

> **Status:** 0.1, not yet on npm. Build it from this repository for now. The API can still change before 1.0
> ([CHANGELOG](CHANGELOG.md)).

## Install

```sh
npm install downsampler
```

The types use the standard WebGPU types, so add [`@webgpu/types`](https://www.npmjs.com/package/@webgpu/types) to
your project. `three` is optional and only needed for `downsampler/three`.

## Quick start

```js
import { Downsampler, requiredLimits } from 'downsampler/webgpu';

const adapter = await navigator.gpu.requestAdapter();
// requiredLimits asks for six storage textures, so one dispatch can write six levels.
const device = await adapter.requestDevice( { requiredLimits: requiredLimits( adapter ) } );

const texture = device.createTexture( {
	format: 'rgba8unorm',
	size: [ 1920, 1080 ],
	mipLevelCount: 11,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
} );
// … fill level 0 …

const downsampler = new Downsampler( device ); // make one per device and reuse it
downsampler.generate( texture );               // records and submits the work
```

Already have a command encoder? Use `downsampler.encode( encoder, texture )` instead, and it records into yours.

## What else it can do

```js
// A conservative min/max depth pyramid for occlusion culling (depth can't be written, so it goes into its own texture).
downsampler.encode( encoder, hiZ, { source: depthTexture, reduction: 'minmax' } );

// Refresh only what a 32×32 brush stroke changed.
downsampler.encode( encoder, texture, { rect: { x: 128, y: 64, width: 32, height: 32 } } );

// A true area average at odd sizes, for lighting and roughness chains.
downsampler.encode( encoder, irradiance, { oddSizes: 'box' } );
```

Each of these has a runnable page in [examples/](examples/) that checks its own output.

## three.js

```js
import { generateMipmaps, requiredLimits } from 'downsampler/three';

const renderer = new THREE.WebGPURenderer( { requiredLimits: requiredLimits( await navigator.gpu.requestAdapter() ) } );
await renderer.init();
generateMipmaps( renderer, texture ); // false if the texture isn't supported; three's own mipmaps still work then
```

This works today for `StorageTexture`s and sRGB color textures. A built-in `mipmapGeneration: 'compute'` option for
three.js itself is in progress.

## How fast is it?

GPU time for a full chain, against one render pass per level:

| Apple M5 Pro | rgba8 | rgba16float |
|---|---|---|
| 512², six layers (a cube map) | 27× faster | |
| 1024² | 7× faster | 4.5× faster |
| 1920×1080 | 3× faster | |

On an NVIDIA RTX 3060 Laptop, from 960×540 to 2560×1440, it takes 5–20 % less time than render passes in rgba8 and up
to 20 % less in rgba16float; a 512² cube map is 3–4× faster. The exception is sRGB color textures, which compute
shaders can't write directly. The library draws those with render passes too, but rounds every level exactly, which
plain render passes don't: they stay 1.05–1.6× faster on the RTX and about 5× on Intel. `srgbRounding: 'device'` skips
the exact rounding and matches them.

The big wins are small textures and cube maps, where the cost of starting many render passes dominates. Very large
textures (4096²) are limited by memory bandwidth, so every method takes about the same time there. If you can accept
SPD's handling of odd sizes, `oddSizes: 'spd'` takes about 25 % less time on the RTX. Methods and full numbers:
[SPEC.md §8](SPEC.md#8-performance-methodology) and [docs/WINDOWS-REPORT.md](docs/WINDOWS-REPORT.md).

## Browser support

Any browser with WebGPU. Tested in Chrome on macOS (Apple M5 Pro) and Windows (NVIDIA RTX 3060, Intel UHD), and on
WebGPU compatibility-mode devices. Not yet tested: mobile, AMD, Safari, Firefox. If you try it there,
`npm run lab` opens a page that runs the test suite on your device.

## Documentation

- [Getting started](docs/getting-started.md): a walkthrough of every feature, with the reasons to pick each option.
- [Reference](docs/guide.md): every class, option and default.
- [Examples](examples/): `npm run examples`, then open http://localhost:8080/examples/.
- [How it compares](docs/differences.md): what changes coming from render passes, SPD, nvpro or a Hi-Z shortcut.
- [Coming from AMD SPD or FSR](docs/spd-users.md).
- Design notes: [SPEC.md](SPEC.md) (decisions and measurements), [DIVERGES.md](DIVERGES.md) (where and why we
  differ from SPD and other libraries).

## Development

```sh
npm install
npm test                                            # build, unit tests, GPU suite on your GPU
npm run test:gpu -- --quick --adapter=swiftshader   # software WebGPU, no GPU needed
npm run test:examples                               # every example page, checked
npm run bench                                       # benchmarks
```

## License

MIT
