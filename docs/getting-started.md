# Getting started

This walks through what downsampler does, one job at a time, and when you'd pick each option. Every section has a
runnable page in [examples/](../examples/) that checks its own output. For the full list of options, see the
[reference](guide.md).

- [Set up a device](#set-up-a-device)
- [Your first mip chain](#your-first-mip-chain)
- [Odd sizes](#odd-sizes)
- [Depth pyramids for occlusion culling](#depth-pyramids-for-occlusion-culling)
- [Updating part of a texture](#updating-part-of-a-texture)
- [Statistics and single values](#statistics-and-single-values)
- [three.js](#threejs)
- [When a texture isn't supported](#when-a-texture-isnt-supported)
- [The same results on every GPU](#the-same-results-on-every-gpu)

## Set up a device

```js
import { Downsampler, requiredLimits } from 'downsampler/webgpu';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice( { requiredLimits: requiredLimits( adapter ) } );
const downsampler = new Downsampler( device );
```

Each level a dispatch writes needs its own storage texture binding, and WebGPU allows only four unless you ask for
more. `requiredLimits( adapter )` asks for up to six, plus the larger workgroups some kernels use, but never for more
than the adapter has. Without it everything still works, just with fewer levels per dispatch.

Make one `Downsampler` per device and keep it. It caches its compiled pipelines, so the first chain of each format
pays a compile and later ones don't.

## Your first mip chain

([mip-chain.html](../examples/mip-chain.html))

```js
const texture = device.createTexture( {
	format: 'rgba8unorm',
	size: [ width, height ],
	mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
} );
device.queue.writeTexture( { texture }, pixels, { bytesPerRow: width * 4 }, [ width, height ] );

downsampler.generate( texture );
```

`generate` records the work and submits it. If you're already building a frame, record into your own encoder
instead, or into a compute pass you've started:

```js
downsampler.encode( encoder, texture );    // its own compute passes in your encoder
downsampler.encodePass( pass, texture );   // dispatches inside your compute pass
```

The texture needs `STORAGE_BINDING` so the shader can write the levels. A few formats can't be storage textures at
all (sRGB colour, `rgb9e5ufloat`); those need `COPY_DST` instead, and the library writes them through a buffer and
copies them in. You don't have to do anything different for them, except that `encodePass` can't record those copies:
use `encode`.

Array textures and cube maps (as 6-layer `'2d-array'` textures) work the same way. Every layer goes through the same
dispatches, which is where most of the speed-up over render passes comes from.

To regenerate only part of the chain, pass `baseMipLevel` (the level to read) and `mipLevelCount`.

## Odd sizes

([odd-sizes.html](../examples/odd-sizes.html))

Halving 1920 × 1080 goes 960, 480, 240, 120, 60, 30, 15, then 7. That last step can't average pairs of texels,
because 15 doesn't split into pairs. Every mipmap generator has to decide what to do there, and they disagree.
`oddSizes` picks one:

| `oddSizes` | What it does | Use it for |
|---|---|---|
| `'exact'` (default) | What render passes and WebGL do: one bilinear sample at each output texel's centre. | Colour textures, anything that should look like normal mipmaps. |
| `'box'` | A true area average: every texel counts by how much of it the output covers. Every level keeps the image's mean. | Lighting, irradiance and roughness chains, anything where the average matters. |
| `'spd'` | AMD SPD's shortcut: drop the last odd row or column. Fewest dispatches. | Bloom and other effects where speed matters more than the exact values. Averages only. |

```js
downsampler.generate( irradiance, { oddSizes: 'box' } );
```

How far apart they are: on a 1129 × 635 test image, the average error against a true area average over all levels is
2.1 steps of 255 for `'exact'`, 5.1 for `'spd'` and 0.3 for `'box'`. And a 31 × 31 texture under `'exact'` ends in a
1 × 1 that is just its centre texel, as render passes give you; under `'box'` it is the mean of all 961.

## Depth pyramids for occlusion culling

([hi-z.html](../examples/hi-z.html), [hi-z-pow2.html](../examples/hi-z-pow2.html))

A Hi-Z pyramid stores, for each block of the screen, the nearest and farthest depth in it. An object is hidden if
its nearest point is behind the farthest depth of every block it covers. That test is only safe if each pyramid texel
really covers every depth texel under it. Here it does, at every size: at odd sizes each texel takes the extra row or
column too.

Depth textures can't be written by compute shaders, so the pyramid is its own texture, and the depth buffer is its
`source`:

```js
const hiZ = device.createTexture( {
	format: 'rg32float', size: [ width, height ], mipLevelCount: Math.floor( Math.log2( Math.max( width, height ) ) ) + 1,
	usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
} );
downsampler.encode( encoder, hiZ, { source: depth, reduction: 'minmax' } ); // r = min, g = max depth
```

Use `'max'` alone (an `r32float` pyramid) if you only need the farthest depth, or `'min'` with reverse-Z.

**A power-of-two pyramid.** Screen sizes make awkward pyramids. With `resample: 'area'`, level 0 of the pyramid can
be a different size from the depth buffer, such as 1024 × 512 for a 1920 × 1080 screen. Each texel still takes
every depth texel its area touches, so it stays conservative, and every later level halves cleanly:

```js
downsampler.encode( encoder, hiZ, { source: { texture: depth, resample: 'area' }, reduction: 'max' } );
```

The example page compares this with sampling 2 × 2 depth texels per pyramid texel, the common shortcut, and counts
the depth texels the shortcut misses.

## Updating part of a texture

([partial-update.html](../examples/partial-update.html))

If only part of level 0 changed, say a decal or a brush stroke, pass the changed rectangle. Each level then rewrites
only the texels that depend on it, and the result is exactly what a full regeneration would give:

```js
downsampler.encode( encoder, texture, { rect: { x: 128, y: 64, width: 32, height: 32 } } );
downsampler.encode( encoder, cubeFaces, { layers: [ 2, 1 ] } );              // one face of a cube
downsampler.encode( encoder, cubeFaces, { layers: [ 4, 1 ], rect: decal } ); // a decal on face 4
```

An empty rectangle does nothing, so you can pass this frame's dirty region without checking it first.

It pays off most on big textures and cube maps. The dispatches still run, so a small change costs about the same
whatever its size: on an RTX 3060, a 32 × 32 rectangle costs 3 % of a full chain at 4096², a fifth to a third at 1080p, and
one cube face a fifth to a third of the cube.

## Statistics and single values

([reducer.html](../examples/reducer.html), [scalar.html](../examples/scalar.html))

The same tile reduction works for more than mipmaps. A `Reducer` reads your image through a WGSL function you write,
reduces each channel its own way, and stores only the levels you ask for. Here, the mean, mean of squares (for
variance), max and texel count of luminance per 8 × 8 and 32 × 32 block, in one dispatch:

```js
const stats = new Reducer( device, {
	reduction: [ 'average', 'average', 'max', 'sum' ],
	levels: [ 3, 5 ],                 // 2³ = 8 and 2⁵ = 32 texel blocks
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

A `ScalarReduction` takes a whole image down to one value per layer in two dispatches, at any size. A `finish`
function runs on the GPU with the result, so values like auto-exposure never need a readback:

```js
const metering = new ScalarReduction( device, {
	reduction: [ 'average', 'max', 'min', 'average' ],
	bindings: { wgsl: '@group(1) @binding(0) var hdr: texture_2d<f32>; @group(1) @binding(1) var<storage, read_write> exposure: array<vec4f, 2>;', entries },
	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f { let l = max( dot( textureLoad( hdr, texel, 0 ).rgb, vec3f( 0.2126, 0.7152, 0.0722 ) ), 1e-4 ); return vec4f( log( l ), l, l, l ); }',
	finish: 'fn finish( value: vec4f, weight: f32 ) { exposure[ 0 ] = value; exposure[ 1 ] = vec4f( exp( value.x ), 0.18 / exp( value.x ), weight, 0.0 ); }',
} );
metering.encode( encoder, { width, height, bindGroup } );
```

(`entries` and `bindGroup` are set up as for the Reducer; the example page has the full code.)

## three.js

([three.html](../examples/three.html))

```js
import { generateMipmaps, requiredLimits, unsupportedReason } from 'downsampler/three';

const renderer = new THREE.WebGPURenderer( { requiredLimits: requiredLimits( await navigator.gpu.requestAdapter() ) } );
await renderer.init();

const texture = new THREE.StorageTexture( width, height );
texture.generateMipmaps = true;     // allocate the full chain
texture.mipmapsAutoUpdate = false;  // don't let three overwrite it with render-pass mipmaps
texture.minFilter = THREE.LinearMipmapLinearFilter;
// … write level 0 …

if ( ! generateMipmaps( renderer, texture ) ) console.warn( unsupportedReason( renderer, texture ) );
```

`generateMipmaps` takes the same options as `encode`, plus `encoder` to record into your own. It returns `false` and
does nothing when it can't handle a texture, so three's own mipmaps remain your fallback.

It works for `StorageTexture`s and for sRGB colour textures. Other regular three.js textures don't get the storage
usage it needs; a built-in compute option for three.js itself is in progress.

For sRGB textures on NVIDIA, three's own mipmaps are 2–3× faster (compute shaders can't write sRGB directly, so the
library takes a slower path), so leave those to three there.

## When a texture isn't supported

```js
const reason = downsampler.unsupportedReason( texture ); // null if it's fine
```

The common reasons:

- **Missing `STORAGE_BINDING` usage.** Add it when you create the texture. The library won't quietly switch to the
  slower buffer path for a format that could have been a storage texture.
- **The format needs a device feature.** The one- and two-channel 8- and 16-bit formats need
  `texture-formats-tier1`, and `bgra8unorm` needs `bgra8unorm-storage`. Request them on the device when the adapter
  has them. The [reference](guide.md#textures-it-takes) lists every format.
- **A depth texture as the destination.** Use a separate pyramid texture with the depth buffer as `source`.

`encode`, `generate` and `prepare` throw with the same message.

WebGPU compatibility mode (older devices, `featureLevel: 'compatibility'`) is handled for you: the library writes
through scratch textures there, with the same results.

## The same results on every GPU

By default, averages use the GPU's texture sampler, as render passes do. Samplers round their filter weights slightly
differently on each vendor, so two GPUs can disagree by a step at odd sizes. If you need identical bits everywhere,
for tests or for values you compare across machines:

```js
new Downsampler( device, { averageAccess: 'load' } );
```

This computes the weights in the shader. It costs 2–13 % on an NVIDIA RTX 3060.

Half-float textures (`rgba16float`) are always rounded to nearest in the shader, because GPUs on Direct3D 12 (Windows)
truncate when they convert. So those are the same everywhere by default.

## Where next

- [Reference](guide.md): every option, default and method.
- [How it compares](differences.md) with render passes, SPD and nvpro.
- [Coming from AMD SPD or FSR](spd-users.md).
- [SPEC.md](../SPEC.md) for why things work the way they do, with measurements.
