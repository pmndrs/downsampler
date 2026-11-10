# downsampler

GPU texture downsampling: mip chains, min/max reduction pyramids (Hi-Z from depth) and custom reductions. WebGPU first, engine-agnostic, zero dependencies. Adapters for three.js and others sit on top.

A WGSL implementation of the ideas in AMD FidelityFX SPD: a 64×64 source tile reduced to six levels per workgroup in workgroup memory, with every level written once. It targets SPD's feature set. See [SPEC.md](SPEC.md) for the full design, parity table and roadmap.

> Status: early (milestone 2 in progress). Destinations `rgba8unorm`, `rgba16float`, `r32float`, `rg32float`, `rgba32float`; any float-sampleable or depth source; 2D textures and array layers; average, min, max, minmax and custom reductions. Not yet published to npm.

## Two things to know first

**Odd sizes default to `'exact'`.** Where a level has an odd, non-unit dimension, the next level is re-sampled with normalized-coordinate bilinear filtering. Output matches render-pass and WebGL mipmaps, so you can switch without visual change. A planned `'fast'` mode will reduce straight through odd sizes the way SPD does. That means fewer dispatches (2 instead of 7 for 1920×1080) but slightly different edge weights. It is opt-in.

**There is no single-dispatch mode.** SPD's final levels are written by "the last workgroup to finish", detected with an atomic counter. In WGSL that handoff isn't guaranteed: barriers are workgroup-scoped and atomics are relaxed, so seeing the counter doesn't guarantee seeing other workgroups' results. We built and measured it anyway, and it was also slower. The atomic tail added ~20 µs, versus ~1–2 µs for the dispatch boundary it saves. So chains up to 4096² use two dispatches instead. Details and numbers are in [SPEC.md §2.2](SPEC.md#22-no-single-dispatch-mode).

## Usage

```js
import { Downsampler } from 'downsampler/webgpu';

const adapter = await navigator.gpu.requestAdapter();
// Six levels per dispatch need six storage textures (the default limit is four).
const device = await adapter.requestDevice( { requiredLimits: Downsampler.requiredLimits( adapter ) } );
const downsampler = new Downsampler( device );

// The texture needs TEXTURE_BINDING | STORAGE_BINDING usage.
downsampler.generate( texture );                    // encode + submit
downsampler.encode( encoder, texture );             // or record into your own encoder
downsampler.encode( encoder, texture, { baseMipLevel: 2, mipLevelCount: 4 } );

// Reduction pyramids. Odd sizes include the extra row/column, so min/max are conservative.
downsampler.generate( texture, { reduction: 'max' } );
downsampler.generate( texture, { reduction: 'minmax' } );  // r/b = min, g/a = max
downsampler.generate( texture, { reduction: { wgsl: `
	fn reduce( a: vec4f, b: vec4f ) -> vec4f { return max( a, b ); }  // associative, commutative, idempotent
` } } );

// Hi-Z from a depth buffer: level 0 is a copy (same size) or the first reduction (half size).
downsampler.generate( hiZ /* rg32float */, { source: { texture: depthTexture }, reduction: 'minmax' } );
```

three.js (`WebGPURenderer`, texture allocated with storage usage, e.g. `StorageTexture`):

```js
import { generateMipmaps } from 'downsampler/three';

if ( ! generateMipmaps( renderer, texture ) ) { /* not allocated or unsupported: use three's own path */ }
```

Native support for all three.js textures is planned as an in-tree three.js option. See [SPEC.md §10](SPEC.md#10-threejs-integration).

## Performance

Apple M5 Pro (Metal 3, Chrome 155), GPU ms per full chain, against one render pass per level:

| Size | Format | Render passes | downsampler | Speedup |
|---|---|---:|---:|---:|
| 512² | rgba8unorm | 0.0882 | 0.0098 | 9.0× |
| 1024² | rgba8unorm | 0.1159 | 0.0161 | 7.2× |
| 4096² | rgba8unorm | 0.4501 | 0.3371 | 1.3× |
| 1920×1080 | rgba8unorm | 0.1176 | 0.0374 | 3.1× |
| 512²×6 layers | rgba8unorm | 0.6337 | 0.0236 | 26.8× |
| 1024² | rgba16float | 0.1178 | 0.0263 | 4.5× |

Large textures are memory-bandwidth bound for every method. Other GPUs haven't been measured yet; device reports are welcome (below).

## Development

```sh
npm install
npm test              # build, unit tests, GPU suite (headless Chrome)
npm run test:gpu -- --quick --adapter=swiftshader   # software WebGPU, no GPU needed
npm run bench         # validation + benchmark
npm run lab           # device lab at http://localhost:8080/test/browser/
```

The device lab runs the full validation suite and benchmark on whatever device opens it and produces a JSON report.

## License

MIT
