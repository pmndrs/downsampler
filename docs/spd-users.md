# Notes for SPD users

For people coming from AMD FidelityFX SPD, its WGSL port webgpu-spd, or the SPD passes in FSR 3. It maps SPD's concepts and options to this library, says what is different and why, and shows how to get SPD-identical results where you need them. The full list of differences, with measurements and the cost of converging, is [DIVERGES.md](../DIVERGES.md); the parity table is [SPEC.md §6](../SPEC.md#6-spd-feature-parity).

## In one paragraph

The kernel is SPD's idea, implemented independently in WGSL: a 64 × 64 source tile reduced through six levels in workgroup memory, every level written once. Two things differ by design. **There is no single dispatch:** SPD's last-workgroup handoff isn't guaranteed by WGSL, so the levels below the sixth come from a second dispatch (2 dispatches up to 4096²). **Odd sizes default to render-pass semantics (`'exact'`)**, not SPD's dropped row or column, so min/max pyramids are conservative. `oddSizes: 'spd'` gives you SPD's semantics for averages.

## Concept map

| SPD | downsampler |
|---|---|
| `SpdSetup( dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo, mips )`, a dispatch | `downsampler.encode( encoder, texture, options )`; the library plans the dispatches. `prepare( texture )` lists them. |
| `mips` (levels below the source, at most 12) | `mipLevelCount` counts the **source level too** in place: SPD `mips = N` is `mipLevelCount: N + 1`. No cap on levels or size. |
| Global atomic counter, `globallycoherent` mip 6, "last workgroup continues" | None: a dispatch boundary. Nothing to allocate, clear or alias. |
| `SpdReduce4( v0, v1, v2, v3 )` | `reduction`: `'average'`, `'min'`, `'max'`, `'minmax'`, or `{ wgsl: 'fn reduce( a: vec4f, b: vec4f ) -> vec4f' }`, a binary fold (`Downsampler`). Sums, counts and weighted means: per-channel ops in a `Reducer`. |
| `SpdLoadSourceImage`, `SpdLoad` | `Reducer` `load` hook: `fn load( texel: vec2i, layer: u32 ) -> vec4f`, reading your own `@group(1)`. |
| `SpdStore` | `Reducer` `store` hook: `fn store( level: u32, texel: vec2i, layer: u32, value: vec4f )`, `level` 1 = half size (SPD's mip index + 1). Or output textures. |
| Choosing which mips to store | `Reducer` `levels: [ 1, 2, 3 ]`: unemitted levels are reduced but get no binding. |
| `SPD_LINEAR_SAMPLER` | `averageAccess: 'sample'` (the default for averages): one bilinear sample per 2 × 2 block. `'load'` applies the weights in the shader. |
| Separate source SRV and destination UAV | `source: texture` or `{ texture, mipLevel }`. Destination level `baseMipLevel` may be the **same size** as the source (a converting copy, e.g. depth → `r32float`) or half, or any size with `resample: 'area'`; SPD always halves. |
| Texture arrays (`slice` in the dispatch z) | All layers of a `'2d-array'` texture in the same dispatches (workgroup z), or a range with `layers: [ first, count ]`. |
| `rectInfo` (sub-rectangle) | `rect: { x, y, width, height }` on `encode`. It rewrites exact per-level regions instead of snapping to 64-texel tiles, bit-identical to a full recompute (DIVERGES #15; [examples/partial-update.html](../examples/partial-update.html)); an empty rect records nothing. |
| sRGB: sample an sRGB view, write a UNORM alias | Not yet. WebGPU forbids sRGB storage even through a view; planned via a scratch copy (M2.3.5). |
| Packed fp16 (`A_HALF`, `SPD_PACKED_ONLY`) | Not offered: f32 math. Under review in M3, kept only if measurably faster. |
| Wave/quad ops (`SpdReduceQuad`, `WaveReadLaneAt`) | Workgroup memory only: WGSL defines quads only in fragment shaders and no lane-to-index mapping in compute. A subgroup kernel is gated on measurement (M3). |
| 4096² and 12-level limit | Any size, by chaining batches. |

## What is different, and why

### No single dispatch

SPD's last workgroup to finish (found with an atomic counter) reduces the final levels from mip 6. In WGSL every atomic is relaxed and every barrier is workgroup-scoped, so seeing the counter doesn't guarantee seeing the other workgroups' mip-6 writes. This isn't theoretical: webgpu-spd 3.0.0's handoff returned stale levels 7+ on an Apple M5 Pro in up to 5 % of frames at render sizes and up to 77 % at 256² (DIVERGES #22). It was also slower: a WGSL atomic handoff built from the same kernel lost to the same kernel split into two dispatches at every size measured except 4096², where they tied. AMD's own SPD on D3D12 showed the same: atomic and split variants within noise (DIVERGES #1).

So a full chain takes 2 dispatches up to 4096² (6 levels each, given six storage textures), more under `'exact'` at odd sizes. If WGSL gains device-scope acquire/release, this gets re-measured.

### Odd sizes

SPD halves with ⌊n/2⌋ and drops the last odd row or column. The default here, `'exact'`, matches render-pass mipmaps instead: an odd level is re-sampled with normalized coordinates, and min/max fold the `2x … 2x+2` footprint. Two consequences matter to SPD users:

- **Averages drift under SPD semantics.** When level 0 or 1 is odd (1129×635, 1477×831, 1707×960), SPD's level-1 texels sit up to a texel away from their normalized footprint, and every deeper level inherits it: up to 0.64 of full scale in rgba8 against `'exact'` on a test scene (SPEC §2.1). At even sizes the two agree to one 8-bit step.
- **SPD's min/max is not a conservative Hi-Z.** A dropped row or column belongs to no texel, so a max pyramid misses it: 0.01–15 % of texels on a scene image at render sizes, up to 52 % on noise. `'exact'` has none, which is what occlusion culling needs. [examples/hi-z.html](../examples/hi-z.html) shows both.

The cost is dispatches: `'exact'` ends a batch at an odd size, 5–9 dispatches at render sizes before the tail kernel, which (on by default) brings 1920×1080 to 4. On an M5 Pro `'spd'` was 1.4–2.1× faster than `'exact'` at render sizes in rgba8, which is tens of µs per chain.

For area-correct averages (SPD's 2×2 box generalized to odd sizes), `oddSizes: 'box'` takes nvpro_pyramid's weights: along an odd size `2n+1`, three taps weighted `(n−x, n, x+1)/(2n+1)`, so every texel counts in proportion to its overlap. Same dispatches as `'exact'`. [examples/odd-sizes.html](../examples/odd-sizes.html) puts the three policies side by side against the area-weighted ideal.

### Edges and rounding

- **No zero reads past the edge.** SPD's unclamped loads read zeros outside the texture unless it is padded; AMD's own D3D12 SPD at 1129×635 got 0.28 instead of 0.50 in its 1×1 level that way (DIVERGES #5). Every read here is clamped.
- **Every level is rounded as stored** under `'exact'`, so a chain equals level-by-level generation and doesn't depend on the batch layout. SPD keeps unrounded values in LDS within a dispatch; `'spd'` mode does the same within a dispatch, and rounds at the dispatch boundary as SPD rounds mip 6.
- **float16 is rounded to nearest even** in the shader (`f16Rounding: 'nearest'`), because D3D12 truncates in `textureStore` and `pack2x16float`.

### Reductions are binary

A custom reduction is `reduce( a, b )`, not `SpdReduce4( a, b, c, d )`. A pairwise reduction composes inside the tile and normalizes once, and everything a 4-way reduction is used for (weighted averages, sums, counts) is a per-channel op in a `Reducer`: `[ 'weighted', 'weighted', 'weighted', 'weight' ]` is an alpha- or coverage-weighted average. In `Downsampler`, a custom fold must be idempotent (min, max, or): duplicated reads at clamped edges and overlapping odd footprints are then harmless.

## Getting SPD-identical results

**Average mip chains:** `oddSizes: 'spd'`.

```js
downsampler.encode( encoder, texture, { oddSizes: 'spd' } );                              // in place, like SPD
downsampler.encode( encoder, texture, { oddSizes: 'spd', mipLevelCount: mips + 1 } );     // SPD's `mips`
```

This is SPD's semantics: clamped loads inside the texture, ⌊n/2⌋ sizes, no rounding between the levels of a dispatch, `⌈levels / levelsPerDispatch⌉` dispatches at any size, one bilinear sample at each 2 × 2 corner for filterable formats. It is checked against `spdReference` (core) in the GPU suite, and level dumps of AMD's native D3D12 SPD at 1129×635 match the same oracle (`results/windows/native-semantics-rtx3060.json`, apart from AMD's zero reads below). It takes `'average'` only.

**Custom loads, stores, sums, or min/max with SPD's edges:** a `Reducer` with `edges: 'clamp'`. That is the SPD tile, one dispatch, up to 6 levels, any hooks:

```js
const pyramid = new Reducer( device, { edges: 'clamp', levels: [ 1, 2, 3 ], reduction: 'average', bindings, load, store } );
pyramid.encodePass( pass, { width, height, bindGroup } );
```

`'clamp'` sizes equal mip sizes, so the outputs can be the mips of one half-resolution texture (outside compatibility mode). For Hi-Z, prefer `edges: 'zero-weight'` (⌈n/2⌉ sizes, every texel counted once, conservative) or the `Downsampler`'s `'exact'`.

What stays different even then: we don't read zeros past the edge (match by padding the source with zeros in your load hook), and hardware sampler weights differ by vendor exactly as they do for SPD's linear-sampler path (`averageAccess: 'load'` makes them identical everywhere).

## FSR 3 passes

FSR 3.1's two SPD passes are rebuilt from `Reducer` and `ScalarReduction` with only tool code in [test/tools/generator.js](../test/tools/generator.js), and each passes an oracle at five render sizes ([test/tools/README.md](../test/tools/README.md)):

| FSR pass | Built as |
|---|---|
| Shading-change pyramid (custom load, average, mips 1–3 stored) | `Reducer` `'clamp'`, `levels: [ 1, 2, 3 ]`, FSR's own `load` and `store`: one dispatch, 0.74–0.86× the SPD port's time on an M5 Pro |
| Luma pyramid, AMD parity (mip 1 depth, mip 6, `FrameInfo` from texel (0,0) of the last mip) | `Reducer` `'clamp'` `[ 1, 6 ]` + a `ScalarReduction` over the top-left `2^(mips−6)` window of mip 6 |
| Luma pyramid over every texel | `Reducer` `'zero-weight'` `[ 1, 6 ]` with sums, then a combine-only `ScalarReduction` over its 64² tile partials: 2 dispatches, one read of the source |

AMD's exposure covers only the top-left `2^mips` square, and part of that window reads zeros; at 1129×635 that biases exposure by +19 %. The every-texel version is the one to use unless you need bit parity.

## Coming from webgpu-spd

| webgpu-spd | downsampler |
|---|---|
| `generateMipmaps( texture )` | `downsampler.generate( texture )` |
| `preparePass( texture, … )` → `SPDPass.encode( pass )` | `downsampler.encodePass( pass, texture, options )`; `prepare()` for the dispatches |
| Filters: average, min, max, minmax | `reduction`: the same names |
| Custom 4-way filter | a binary `{ wgsl }` fold, or a `Reducer` |
| Separate target | `source: texture` or `{ texture, mipLevel }` (also same-size converting copies) |
| `maxMipsPerPass` (above 6 uses the atomic handoff) | `levelsPerDispatch` (at most 6; there is no handoff) |
| Integer formats, `rgb10a2unorm` | Not yet (M2.3.7, M2.3.3) |
| Sub-rectangles | `rect: { x, y, width, height }`, exact per-level regions |

If you stay on webgpu-spd: set `maxMipsPerPass: 6` (its handoff returned stale levels on Apple), and don't use its min/max for Hi-Z at odd sizes. Measured with the same semantics, our `'spd'` was 1.2–2.1× faster than webgpu-spd on the M5 Pro, and 1.6× / 3.5× faster than its default on an RTX 3060 / Intel UHD in rgba8 ([results/compare/webgpu-spd.md](../results/compare/webgpu-spd.md)).

## Performance against native SPD

On an RTX 3060 Laptop (D3D12, clocks locked), our `'spd'` takes 1.26 / 1.27× (rgba8 / rgba16float) the time of AMD's best native D3D12 SPD build at render sizes, and 1.00× at 4096²: per-texel work is at parity, and the gap is WebGPU's fixed cost per dispatch (the same render-pass chain costs 1.45 / 1.34× in WebGPU what it costs natively). On Intel UHD it is 1.28 / 1.50×. Details: [docs/WINDOWS-REPORT.md](WINDOWS-REPORT.md#speed-against-native).
