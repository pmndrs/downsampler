# Divergences from the primary sources

Where this library does something different from the sources in SPEC §13: AMD FidelityFX SPD, nvpro_pyramid (vk_compute_mipmaps), the WebGPU community ports (webgpu-spd, `greggman`, `davidar`, the `Volcomix` fiddle) and the requests in gpuweb #386/#5253.

Each entry says what they do, what we do, why, what it would take to converge, and the expected effect. Performance figures marked *measured* come from the Apple M5 Pro runs in SPEC §2.2, §4 and §8. Nothing has been measured on other vendors yet, so anything else is an *estimate* or *unknown*.

**Verdicts**
- **Keep:** a deliberate difference, with no plan to converge.
- **Planned:** convergence or an alternative is on the milestone list (SPEC §11).
- **Candidate:** worth doing, but not scheduled.
- **Risk:** an open question that could change a default.

Effort: **S** is under a day, **M** a few days including oracle and GPU cases, **L** a week or more.

Last updated: 2026-11-11 (M2T.1)

## Summary

| # | Divergence | Source | Verdict | Effort to converge | Quality | Performance |
|---|---|---|---|---|---|---|
| 1 | No single dispatch / atomic handoff | SPD, webgpu-spd | Keep | S (prototype exists) | Converging risks missing mips | Measured: ours 1.5–2.4× faster at ≤1024² |
| 2 | Batch ends at every odd size | SPD, nvpro | Planned (tail kernel); halo batches candidate | M / M–L | None | Estimate: a few µs at 1080p |
| 3 | Odd sizes match render passes, not area-weighted or dropped | SPD, nvpro | Keep default; `'box'` planned | M | `'box'` fixes the 31×31 case | ~None |
| 4 | Conservative min/max footprint at odd edges | SPD | Keep | S (but wrong) | SPD's Hi-Z can false-cull | One extra load per odd edge texel |
| 5 | Clamp at edges, no zero reads | SPD | Keep | S | Ours: no dark or zero borders | ~None |
| 6 | Clamp only, no wrap edges | gpuweb#5253 | Candidate | S–M | Seams on tiling textures at odd sizes | ~None |
| 7 | Every level rounded as stored | SPD | Keep; unrounded opt-in is a candidate | S | Ours equals level-by-level generation | Unknown, likely noise |
| 8 | Binary `reduce(a, b)` (fold or sum), not 4-way `SpdReduce4` | SPD | Mostly converged (M2T.1) | — | Sums, counts and weighted means now supported | None |
| 9 | No alpha-coverage preservation | gpuweb#5253 | Candidate | M | Cutout foliage thins out with distance | One extra pass if added |
| 10 | No subgroup / wave ops | SPD, nvpro, fiddle | Planned (M3, gated) | L | Butterfly order changes float sums | Unknown; plausible gain at ≤512² only |
| 11 | No packed fp16 | SPD | Planned (M3) | M | f16 math, slightly less precise | Unknown; nvpro saw no gain |
| 12 | One `vec4f` workgroup array, not per channel | SPD | Planned (M3) | S | None | Unknown, vendor-dependent |
| 13 | float32 loads instead of sampling | nvpro | Planned (M3) | S | None | nvpro: sampler 26–32 % faster (NVIDIA) |
| 14 | No `texture_external` source | fiddle | Candidate | M | None | Saves one full-frame copy per frame |
| 15 | No sub-rectangle yet | SPD | Planned (M2.4) | M | Depends on the snapping decision | Saves work on partial updates |
| 16 | sRGB via scratch copy or buffer, not a UNORM alias | SPD | Planned (M2.3.5) | M | Same results | Extra copy of levels 1+ |
| 17 | Compute only; no per-device render fallback | `greggman`, `toji` | Risk (M3) | M | None | Unknown on NVIDIA/D3D12: reported up to 10× slower there |
| 18 | Needs `STORAGE_BINDING` on the texture | gpuweb#5253 (`Kangz`) | Risk (M3) | — | None | Unknown sampling cost |
| 19 | Timestamp benchmark only | `greggman`, nvpro | Planned (M3) | S–M | — | Credibility of every claim |
| 20 | API conventions: mip count, levels written, no counter buffer | SPD | Keep | S (docs) | — | — |
| 21 | No size or level cap | SPD | Keep | — | — | — |

## Dispatch structure

### 1. No single dispatch or atomic handoff
- **They:** SPD and webgpu-spd write every level in one dispatch. The last workgroup to finish, found with a global atomic counter, reduces the final levels.
- **We:** use one dispatch per batch of up to 6 levels, so two dispatches up to 4096².
- **Why:**
  - The WGSL memory model doesn't guarantee the handoff: atomics are relaxed and barriers are workgroup-scoped (SPEC §2.2).
  - webgpu-spd#7 saw missing mips on M1 and "fixed" it with a `storageBarrier`, which isn't guaranteed in WGSL either.
  - Our own prototype was slower.
- **Effort to converge:** S. The prototype is in the three.js `webgpu-spd-audit` branch.
- **Quality:** converging could produce stale or missing levels on hardware that reorders.
- **Performance (measured):** single dispatch vs ours, in ms per chain:
  - 256² rgba8: 0.0099 vs 0.0068.
  - 1024² rgba8: 0.0305 vs 0.0151.
  - 1024² rgba16f: 0.0592 vs 0.0246.
  - The atomic tail costs about 20 µs. The dispatch boundary it replaces costs 1–2 µs.
- **Verdict:** keep. Revisit only if WGSL gains device-scope acquire/release.

### 2. A batch ends at every odd size
- **They:**
  - SPD carries every level through odd sizes in one pass by dropping the last row or column.
  - nvpro carries 2 levels per dispatch through odd sizes, recomputing a 1-texel halo.
- **We:** end the batch where the next reduction is inexact, so 1920×1080 takes 7 dispatches.
- **Why:** it is the simplest structure that keeps `'exact'` results identical to render passes.
- **Effort to converge:**
  - Tail kernel, M (planned for M2.5): once a level fits one workgroup's memory, reduce every remaining level there with exact weights. Target: 3–4 dispatches at 1080p.
  - Halo batches, M–L (candidate): our odd taps fall in `2x … 2x+2`, the same support as nvpro's, so their halo structure would work without changing results.
- **Quality:** none. Results are unchanged by design.
- **Performance (estimate):** 3–4 fewer boundaries at 1–2 µs each, so about 3–8 µs on a 0.037 ms 1080p chain. Must be measured.

## Odd sizes and edges

### 3. Odd sizes match render passes
- **They:**
  - SPD drops the last row or column.
  - nvpro weights by area, `(n−x, n, x+1)/(2n+1)` for size `2n+1`, so 3→1 averages all three texels.
- **We:** `'exact'` by default: normalized-coordinate bilinear, identical to render-pass and WebGL mipmaps. `'box'` (nvpro's weights) is a planned opt-in. SPD's behaviour (`'fast'`) was dropped.
- **Why:** three.js and other engines can switch to us without any visual change. Hi-Z stays conservative.
- **Effort to converge on nvpro:** M, already planned as `'box'` in M2.5. It needs its own oracle.
- **Quality:**
  - `'exact'` shares the render-pass flaw: a 31×31 texture can reduce to an unrepresentative 1×1.
  - `'box'` is area-correct; recommend it for GI, irradiance and roughness chains.
  - SPD's mode loses edge data.
- **Performance (estimate):** `'box'` has the same dispatch structure and footprint as the folds, so it adds a few ALU ops at most.

### 4. Conservative min/max footprint at odd edges
- **They:** SPD's min/max skip the last row or column at odd sizes.
- **We:** each fold covers texels `2x … 2x+2` along an odd dimension, the smallest footprint that is conservative.
- **Why:** Hi-Z occlusion culling needs it; a non-conservative max can cull visible objects.
- **Effort to converge:** S, but it would reintroduce the bug.
- **Performance:** one extra load per output texel along odd edges, in the first output of a batch only. Not measured separately.
- **Verdict:** keep.

### 5. Clamped edges, no zero reads
- **They:** SPD's unclamped loads read zeros past the texture edge unless the texture is padded.
- **We:** clamp every read inside the texture.
- **Quality:** no darkened borders, and no zero depth leaking into Hi-Z.
- **Verdict:** keep.

### 6. No wrap edge mode
- **They:** gpuweb#5253 asked for wrap or clamp edge handling.
- **We:** clamp only.
- **Why it rarely matters:** even sizes never read past an edge. Wrap only changes odd-size footprints at the border.
- **Effort:** S for folds and `'box'`. Under `'exact'` it would break render-pass parity, so only offer it with `'box'`.
- **Quality:** tiling textures with odd sizes get a faint seam at the border.
- **Verdict:** candidate. Add to SPEC §12 if a user asks.

## Precision

### 7. Every level is rounded as stored
- **They:** SPD keeps unrounded values in workgroup memory between levels and rounds once on store.
- **We:** quantize each level exactly as stored. For 8-bit and 16-bit normalized formats, the math runs in exact integer units with ties to even.
- **Why:** a chain equals level-by-level generation, so results match render passes and are deterministic across batch sizes and devices.
- **Effort to converge:** S, as an opt-in that skips intermediate quantization. It would need an oracle mode.
- **Quality:** SPD's output is closer to a true box average of the base level on deep 8-bit chains. Ours matches what every render-pass generator produces, and doesn't depend on the batch layout.
- **Performance:** unknown; a few ALU ops per texel. Likely noise at bandwidth-bound sizes.
- **Verdict:** keep as the default; the opt-in is a candidate.

## Reductions

### 8. Binary reductions, not a 4-way `SpdReduce4`
- **They:** SPD's `SpdReduce4(a, b, c, d)` is arbitrary, so it can be a weighted average, a sum or a count.
- **We (since M2T.1):** `Reducer` takes a reduction per channel: sum, average, min, max, a weight channel for weighted means, or a custom binary `{ wgsl, kind: 'fold' | 'sum' }`. `Downsampler` keeps idempotent folds only, because its `'exact'` odd sizes have no defined meaning for sums.
- **Why binary:** a pairwise reduction composes inside the tile, and normalisation happens once when a level is emitted. A 4-way custom reduction adds nothing that a binary sum plus a weight channel can't express.
- **Still missing:** argmin/argmax with a payload (ssgiDev's normal taken from the min-depth texel), and a non-commutative "first valid" scan (GI-1 probe masks).
- **Verdict:** mostly converged. The payload fold is a candidate.

### 9. No alpha-coverage preservation
- **They:** gpuweb#5253 lists it as a requirement for a core API.
- **We:** don't offer it.
- **Effort:** M. Each level needs a search for the alpha scale that keeps the base level's coverage at a cutoff, which means an extra pass or a per-level reduction.
- **Quality:** alpha-tested foliage and fences thin out in the distance.
- **Verdict:** candidate for after M4. It is a natural fit for the transform hooks.

## Hardware features

### 10. No subgroup / wave operations
- **They:**
  - SPD uses wave ops with a fixed quad layout (`ARmpRed8x8` and `WaveReadLaneAt`).
  - nvpro uses `subgroupShuffleXor`, assuming lane equals local index.
  - The fiddle uses quad ops in compute.
- **We:** use workgroup memory only.
- **Why:**
  - WGSL defines quads only in fragment shaders and gives no mapping between lanes and `local_invocation_index`, so those layouts are undefined behaviour.
  - Firefox lacks `subgroups`; Chrome and Safari 27.3 have it.
- **Effort:** L. The portable scheme in SPEC §13 builds a linear id from `subgroup_id` and uses `subgroupShuffleXor`. It ships with the workgroup-memory fallback.
- **Quality:** butterfly order pairs sums as `(a+b)+(c+d)`. That doesn't matter for integer units, but the float oracle must pair the same way.
- **Performance:** unknown. Barriers would fall from 10 to 1–2 and workgroup memory from 16 KiB to about 1 KiB. A gain is plausible at ≤512², where the kernel is latency-bound, and unlikely at 4096², where it is bandwidth-bound.
- **Verdict:** planned for M3, behind an upper-bound test; build it only if that shows ≥5 %.

### 11. No packed fp16 math
- **They:** SPD has a packed half-precision path.
- **We:** compute in f32.
- **Effort:** M. It needs a `shader-f16` kernel variant and oracle pairing.
- **Performance:** unknown. nvpro saw no gain from f16 shared memory on an RTX 3090.
- **Verdict:** planned for M3; keep only if measurably faster.

### 12. Workgroup memory layout
- **They:** SPD stores intermediates in separate per-channel arrays.
- **We:** use one `array<vec4f>`.
- **Effort:** S.
- **Performance:** unknown. Bank behaviour differs by vendor.
- **Verdict:** planned for M3; measure it.

### 13. float32 formats load instead of sampling
- **They:** nvpro measured the bilinear sampler 26–32 % faster than loads on NVIDIA.
- **We:** load float32 formats, because sampling them needs `float32-filterable`.
- **Effort:** S. Use the sample path when the device has the feature.
- **Performance:** for rgba8, our own load path measured 12–50 % slower than sampling (Apple, under load). The same is likely for float32 where filterable.
- **Verdict:** planned for M3.

## Sources and inputs

### 14. No `texture_external` source
- **They:** the fiddle samples level 1 straight from a video frame's `texture_external`.
- **We:** need the frame copied first with `copyExternalImageToTexture`, into level 0 or a separate source texture.
- **Effort:** M. Add a source-batch variant that binds `texture_external` on the sample path. That gives level 1 onward, with no level 0.
- **Performance (estimate):** saves one full-frame write and read per frame for video pyramids.
- **Verdict:** candidate. It matters for video, the upscaler or ML preprocessing.

### 15. No sub-rectangle yet
- **They:** SPD's `rectInfo` snaps the rect to 64-texel source tiles, and recomputes levels below 64×64 for the whole texture.
- **We:** don't have it yet; it is planned for M2.4.
- **Effort:** M. Matching SPD's snapping is simplest. A tighter rect saves more work but needs per-level rounding rules.
- **Verdict:** planned. Decide the snapping in M2.4.

### 16. sRGB through a scratch copy or buffer
- **They:** SPD samples an sRGB view and writes a UNORM alias of the same texture.
- **We:** can't do that, because WebGPU forbids sRGB storage even through a view. We'll reduce in linear, write to an rgba8unorm scratch texture or buffer, and copy.
- **Effort:** M (M2.3.5).
- **Performance (estimate):** one extra copy of levels 1 and up, about a third of the base level's bytes.
- **Verdict:** planned. Choose between scratch texture and buffer by measurement.

## Compute vs render, and measurement

### 17. Compute only, with no per-device render fallback
- **They:** `greggman` found compute generators slower than render passes on NVIDIA/D3D12, up to 10× for SPD-style ports, and `toji` reported a partner seeing the same. The community default is render passes.
- **We:** are compute only. The three.js adapter falls back to render passes only for textures we can't handle.
- **Effort:** M. Collect cross-vendor data (M3), then either choose the path per device in the adapters or add a render-pass backend here (SPEC §3 leaves room for one).
- **Performance (measured on Apple only):** 1.3–27× faster than render passes. Unknown on NVIDIA, AMD and Intel.
- **Verdict:** risk. This is the biggest open question, and it decides the three.js default.

### 18. Textures need `STORAGE_BINDING`
- **They:** render-pass generators need only `RENDER_ATTACHMENT`. `Kangz` flagged that adding storage usage to sampled textures has an unmeasured cost.
- **We:** need `STORAGE_BINDING` on every destination; the three.js integration would add it automatically.
- **Effort:** none to measure the cost (M3). Avoiding it would require the buffer output path for every format.
- **Performance:** unknown. Some drivers may disable compression on storage textures.
- **Verdict:** risk; measure in M3.

### 19. Benchmark method
- **They:**
  - `greggman` measures work per frame, because he found timestamp queries unreliable.
  - nvpro interleaves alternatives, takes the min and median over 256 batches, and compares against a speed-of-light kernel.
- **We:** use timestamps bracketing 16 repeats, with medians of 11 rotated rounds.
- **Effort:** S–M. Add a throughput mode and a speed-of-light copy kernel in M3.
- **Verdict:** planned. Every cross-vendor claim depends on it.

## API conventions

### 20. Mip count, levels written, no counter buffer
- **They:** SPD counts `mips` excluding the source level and always halves, and needs an atomic counter buffer allocated and reset.
- **We:**
  - Count `mipLevelCount` including the source level in place, so SPD's `mips = N` is `N + 1` here.
  - With a separate source, write destination `baseMipLevel` at the same size (a copy) or half.
  - Need no counter buffer.
- **Effort:** S, docs only (M4 notes for SPD users).
- **Verdict:** keep.

### 21. No size or level cap
- **They:** SPD handles at most 4096² and 12 levels.
- **We:** handle any size, by chaining batches.
- **Verdict:** keep.
