# Benchmarks

`npm run bench -- <flags>` builds, then runs `node test/run.js --bench <flags>` in headless Chrome on the real GPU
(add `--skip-validation` to skip the suite first, `--power=high|low` on dual-GPU laptops, `--output=results/….json` to
keep the report). The method shared by every mode is in SPEC.md §8. Differences under about 5 % are noise; only
same-run ratios are trustworthy on a loaded machine.

| Mode | File | What it answers |
|---|---|---|
| (default) | `bench.js` | render passes vs the library at 4 and 6 levels per dispatch, `'load'`, `'max'`, buffer output |
| `--compare=spd` | `bench.js`, `atomic-spd.js` | odd-size strategies: `'exact'`, tail, `'spd'`, the atomic probe, render (M2T.4) |
| `--compare=render` | `bench.js` | render passes vs `'exact'`, hybrids, level 1 alone, tile sweep (DIVERGES #17), `halo` and `haloAny` (halo batches, capped and uncapped; DIVERGES #2) |
| `--compare=webgpu-spd` | `webgpu-spd.js` | webgpu-spd 3.0.0 vs ours (results/compare/webgpu-spd.md) |
| `--compare=throughput` | `throughput.js`, `bench.js` | chains per second without timestamps (M3) |
| `--compare=storage-usage` | `storage-usage.js` | does STORAGE_BINDING make sampling slower? (M3) |
| `--compare=subgroups` | `kernel-variants.js` | upper bound for a subgroup kernel (M3) |
| `--compare=f16` | `kernel-variants.js` | averaging in f16: speed and precision (M3) |

`--strategies=a,b,c` times a subset in every `--compare` mode. The strategy a mode's ratios divide by is always kept:
`render` for `render` and `throughput`, `exact`/`exactMax` for `spd` and `webgpu-spd`, the library kernel for the probes.
For `storage-usage` the names are its shaders. An unknown name fails and lists the valid ones.

The `exact*` strategies pin `tail: false`. The library's default is the tail kernel (since 2026-11-13), and `tail` has
its own column.

## Throughput (`--compare=throughput`, or `--throughput`)

This is greggman's method from gpuweb#5253: count work per frame instead of reading timestamps, so it runs on devices
without `timestamp-query` and doesn't depend on how a driver implements timestamps.

- **Unit:** one mip chain: `render` passes, `exact` (`tail: false`), `spd`, `tail`. Defaults: the render sizes plus
  1024² and 4096², rgba8unorm and rgba16float.
- **Calibration:** N chains per submit, doubling until one submit takes `--batch-ms` (8 ms; 25 ms with rAF), capped
  at `--max-batch` (1024).
- **Window:** for `--window` ms (1000), submits are encoded and kept in flight: `--in-flight` (2) outstanding,
  each awaited with `queue.onSubmittedWorkDone()`. With `--pacing=raf`, it is one submit per `requestAnimationFrame`,
  as an app records a frame. N is then sized to exceed a 60 Hz frame, or the rate would be the display's.
- **Result:** wall ms per chain (and chains/s), median over `--rounds` (5) windows in rotated order. Ratios to
  `render` are paired per round. `--textures=k` cycles k identical textures per strategy, giving independent chains, as
  when an app generates mips for several textures per frame.
- **CPU share:** the time spent in JS encoding and submitting, as a fraction of the window. Above 80 % the row is
  flagged `BOUND`: the number then measures encoding, not the GPU. JS can't see Chrome's GPU process (Dawn), which works
  between the page and the GPU, so the check below matters.
- **Check:** where timestamps exist, the same strategies, sizes and formats are then timed with timestamps
  (`--no-timestamps` skips this). `agreement` counts strategy pairs that both methods order the same way; pairs within
  5 % by timestamps count as ties. It also reports Kendall's τ. `--no-timestamp-query` requests a device without the
  feature, to exercise the throughput-only path on any GPU.
- **JSON:** `timing.results[]` has `msPerChain`, `throughput[name]` (`msPerUnit`, `unitsPerSecond`, `cpuMsPerUnit`,
  `cpuShare`, `cpuBound`, `batch`, `windows`, `ratio`), `gpuMs`, `gpuRatios` and `agreement`. Summary: `timing.agreement`.

## STORAGE_BINDING sampling cost (`--compare=storage-usage`)

This is Kangz's question in gpuweb#5253. Three.js would add the usage to every texture whose mips we generate (SPEC §10).

- **Textures:** two per case, identical except for `STORAGE_BINDING`. Both have TEXTURE_BINDING, RENDER_ATTACHMENT,
  COPY_SRC, COPY_DST and a full mip chain. Formats: rgba8unorm, rgba16float and r32float (sampled with a non-filtering
  sampler). Sizes: the render sizes, 1024² and 4096².
- **Content:**
  - `rendered`: drawn by a render pass, with flat bands and gradients. This is the compressible kind, and lossless
    framebuffer compression (which a storage usage may turn off) only pays on content like it.
  - `noise`: uploaded random texels. Incompressible, so it is the control.

  Levels 1+ are rendered 2×2 averages for both textures.
- **Shaders** (`--strategies` picks among them):
  - `fullscreen`: one tap per pixel at level 0, like a post pass.
  - `material`: nine trilinear taps with implicit derivatives at about 1.6× minification, like a material.
  - `write`: the content pass itself, rendering into level 0 (the cost of the usage on a render target). `rendered` only.
- **Timing:** `--repeats` passes into one rgba8unorm target, with `timestampWrites` on the first and last pass (never
  empty passes). Rounds alternate the order. Medians; storage ÷ plain is paired per round.
- **Throughput:** `--throughput` adds per-pass throughput (`throughputRatio`). Without timestamp-query, this mode
  measures throughput only.
- **JSON:** `results[]` (`gpuMs.plain`, `gpuMs.storage`, `ratio`, `samples`, `throughput`) and `summary["<format>
  <content> <shader>"]` (the median ratio over sizes, with its range).

## Subgroup upper bound (`--compare=subgroups`)

The SPEC M3 rule: build a subgroup kernel (with the workgroup-memory fallback) only if it could win at least 5 %.
Windows showed Intel losing about 2× with subgroup or quad reductions, so the decision rests on Apple and NVIDIA.

The probe runs the library's own chain: `Downsampler.prepare` with the defaults, so the tail kernel and the device's
`exactTile`. The variants replace `main()` of every `'exact'` batch of two or more levels. Everything else is the
library's WGSL and bind groups.

| Variant | What | Results |
|---|---|---|
| `lds` | the library kernel: two barriers per level (10 in a 6-level batch) | reference |
| `slot` | one barrier per level: level i's texel goes to the slot of its first input at level i−1's stride, so no reader is overwritten (SPEC §13 "cheaper first"); no subgroups | must be identical |
| `shuffle` | the first reduction from `subgroupShuffleXor` (masks 1 and the workgroup side), then `slot`; 4 barriers per 6-level batch. Assumes lane = `local_invocation_index` mod subgroup size and subgroups of at least two rows, which WGSL doesn't guarantee | identical where the assumption holds (it did on the M5 Pro); otherwise timing only |
| `nosync` | levels 2+ from each invocation's own registers: no barriers, no workgroup memory, stores unchanged | timing only (wrong values) |

No reduction scheme is cheaper than `nosync`, so **(lds − nosync) ÷ lds bounds what any subgroup kernel can gain**. If
that bound is under 5 % at render sizes, drop the subgroup kernel. If it is above, `shuffle` and `slot` show how much
of it real kernels recover. `slot` needs no feature.

- **Check:** each variant except `nosync` is compared level by level with the library's chain. Two inputs are used:
  the timed one (`random`) and `tiles`/`large`. The check reports values that differ and the worst difference in format
  steps.
- **Timing:** as the main bench (`timeGpu`: render markers, marker cost subtracted), rotated, medians, ratios paired per
  round. `--throughput` adds throughput.
- **JSON:** `results[]` (`batches`, `gpuMs`, `ratios`, `checks`) and `summary["<format> <variant>"]` (median ratio
  over render sizes and over squares, and whether every check was identical).

## fp16 arithmetic (`--compare=f16`)

The same harness as the subgroup probe, with `shader-f16`.

| Variant | What |
|---|---|
| `f32` | the library kernel |
| `f16` | workgroup memory as `vec4h` (half the bytes), levels 2+ averaged in f16 as `(a + b + c + d) * 0.25` |
| `f16safe` | the tile holds texel × 0.25, so a level is the sum of four quarters: no overflow below 65504 |

- **First output:** still sampled and rounded in f32, then converted. The conversion is exact: integer units for unorm8,
  halves for float16.
- **Fast math:** nothing here uses `(x + c) − c` rounding, which Metal's fast math folds away. Scaling inside the sum
  (`a * 0.25 + b * 0.25 + …`) overflowed on the M5 Pro just like `f16`, because fast math refactored it to
  `(a + b + …) * 0.25`. Hence quarters in memory.
- **Precision cost:** the difference from the library's f32 chain, which the GPU suite checks against the CPU oracle.
  Counted per level, in unorm8 steps or f16 ulps, plus non-finite values. Inputs: `random` (HDR in [−4, 12] for
  float16) and `large` (halves in [16384, 65504], where f16 sums overflow), or `tiles` for unorm8. unorm8 sums (≤ 1020)
  and quarters are exact in f16, so rgba8 should be identical.

Only rgba8unorm and rgba16float are covered (`--formats=`). On devices without `shader-f16`, the f16 variants are
skipped and reported.
