# Windows results (RTX 3060 Laptop + Intel UHD), 2026-11-12/13

Handoff: `docs/WINDOWS-HANDOFF.md`. Two branches:

| Branch | What | Library code (`src/`) |
|---|---|---|
| `windows-testing` | Harness fixes for Windows, `test/parallel.js`, native harnesses (`native/`), first correctness runs, the sampler probes | unchanged, per the handoff |
| **`f16-rounding`** (from `windows-testing`) | Everything after the owner's decisions: `f16Rounding`, the copy fix, the sampler tolerance model, all timings, this file | **changed**: see below |

**Read `f16-rounding`** for the current state; `windows-testing` stops at the point where the correctness failures were
explained and the owner decided what to change.

## Library changes on `f16-rounding`

- **`f16Rounding: 'nearest' | 'device'`** (Downsampler, Reducer; default `'nearest'`). D3D12 truncates f32 → f16 in both
  `textureStore` and `pack2x16float` (NVIDIA and Intel), so float16 levels drifted low. Levels now round to nearest even
  in the shader (one f32 add and subtract; bit-exact against core `toHalf` on both GPUs). Cost against `'device'`: noise
  on the RTX, 14 % on Intel. Owner decision 2026-11-12.
- **Converting copies load.** A same-size level from a separate source used the sampler at texel centres, which isn't exact
  on Intel (it truncates the coordinate a hair low and blends in 1/256 of the neighbour). It now reads the texel.
- **Sampler tolerance model** (`src/core/reference.ts`, `spd.ts`): `ReferenceOptions.weights: 'sampler' | 'shader'`.
  Sampled levels allow 1.5 weight steps of 1/256 per axis times the local contrast, at any weight, including the texel
  below the footprint (NVIDIA rounds each 2D tap weight; Intel truncates axis weights, even at exact halves and zeros). Shader weights (`averageAccess: 'load'`, in-tile and
  tail levels) keep the strict model. Owner decision: keep `'sample'` the default, document `'load'` (README).
- **`exactTile` (default 32 on hardware adapters with 256 invocations, else 8).** `'exact'` batches of up to four levels
  ran 8 × 8 workgroups, up to 1.6× slower on Intel and 1.3× on the RTX; they now use the 6-level kernel's 32 × 32 tile, and
  level sizes are derived instead of queried from storage textures. Same results (full RTX matrix). `--compare=render` and
  `--dawn-features=` added to the bench (timings.md, "Render passes vs `'exact'`").
- **Docs:** README "Identical results on every GPU" (`averageAccess`, `f16Rounding`); SPEC §Quantization and findings.

## Files

| File | What |
|---|---|
| [correctness.md](correctness.md) | Every failure on both GPUs, traced to f16 truncation and sampler weight rounding; the probes |
| [timings.md](timings.md) | Sampler vs load, f16 rounding cost, native platform paths, locked-clock rerun, webgpu-spd |
| [native-tables.md](native-tables.md) | Every native and library number (`node native/summarize.mjs`) |
| `parallel-*/`, `f16*-*/` | Suite summaries from `test/parallel.js` (summary.md, every failure, per-device durations) |
| `native-*.json`, `*-spd*.json`, `*-webgpu-spd*.json` | Raw results; `logs/` has every job's output and GPU clock samples |
| `sampler-*.json`, `kernel-ramp-*.json`, `f16-store-rounding-*.json` | Probe outputs (`results/probes/`) |

## Headlines

- **Correctness:** no structural failures (levels, layouts, layers, races) in 18,291 cases on the RTX and 10,446 quick on
  Intel. Every failure was f16 truncation (fixed), a sampled converting copy (fixed) or sampler weight rounding (tolerance
  model; `averageAccess: 'load'` is exact everywhere). Quick suites now: 2 failures per GPU, both AMD's FSR hooks.
- **Ours vs native AMD SPD** (RTX, locked clocks; Intel directly): our `'spd'` is 1.26–1.27× AMD's best D3D12 build on the
  RTX, 1.28–1.50× on Intel. WebGPU itself costs ~1.4× for the same render-pass chain on the RTX.
- **Ours vs webgpu-spd:** 1.6× faster in rgba8 on the RTX (level in rgba16f), 2.7–3.5× faster on Intel, where subgroups hurt.
  Its handoff showed 0 stale frames of 41,472 on each GPU (up to 77 % on the M5).
- **Platform paths:** on the RTX compute SPD beats render passes / `GenerateMips` by 15–30 %; on Intel render passes and
  `GenerateMips` beat every compute variant by ~27 % in rgba8. In WebGPU, after the exact-tile fix, our `'exact'` is 1.04×
  render passes on Intel and 1.19× on the RTX in rgba8 (the tail kernel 0.98× there); no render path needed (DIVERGES #17).
- **Process:** the full suite took 35+ min in one page on this laptop (CPU-bound oracles); `test/parallel.js` runs it in
  15–17 min, `--quick` in 2. No run over 30 min (owner).

## Not done

- Upscaler runs (handoff §4): `perf/exposure-parallel` isn't on the remote.
- `npm run bench` and `npm run tools` timings per GPU (prepared in `run-benches.sh`, each capped at 25 min).
- FSR3 native sample (optional).
- AMD's FSR luma-pyramid hooks store f16 with `textureStore`, so they truncate on D3D12 (fixture `fsr/luma-pyramid`,
  `generator/luma-pyramid-clamp`): AMD's own code, the tool owner's call (`float16Wgsl` is exported for it).
