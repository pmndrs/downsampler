# Windows report: RTX 3060 Laptop and Intel UHD (2026-11-12 – 13)

This page is for anyone, person or agent, who wants to know what was measured on Windows, what it showed, how much to
trust it, and how to re-run or extend it. It holds the answers, the method and the open questions. The raw data and the
per-topic detail are in [results/windows/](../results/windows/), linked from each answer.

The brief was [WINDOWS-HANDOFF.md](WINDOWS-HANDOFF.md). Everything here is on branch **`f16-rounding`**, which branches from
`windows-testing` and changes `src/` under owner decisions (see [Library changes](#library-changes-that-came-out-of-this)).
Until it merges, read that branch.

## Setup

| | |
|---|---|
| Machine | Laptop, Windows 11 Home 10.0.26200, i9-11900H (8 cores), 40 GB, on AC, power plan "Turbo" |
| Browser | puppeteer's Chrome 148.0.7778.97, headless, WebGPU on D3D12 |
| GPU 1 | NVIDIA GeForce RTX 3060 Laptop, driver 32.0.15.9200 (NVIDIA 592.00), adapter `nvidia ampere`, 8 storage textures, `texture-formats-tier1` |
| GPU 2 | Intel UHD Graphics (gen-12lp, in the i9), driver 31.0.101.5333 |
| GPU choice | `--power=high` (RTX) or `--power=low` (Intel) on every runner. The browser's GPU is pinned by LUID (`test/gpu-choice.js`), the native harnesses use the DXGI preference or the Vulkan device type |
| Native | Visual Studio 2022, CMake, Windows SDK `dxc`, Vulkan SDK 1.3.283. Upstream code is fetched at pinned commits, not vendored ([native/README.md](../native/README.md)) |
| Timing lock | Every timed job ran alone under the shared GPU lock (`bench-lock.ts` from webgpu-ddgi), which waits for < 70 °C, samples clocks every 200 ms and gives a VALID / INVALID throttling verdict. Logs and clock samples are in `results/windows/logs/` |

**Clocks matter for one kind of comparison.** Unlocked, the RTX ran native jobs at a median of 1460–1630 MHz but the
browser's short bursts at only 620–920 MHz. Cross-run ratios of WebGPU ÷ native came out **1.2–2× too high**. Every
WebGPU ÷ native number on this page for the RTX comes from runs with the clocks locked at 1200 MHz (`*-locked.json`).
Same-run ratios (one strategy ÷ another in one bench) don't move with clocks. Intel has no clock telemetry; its ratios
move about ±10 % between runs.

## Answers

Medians over the render sizes 960×540 … 2560×1440 unless stated. Under ~5 % is noise. "rgba8 / rgba16f" pairs are
average reductions of rgba8unorm and rgba16float.

### Correctness

**Is the library correct on D3D12?** Yes, structurally. The RTX full matrix (18,292 cases) and the Intel quick suite
(10,437) found no failures in levels, layouts, layers, batch plans or races. Every failure was a precision mismatch with
one of three causes. Two were fixed in the library and one in the oracle. Two failures remain on each GPU: AMD's FSR
luma-pyramid hooks, which call `textureStore` for f16 themselves (cause 1). That is AMD's code, and `float16Wgsl` is
exported for it. [correctness.md](../results/windows/correctness.md)

1. **f32 → f16 truncates toward zero on D3D12**, in both `textureStore` and `pack2x16float`, on NVIDIA and Intel. On the M5
   Pro only `textureStore` truncated, so the old quantize step didn't help here. Fix: `f16Rounding: 'nearest'` (default)
   rounds to nearest even in the shader with one f32 add and subtract, bit-exact against core `toHalf` on both GPUs.
   Cost against `'device'`: noise on the RTX, **14 % on Intel**. Probe: `results/probes/f16-store-rounding.mjs`.
2. **Hardware bilinear weights differ by vendor.** NVIDIA rounds axis weights to 8 bits and then rounds each of the four
   2D tap products to 1/256. Intel truncates axis weights, even at exact halves: `(x + 0.5) / n` lands a float ulp below
   the texel edge, so at 1920 → 960 every weight is 127/256. The native platforms' own mip paths (D3D12 render passes,
   `GenerateMips`, Vulkan blit) show the same deviation, so the cause is the hardware sampler, not our kernel. Fix: the
   oracle models samplers (`ReferenceOptions.weights: 'sampler'`, 1.5 weight steps of 1/256 per axis × local contrast).
   Probes: `sampler-weights`, `sampler-2d-weights`, `sampler-ramp`, `kernel-ramp`.
3. **Same-size converting copies sampled at texel centres**, which isn't exact on Intel (cause 2). They now load the
   texel.

**Are the results identical on every GPU?** Not with the default `averageAccess: 'sample'`: the sampler's weights are
the vendor's. With `averageAccess: 'load'` the shader applies the weights, and every case passes the strict oracle on both
GPUs. Cost of `'load'`: RTX 1.13 / 1.02, Intel 1.31 / 1.22. The owner kept `'sample'` as the default and documented
`'load'` (README "Identical results on every GPU").

### Speed against native

**How far is our `'spd'` mode from AMD's own SPD?** RTX (locked clocks): **1.26 / 1.27×** AMD's best D3D12 build.
Intel: **1.28 / 1.50×**. At 4096² it is 1.00 on the RTX, so per-texel work is at parity. The gap at render sizes is fixed
cost per dispatch. [timings.md](../results/windows/timings.md), [native-tables.md](../results/windows/native-tables.md)

**How much of that is WebGPU itself?** The same render-pass chain costs **1.45 / 1.34×** in WebGPU what it costs in native
D3D12 on the RTX (locked), and **2.04 / 2.59×** on Intel. Part of every WebGPU ÷ native gap is the platform: Dawn's
translation, barriers and per-pass overhead.

**Does the native single dispatch (atomic handoff) pay?** Not on NVIDIA: all eight AMD SPD variants are within noise of
each other there (atomic or split, quad ops or LDS, load or sampler). On Intel, AMD's quad-op path as shipped is 2× its
LDS path. DIVERGES #1.

**Do the vendors prefer different paths natively?** Yes. On NVIDIA, compute (SPD) beats D3D12 render passes and D3D11
`GenerateMips` by 15–27 %. On Intel, render passes and `GenerateMips` beat every compute variant in rgba8 by about 27 %,
and nvpro's compute is 3–5× slower than `vkCmdBlitImage`.

### Speed inside WebGPU

**Should we ship a render-pass path for some devices (DIVERGES #17)?** No, after the exact-tile fix (next answer). In
WebGPU, render passes save fixed cost per step (about 10 µs per dispatch at render sizes on the RTX), not per texel: at
4096², level 1 costs the same either way. Rendering the first 1–3 levels and finishing with `'exact'` was never faster
than `'exact'` alone.

| ÷ WebGPU render passes (same run) | RTX rgba8 | RTX rgba16f | Intel rgba8 | Intel rgba16f |
|---|---|---|---|---|
| `'exact'` before the tile fix | 1.34 | 1.04 | 1.62 | 0.98 |
| `'exact'` now | 1.19 | 1.06 | 1.04 | 0.66 |
| `'exact'` + tail kernel | **0.98** | **0.90** | 1.02 | 0.63 |
| `'spd'` | 0.75 | 0.75 | 0.78 | 0.68 |

**Why was `'exact'` slow on Intel?** Batches of up to four levels ran 8 × 8 workgroups. They now use the 6-level kernel's
32 × 32 tile (16 × 16 invocations, 2 × 2 texels each), and level sizes are derived instead of queried with
`textureDimensions` on storage textures. Results are the same bits. New ÷ old: RTX 0.95 / 1.03, Intel **0.81 / 0.94**
(down to 0.61 at 1280×720, 1080p and 1440p). The size queries alone were worth 5–25 % on Intel rgba8. The default is
`exactTile` 32 on hardware adapters with 256 invocations per workgroup, and 8 in compatibility mode and on fallback
adapters. SwiftShader takes 30 s or more to compile a 32 × 32 pipeline.

**Are Dawn's robustness checks or validation the gap?** No change seen. `--dawn-features=disable_robustness,skip_validation`
moved every strategy together: RTX 1.00 ± 0.02, Intel ± 10 %. Chrome doesn't report whether a toggle took effect, so this
isn't a confirmed ablation.

**Is the tail kernel worth making the default?** It is faster on the RTX and neutral on Intel. Tail ÷ `'exact'`: RTX
**0.82 / 0.90**, Intel 0.97 / 0.97. On the M5 Pro it was 0.83–1.07, but that run was on a loaded machine with the old
8 × 8 tiles, so the baseline has changed. Results match `'exact'` within one rgba8 step: the tail levels use shader
weights, which are the same on every GPU. SPEC §2.1's rule is "default if an idle run shows ≥ 5 % at render sizes". The
RTX meets it and Intel doesn't; the M5 hasn't been re-run.

**Is webgpu-spd (third party, 3.0.0) faster?** No. Its default ÷ our `'spd'`: RTX 1.64 / 1.07, Intel **3.52 / 2.68**. Most
of Intel's gap is its subgroup reductions: without them it is 1.78 / 1.35, which matches AMD's quad-op path natively.
Its atomic handoff gave 0 stale frames of 41,472 on each GPU here, against up to 77 % stale frames on the M5 Pro
(DIVERGES #22). There is nothing in it for us to adopt.

## Library changes that came out of this

All are on `f16-rounding` and were verified on the RTX full matrix, the quick suites on both GPUs and a SwiftShader CI
shard. They touch seven files in `src/` (about 220 lines; `git diff origin/m2-reductions...f16-rounding -- src`).

| Change | Files | Why | Owner decision |
|---|---|---|---|
| `f16Rounding: 'nearest' \| 'device'` (Downsampler, Reducer; default `'nearest'`) | `kernel.ts`, `downsampler.ts`, `reducer.ts`, `index.ts` | D3D12 truncates f16 | 2026-11-12 |
| Converting copies load instead of sampling | `kernel.ts` | Intel sampler at texel centres | Part of the same fix |
| `ReferenceOptions.weights: 'sampler' \| 'shader'` | `core/reference.ts`, `core/spd.ts` | Oracle for vendor samplers | Keep `'sample'` default, document `'load'` |
| `exactTile` (default 32 / 8), derived level sizes | `core/plan.ts`, `kernel.ts`, `downsampler.ts` | Intel 1.6×, RTX 1.3× | Not asked: same bits; reported 2026-11-13 |

## Ruled out (don't redo these)

- **Render passes for the large levels, then compute** (`hybrid1`–`3`): no faster than `'exact'`.
- **Dawn's robustness and validation**: no change seen on either GPU.
- **One pass per batch vs one encoder** (`encodePass` vs `encode`): the cost is per dependent dispatch, not per pass.
- **Our kernel arithmetic**: `'load'` passes the strict oracle on both GPUs. The sampled deviations come from the samplers.
- **webgpu-spd's handoff race on Windows**: not observed (0 / 41,472). It is real on the M5.
- **Subgroup or quad reductions on Intel**: about 2× slower, natively and in webgpu-spd.
- **Comparing unlocked RTX runs across WebGPU and native**: misleading by 1.2–2×.

## Open questions and things to try

| Question | Why it matters | How |
|---|---|---|
| Tail kernel as the default? | 10–18 % on the RTX; the M5 decides | M5: `npm run bench -- --compare=render --skip-validation` (has `tail` and `exact8` columns) |
| `exactTile` 32 on Apple? | Default changed without an M5 number | Same run, `exact` ÷ `exact8` |
| Fewer dispatches at odd sizes (halo batches under `'exact'`) | The only lever left on the native gap; upper bound is `'spd'` (0.75 of render here) | SPEC §13 item 3. Needs its own oracle and GPU cases first |
| Intel f16 rounding costs 14 % | Largest remaining cost of a correctness default | Measure `'nearest'` ÷ `'device'` after the tile fix (not re-measured); look for a cheaper rounding |
| Intel full matrix | Only `--quick` ran on Intel; the first full attempt lost the device after 580 s | `node test/parallel.js --jobs=4 --cap=10 --budget=30 -- --power=low`. Size it first; it may need two halves with `--only=` |
| `npm run bench` and `npm run tools` per GPU | Not run (handoff §2) | `sh results/windows/run-benches.sh rtx3060-bench`, then `-tools`, `intel-uhd-…` |
| AMD, Intel Arc, Qualcomm GPUs | Only NVIDIA Ampere and Intel gen-12 measured on Windows | Same harnesses; `--power` picks the GPU on a laptop |
| Chrome's Vulkan backend on Windows | Every WebGPU number here is D3D12 | Not tried |
| Upscaler runs (handoff §4) | `perf/exposure-parallel` wasn't on the remote | Upscaler repo |

## Reproduce or extend

```sh
npm run build                         # tests and benches load dist/
npm run test:core                     # backend-neutral unit tests
node test/parallel.js --jobs=6 --cap=10 --budget=30 -- --quick --power=high   # RTX quick, ~2.5 min
node test/parallel.js --jobs=6 --cap=10 --budget=30 -- --power=high           # RTX full, ~17 min
npm run bench -- --compare=render --skip-validation --power=high --output=results/windows/x.json
npm run bench -- --compare=spd --skip-validation --power=low                  # sample/load, f16 rounding, tail
npm run bench -- --compare=webgpu-spd --formats=rgba8unorm,rgba16float,r32float --skip-validation
node test/run.js --webgpu-spd-check --skip-validation --frames=256
node results/probes/sampler-weights.mjs --power=low                           # any probe in results/probes/
sh results/windows/run-benches.sh <job-filter>                               # every timed job, under the GPU lock
node native/summarize.mjs > results/windows/native-tables.md                  # tables from all native + WebGPU JSON
```

The native build and run commands are in [native/README.md](../native/README.md). Gotchas we hit:

- **Never start a run that can exceed 30 minutes** (owner rule). Use `test/parallel.js` with `--cap` and `--budget`. One
  page at a time, the full suite takes 35+ min, because the oracles are CPU-bound JavaScript.
- **Don't run `npm run build` while a job runs.** Every bench job rebuilds `dist/`, and running pages load it.
- **Don't edit `run-benches.sh` while `sh` is running it.** `sh` reads the script as it goes, so an edit can break the jobs still to come.
- **Lock the RTX clocks** for any WebGPU vs native comparison: `nvidia-smi -lgc 1200,1200` (admin); reset with
  `nvidia-smi -rgc`. The lock script is outside this repo; point `BENCH_LOCK` at `webgpu-ddgi/tools/reference/bench-lock.ts`.
- **SwiftShader** exposes 4 storage textures, so it skips the 5- and 6-level kernels, and it compiles 32 × 32 tiles in
  30 s or more. A real GPU is required for kernel changes.
- **Native build:** sign DXIL with the Windows SDK's `dxc`, not the Vulkan SDK's. A global `CMAKE_TOOLCHAIN_FILE` that
  points at a missing vcpkg breaks configure (`env -u CMAKE_TOOLCHAIN_FILE cmake …`).
- **Dual-GPU laptops:** pass `--power` to every runner, or the browser and the native harness may pick different GPUs.

## Where the data is

| What | Where |
|---|---|
| Summary for this branch and file index | [results/windows/README.md](../results/windows/README.md) |
| Every failure, its cause and the probes | [correctness.md](../results/windows/correctness.md) |
| Timing tables with the method and caveats per section | [timings.md](../results/windows/timings.md) |
| Every native and WebGPU number per size | [native-tables.md](../results/windows/native-tables.md) |
| Raw bench JSON | `results/windows/*-spd*.json`, `*-render*.json`, `*-webgpu-spd*.json`, `native-*.json` |
| Suite runs (every failure, per-shard durations) | `results/windows/{parallel,f16,f16fast,tolerance}-*/` |
| Probe scripts and outputs | `results/probes/*.mjs`, `results/windows/{sampler,kernel-ramp,f16-store-rounding}-*.json` |
| Job logs and clock samples | `results/windows/logs/` |
| Where we differ from SPD, nvpro and the community, and why | [DIVERGES.md](../DIVERGES.md) #1, #2, #17, #22 |
