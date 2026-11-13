# Windows handoff (branch `windows-testing`)

This is for a Claude Code agent on the owner's Windows laptop. The laptop has an NVIDIA RTX 3060 laptop GPU and Intel UHD integrated graphics. AMD's native benchmarks are also installed there, for the owner's GI work.

Everything up to here was built and verified on an Apple M5 Pro only. This branch asks for three things:
1. Does it pass on D3D12?
2. How fast is it on NVIDIA and Intel?
3. How does it compare with native D3D12/Vulkan implementations on the same GPU?

Read `CLAUDE.md`, `SPEC.md` (§2, §8, §13, §14) and `PROGRESS.md` first. `DIVERGES.md` has the open questions this run answers: entries 1–3 and 17.

## Rules

- Work on `windows-testing` and commit results to it. Push only that branch. Never push `main` or `m2-reductions`.
- **Don't change library kernels** (`src/`). If something fails on D3D12, record it: the failing case, the full output, the adapter info and the driver version. Then stop and report. A kernel fix needs the full GPU matrix on a real GPU, and this branch isn't where that happens.
- **Name every number.** Give the GPU, the driver version, the Chrome version, the power state (plugged in, power plan) and the run's load. Differences under ~5 % are noise.
- **Commit raw JSON** to `results/windows/`, with a short `.md` next to it summarising what it shows.
- Draft nothing for outside repos or forums. Put questions for the owner in `PROGRESS.md` under *Waiting on owner*.

## Setup

```
git clone https://github.com/pmndrs/downsampler && cd downsampler
git checkout windows-testing
npm ci
npm run build
```

`--power=high` selects the RTX 3060; `--power=low` selects the Intel UHD. Check the adapter line each run prints. If it shows the wrong GPU, also set the GPU for Chrome in Windows Graphics settings, then rerun.

Each command below prints `load average`. On Windows that reads 0, so also note anything else running. Close other GPU apps first, including the GI benches.

## 1. Correctness (do this first)

```
npm run test:gpu -- --power=high
npm run test:gpu -- --power=low
npm run test:gpu -- --power=high --compat
```

- Expect "validation passed" with about 10–20k cases. Each full run takes 15–25 min.
- A failure here matters more than any timing.
- Record each run in the `PROGRESS.md` verification log: date, commit, device, suite, cases.

## 2. Library timings

```
npm run bench -- --power=high --output=results/windows/rtx3060-bench.json
npm run bench -- --power=low --output=results/windows/intel-uhd-bench.json
npm run bench -- --compare=spd --skip-validation --power=high --output=results/windows/rtx3060-spd.json
npm run bench -- --compare=spd --skip-validation --power=low --output=results/windows/intel-uhd-spd.json
npm run bench -- --compare=webgpu-spd --formats=rgba8unorm,rgba16float,r32float --skip-validation --power=high --output=results/windows/rtx3060-webgpu-spd.json
node test/run.js --webgpu-spd-check --skip-validation --frames=256 --power=high
npm run tools -- --power=high --output=results/windows/rtx3060-tools.json
npm run tools -- --power=low --output=results/windows/intel-uhd-tools.json
```

**What the owner needs from these:**
- **Compute vs render passes on NVIDIA/D3D12.** The `render` column of `--compare=spd`. gpuweb#5253 reported compute slower there. If ours is too, say so plainly; that decides DIVERGES 17.
- **The tail kernel.** Is `tail` 5 % or more faster than `exact` at render sizes? If not, it stays opt-in.
- **`'spd'` vs `'exact'`.** Does the 1.4–2.1× gap seen on Apple hold?
- **webgpu-spd's handoff.** Does it also return stale levels 7+ on NVIDIA or Intel? Give its stale-frame counts per size.
- **Tool fixtures.** Do the generator versions still tie or beat the upscaler's hand-written passes (`test/tools/README.md`)? And the 8-channel vs sign-bit gap at 1129×635.

## 3. Native matches (new)

**Goal:** a native D3D12 or Vulkan number on the same GPU for the same work, so the WebGPU numbers have a ceiling and a baseline. Use what is already installed. Look first for the AMD FidelityFX SDK and Capsaicin, which the GI work uses; fetch only what is missing.

Match the work exactly:
- **Sizes:** 960×540, 1280×720, 1920×1080, 2560×1440, 1024², 4096², plus 1129×635 and 1707×960 (odd, upscaler internal resolutions).
- **Formats:** rgba8unorm and rgba16float. For Hi-Z, r32float max.
- **Work:** a full mip chain unless noted.
- **Timing:** GPU timestamp queries around the chain, warm-up first, median over at least 100 runs. Run in the same session as the WebGPU runs, so the clocks are comparable.
- **Record** the API (D3D12 or Vulkan), the driver version and the exact sample commit.

In priority order:

1. **AMD FidelityFX SPD, native (D3D12).** This is the reference our `oddSizes: 'spd'` and webgpu-spd copy.
   - Time it with its single-dispatch atomic handoff on (the default). Also time it with 6 levels per dispatch, if its sample or code allows.
   - Compare with `spd` and `atomic` in `--compare=spd`.
   - Read back its levels at one odd size (1129×635). Diff them against `spdReference` (`src/core`) to confirm the semantics really match.
2. **D3D11 `GenerateMips`** (the driver's own path) or a native render-pass blit chain in D3D12.
   - This tests toji's hypothesis in gpuweb#5253, that drivers fast-path the render-pass pattern.
   - Compare with our `render` and `exact`.
3. **nvpro `vk_compute_mipmaps`** (Vulkan, NVIDIA only). Its area-weighted odd sizes are what our planned `'box'` will be, and it is NVIDIA's own tuned compute generator. Compare with `exact` and `spd` on the RTX 3060.
4. **FSR 3.1 native passes,** if the FidelityFX SDK FSR3 sample builds. Get the luma pyramid / exposure and the shading-change pyramid GPU times from its profiler, at the upscaler's sizes. Compare with `fsr/*` and `generator/*` in `npm run tools`.

**Where things go:**
- Put any native harness code you write under `native/<name>/`, with a README giving the build steps and the upstream commit.
- Don't vendor large SDKs: reference them by path and commit.
- Results go in `results/windows/native-<name>-<gpu>.json`, plus a summary `results/windows/native.md`. The summary is a table per size and format: native ms, our ms, and the ratio.

## 4. Upscaler (optional)

This needs the upscaler's `perf/exposure-parallel` branch. It is only on the owner's Mac unless the owner pushes it to pmndrs/upscaler. If it's there:

```
node scripts/measure-exposure-parallel.mjs --power-preference high-performance --expected-vendor nvidia
node scripts/measure-exposure-parallel.mjs --power-preference low-power --expected-vendor intel
node scripts/measure-exposure-compile.mjs
node scripts/run-benchmark.mjs --smoke --ratios 1,1.5,2,3 --blocks 4 --warmup 240 --samples 300 --variant baseline --comparison exposure-serial-v1
```

It should show the 6–10× exposure speedup off Apple, with DXC compile time at or under 39 ms.

## Report

- Update `PROGRESS.md`: the verification log rows, plus what is answered under *Waiting on owner*.
- Update `DIVERGES.md` entries 1–3 and 17 with the new numbers. Keep `docs/diverges.html` in step.
- Commit, push `windows-testing`, and summarise for the owner. Cover:
  - correctness on both GPUs;
  - compute vs render vs native;
  - which recommendations the numbers change.
