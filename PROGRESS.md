# Progress

Live status board. SPEC.md holds the design and the reasons behind it; this file holds where things stand.

**For the owner:** open decisions and the Windows run sheet are also in the artifact (`docs/diverges.html`), with a recommendation on each.

**For agents:** read this after SPEC.md. When you start a task, move it to *Now* with your branch. When you finish, tick it, add the commit, and log the verification run. Keep entries to one line and link SPEC sections instead of restating them. Anything that needs the owner goes under *Waiting on owner*.

Last updated: 2026-11-12 · Branch: `m2-reductions`

## Now

- [ ] Windows runs (owner + a Windows agent, RTX 3060 + Intel UHD): branch `windows-testing`, handoff in `docs/WINDOWS-HANDOFF.md`. Verification, cross-vendor timings, and native D3D12/Vulkan matches (FidelityFX SPD, D3D11 `GenerateMips`, nvpro) for the same sizes
  - [x] Correctness: RTX full + quick + compat quick, Intel quick, all groups run; failures traced to f16 truncation and sampler precision, both explained by probes ([results/windows/correctness.md](results/windows/correctness.md))
  - [x] Native harnesses built (`native/`): D3D12 AMD SPD (8 variants), D3D12 render, D3D11 `GenerateMips`, Vulkan nvpro + blit; level dumps at 1129×635 match our oracles ([native-semantics-rtx3060.json](results/windows/native-semantics-rtx3060.json))
  - [x] `f16Rounding` (branch `f16-rounding`, `1af33de`): RTX full matrix 287 → 85 failures, every f16 one gone on both GPUs
  - [x] Timings, native + `--compare=spd` on both GPUs ([results/windows/timings.md](results/windows/timings.md)); RTX WebGPU ran at 620–920 MHz vs native ~1600, so cross-API ratios there wait for locked clocks
  - [ ] Remaining handoff jobs: `--compare=webgpu-spd`, `bench`, `tools`, `--webgpu-spd-check` per GPU (prepared in `run-benches.sh`)
  - [x] `test/parallel.js`: the suite in capped parallel shards, one summary; full matrix 35+ → 17 min, quick 4.6 → 2 min (RTX 3060)
- [ ] CI: first GitHub run of the shards (`test/ci/README.md`): real shard times, Linux SwiftShader compat device, artifact collection. Then commit its `gpu-durations` artifact

## Next (SPEC §11)

- [ ] Upscaler on the library through a local `file:` reference (owner: the upscaler releases after this is on npm)
- [ ] M2.4: layer range + rect
- [ ] M2.5: `oddSizes: 'box'`, power-of-two destination from a separate source (SPEC §2.1, §13)

## Deferred (three.js-driven)

- [ ] M2.3.3: rgb10a2unorm
- [ ] M2.3.5: sRGB + transform hooks for colour textures (hooks themselves arrive with M2T.1)
- [ ] M2.3.7: integer formats

## Later

- **M3, dial in:** per-frame throughput benchmark; cross-vendor device lab runs (NVIDIA/D3D12 first); `STORAGE_BINDING` sampling cost; float32 sampler vs load; subgroup kernel with fallback (upper-bound test first); fp16
- **M4, examples and docs:** usage docs, notes for SPD users, examples (mip chain, Hi-Z from depth, custom reduction, three.js)
- **M5, ship:** npm publish, the upscaler and other tools as consumers, three.js in-tree PR, the gpuweb#5253 comment

## Done

- [x] CI shards like homefig: `--shard=i/N` over 323 SwiftShader case groups, LPT-balanced by `test/ci/durations.json`; 14 core shards (+14 compat on the full tier); `gpu-suite` aggregate checks coverage (`38beb25`, merge `44b25bc`)
- [x] webgpu-spd 3.0.0 compared (`results/compare/webgpu-spd.md`, DIVERGES #22): its default single-dispatch handoff returned stale levels 7+ on M5 Pro (up to 5 % of frames at render sizes, 25–41 % at 4096²); correct with `maxMipsPerPass: 6`; our `'spd'` 1.2–2.1× faster for the same semantics (`805c21f`)
- [x] Fixed `--compare=spd` parity check: the atomic probe now rounds f16 before `textureStore`, like the library
- [x] M2T.4: `oddSizes: 'spd'` (average only; folds rejected), exact tail kernel (`tail: true`, opt-in), atomic single-dispatch probe (test-only); both compose with the buffer path and compatibility mode; 12/12 mutations caught (`b4ef2dc`…`1b7e897`)
- [x] M2.3.4 + M2.3.6: buffer output path (6 levels per dispatch at any storage limit), rgb9e5ufloat, rg11b10ufloat; composes with compatibility mode; Reducer f16 outputs round to nearest even; 12/14 mutations caught, 2 identical on Apple (`9d6b00d`, `213cb23`, `b2ec515`)
- [x] Tool API: shared group-1 plumbing (`tool.ts`), combine-only scalar (fused luma pyramid 3 → 2 dispatches), 8-channel `DsWide` accumulators, argmin/argmax payload folds, compatibility mode (D4100; 8599 cases on a real compat device); 19/19 mutations caught (`a00e291`…`af98701`)
- [x] M2T.3: generator versions of every FSR 3.1 and upscaler pass pass their oracles; upscaler shading change ties the hand-fused pass (0.95–1.02×), every-texel exposure 0.33–0.60× of the upscaler's, fused luma pyramid 0.89–1.15× of FSR's every-texel (M5 Pro, loaded, paired ratios; `bfcdaea`)
- [x] M2T.1: `Reducer` tile kernels: tool hooks in group 1, per-channel sums/weights, level selection, `'zero-weight'`/`'clamp'` edges, self-dispatch; 10/10 mutations caught (`97664d7`, `546692b`)
- [x] Upscaler: parallel exposure, same 1024 taps, 6–10× faster on M5 Pro, output bit-identical on 1440 bench frames (exposure channel); upscaler branch `perf/exposure-parallel` (`22468ae`…`4715027`, not pushed)
- [x] M2T.3 groundwork: FSR 3.1.6 luma and shading-change pyramids (AMD-exact and every-texel), upscaler passes verbatim, oracles, `npm run tools` (`f67a2a3`)
- [x] M2T.2: `ScalarReduction`: any size to one value per layer in 2 dispatches, every texel counted; load/finish hooks; 18/18 mutations caught (`50c6368`, `cb6ce7b`)
- [x] `encodePass()`: record the chain into a caller-owned compute pass, for @pmndrs/reflections (branch `reflections-integration`, `a996cc4`)
- [x] Tools survey: FSR 3.1.6 SPD passes (AMD source), upscaler, reflections, ssgiDev, Denoiser, webgpu-ddgi, Capsaicin → SPEC §14
- [x] DIVERGES.md: 21 divergences from SPD, nvpro and the WebGPU community, with effort and impact; readable copy at `docs/diverges.html` (published as an artifact)
- [x] M1: kernel, Downsampler, three adapter, tests, device lab, benchmark, CI (`f0f684c`)
- [x] M2.1: min/max/minmax/custom reductions, textureLoad path, explicit layouts (`9cf50f2`)
- [x] Test suite 8× faster: 341 s → 42 s (`dfc3fa5`)
- [x] M2.2: separate source (depth → Hi-Z), r32float/rg32float/rgba32float (`60ff7d3`)
- [x] Research: gpuweb #386/#5253, nvpro_pyramid, SPD manual, subgroups → SPEC §13 (`c843165`, `f2e70b8`)
- [x] Format plan probed (`190f809`); rgb9e5ufloat buffer path verified (`4229752`)
- [x] Plan update: PROGRESS.md, odd-size decision, build-first milestones (`31ff437`)
- [x] M2.3.2: rgba8snorm, r8/rg8snorm, 16-bit unorm/snorm; exact checks for in-tile integer levels; oracle tie fix (`619b60d`)
- [x] M2.3.1: bgra8unorm, r8unorm, rg8unorm, r16float, rg16float behind their features; skipped formats reported (`ab67d0f`)

## Decisions (dated; details in SPEC)

- 2026-11-11 · Primary users are our tools (upscaler, denoiser, GI); three.js mipmaps are a beneficiary, not the driver. Speed at render resolutions (non-power-of-two) is a first-class goal; single dispatch is not (WGSL forbids it). Plan reworked into M2T (§11)
- 2026-11-11 · Fold rule relaxed (owner): sums, counts and weighted means become reductions for the denoiser / GI / upscaler block statistics; idempotent folds keep their contract (§14)
- 2026-11-11 · `reflections-integration` (`encodePass`) merged into `m2-reductions`
- 2026-11-11 · SPD-style mode is deferred until M2T.1–3 clear, then built at least as a measured comparison
- 2026-11-10 · `'exact'` stays the default odd-size policy; `'box'` is added as an opt-in; SPD-style `'fast'` is dropped in favour of the tail kernel (§2.1)
- 2026-11-10 · Build features first, benchmark once coverage is broad (§11)
- 2026-11-10 · `'average'` keeps the sampler: loads measured 12–50 % slower for rgba8 (§4 Source access)
- 2026-11-10 · Subgroups ship in Chrome and Safari 27.3; any subgroup kernel keeps the workgroup-memory fallback for Firefox (§13)
- 2026-11-10 · rgb9e5ufloat is in scope through a buffer output path (§6.1)

## Waiting on owner

- Payload-fold ties (smaller payload vs position): examples sent 2026-11-12; awaiting a call
- Reducer defaults (`'zero-weight'` edges, no rounding between levels, scalar `finish` layer argument): impact explained 2026-11-12; awaiting a call
- Upscaler exposure, every texel vs 1024 taps: pros/cons and sources sent 2026-11-12. Agreed: merge the bit-identical `perf/exposure-parallel` first
- Review drafts (local only, gitignored, never pushed): `drafts/webgpu-spd-issues.md` (6 issues), `drafts/gpuweb-5253-comment.md` (numbers to fill from Windows)
- Upscaler branch `perf/exposure-parallel` is local only: push it to pmndrs/upscaler so the Windows agent can run its measurements, or skip that run
- Windows, sampler precision ([correctness.md](results/windows/correctness.md)): NVIDIA rounds each 2D tap weight to 1/256, Intel truncates axis weights (even sizes too); `averageAccess: 'load'` passes on both. Timed ([timings.md](results/windows/timings.md)): `'load'` costs 13 % / 2 % (RTX rgba8 / rgba16f), 31 % / 22 % (Intel). Default `'sample'`, `'load'`, or `'load'` only after odd sizes?
- Windows, f16 rounding cost: `'nearest'` costs 4 % on the RTX but 29 % on Intel (rgba16f, render sizes). Try a cheaper formulation (f32 magic-number add), keep as is, or default to `'device'` on some devices?
- Windows, RTX clocks: WebGPU runs sat at 620–920 MHz, native at ~1600. Lock clocks (`nvidia-smi -lgc`, needs admin) and rerun for fair WebGPU ÷ native ratios?

## Decided (owner, 2026-11-12)

- Upscaler uses the library through a local reference until npm; the upscaler releases after this is published
- Package: npm `downsampler` (unscoped; free on 2026-11-12), else `@pmndrs/downsampler`; GitHub `pmndrs/downsampler`
- Devices: RTX 3060 laptop + Intel UHD now (maybe more NVIDIA); wider coverage after publish
- `oddSizes: 'spd'` stays a public opt-in
- Tail kernel stays opt-in, documented
- Upscaler shading change keeps the 8-channel version (documented); the sign-bit version stays an experiment
- f16 outputs round to nearest even in the shader (D3D12 truncates in `textureStore` and `pack2x16float`), as a configurable default (branch `f16-rounding`)
- Order on Windows: native timings now; library timings after the f16 change, with `'sample'` vs `'load'` and f16 rounding on/off in the comparison
## Verification log

| Date | Commit | Device | Suite | Result |
|---|---|---|---|---|
| 2026-11-10 | `60ff7d3` | Apple M5 Pro, Chrome (puppeteer) | full GPU matrix | ✅ 2699 cases, 8448 pattern runs |
| 2026-11-10 | `60ff7d3` | SwiftShader | quick | ✅ 1225 cases |
| 2026-11-10 | `c843165` | Apple M5 Pro | quick | ✅ 2359 cases |
| 2026-11-10 | `ab67d0f` | Apple M5 Pro, Chrome 148 | full GPU matrix | ✅ 5579 cases, 18048 pattern runs |
| 2026-11-10 | `ab67d0f` | SwiftShader | quick | ✅ 1477 cases; skipped r8unorm, rg8unorm, r16float, rg16float (no tier1) |
| 2026-11-10 | `619b60d` | Apple M5 Pro, Chrome 148 | full GPU matrix, untrimmed (every format × every shape) | ✅ 10187 cases, 33024 pattern runs, 600 s |
| 2026-11-10 | `619b60d` | Apple M5 Pro, Chrome 148 | full GPU matrix (family-trimmed) | ✅ 6443 cases, 20752 pattern runs, 361 s |
| 2026-11-10 | `619b60d` | SwiftShader | quick | ✅ 1729 cases; 12 tier1 formats skipped |
| 2026-11-11 | `cb6ce7b` | Apple M5 Pro, Chrome 148 | full GPU matrix (agents running concurrently) | ✅ 6774 cases (330 scalar), 310 s |
| 2026-11-11 | `cb6ce7b` | SwiftShader | quick | ✅ 1660 cases |
| 2026-11-11 | `38d2bc1` | Apple M5 Pro, Chrome 148 | full GPU matrix (generator + scalar + reflections merged) | ✅ 8144 cases, 22112 pattern runs, 353 s |
| 2026-11-11 | `af98701` | Apple M5 Pro, Chrome 148 | full GPU matrix (agent) / compat device | ✅ 9326 cases / ✅ 8599 + 452 scalar |
| 2026-11-11 | merge of `m2t-api` | Apple M5 Pro, Chrome 148 | quick + `npm run tools --check` | ✅ 8591 cases; fixtures ok |
| 2026-11-12 | `b2ec515` | Apple M5 Pro, Chrome 148 | full GPU matrix (agent) / compat quick | ✅ 13217 cases, 572 s / ✅ 9449 cases |
| 2026-11-12 | `b2ec515` | SwiftShader | quick per part (parallel) | ✅ downsampler 1849 / 500 s, reducer 2243 / 991 s, scalar 319 / 86 s |
| 2026-11-12 | merge of `m2-gi-formats` | Apple M5 Pro, Chrome 148 | quick | ✅ 9538 cases |
| 2026-11-12 | `1b7e897` | Apple M5 Pro, Chrome 148 | full GPU matrix (agent) / compat quick | ✅ 18290 cases, 16 min / ✅ 10416 cases |
| 2026-11-12 | `1b7e897` | SwiftShader | quick per part (one at a time) | ✅ downsampler 2194 / 8.6 min, reducer 2243 / 16.9 min, scalar 319 / 1.4 min |
| 2026-11-12 | merge of `m2t-spd` | Apple M5 Pro, Chrome 148 | quick + core | ✅ 10445 cases; 67 core |
| 2026-11-10 | `a996cc4` | Apple M5 Pro, Chrome (puppeteer) | quick | ✅ 6032 cases, 19456 pattern runs (branch `reflections-integration`) |
| 2026-11-12 | `38beb25` | SwiftShader | quick, unsharded, `--record-durations` | ✅ 4784 cases, 323 groups, 25.4 min; shards 1, 7, 14 of 14 match their recorded case counts (21, 301, 366) |
| 2026-11-12 | `44b25bc` | Apple M5 Pro, Chrome 148 | quick (unsharded) + `--list --shards=14` | ✅ 10475 cases, 25548 pattern runs; partition disjoint and covering |
| 2026-11-12 | `b6aca67` + harness fixes | RTX 3060 Laptop, D3D12, driver 592.00, Chrome 148 | full GPU matrix, `test/parallel.js` (36 shards × 6) | ❌ 18289 cases, 1674/1674 groups, 17 min; 287 failures: f16 truncation, sampler precision |
| 2026-11-12 | `b6aca67` + harness fixes | RTX 3060 Laptop, D3D12 | quick / compat quick, parallel | ❌ 10444 cases, 2 min, 189 failures / ❌ 10355 cases, 2 min, 187 (same causes) |
| 2026-11-12 | `b6aca67` + harness fixes | Intel UHD, D3D12, driver 31.0.101.5333 | quick, parallel (12 × 4) | ❌ 10437 cases, 2.4 min, 629 failures (f16 truncation + Intel-only deviations) |

## Notes

- `textureStore` to rgba16float truncates instead of rounding on the M5 Pro (probe `results/probes/f16-store-rounding.mjs`); Downsampler quantizes first, the Reducer is being fixed to do the same.
- The dev machine is often loaded (load average 12–18 from other projects): only trust ratios from the same run, and don't update SPEC §8 absolute numbers until it's idle.
- Whole-suite correctness: `node test/parallel.js -- <run.js flags>` (shards in parallel, `--cap` per shard, `--budget` per run, one summary.md). The suite is CPU-bound (JS oracles), so one page uses one core: on the RTX 3060 laptop the full matrix is 35+ min alone, 17 min in 6 jobs; `--quick` 2 min. No run over 30 min (owner, 2026-11-12).
- The full matrix takes about 6 minutes under load (family-trimmed). The six family formats on 2048², 96×4096 and 1920×1080 dominate. If it grows further, run big shapes with fewer patterns or widths.
- In-tile levels of normalized formats are checked bit-exactly. Before that, a tie-rounding bug in the oracle (present since M1) was hidden by the 2-step tolerance.
- SwiftShader exposes 4 storage textures, so CI can't cover the 5- and 6-level kernels (CLAUDE.md).
