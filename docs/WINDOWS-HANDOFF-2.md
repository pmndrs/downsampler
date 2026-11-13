# Windows handoff, round 2: verify, then own M3 (RTX 3060 Laptop + Intel UHD)

For the Claude Code agent on the owner's Windows laptop, after round 1 ([WINDOWS-REPORT.md](WINDOWS-REPORT.md)). Same machine and tools (`test/parallel.js`, `--power=high|low`, the GPU lock, `run-benches.sh`).

**What changed in scope (owner, 2026-11-13):** M3 ("dial in": measurement and tuning) now happens on this machine, including library changes. The Mac works on three.js, the remaining formats, docs and ship prep at the same time. The goal is to finish the remaining phases today, so keep moving: measure, decide by the rules below, build, verify, push, next.

## What's new since round 1 (verified on the M5 Pro only)

| Change | Commit | Why it needs Windows |
|---|---|---|
| f16 rounding rewritten: branch-free integer rounding instead of `(x + c) − c` | `5f97c86` | The add trick was folded away by Metal's fast math (no rounding on Apple). The new one must be bit-exact on D3D12; its Intel cost is unknown (the add trick cost 14 %) |
| `tail: true` by default; `exactTile` 16 on Apple, 32 elsewhere | `198728a` | The default chain is now the tail kernel |
| Partial updates (`rect`) and layer ranges (M2.4) | `f78847b` | New kernel variants (region uniform at binding 11) |
| API consistency (`reduction` on ScalarReduction, `release()` destroys owned buffers, `requiredLimits()`) | `9a49043` | Buffers destroyed after `onSubmittedWorkDone` |
| `oddSizes: 'box'` and `source.resample: 'area'` (M2.5) | `bb16237`, `a39022b` | New box/area kernels |
| M3 bench modes: throughput, storage-usage, subgroups, f16; `--strategies` | `5b137a3` | Cross-vendor answers are the point of M3 |
| README, docs/guide.md, docs/spd-users.md, examples/ (`npm run examples`, `node test/examples.js`) | M4 | Examples should run on both GPUs |

Read `CLAUDE.md`, `PROGRESS.md` (owner decisions of 2026-11-13), SPEC §2.1, §4, §8, §13 (halo batches, subgroups), `test/bench/README.md` and your round-1 report first.

## Rules

- Work on **`windows-testing`** (this branch; it equals `m2-reductions` plus this file). Commit small, push `windows-testing` often. Never push `main` or `m2-reductions`.
- **`src/` changes are allowed for M3**, under CLAUDE.md: new features add their CPU oracle and GPU cases first; every kernel or planning change passes the RTX full matrix and the Intel quick suite (`test/parallel.js`) before you push it; performance claims come with bench numbers and the device named; differences under ~5 % are noise. The Mac re-runs its own full matrix when it merges your branch (Metal compiles WGSL with fast math: avoid floating-point tricks that rely on exact `a + b − b` style evaluation).
- **Avoid conflicts:** the Mac is changing `src/core/formats.ts`, the format parts of `src/webgpu/kernel.ts`, `src/three/`, the `source` option shapes and the docs. Keep your kernel edits to the exact/tail/box kernel bodies and `src/core/plan.ts` where you can. Run `git fetch && git merge origin/m2-reductions` before each new task.
- Never start a run that can exceed 30 minutes. Timed jobs run alone under the GPU lock. Raw JSON to `results/windows/round2/`, a short `.md` per topic.
- For each task, add a line to `PROGRESS.md` (*Now* while working, *Done* with the commit and verification row when finished). Questions for the owner go under *Waiting on owner*.

## 1. Correctness of the merged work (first, ~30 min)

```sh
git fetch && git checkout windows-testing && git pull
npm ci && npm run build && npm run test:core
node test/parallel.js --jobs=6 --cap=10 --budget=30 --out=results/windows/round2/rtx-full -- --power=high
node test/parallel.js --jobs=6 --cap=10 --budget=30 --out=results/windows/round2/intel-quick -- --quick --power=low
node test/parallel.js --jobs=6 --cap=10 --budget=30 --out=results/windows/round2/rtx-compat-quick -- --quick --compat --power=high
node test/examples.js --power=high && node test/examples.js --power=low
```

Expected: no failures except, as in round 1, the two FSR luma-pyramid fixtures that store f16 themselves. Failures in `region/*`, `box/*`, the f16 codec checks or the tail path are new: fix them if the cause is clear (with a test that fails before), otherwise report them with the case, output, adapter and driver.

## 2. Defaults and the f16 cost (~20 min)

```sh
npm run bench -- --compare=render --skip-validation --power=high --output=results/windows/round2/rtx-render.json
npm run bench -- --compare=render --skip-validation --power=low  --output=results/windows/round2/intel-render.json
npm run bench -- --compare=spd --skip-validation --power=high --output=results/windows/round2/rtx-spd.json
npm run bench -- --compare=spd --skip-validation --power=low  --output=results/windows/round2/intel-spd.json
```

Decide by these rules, and make the change if a rule says so:
- **Tile:** if `exact16 ÷ exact` and `tail16 ÷ tail` are within 5 % on both GPUs at the render sizes, make 16 the default everywhere (drop the Apple vendor check in `Downsampler`).
- **Tail default:** stays on unless it now loses ≥ 5 % on either GPU.
- **f16 rounding** (owner: keep `'nearest'` everywhere, document `'device'`): report `'nearest'` ÷ `'device'`. If Intel pays more than ~10 %, try a cheaper rounding that stays bit-exact against core `toHalf` and survives fast math; the codec check in `validate-packed.js` is the test.
- **`'box'`** (`box`, `tailBox` columns): report its cost against `'exact'`; nothing to decide.

## 3. M3 measurements (~40 min, both GPUs)

Commands are in `test/bench/README.md` (add `--power=high` / `--power=low`):
1. `--compare=throughput`, then `--compare=throughput --pacing=raf --no-timestamps`: does work-per-frame order the strategies like the timestamps?
2. `--compare=storage-usage`: does `STORAGE_BINDING` usage slow sampling (Kangz, gpuweb#5253)?
3. `--compare=subgroups`: `nosync` bounds any subgroup kernel. The bar is ≥ 5 % at render sizes. (Intel's `shuffle` may report "differ": smaller subgroups, timing only.)
4. `--compare=f16`: f16 arithmetic speed against its precision columns.

## 4. M3 builds, in priority order (decide each from the data)

1. **Halo batches for `'exact'` and `'box'`** (SPEC §13 item 3; DIVERGES #2). Carry an odd step inside a batch with nvpro's halo (each tile reads its neighbours' border texels; 1, 3, 7 … texels per extra level), so render sizes stop splitting a dispatch at every odd level. Results must stay bit-identical to today's `'exact'` and `'box'` (the existing oracles cover them); add plan tests for dispatch counts and GPU cases that compare against the current kernels. The upper bound is `'spd'` (0.75 of render passes in round 1). Ship it as the default only if it wins ≥ 5 % at render sizes on both GPUs; otherwise keep it as an option and report.
2. **A cheaper exact kernel**: at 1024², where both take 2 dispatches, `'spd'` was 0.81× of `'exact'` on the M5 (integer tie rounding, odd-footprint branches). Profile and trim; same bits.
3. **Subgroup kernel** only if step 3's `nosync` bound is ≥ 5 % on the RTX. It needs the workgroup-memory fallback (Firefox and devices without subgroups) and a per-device choice (Intel was 2× slower with subgroups in round 1).
4. **fp16 arithmetic** (`shader-f16`) only if `--compare=f16` shows ≥ 5 % with `f16safe` precision acceptable; opt-in, never default.
5. **Leftover round-1 jobs:** `sh results/windows/run-benches.sh rtx3060-bench`, then `-tools`, then the `intel-uhd-…` jobs.

## Report

Add a "Round 2" section to `docs/WINDOWS-REPORT.md` (answers first, then method and data links), keep `PROGRESS.md` current, and push `windows-testing` after each finished task, so the Mac can merge as you go.
