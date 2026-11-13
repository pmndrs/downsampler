# Windows handoff, round 2 (RTX 3060 Laptop + Intel UHD)

For a Claude Code agent on the owner's Windows laptop, after round 1 ([WINDOWS-REPORT.md](WINDOWS-REPORT.md)). Same machine, same rules, same tools (`test/parallel.js`, `--power=high|low`, the GPU lock, `run-benches.sh`).

Since round 1, `m2-reductions` gained library changes that were verified on the M5 Pro only:

| Change | Commit | Why it needs Windows |
|---|---|---|
| f16 rounding rewritten: branch-free integer rounding instead of `(x + c) − c` | `5f97c86` | The old trick was folded away by Metal's fast math (no rounding on Apple). The new one must still be bit-exact on D3D12, and its cost on Intel was 14 % before |
| `tail: true` by default; `exactTile` 16 on Apple, 32 elsewhere | `198728a` | The default chain on Windows is now the tail kernel |
| Partial updates (`rect`) and layer ranges (M2.4) | `f78847b` | New kernel variants (a region uniform at binding 11) |
| API consistency: `reduction` on ScalarReduction, `release()` destroys owned buffers, `requiredLimits()` | `9a49043` | Eviction now destroys buffers after `onSubmittedWorkDone` |
| `oddSizes: 'box'` and `source.resample: 'area'` (M2.5), merged with M2.4 | `bb16237`, `a39022b` | New box/area kernels; `rect` with resampled sources |
| M3 bench modes: throughput, storage-usage, subgroups, f16 | `5b137a3` | Cross-vendor answers are the point of M3 |

Read `CLAUDE.md`, `PROGRESS.md`, `docs/WINDOWS-REPORT.md` (your own round 1) and `test/bench/README.md` first.

## Rules

- `git fetch && git checkout -b windows-round2 origin/m2-reductions`. Commit results there and push only that branch. Never push `main` or `m2-reductions`.
- **Don't change `src/`.** If something fails, record the case, the full output, adapter and driver, then report. (Round 1 changed `src/` under owner decisions; this round only measures unless the owner says otherwise.)
- Never start a run that can exceed 30 minutes. Timed jobs run alone under the GPU lock; same-run ratios don't need locked clocks, WebGPU ÷ native ones do.
- Name every number (GPU, driver, Chrome, power state, load). Under ~5 % is noise. Raw JSON to `results/windows/round2/`, with a short `.md` per topic.

## 1. Correctness (first)

```sh
npm ci && npm run build && npm run test:core
node test/parallel.js --jobs=6 --cap=10 --budget=30 --out=results/windows/round2/rtx-full -- --power=high
node test/parallel.js --jobs=6 --cap=10 --budget=30 --out=results/windows/round2/intel-quick -- --quick --power=low
node test/parallel.js --jobs=6 --cap=10 --budget=30 --out=results/windows/round2/rtx-compat-quick -- --quick --compat --power=high
node test/examples.js --power=high && node test/examples.js --power=low
```

Expected: no failures except, as in round 1, the two FSR luma-pyramid fixtures that store f16 themselves. Anything in `region/*`, `box`, the f16 codec checks, or the tail path is new and is the main thing to report. If the Intel full matrix fits (round 1 lost the device after 580 s), try it in two halves with `--only=downsampler` and `--only=reducer,scalar,fixtures`.

## 2. Defaults: confirm or correct them

```sh
npm run bench -- --compare=render --skip-validation --power=high --output=results/windows/round2/rtx-render.json
npm run bench -- --compare=render --skip-validation --power=low  --output=results/windows/round2/intel-render.json
```

Also time `'box'`: `npm run bench -- --compare=spd --strategies=box,tailBox --skip-validation --power=high|low` (the reference column is kept).

The render compare now has `exact8`, `exact16`, `tail`, `tail8`, `tail16` columns (`exact` is the device default tile, 32 here). Report, at the render sizes:
- `exact16 ÷ exact` and `tail16 ÷ tail`: if 16 ties 32 on both GPUs (within 5 %), the default can be 16 everywhere, which is simpler.
- `tail ÷ exact` with the new f16 rounding: does the default still win on the RTX and stay neutral on Intel?

## 3. f16 rounding cost (owner question)

```sh
npm run bench -- --compare=spd --skip-validation --power=low  --output=results/windows/round2/intel-spd.json
npm run bench -- --compare=spd --skip-validation --power=high --output=results/windows/round2/rtx-spd.json
```

`'nearest'` ÷ `'device'` per format. Round 1: 14 % on Intel with the add trick (29 % with the first integer version), noise on the RTX. The owner decides from this whether any device should default to `'device'`.

## 4. M3 measurements

Commands are in `test/bench/README.md`; add `--power=high` / `--power=low`. In order:
1. `--compare=throughput`, then `--compare=throughput --pacing=raf --no-timestamps` (greggman's method; does it order strategies like the timestamps?)
2. `--compare=storage-usage` (Kangz: does `STORAGE_BINDING` slow sampling?)
3. `--compare=subgroups` (the `nosync` column bounds any subgroup kernel; ≥ 5 % at render sizes is the bar. On Intel `shuffle` may report "differ": smaller subgroups, timing only)
4. `--compare=f16` (f16 arithmetic: speed vs the precision columns)

## 5. Left over from round 1

`sh results/windows/run-benches.sh rtx3060-bench`, then `-tools`, then the `intel-uhd-…` jobs (`npm run bench` and `npm run tools` per GPU).

## Report

Add a "Round 2" section to `docs/WINDOWS-REPORT.md` (answers first, then method and data links), update `PROGRESS.md` (*Now*, the verification log, *Waiting on owner*), commit and push `windows-round2`.
