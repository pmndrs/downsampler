# Windows round 3: verify `renderPathMinTexels` and re-time `colorSpace`

Branch `srgb-store`. The M5 Pro pass is done ([results/m5/srgb-m5.md](../results/m5/srgb-m5.md)); this is what only the
Windows machines can do before the branch merges. Read CLAUDE.md, then PROGRESS.md (*Now*, *Decided (owner, 2026-11-17)*).

## What changed since round 2

- **The bench timing markers now bracket the texture a strategy writes** (`target` in `test/bench/srgb.js`, used by
  `test/bench/bench.js`). `colorSpace` (and the `alias` prototype and `srgb-cost`'s linear strategies) write an aliased
  twin, which the markers didn't depend on; on Metal the end marker ran early and `colorSpace` timed at 0 ms. D3D12
  probably ordered the passes anyway, so the round-2 Windows `colorSpace` numbers are likely right, but unconfirmed.
  The `colorSpace` strategy now asks for `output: 'storage'` (its twins gained `RENDER_ATTACHMENT` for the markers).
- **`renderPathMinTexels`** (owner, 2026-11-17; SPEC §6.1, DIVERGES #16): under `output: 'auto'`, sRGB chains take the
  render path only when their first level (base level, or a partial update's `rect`) has at least that many texels per
  layer, default 1920 × 1080; smaller ones take the buffer path (`colorSpace: 'srgb'` its storage path) unless the
  texture can only render. On the M5 the render path lost to the buffer path below 1080p (1.25–2.18×); here it wins at
  every size, so below 1080p the RTX and Intel give up that gain (`renderPathMinTexels: 0` restores it).
- `srgb/render` is pinned to `renderPathMinTexels: 0`, and fails instead of skipping when the render path doesn't apply.
  `srgb/render-api` has the threshold cases (boundary 1920×1079 / 1080, sub-range, rect, render-only, `colorSpace`,
  both paths against the oracle).

## Run

1. Setup: `git fetch && git checkout srgb-store && git pull`, then `npm ci && npm run build && npm run test:core` (129).
2. The changed groups first, on the RTX (`--power=high`) and Intel (`--power=low`):
   ```bash
   node test/run.js --groups=srgb/render-api,srgb/render,srgb/alias,downsampler/options --keep-going --power=high
   ```
   Note `srgb/render-api`'s `thresholdNearTies` / `thresholdDiffering` (`--output=<file>.json`); the M5 had 88 near-ties
   and 2 values a code apart between the paths at 1920 × 1080, all near-ties.
3. **RTX full matrix** (required before merging: this changes planning), in three parts, as in CLAUDE.md:
   ```bash
   node test/parallel.js --shards=48 --part=1/3 --cap=15 --budget=28 --out=results/windows/round3/threshold-rtx-full-p1 -- --power=high --durations=results/windows/durations-rtx3060-full.json
   ```
   then `2/3` and `3/3` (`-p2`, `-p3`). Expected failures: the FSR fixtures only. Then Intel quick and RTX compat quick
   (`-- --quick --power=low`, `-- --quick --compat --power=high`).
4. **Re-time `colorSpace`** with the fixed markers (VALID under the GPU lock as in round 2; Intel paired per round):
   ```bash
   npm run bench -- --compare=srgb --skip-validation --strategies=render,buffer,colorSpace,renderPath,renderDevice --rounds=11 --repeats=32 --output=results/windows/round3/rtx3060-srgb-renderpath.json
   ```
   Compare `colorSpace` with round 2 ([rtx3060-srgb-renderpath.json](../results/windows/round2/rtx3060-srgb-renderpath.json):
   0.73–0.83× of the buffer path). If it moved by more than ~5 %, update DIVERGES #16, the guide's `colorSpace` paragraph
   and `docs/differences.md`.

## Record

PROGRESS: tick the `renderPathMinTexels` and `colorSpace` re-run tasks, add verification rows (device, suite, case
count, link). For suite runs commit only `summary.md`. American English, no attribution lines, commit and push only
when the owner asks.
