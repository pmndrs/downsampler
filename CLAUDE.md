# downsampler — working notes for agents

Read SPEC.md first. It holds the scope, design decisions (odd sizes default to `'exact'`; no single-dispatch mode, and why), the SPD parity table, the correctness matrix, milestones and research notes (§13). Update it when a decision changes.

Then read PROGRESS.md, the live status board. Keep it current: move a task to *Now* when you start it; when you finish, tick it with its commit and log the verification run (device, suite, case count). Put questions for the owner under *Waiting on owner*, not only in chat.

DIVERGES.md lists every place we differ from SPD, nvpro and the WebGPU community, with the reason and the cost of converging. When a change adds, removes or resolves a divergence, update it, and `docs/differences.md` (the plain-language version for users) if users would notice.

## Commands

- `npm run build`: tsc to `dist/` (tests import from `dist/`, so build first).
- `npm run test:core`: backend-neutral unit tests (node:test).
- `npm run test:gpu`: full GPU suite in headless Chrome on the real adapter. Add `--quick` for small shapes, `--adapter=swiftshader` for software WebGPU, `--verbose` for progress, `--only=downsampler|reducer|scalar|fixtures` for one part (the four partition the suite; `fixtures` is the tool fixtures' oracle checks), `--compat` for a compatibility-mode device.
- Local whole-suite runs: `node test/parallel.js [--jobs=6] [--cap=10] [--budget=30] -- <run.js flags>` runs the shards in parallel headless Chromes with `--keep-going`, kills a shard past `--cap` minutes and the run past `--budget`, and writes one `summary.md`/`summary.json` (every failure) plus this device's `durations.json`; only `summary.md` is committed (`.gitignore`), and durations worth keeping are merged into one named file. Bench output (`--output`) is written as compact JSON. Use it rather than one long `test:gpu`: the suite is CPU-bound (JS oracles). On a laptop RTX 3060: `--quick` 9–11 min; the full matrix is about an hour, so run it in three parts: `node test/parallel.js --shards=48 --part=1/3 --cap=15 --budget=28 -- --power=high --durations=results/windows/durations-rtx3060-full.json` (then `2/3`, `3/3`; 18–21 min each). Never start a run that can exceed 30 minutes. On Intel UHD add `--isolate-devices` after the `--`: the groups that open a second device get shards of their own (the Intel driver loses the device otherwise).
- Shards (CI; test/ci/README.md): `node test/run.js --quick --adapter=swiftshader --shard=3/14` runs one of 14 disjoint shards of the case groups, balanced by `test/ci/durations.json`. `--list --shards=14` checks a partition without running anything (`--verbose` prints the ids). After adding or reshaping cases, re-record durations with `node test/run.js --quick --adapter=swiftshader --record-durations` (about 30 min) or from a CI run's `gpu-durations` artifact.
- `npm run bench`: validation + benchmark (`--sizes=1024x1024,512x512x6 --rounds=11 --repeats=16 --output=results/x.json`).
- `npm run lab`: device lab server.

## Rules

- `src/core` stays backend-neutral: no WebGPU types or globals.
- Zero runtime dependencies. `three` is an optional peer, typed structurally in `src/three`.
- Every kernel or planning change must pass the full GPU matrix (SPEC.md §7) on a real GPU, not just SwiftShader (which only exposes 4 storage textures, so it skips the 5- and 6-level kernels).
- New features add their own CPU oracle and GPU cases first.
- Performance claims need benchmark numbers from `npm run bench`, with the device named. Differences under ~5 % are noise.
- Style: tabs, spaces inside parentheses and brackets (three.js style).
