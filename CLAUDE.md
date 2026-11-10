# downsampler — working notes for agents

Read SPEC.md first. It holds the scope, design decisions (odd sizes default to `'exact'`; no single-dispatch mode, and why), the SPD parity table, the correctness matrix and milestones. Update it when a decision changes.

## Commands

- `npm run build`: tsc to `dist/` (tests import from `dist/`, so build first).
- `npm run test:core`: backend-neutral unit tests (node:test).
- `npm run test:gpu`: full GPU suite in headless Chrome on the real adapter. Add `--quick` for small shapes, `--adapter=swiftshader` for software WebGPU, `--verbose` for progress.
- `npm run bench`: validation + benchmark (`--sizes=1024x1024,512x512x6 --rounds=11 --repeats=16 --output=results/x.json`).
- `npm run lab`: device lab server.

## Rules

- `src/core` stays backend-neutral: no WebGPU types or globals.
- Zero runtime dependencies. `three` is an optional peer, typed structurally in `src/three`.
- Every kernel or planning change must pass the full GPU matrix (SPEC.md §7) on a real GPU, not just SwiftShader (which only exposes 4 storage textures, so it skips the 5- and 6-level kernels).
- New features add their own CPU oracle and GPU cases first.
- Performance claims need benchmark numbers from `npm run bench`, with the device named. Differences under ~5 % are noise.
- Style: tabs, spaces inside parentheses and brackets (three.js style).
