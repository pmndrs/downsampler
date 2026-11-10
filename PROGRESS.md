# Progress

Live status board. SPEC.md holds the design and the reasons behind it; this file holds where things stand.

**For agents:** read this after SPEC.md. When you start a task, move it to *Now* with your branch. When you finish, tick it, add the commit, and log the verification run. Keep entries to one line and link SPEC sections instead of restating them. Anything that needs the owner goes under *Waiting on owner*.

Last updated: 2026-11-10 · Branch: `m2-reductions`

## Now

- [ ] M2.3.3: rgb10a2unorm (SPEC §6.1)

## Next (M2, features: SPEC §11)

- [ ] M2.3.4: buffer output path + rgb9e5ufloat (probe: `results/probes/rgb9e5-buffer-copy.mjs`)
- [ ] M2.3.5: sRGB + load/store transform hooks (scratch texture vs buffer output, by measurement)
- [ ] M2.3.6: rg11b10ufloat
- [ ] M2.3.7: integer formats (folds first)
- [ ] M2.4: layer range + rect
- [ ] M2.5: `oddSizes: 'box'`, then the exact tail kernel (SPEC §2.1, §13)

## Later

- **M3, dial in:** per-frame throughput benchmark; cross-vendor device lab runs (NVIDIA/D3D12 first); `STORAGE_BINDING` sampling cost; float32 sampler vs load; subgroup kernel with fallback (upper-bound test first); fp16
- **M4, examples and docs:** usage docs, notes for SPD users, examples (mip chain, Hi-Z from depth, custom reduction, three.js)
- **M5, ship:** three.js in-tree PR, npm publish, the upscaler as a consumer, the gpuweb#5253 comment

## Done

- [x] M1: kernel, Downsampler, three adapter, tests, device lab, benchmark, CI (`f0f684c`)
- [x] M2.1: min/max/minmax/custom reductions, textureLoad path, explicit layouts (`9cf50f2`)
- [x] Test suite 8× faster: 341 s → 42 s (`dfc3fa5`)
- [x] M2.2: separate source (depth → Hi-Z), r32float/rg32float/rgba32float (`60ff7d3`)
- [x] Research: gpuweb #386/#5253, nvpro_pyramid, SPD manual, subgroups → SPEC §13 (`c843165`, `f2e70b8`)
- [x] Format plan probed (`190f809`); rgb9e5ufloat buffer path verified (`4229752`)
- [x] Plan update: PROGRESS.md, odd-size decision, build-first milestones (`31ff437`)
- [x] M2.3.2: rgba8snorm, r8/rg8snorm, 16-bit unorm/snorm; exact checks for in-tile integer levels; oracle tie fix (M2.3.2)
- [x] M2.3.1: bgra8unorm, r8unorm, rg8unorm, r16float, rg16float behind their features; skipped formats reported (`ab67d0f`)

## Decisions (dated; details in SPEC)

- 2026-11-10 · `'exact'` stays the default odd-size policy; `'box'` is added as an opt-in; SPD-style `'fast'` is dropped in favour of the tail kernel (§2.1)
- 2026-11-10 · Build features first, benchmark once coverage is broad (§11)
- 2026-11-10 · `'average'` keeps the sampler: loads measured 12–50 % slower for rgba8 (§4 Source access)
- 2026-11-10 · Subgroups ship in Chrome and Safari 27.3; any subgroup kernel keeps the workgroup-memory fallback for Firefox (§13)
- 2026-11-10 · rgb9e5ufloat is in scope through a buffer output path (§6.1)

## Waiting on owner

- Cross-vendor hardware for M3: which NVIDIA/AMD/Intel (Windows) and Android devices are available?
- Package name: `downsampler` or `@pmndrs/downsampler` (SPEC §12)
- Review of the gpuweb#5253 comment draft before anything is posted (M5)

## Verification log

| Date | Commit | Device | Suite | Result |
|---|---|---|---|---|
| 2026-11-10 | `60ff7d3` | Apple M5 Pro, Chrome (puppeteer) | full GPU matrix | ✅ 2699 cases, 8448 pattern runs |
| 2026-11-10 | `60ff7d3` | SwiftShader | quick | ✅ 1225 cases |
| 2026-11-10 | `c843165` | Apple M5 Pro | quick | ✅ 2359 cases |
| 2026-11-10 | `ab67d0f` | Apple M5 Pro, Chrome 148 | full GPU matrix | ✅ 5579 cases, 18048 pattern runs |
| 2026-11-10 | `ab67d0f` | SwiftShader | quick | ✅ 1477 cases; skipped r8unorm, rg8unorm, r16float, rg16float (no tier1) |
| 2026-11-10 | M2.3.2 | Apple M5 Pro, Chrome 148 | full GPU matrix, untrimmed (every format × every shape) | ✅ 10187 cases, 33024 pattern runs, 600 s |
| 2026-11-10 | M2.3.2 | Apple M5 Pro, Chrome 148 | full GPU matrix (family-trimmed) | ✅ 6443 cases, 20752 pattern runs, 361 s |
| 2026-11-10 | M2.3.2 | SwiftShader | quick | ✅ 1729 cases; 12 tier1 formats skipped |

## Notes

- The dev machine is often loaded (load average 12–18 from other projects): only trust ratios from the same run, and don't update SPEC §8 absolute numbers until it's idle.
- The full matrix takes about 6 minutes under load (family-trimmed). The six family formats on 2048², 96×4096 and 1920×1080 dominate. If it grows further, run big shapes with fewer patterns or widths.
- In-tile levels of normalized formats are checked bit-exactly. Before that, a tie-rounding bug in the oracle (present since M1) was hidden by the 2-step tolerance.
- SwiftShader exposes 4 storage textures, so CI can't cover the 5- and 6-level kernels (CLAUDE.md).
