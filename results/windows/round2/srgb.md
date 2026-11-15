# sRGB lookup tables on D3D12 (round 2, 2026-11-14)

RTX 3060 Laptop (nvidia ampere, driver 592.00), Chrome 148 headless, D3D12, `--power=high`, on AC, Turbo plan.
GPU idle apart from the desktop (Cursor, Chrome, Camera Hub open; 0 % utilisation, 62 °C at the start).

## What happened

The round-2 RTX full matrix (`test/parallel.js`, 18 shards, 10 min cap) finished no shard. The sRGB groups were
the cause: on the committed code a 2×2 `rgba8unorm-srgb` group took 71 s and a 16×16 one 87 s
(`probe-srgb.log`), where other formats take a second or two.

`srgbWgsl` declared its two exact tables (256-entry decode, 255-entry code bounds) as module-scope `const`
arrays, indexed at run time. On D3D12 that costs at compile time and at run time.

## Compile time per pipeline (`srgb-compile-time-rtx3060.json`, `results/probes/srgb-compile-time.mjs`)

`createComputePipelineAsync`, a unique constant per compile so no cache hits.

| Kernel | rgba8unorm | sRGB, `const` tables | `var<private>` | storage buffer |
|---|---:|---:|---:|---:|
| 1 level | 32 ms | 551 ms | 54 ms | 58 ms |
| 1 level, tail | 46 | 2272 | 172 | 205 |
| 4 levels | 74 | 2334 | 140 | 162 |
| 4 levels, tail | 84 | 3996 | 248 | 317 |
| 6 levels | 105 | 4949 | 311 | 397 |
| 6 levels, tail | 125 | 5338 | 492 | 869 |

## Run time (`--compare=srgb`, rgba8unorm-srgb, medians of 11 rounds × 16 repeats; ms)

| Size | `const` (shipped) | `var<private>` | no lookups (wrong bits, timing only) | storage buffer (now) | render passes |
|---|---:|---:|---:|---:|---:|
| 1920×1080 | 0.901 | 0.902 | 0.169 | **0.239** | 0.079 |
| 2560×1440 | 1.512 | 1.512 | — | **0.352** | 0.116–0.129 |
| 4096² | 5.205 | 5.203 | 0.881 | **1.311** | 0.381–0.390 |
| 1280×720 | 0.480 | 0.483 | — | **0.145** | 0.055 |

Files: `rtx-srgb-const.json`, `rtx-srgb-private.json`, `rtx-srgb-table-buffer.json` (the no-lookup run wasn't saved).

- `var<private>` fixes compiling (11–16× faster) but not running: a private array indexed at run time lives in
  per-invocation local memory on NVIDIA either way.
- A read-only storage buffer fixes both: 3.8–4× faster chains, 8–19× faster compiles than `const`.
- **Render passes stay 2.6–3.3× faster** than our sRGB path on this GPU (they were 0.55–1.25× ours on the M5 Pro).
  three.js should keep render-pass mipmaps for sRGB textures on NVIDIA.

## The change

`srgbWgsl` reads both tables from `@group(0) @binding(12) var<storage, read> ds_srgb_table: array<f32, 512>`
(decode at 0, bounds at 256). `srgbTable( device )` creates and fills it once per device (2 KiB);
`usesSrgbTable( options )` says whether a kernel needs it; `kernelLayoutEntries` declares it; `Downsampler` and
`Reducer` bind it. Same table values, so the same bits.

## Verification

- `npm run test:core`: 120/120.
- `--only=srgb` (full shapes, RTX): every group that ran passed; the run was stopped at 20 min with the 2048² and
  1080p groups of both formats done (`srgb-only-rtx.log`). Those take 130–290 s each: JS oracle time.
- `--only=srgb --quick` (RTX): 16 failing checks, all `rgba8unorm-srgb → rgba16float` separate sources at half size,
  sampled: errors 0.0029297 against a bound of 0.0029253 (1.5 % over). Those kernels don't call the table
  functions (the change only adds an unused declaration to them), so they predate it: the RTX's filtered sRGB
  decode is slightly outside the oracle's sampler tolerance. Open item.
- The first quick run found a real bug in the change: `Downsampler` cached bind group layouts without the source
  format, so an sRGB source and a linear one shared a layout. The key now includes the table.
