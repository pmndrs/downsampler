# DS3 pipeline warm-up measurements

Same-build A/B compares synchronous first encode with awaited `warmup(texture)` before first encode. Each pair uses separate private profiles for both arms, followed by a same-browser reload. OS/driver caches remain. Values below are medians in milliseconds; traces add overhead when enabled.

| Dataset/cache | Path | First recording sync → warmed | First submit-to-done sync → warmed | Warm-up wait | Second record-to-done sync → warmed | Compiler events sync / warmed | Compiler CPU ms sync → warmed |
|---|---|---:|---:|---:|---:|---:|---:|
| review-warmup/fresh-profile | storage | 1.7 → 0.6 | 2.8 → 1.0 | 2.9 | 0.6 → 0.8 | n/a | n/a |
| review-warmup/fresh-profile | buffer | 1.8 → 0.8 | 2.7 → 0.8 | 2.9 | 0.6 → 0.6 | n/a | n/a |
| review-warmup/fresh-profile | render | 1.4 → 0.6 | 2.4 → 1.3 | 2.3 | 0.7 → 0.9 | n/a | n/a |
| review-warmup/same-browser-reload | storage | 0.3 → 0.2 | 1.1 → 0.8 | 0.8 | 0.4 → 1.1 | n/a | n/a |
| review-warmup/same-browser-reload | buffer | 0.4 → 0.3 | 2.0 → 1.0 | 0.8 | 0.6 → 0.9 | n/a | n/a |
| review-warmup/same-browser-reload | render | 0.5 → 0.3 | 1.2 → 1.3 | 0.8 | 0.8 → 0.7 | n/a | n/a |

**review-warmup**: 36 page runs, 3 alternating pairs per path, 324 generated mip readbacks. Browser Chrome/148.0.7778.97; platform darwin; adapter {"vendor":"apple","architecture":"metal-3","device":"","description":""}. Provenance: immutable dist snapshot; SHA256 `e4917edffd6288e71b620817bb849c256fc214d3ff9c81b8c2db1aa154d0d7c8`; captured 2026-11-18T02:08:35.606Z.

First-use maximum rAF gap sync → warmed: 33.4 → 16.8 ms; preceding warm-up maximum 16.8 ms. First-use heartbeat sync → warmed: 5.5 → 4.9 ms; warm-up maximum 5.3 ms. First submit() call medians sync → warmed: 0.0 → 0.0 ms.

storage: 2 modules, 6947 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: none or tracing disabled. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

buffer: 2 modules, 7192 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: none or tracing disabled. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

render: 1 modules, 1945 WGSL bytes, 1 pipelines. API modes: sync: {"computeSync":0,"computeAsync":0,"renderSync":1,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":0,"renderSync":0,"renderAsync":1}. Observed compiler events: none or tracing disabled. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm-srgb"}.

Gaps above 50 ms, first-use: sync 0 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat; preparation: sync 0 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat.

Validation passed: constant-input oracle, identical per-path A/B mip hashes, zero pipeline calls at first encode after warm-up, and zero pipeline calls at second encode in either arm.

Cross-dataset build digests: 1. Compare timing deltas within an adapter. Different digests require reviewing implementation changes before comparing builds.

Compiler events count DXC/FXC complete events or begin events. CPU duration sums complete spans and reconstructed begin/end spans when available. No observed DXC/FXC event does not prove a cache hit: backend support and trace availability differ. Pipeline API calls are distinct from shader compiler invocations.

Warm-up moves compilation before first submission; it does not remove compilation or guarantee lower total startup time. Headless rAF describes browser scheduling, not physical presentation. The constant-input fixture is a narrow execution check; nonconstant GPU oracle tests remain necessary.

Reproduce with `npm run build`, then `node results/probes/ds3-warmup.mjs --repeats=3 --trace --output=PATH.json` and optional `--power=high|low`, `--adapter=swiftshader` or `--chrome=PATH`. Run adapters sequentially. Regenerate using `node results/probes/ds3-report.mjs PATH.json ... --output=SUMMARY.md`.

Skill feedback: none.
