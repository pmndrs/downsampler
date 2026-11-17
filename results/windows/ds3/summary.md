# DS3 pipeline warm-up measurements

Same-build A/B compares synchronous first encode with awaited `warmup(texture)` before first encode. Each pair uses separate private profiles for both arms, followed by a same-browser reload. OS/driver caches remain. Values below are medians in milliseconds; traces add overhead when enabled.

| Dataset/cache | Path | First recording sync → warmed | First submit-to-done sync → warmed | Warm-up wait | Second record-to-done sync → warmed | Compiler events sync / warmed | Compiler CPU ms sync → warmed |
|---|---|---:|---:|---:|---:|---:|---:|
| nvidia/fresh-profile | storage | 3.1 → 1.0 | 128.0 → 3.2 | 70.5 | 1.5 → 1.7 | 2.0 → 2.0 | 64.1 → 67.1 |
| nvidia/fresh-profile | buffer | 3.7 → 1.2 | 174.5 → 10.6 | 120.4 | 1.5 → 1.4 | 2.0 → 2.0 | 93.4 → 148.7 |
| nvidia/fresh-profile | render | 2.5 → 1.2 | 48.9 → 2.6 | 42.8 | 1.4 → 1.7 | 2.0 → 2.0 | 28.5 → 23.4 |
| nvidia/same-browser-reload | storage | 0.6 → 0.3 | 18.3 → 2.5 | 10.3 | 1.5 → 1.3 | 0.0 → 0.0 | 0.0 → 0.0 |
| nvidia/same-browser-reload | buffer | 0.8 → 0.5 | 27.1 → 10.3 | 10.2 | 1.5 → 1.4 | 0.0 → 0.0 | 0.0 → 0.0 |
| nvidia/same-browser-reload | render | 0.6 → 0.6 | 4.9 → 3.0 | 4.0 | 1.4 → 1.5 | 0.0 → 0.0 | 0.0 → 0.0 |
| intel/fresh-profile | storage | 2.8 → 0.8 | 128.1 → 4.1 | 73.7 | 1.0 → 3.2 | 2.0 → 2.0 | 63.6 → 69.7 |
| intel/fresh-profile | buffer | 3.7 → 1.2 | 173.5 → 4.0 | 93.1 | 4.0 → 3.7 | 2.0 → 2.0 | 91.5 → 95.5 |
| intel/fresh-profile | render | 2.8 → 1.3 | 42.3 → 4.7 | 41.5 | 4.1 → 4.2 | 2.0 → 2.0 | 20.1 → 22.0 |
| intel/same-browser-reload | storage | 0.8 → 0.3 | 4.5 → 3.9 | 1.7 | 3.1 → 3.2 | 0.0 → 0.0 | 0.0 → 0.0 |
| intel/same-browser-reload | buffer | 0.8 → 0.4 | 4.8 → 3.9 | 1.9 | 4.2 → 4.2 | 0.0 → 0.0 | 0.0 → 0.0 |
| intel/same-browser-reload | render | 0.5 → 0.3 | 5.6 → 4.7 | 1.7 | 3.8 → 4.2 | 0.0 → 0.0 | 0.0 → 0.0 |
| swiftshader/fresh-profile | storage | 3.2 → 0.8 | 414.8 → 19.2 | 365.4 | 8.2 → 10.2 | 0.0 → 0.0 | 0.0 → 0.0 |
| swiftshader/fresh-profile | buffer | 3.4 → 1.2 | 488.9 → 30.8 | 480.2 | 18.7 → 17.7 | 0.0 → 0.0 | 0.0 → 0.0 |
| swiftshader/fresh-profile | render | 2.9 → 1.4 | 58.2 → 44.7 | 13.6 | 3.1 → 3.1 | 0.0 → 0.0 | 0.0 → 0.0 |
| swiftshader/same-browser-reload | storage | 0.6 → 0.4 | 362.9 → 19.9 | 340.3 | 8.4 → 9.2 | 0.0 → 0.0 | 0.0 → 0.0 |
| swiftshader/same-browser-reload | buffer | 0.8 → 0.4 | 454.8 → 31.5 | 422.3 | 18.8 → 16.3 | 0.0 → 0.0 | 0.0 → 0.0 |
| swiftshader/same-browser-reload | render | 0.9 → 0.4 | 58.0 → 45.8 | 8.6 | 3.2 → 3.1 | 0.0 → 0.0 | 0.0 → 0.0 |

**nvidia**: 36 page runs, 3 alternating pairs per path, 324 generated mip readbacks. Browser Chrome/157.0.8103.79; platform win32; adapter {"vendor":"nvidia","architecture":"ampere","device":"","description":""}. Provenance: immutable dist snapshot; SHA256 `98442b581658c0606ce60581dacf4c7aa812bd667e75ca6b0d18558a004239de`; captured 2026-11-17T12:17:55.636Z.

First-use maximum rAF gap sync → warmed: 16.9 → 16.9 ms; preceding warm-up maximum 16.9 ms. First-use heartbeat sync → warmed: 16.5 → 5.6 ms; warm-up maximum 6.5 ms. First submit() call medians sync → warmed: 0.0 → 0.1 ms.

storage: 2 modules, 8457 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: CompileShaderDXC. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

buffer: 2 modules, 8618 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: CompileShaderDXC. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

render: 1 modules, 1945 WGSL bytes, 1 pipelines. API modes: sync: {"computeSync":0,"computeAsync":0,"renderSync":1,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":0,"renderSync":0,"renderAsync":1}. Observed compiler events: CompileShaderDXC. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm-srgb"}.

Gaps above 50 ms, first-use: sync 0 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat; preparation: sync 0 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat.

Validation passed: constant-input oracle, identical per-path A/B mip hashes, zero pipeline calls at first encode after warm-up, and zero pipeline calls at second encode in either arm.

**intel**: 36 page runs, 3 alternating pairs per path, 324 generated mip readbacks. Browser Chrome/157.0.8103.79; platform win32; adapter {"vendor":"intel","architecture":"gen-12lp","device":"","description":""}. Provenance: immutable dist snapshot; SHA256 `98442b581658c0606ce60581dacf4c7aa812bd667e75ca6b0d18558a004239de`; captured 2026-11-17T12:18:32.183Z.

First-use maximum rAF gap sync → warmed: 150.2 → 16.9 ms; preceding warm-up maximum 16.9 ms. First-use heartbeat sync → warmed: 6.0 → 5.6 ms; warm-up maximum 5.9 ms. First submit() call medians sync → warmed: 0.0 → 0.0 ms.

storage: 2 modules, 8457 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: CompileShaderDXC. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

buffer: 2 modules, 8618 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: CompileShaderDXC. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

render: 1 modules, 1945 WGSL bytes, 1 pipelines. API modes: sync: {"computeSync":0,"computeAsync":0,"renderSync":1,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":0,"renderSync":0,"renderAsync":1}. Observed compiler events: CompileShaderDXC. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm-srgb"}.

Gaps above 50 ms, first-use: sync 6 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat; preparation: sync 0 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat.

Validation passed: constant-input oracle, identical per-path A/B mip hashes, zero pipeline calls at first encode after warm-up, and zero pipeline calls at second encode in either arm.

**swiftshader**: 36 page runs, 3 alternating pairs per path, 324 generated mip readbacks. Browser Chrome/157.0.8103.79; platform win32; adapter {"vendor":"google","architecture":"swiftshader","device":"","description":""}. Provenance: immutable dist snapshot; SHA256 `98442b581658c0606ce60581dacf4c7aa812bd667e75ca6b0d18558a004239de`; captured 2026-11-17T12:19:26.463Z.

First-use maximum rAF gap sync → warmed: 449.9 → 23.1 ms; preceding warm-up maximum 16.9 ms. First-use heartbeat sync → warmed: 15.7 → 10.7 ms; warm-up maximum 6.0 ms. First submit() call medians sync → warmed: 0.0 → 0.1 ms.

storage: 3 modules, 7005 WGSL bytes, 3 pipelines. API modes: sync: {"computeSync":3,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":3,"renderSync":0,"renderAsync":0}. Observed compiler events: none or tracing disabled. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

buffer: 2 modules, 7171 WGSL bytes, 2 pipelines. API modes: sync: {"computeSync":2,"computeAsync":0,"renderSync":0,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":2,"renderSync":0,"renderAsync":0}. Observed compiler events: none or tracing disabled. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm"}.

render: 1 modules, 1945 WGSL bytes, 1 pipelines. API modes: sync: {"computeSync":0,"computeAsync":0,"renderSync":1,"renderAsync":0}; warmup: {"computeSync":0,"computeAsync":0,"renderSync":0,"renderAsync":1}. Observed compiler events: none or tracing disabled. Fixture {"width":512,"height":288,"levels":10,"constant":[128,64,32,255],"tolerance":1,"format":"rgba8unorm-srgb"}.

Gaps above 50 ms, first-use: sync 6 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat; preparation: sync 0 rAF / 0 heartbeat; warmup 0 rAF / 0 heartbeat.

Validation passed: constant-input oracle, identical per-path A/B mip hashes, zero pipeline calls at first encode after warm-up, and zero pipeline calls at second encode in either arm.

Cross-dataset build digests: 1. Compare timing deltas within an adapter. Different digests require reviewing implementation changes before comparing builds.

Compiler events count DXC/FXC complete events or begin events. CPU duration sums complete spans and reconstructed begin/end spans when available. No observed DXC/FXC event does not prove a cache hit: backend support and trace availability differ. Pipeline API calls are distinct from shader compiler invocations.

Warm-up moves compilation before first submission; it does not remove compilation or guarantee lower total startup time. Headless rAF describes browser scheduling, not physical presentation. The constant-input fixture is a narrow execution check; nonconstant GPU oracle tests remain necessary.

Reproduce with `npm run build`, then `node results/probes/ds3-warmup.mjs --repeats=3 --trace --output=PATH.json` and optional `--power=high|low`, `--adapter=swiftshader` or `--chrome=PATH`. Run adapters sequentially. Regenerate using `node results/probes/ds3-report.mjs PATH.json ... --output=SUMMARY.md`.

Skill feedback: none.
