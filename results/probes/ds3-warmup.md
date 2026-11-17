# DS3 pipeline warm-up probe

Build once with `npm run build`, then run adapters sequentially while the GPU is idle:

```powershell
node results/probes/ds3-warmup.mjs --power=high --repeats=3 --trace --output=results/windows/ds3/nvidia.json
node results/probes/ds3-warmup.mjs --power=low --repeats=3 --trace --output=results/windows/ds3/intel.json
node results/probes/ds3-warmup.mjs --adapter=swiftshader --repeats=3 --trace --output=results/windows/ds3/swiftshader.json
node results/probes/ds3-report.mjs results/windows/ds3/nvidia.json results/windows/ds3/intel.json results/windows/ds3/swiftshader.json --output=results/windows/ds3/summary.md
```

On macOS/Linux omit `--power` to use the default adapter; Puppeteer's browser is used unless `--chrome=PATH` is supplied. Windows defaults to installed system Chrome when present, otherwise Puppeteer. Output paths are arbitrary; the scripts do not require Windows paths. `--help` lists all options. `--paths=storage --repeats=1` provides a short GPU smoke test.

Before launching Chrome, the runner snapshots the complete dist directory into its own temporary directory. Every arm is served from that immutable snapshot. Result schema 2 records UTC start/capture/finish timestamps, platform, command, flags, browser version, and a complete file manifest with per-file and aggregate SHA256 digests. Build changes elsewhere cannot alter an in-flight run.

Each output path and A/B arm gets a separate fresh Chrome profile per pair, followed by a reload at the same origin in that browser. Arm order alternates by pair. OS and driver shader caches are **not cleared**. A fresh profile does not mean cold driver caches. The comparison uses one build: synchronous `encode()` versus `await warmup(texture)` before `encode()`.

Private browser profiles and the dist snapshot are removed in `finally` by default, including failures. Cleanup resolves each exact owned directory and checks it remains beneath the operating system temp directory. `--keep-temp` retains these paths for debugging. Raw traces are opt-in with `--trace`; their retained directory is printed and recorded. Set `--trace-dir=PATH` to choose it. JSON contains compact compiler events/spans so reporting does not depend on raw trace files. Trace start, decoding and writing failures fail the run explicitly and remain in result JSON.

Measurements include warm-up elapsed time, first and second command recording/finish, `queue.submit()` call time and submit-to-`onSubmittedWorkDone()` time. Four-millisecond interval heartbeat and rAF gaps include 35 ms observation margins around each phase. Headless rAF measures browser scheduling rather than physical display presentation. API counters distinguish sync/async compute/render pipeline creation, modules and emitted WGSL bytes. Pipeline API calls differ from shader compiler invocations. Traces preserve available DXC/FXC complete events and reconstruct begin/end spans. Absent compiler events do not prove cache hits; backend support and trace availability vary.

The fixture is a 512x288 texture with ten mip levels, constant RGBA `[128,64,32,255]`. All generated levels are read back against the constant-input oracle (one-code tolerance), with per-level FNV hashes for A/B comparison. Storage and buffer use rgba8unorm; render uses rgba8unorm-srgb. This narrow execution/pixel check supplements the dedicated nonconstant GPU oracle suite. It does not measure steady-state GPU throughput; use the repository benchmark for that.

The reporter derives timing medians, frame/heartbeat events, compiler counts and CPU durations, WGSL bytes and pipeline counters from result JSON. It rejects incomplete/duplicate A/B combinations, pixel errors/hash mismatches and unexpected warm/second-use pipeline calls. Tracing-disabled runs produce `n/a` compiler columns. Legacy schema 1 results are labelled as lacking verified build isolation. Unclassified CDP error logs are surfaced for review rather than silently discarded.

Offline checks without a browser or GPU:

```powershell
node --check results/probes/ds3-warmup.mjs
node --check results/probes/ds3-warmup.js
node --check results/probes/ds3-report.mjs
node results/probes/ds3-warmup.mjs --dry-run --output=OFFLINE-SMOKE.json
```

`--dry-run` validates options, snapshots/hashes dist, fetches the served build and browser probe assets, and cleans up. It skips adapter queries and browser launch. The report JSON records `options.dryRun`; the statistical reporter rejects an empty measurement set.

Skill feedback: none.
