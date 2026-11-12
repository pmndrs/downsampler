# CI: the sharded GPU suite

CI runs the quick GPU suite on SwiftShader (software WebGPU) in parallel shards, following the owner's homefig pattern (`.github/workflows/ci.yml`):

- **`layout`** picks the shard counts for the tier.
  - Fast tier (pull requests): 14 core shards.
  - Full tier (push to main, the weekly schedule, manual runs): 14 core shards plus 14 compatibility-mode shards.
- **`core`** runs the typecheck, the build and `test:core`. It also checks the real partition (`--list`), which runs nothing on the GPU.
- **`gpu` / `gpu-compat`** are matrix jobs. Each runs `node test/run.js --quick --adapter=swiftshader [--compat] --shard=i/N --timing=ci-timing.json`. Each writes its time and slowest groups to the step summary and uploads `ci-timing.json` as `gpu-timing-<suite>-<i>`.
- **`gpu-suite`** (`if: always()`) is the stable check for branch protection. It fails unless every shard job succeeded (`report.js needs`, like homefig's `assert-needs-results`). It also fails unless the timing artifacts show N passing shards per suite that saw one listing, and ran disjoint case groups covering it (`report.js aggregate`). It puts a per-shard table in the summary and uploads `gpu-durations`: the committed durations with this run's times merged in.

## Case groups and shards

The suite is a list of **case groups** (`test/browser/validate.js`). Each has a stable id and owns its cases; every result belongs to exactly one group:

| ids | group |
| --- | --- |
| `downsampler/<format>/<output>/L<width>/<variant>` | main matrix, all shapes |
| `source/<from>/<to>/<relation>` | separate sources |
| `reducer/<edges>/<variant>/L<width>/<path>` | Reducer matrix, all shapes |
| `downsampler/options`, `reducer/options`, `packed/{downsampler,reducer}`, `wide`, `payload` | API and packed checks |
| `tools/…` | the shading-change, denoiser and cross-level shapes |
| `scalar/…` | one per ScalarReduction section |
| `fixtures/<W>x<H>` | the tool fixtures' oracle checks (`test/tools`), one per size; `--quick` runs two sizes |

Groups follow pipelines: a group compiles its own kernels, so shards rarely compile the same pipeline twice. On SwiftShader, compiling is most of the cost.

A sharded run first makes a listing pass. It walks the suite, registers every id and runs nothing: no textures, pipelines or oracles, about a second. Then `assignShards` (`shard.js`) splits the ids into N disjoint shards that cover the listing:

- With durations, longest first onto the least-loaded shard (LPT). Ties go by id, then to the lower shard, so the split depends only on the ids and durations, not on their order.
- Groups without a recorded duration count as the median.
- With no durations at all (missing file), round-robin in listing order.

Each shard then runs only its groups, in suite order. The listing depends on the device's features (SwiftShader lacks tier1; compatibility mode drops some storage paths). So every shard computes it on its own runner, and `gpu-suite` checks that all shards saw the same digest.

`npm run test:core` covers `shard.js` (`test/core/shard.test.js`): disjoint and covering for 1–24 shards with and without durations, order-independent, and LPT-bounded. The `core` job checks the real listing.

## Durations

`durations.json` records each group's wall time (ms) and case count from one unsharded `--quick` SwiftShader run. Time between groups (shared textures, inputs, oracles) is charged to the next group, so the times add up to the run.

Re-record after adding, removing or reshaping cases. The shards stay correct without it (new groups weigh the median), but drift apart in length. `core` warns when groups have no recorded duration.

```
node test/run.js --quick --adapter=swiftshader --record-durations                  # ~30 min on an M5 Pro: rewrites the file
node test/run.js --quick --adapter=swiftshader --shard=2/4 --record-durations     # merges one shard's groups
```

Or take the `gpu-durations` artifact of a CI run: the times measured on GitHub's runners, merged into the committed file. Copy it over `test/ci/durations.json` and commit. `node test/ci/report.js aggregate <dir> --write-durations=<file>` does the same merge from downloaded `gpu-timing-*` artifacts.

## Running one shard locally

```
npm run build
node test/run.js --quick --adapter=swiftshader --shard=3/14              # what CI shard 3 runs
node test/run.js --quick --adapter=swiftshader --list --shard=3/14       # its case groups, without running them
node test/run.js --quick --adapter=swiftshader --list --shards=14 --verbose  # every shard's groups and recorded time
node test/run.js --quick --shard=3/14                                     # the same shard on the real GPU (its own listing)
node test/run.js --quick --adapter=swiftshader --compat --shard=3/14    # a compatibility-mode shard
```

The listing depends on the adapter, so a real GPU's shard 3 is not SwiftShader's shard 3. On any one adapter the shards add up to the unsharded run: same cases, same pattern runs.

## Shard counts

Both tiers run 14 core shards. The full tier adds 14 compatibility-mode shards. They are set in the `layout` job.

How the counts were estimated, before any CI timing existed:

- An unsharded `--quick` SwiftShader run on an Apple M5 Pro (18 cores) takes 25.4 min (4784 cases, 323 groups, recorded 2026-11-12). The recorded group times add up to that.
- A 2-vCPU GitHub runner is assumed to be about 4× slower. SwiftShader mostly compiles pipelines on one thread, which is about 2–2.5× slower there. The renderer's oracles and the GPU process also share two vCPUs instead of 18 cores. That puts the whole suite at about 100 min of runner time.
- 14 shards balanced by LPT come to 1.8–2.3 min each locally, about 7–9 min on CI, under the 10 min target. That leaves room for each runner's checkout, `npm ci`, build and Chrome start (about 1–2 min outside the test step).
- The longest shard is set by the longest group: `downsampler/rgba16float/storage/L1/average`, 135 s locally (about 9 min on CI). It runs first, so it is charged the suite's shared setup (inputs, oracles, first compiles). More shards don't shorten the longest one: 12 and 16 shards have the same maximum. Splitting that group, or moving the shared setup into its own group, is the next lever if CI shows it. Run alone (shard 1/14 locally) that group took 0.5 s, so most of its 135 s is setup that serves later groups: shard 1 is underloaded until CI timings replace the local record.
- Compatibility mode runs the same groups with extra copy dispatches, so it gets the same count.

The jobs' 75 min limit is only a safety net.

After the first CI runs, read the `gpu-suite` summary (longest shard, slowest groups). Commit the `gpu-durations` artifact to balance on runner times. Then move the counts so the longest shard sits near 8 min.
