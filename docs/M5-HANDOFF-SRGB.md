# Mac handoff for the downsampler

The portable bundle's `codex/web-performance-review` branch combines `srgb-store` through `1ff808b` (including
`windows-3`) with the DS review fixes. To use it in a fresh checkout:

```sh
git clone -b codex/web-performance-review downsampler-mac-handoff.bundle downsampler-review
cd downsampler-review
npm ci && npm run build && npm run test:core
```

In an existing Mac checkout:

```sh
git fetch /path/to/downsampler-mac-handoff.bundle refs/heads/codex/web-performance-review:refs/remotes/handoff/downsampler
git log --left-right --oneline HEAD...handoff/downsampler
```

The bundle carries both histories and works offline once dependencies are installed. Use an isolated checkout to
accept the combined branch before integrating it into ongoing Mac work.

## Accept the new review changes on Metal

```sh
node test/run.js --quick --groups=downsampler/options,downsampler/warmup,region/options,srgb/render-api
node test/examples.js --only=partial-update,hi-z
```

These groups cover failed-plan recovery, immutable moving-region bindings, native asynchronous pipeline rejection
and retry, sync-first render warmup, and warmup/encode agreement on threshold-selected paths, including full versus
clipped regions. The examples exercise paired brush replay with CPU/GPU timing and timestamp fallback, plus Hi-Z
culling decisions against full-resolution depth. Check the adapter is Apple and record the combined revision.

For new Metal startup measurements, use the existing same-build A/B probe (fresh and reloaded profiles):

```sh
node results/probes/ds3-warmup.mjs --repeats=3 --output=results/m5/review-warmup.json
node results/probes/ds3-report.mjs results/m5/review-warmup.json --output=results/m5/review-warmup.md
```

Compiler trace events may be unavailable on Metal; API counts, first-use timing and pixel checks remain useful.
See the [probe method](../results/probes/ds3-warmup.md) and [Windows review](../WEB-PERFORMANCE-REVIEW.md).

## Existing M5 decisions are complete

The imported [November 17 M5 report](../results/m5/srgb-m5.md) (`95f13a8`) covers sRGB paths, aliased sampling,
float32 and exact-policy halo. The [threshold matrix, part 1](../results/m5/threshold-full-p1/summary.md) and
[part 2](../results/m5/threshold-full-p2/summary.md) (`c84a90b`, recorded by `ce6abcf`) passed **35,908 cases**.

- **Halo stays opt-in.** Exact-policy halo/tail was 0.98–0.99 median across two runs; rgba8 at 1024² lost 3–4%.
  The owner's requirement (at least 5% gain and no losses) was not met. This default decision needs no rerun.
  Box-policy halo was not measured in that M5 pass; a matched `tailBox,haloBox` comparison is an optional experiment,
  not a release requirement or a reason to reopen the exact-policy decision.
- **sRGB auto selection follows size.** The owner chose `renderPathMinTexels: 1920 * 1080` per layer of the first
  processed level/dirty rectangle; smaller sRGB chains use buffer output (aliased `colorSpace: 'srgb'` uses storage),
  unless only rendering is supported. Overrides are `0` and `Infinity`. M5 render/buffer was 1.25–2.18 below 1080p,
  1.01 at 1080p, and 0.79–0.83 at the larger tested sizes. Exact rounding remains the default.
- **Apple alias and float32 questions are answered.** Aliased sRGB sampling was 0.970–1.042 (medians about 1.00);
  `colorSpace`/buffer was 0.40–0.68. Float32 sample/load was 0.71–1.01. No outstanding Mac rerun is needed for these.

The corrected Windows measurements and threshold validation are tracked in the [combined review](../WEB-PERFORMANCE-REVIEW.md#m5-and-windows-srgb-selection).
