# Mac handoff for the downsampler

The handoff branch is `codex/web-performance-review`. It includes `windows-3` at `0210cd1` and the Windows review changes. To use the portable Git bundle in a fresh checkout:

```sh
git clone -b codex/web-performance-review downsampler-mac-handoff.bundle downsampler-review
cd downsampler-review
npm ci && npm run build && npm run test:core
```

In an existing Mac checkout, `git fetch /path/to/downsampler-mac-handoff.bundle refs/heads/codex/web-performance-review:refs/remotes/handoff/downsampler` brings in the same history. Compare it with your current branch using `git log --left-right --oneline HEAD...handoff/downsampler`. The bundle carries the Windows history back to the shared ancestor, so it also works offline once dependencies are installed.

## First checks on the Mac

Run the three targeted groups and open the same brush and Hi-Z examples exercised on Windows.

```sh
node test/run.js --quick --groups=downsampler/options,downsampler/warmup,region/options
node test/examples.js --only=partial-update,hi-z
```

The warmup group reads back a patterned 65×33 texture with two layers on the storage, buffer and render paths. The brush replay compares the edited mips with a full recompute. Hi-Z compares culling decisions with the full-resolution depth buffer. See the [Windows review](../WEB-PERFORMANCE-REVIEW.md) for the implementation revisions and measurements.

## Capped halo on Apple M5 Pro

Run both policies on the same idle machine. The first pair compares `halo` with `tail` under the default exact policy. The second compares `haloBox` with `tailBox` under the area-weighted box policy. Each compares the same texture, format and policy with the cap on or the halo disabled. Include the odd render sizes; square textures alone miss the dispatch changes.

```sh
sizes=960x540,1129x635,1280x720,1477x831,1707x960,1920x1080,2560x1440,3840x2160
node test/run.js --bench --skip-validation --compare=render --strategies=tail,halo --sizes="$sizes" --formats=rgba8unorm,rgba16float --rounds=11 --repeats=16 --output=results/m5/halo-exact.json
node test/run.js --bench --skip-validation --compare=spd --strategies=tailBox,haloBox --sizes="$sizes" --formats=rgba8unorm,rgba16float --rounds=11 --repeats=16 --output=results/m5/halo-box.json
node results/probes/halo-pairs.mjs results/m5/halo-exact.json results/m5/halo-box.json
```

A ratio below 1 means the capped halo took less GPU time. Record the Apple adapter, browser and load. Repeat before changing the default. The owner’s criterion is at least a 5% gain on M5 and no losses. Keep the existing opt-in setting while the Apple result is pending. If it qualifies, update the vendored `src/mipmap.ts` and its goldens and run the real-GPU matrix. The Windows reference has all 16 size/format pairs per policy. [Windows review results](../WEB-PERFORMANCE-REVIEW.md)

## sRGB and float32

Background: [DIVERGES #16](../DIVERGES.md#16-srgb-through-the-buffer-output-path), [srgb-renderpath.md](../results/windows/round2/srgb-renderpath.md), [srgb-alias.md](../results/windows/round2/srgb-alias.md).

### What changed

- sRGB textures with `RENDER_ATTACHMENT` now take a **render path** by default: a render pass per level, with the
  compute kernel's rounding in the fragment shader. Same bits as before, 0.63–0.79× the time on the RTX, 0.74–0.92× on Intel.
- `srgbRounding: 'device'` (opt-in): the same render path without the exact rounding. Plain render passes' speed;
  codes up to 3 apart.
- `colorSpace: 'srgb'`: rgba8unorm textures with an sRGB view format, written directly.

### Commands

1. Check sRGB correctness after building:
   ```bash
   node test/run.js --only=srgb --keep-going
   ```
   The new groups are `srgb/render`, `srgb/render-api` and `srgb/alias`. For the full matrix, use `test/parallel.js` with a 28-minute budget and split the run with `--part=i/n` as needed.
2. Speed (machine as idle as it gets; note the load average):
   ```bash
   npm run bench -- --compare=srgb --skip-validation --strategies=render,buffer,colorSpace,renderPath,renderDevice --rounds=11 --repeats=32 --output=results/m5/m5-pro-srgb-renderpath.json
   ```
   `renderDevice`'s `check` in the JSON reports how many bytes differ from the exact path.
3. Sampling cost of the aliased texture (`colorSpace: 'srgb'`): Apple can lose compression on reinterpretable textures.
   ```bash
   npm run bench -- --compare=storage-usage --formats=rgba8unorm-srgb --skip-validation --output=results/m5/m5-pro-srgb-alias-sampling.json
   ```

4. float32 averages, sampled vs loaded (`requiredFeatures` asks for `float32-filterable`; RTX and Intel: sampling 0.69–1.00× of loading):
   ```bash
   npm run bench -- --compare=float32 --skip-validation --output=results/m5/m5-pro-float32.json
   ```

### Decisions

- **Render path as the sRGB default.** It stays if `renderPath` beats `buffer` on the M5, as it does on both Windows
  GPUs. If it loses there, that's a split by vendor, which we avoid: bring it to the owner.
- **`srgbRounding`** stays `'exact'` by default (decided 2026-11-16). Record the gap between `renderPath` and
  `renderDevice` (the price of exact rounding: about 5× on Intel, up to 1.6× on the RTX); bring it back only if the M5
  changes the picture.
- **`colorSpace: 'srgb'`**: worth recommending only if it beats the render path somewhere and sampling stays at about 1.00×.

Record results in PROGRESS (verification log) and DIVERGES #16.
