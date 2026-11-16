# M5 Pro: sRGB render path and `colorSpace: 'srgb'`

Branch `srgb-store`. Built and verified on Windows (RTX 3060 Laptop, Intel UHD); this is what's left for the Mac.
Background: [DIVERGES #16](../DIVERGES.md#16-srgb-through-the-buffer-output-path), [srgb-renderpath.md](../results/windows/round2/srgb-renderpath.md), [srgb-alias.md](../results/windows/round2/srgb-alias.md).

## What changed

- sRGB textures with `RENDER_ATTACHMENT` now take a **render path** by default: a render pass per level, with the
  compute kernel's rounding in the fragment shader. Same bits as before, 0.63–0.79× the time on the RTX, 0.74–0.92× on Intel.
- `srgbRounding: 'device'` (opt-in): the same render path without the exact rounding. Plain render passes' speed;
  codes up to 3 apart.
- `colorSpace: 'srgb'`: rgba8unorm textures with an sRGB view format, written directly.

## Run

1. Correctness:
   ```bash
   npm ci && npm run build && npm run test:core
   ```
   ```bash
   node test/run.js --only=srgb --keep-going
   ```
   The new groups are `srgb/render`, `srgb/render-api` and `srgb/alias`. Then the usual full matrix before merging.
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

## What the numbers decide

- **Render path as the sRGB default.** It stays if `renderPath` beats `buffer` on the M5, as it does on both Windows
  GPUs. If it loses there, that's a split by vendor, which we avoid: bring it to the owner.
- **`srgbRounding`** stays `'exact'` by default (decided 2026-11-16). Record the gap between `renderPath` and
  `renderDevice` (the price of exact rounding: about 5× on Intel, up to 1.6× on the RTX); bring it back only if the M5
  changes the picture.
- **Halo batches** become the default if they gain at least 5 % here and lose nowhere:
  ```bash
  npm run bench -- --compare=render --skip-validation --strategies=render,tail,halo,haloAny --output=results/m5/m5-pro-halo.json
  ```
- **`colorSpace: 'srgb'`**: worth recommending only if it beats the render path somewhere and sampling stays at about 1.00×.

Record results in PROGRESS (verification log) and DIVERGES #16.
