# M3 measurements: Apple M5 Pro (2026-11-13)

Chrome 148 headless, Metal; load average 2.2–4.6 (other projects idle-ish, no agent GPU suites during the runs). `npm run bench -- --compare=<mode> --skip-validation`, default rounds/repeats; raw JSON next to this file (`m5-pro-<mode>.json`). Ratios are same-run medians over the render sizes 960×540 … 2560×1440 unless stated; under ~5 % is noise. Windows numbers come from round 2 (`docs/WINDOWS-HANDOFF-2.md`).

| Question | Answer on the M5 Pro | Decision rule → outcome |
|---|---|---|
| Do timestamps order strategies like work-per-frame throughput (greggman's concern)? | Yes: 80 of 83 strategy pairs ordered alike, mean Kendall τ 0.81; the 3 disagreements are within-noise pairs at 1280×720 / 1024² rgba16f | Timestamps are trustworthy here; keep them as the primary method, throughput as the cross-check |
| Does `STORAGE_BINDING` usage slow sampling (Kangz)? | No: storage ÷ plain 0.999–1.006 median for rgba8, rgba16f, r32f, fullscreen and 9-tap material passes, rendered and noise content | No cost on Apple; Windows round 2 decides for NVIDIA/Intel |
| Subgroup kernel upper bound (`nosync`, no barriers or shared memory past level 1) | 0.983 (rgba8) / 0.945 (rgba16f) at render sizes, 0.887 / 0.924 on squares. `shuffle` itself: 1.009 / 0.999 (no gain) | Bar is ≥ 5 % at render sizes: not met for rgba8, borderline for rgba16f, and the real kernel can only be slower than the bound → **no subgroup kernel for Apple** |
| fp16 arithmetic (`shader-f16`) | rgba8: 0.995 (f16), 0.999 (f16safe), identical results. rgba16f: 0.98 / 0.98, with up to 4 f16 steps of error on 3–13 % of values, and plain f16 overflows to infinity on large inputs | ~2 % for a precision cost → **not built** |
| f16 rounding cost (`'nearest'` vs `'device'`) | `exactDevice` ÷ `exact` 1.00 (rgba8) / 0.97 (rgba16f): rounding costs ~3 % on rgba16f | Keep `'nearest'` (owner, 2026-11-13) |
| `'box'` cost | `box` ÷ `exact` 1.28 (rgba8, 1.04–1.99) / 1.23 (rgba16f, 1.01–2.52); `tailBox` 1.34 / 1.51 (up to 2.6×) | Opt-in quality mode, as documented. Follow-up: the box tail kernel is the slow part (shader weights at every odd step); measure on Windows and consider `tail: false` by default under `'box'` |
| Tail kernel (default) | 0.90 (rgba8, 0.84–1.01) / 1.00 (rgba16f) of `'exact'` without it | Default stays |
| `'spd'` | 0.60 (rgba8) / 0.79 (rgba16f) of `'exact'` | Upper bound for halo batches (Windows round 2) |
| sRGB (buffer path) vs render passes | Render passes faster from 1080p up (4096²: 0.58×); ours faster at 1024² (render 1.95×) | three.js keeps render passes for large sRGB textures until Windows numbers |
