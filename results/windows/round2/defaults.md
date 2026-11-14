# Defaults on Windows (round 2, handoff §2; 2026-11-14)

RTX 3060 Laptop (nvidia ampere, 592.00) and Intel UHD (gen-12lp, 31.0.101.5333), Chrome 148 headless, D3D12,
on AC, Turbo plan, GPU idle (45–54 °C). Same-run ratios of medians (11 rounds × 16 repeats), so clocks
don't matter. "Render sizes" are 960×540 … 2560×1440; under ~5 % is noise.

Files: `rtx-render.json`, `intel-render.json` (`--compare=render`), `rtx-spd.json`, `intel-spd.json`
(`--compare=spd`). The spd runs had a few outlier rounds (ranges like 0.40–3.08 for one ratio); medians
over the render sizes are quoted.

| Question (handoff rule) | RTX rgba8 / rgba16f | Intel rgba8 / rgba16f | Decision |
|---|---|---|---|
| Tile 16 everywhere? (`exact16 ÷ exact`, `tail16 ÷ tail` within 5 % on both) | 1.03 / 0.96 and **1.12** / 1.01 (up to 1.31) | **1.14** / 1.04 and **1.14** / 1.04 (up to 1.41) | **No change:** 32 off Apple, 16 on Apple |
| Tail default (off if it loses ≥ 5 % on either) | `tail ÷ exact` **0.82 / 0.86** | 0.99 / 0.99 | **Stays on** |
| f16 rounding (`'nearest' ÷ 'device'`, act if Intel > ~10 %) | — / 1.05 | — / 1.085 | **Kept;** under the threshold, no cheaper rounding needed |
| `'box'` cost (`box ÷ exact`; report) | 2.15 / 1.45 | 1.54 / 1.58 | Reported (M5: 1.23–1.28) |
| `'box'` tail (`tailBox ÷ tail`) | 1.57 / 1.35 | 1.41 / 1.49 | Reported; see below |
| `'spd' ÷ 'exact'` | 0.73 / 0.76 | 0.76 / 1.04 | — |
| Halo batches | see [halo.md](halo.md) | | Opt-in |

- **Tile.** Round 1's change to 32 off Apple stands: 16 costs 3–14 % at render sizes (rgba8 most), up to
  31 % on the RTX and 41 % on Intel at 2560×1440. The rule for dropping the vendor check isn't met.
- **Tail.** On the RTX it removes 14–18 %; on Intel it's neutral. It stays the default.
- **f16.** The branch-free integer rounding (`5f97c86`) costs 5 % (RTX) and 8.5 % (Intel) of an rgba16float
  chain against the device's (truncating) conversion; round 1's add trick cost 14 % on Intel. rgba8 chains
  are unaffected (0.999 / 1.04, noise).
- **`'box'`** is dearer on Windows than on the M5 (1.45–2.15× against 1.23–1.28×): its odd levels load
  and weight in the shader. `tailBox` is 0.49–2.82× of `tail` by size, so PROGRESS's follow-up ("the box
  tail kernel is slow: consider `tail: false` under `'box'`") needs a per-size look, not a blanket switch:
  `box` (tail off) ÷ `tailBox` is in the JSON for that.
