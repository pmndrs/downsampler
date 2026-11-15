# A cheaper exact kernel (handoff-2 §4.2, 2026-11-15)

**Question.** At power-of-two sizes `'exact'` and `'spd'` take the same two dispatches, yet `'spd'` was faster in
rgba8: 0.84–0.95× of `'exact'` on the RTX 3060 and 0.66–0.83× on Intel UHD at 1024²–2048² (round 2 `--compare=spd`,
[rtx-spd.json](rtx-spd.json), [intel-spd.json](intel-spd.json)); in rgba16float they were equal. So the cost was in
the exact kernel's per-value arithmetic for normalized formats, not its structure (a one-barrier-per-level kernel,
`slot`, was already 0.99×: [m3.md](m3.md)).

**Probe.** `npm run bench -- --compare=trim` (test/bench/kernel-variants.js): the library's own kernel with its
arithmetic edited, structure unchanged, every level compared with the library's chain on two patterns.
- `round`: normalized `quantize` as `round(clamp(v, min, max))`. WGSL's `round()` is ties to even (Metal `rint`,
  HLSL `round_ne`, SPIR-V `RoundEven`), the library's rule; the old version took floor, a subtraction, two
  compares and two selects.
- `mul`: stores as `value * (1.0 / SCALE)` instead of `value / SCALE`.
- `trim`: both.

**Results** (median ratio to the library kernel; paired per round):

| GPU | Format | `round` render sizes / squares | `mul` | `trim` | Check |
|---|---|---|---|---|---|
| RTX 3060 ([rtx3060-trim.json](rtx3060-trim.json), VALID) | rgba8unorm | **0.962 / 0.912** (2048²: 0.858) | 1.001 / 1.001 | 0.964 / 0.912 | identical, every size and pattern |
| RTX 3060 | rgba16float | 1.000 / 1.000 | 1.000 / 1.000 | 0.999 / 1.001 | identical |
| Intel UHD ([intel-uhd-trim.json](intel-uhd-trim.json)) | rgba8unorm | **0.730 / 0.763** | | | identical |

(Intel timings have no clock verdict: the sampler reads NVIDIA clocks only.)

**Shipped:** `round()` in `quantize` for every normalized format, and for sRGB alpha, in `src/webgpu/kernel.ts` and
`src/mipmap.ts` (golden WGSL updated). Not shipped: `mul` (no gain). rgba16float already rounds with
`ds_round_f16`, so it doesn't change.

**Not yet measured on the M5 Pro.** Tint emits `rint` for Metal, which rounds to even under fast math too, and
the suite's in-tile levels are checked bit for bit against the oracle, so a different tie rule would fail there.
