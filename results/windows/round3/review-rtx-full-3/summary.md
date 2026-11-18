# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 3/4 (shards 25–36) of 48 shards, 6 at a time, cap 15 min
- 9279 cases, 20108 pattern runs, 820 of 3239 case groups; wall 13 min 47 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 84.8 s, 49 cases
- `region/source/depth32float/rg32float/pow2-up` 58.2 s, 15 cases
- `region/box/rgba16float/average` 51.3 s, 12 cases
- `region/tail/rgba16float/max` 48.9 s, 12 cases
- `integer/rgba32uint/storage/L5/box-average` 45.9 s, 24 cases
- `integer/rgba32uint/storage/L1/average` 44.2 s, 24 cases
- `integer/rgba8uint/storage/L5/box-average` 42.4 s, 24 cases
- `box/resample/rgba16float/rg11b10ufloat` 42.2 s, 16 cases
- `srgb/render` 41.8 s, 1 cases
- `integer/rgba32sint/storage/L5/box-average` 40.1 s, 24 cases
