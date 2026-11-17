# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 2/3 (shards 17–32) of 48 shards, 6 at a time, cap 15 min
- 12213 cases, 27011 pattern runs, 1089 of 3238 case groups; wall 17 min 14 s

## Failures (3)

- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/depth32float/rg32float/half` 81.0 s, 13 cases
- `halo/rgba8unorm-srgb/auto` 62.6 s, 49 cases
- `source/r32float/rg32float/same` 61.7 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 56.5 s, 15 cases
- `integer/rgba32uint/storage/L5/average` 47.3 s, 24 cases
- `integer/rgba32uint/storage/L1/average` 44.9 s, 24 cases
- `region/tail/rgba8unorm-srgb/max` 44.4 s, 12 cases
- `integer/rgba32sint/storage/L5/box-average` 42.4 s, 24 cases
- `integer/rgba32uint/storage/L1/box-average` 42.2 s, 24 cases
- `region/exact/rgba16float/max` 42.1 s, 12 cases
