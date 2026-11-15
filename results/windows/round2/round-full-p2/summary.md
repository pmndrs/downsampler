# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 2/3 (shards 17–32) of 48 shards, 6 at a time, cap 15 min
- 12213 cases, 26848 pattern runs, 1089 of 3234 case groups; wall 18 min 12 s

## Failures (3)

- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/depth32float/rg32float/half` 85.9 s, 13 cases
- `source/r32float/rg32float/same` 66.4 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 60.1 s, 15 cases
- `integer/rgba32uint/storage/L1/box-average` 50.3 s, 24 cases
- `region/tail/rgba8unorm-srgb/max` 50.1 s, 12 cases
- `integer/rgba32sint/storage/L4/average` 45.9 s, 24 cases
- `integer/rgba32uint/storage/L1/average` 45.0 s, 24 cases
- `region/exact/rgba16float/max` 44.8 s, 12 cases
- `integer/rgba32uint/storage/L5/box-average` 44.2 s, 24 cases
- `box/resample/depth16unorm/r32float` 44.0 s, 16 cases
