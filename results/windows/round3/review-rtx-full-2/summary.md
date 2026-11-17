# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 2/4 (shards 13–24) of 48 shards, 6 at a time, cap 15 min
- 8931 cases, 19886 pattern runs, 814 of 3239 case groups; wall 12 min 48 s

## Failures (2)

- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/depth32float/rg32float/half` 80.9 s, 13 cases
- `source/depth16unorm/r32float/half` 68.8 s, 13 cases
- `source/rgba8unorm-srgb/rgba16float/half` 67.5 s, 13 cases
- `box/resample/rgba16float/rgba8unorm` 63.1 s, 16 cases
- `source/r32float/rg32float/same` 60.3 s, 13 cases
- `region/source/r32float/r32float/pow2-up/box` 57.2 s, 15 cases
- `integer/rgba32sint/storage/L4/average` 51.7 s, 24 cases
- `region/tail/rgba8unorm-srgb/max` 43.9 s, 12 cases
- `region/exact/rgba16float/max` 39.8 s, 12 cases
- `integer/rgba32uint/storage/L1/box-average` 38.7 s, 24 cases
