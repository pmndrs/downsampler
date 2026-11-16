# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 2/3 (shards 17–32) of 48 shards, 6 at a time, cap 15 min
- 12177 cases, 27058 pattern runs, 1093 of 3237 case groups; wall 21 min 4 s

## Failures (3)

- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `srgb/alias` 130.4 s, 1 cases
- `source/depth32float/rg32float/half` 83.4 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 80.1 s, 15 cases
- `source/r32float/rg32float/same` 66.6 s, 13 cases
- `integer/rgba32uint/storage/L5/average` 59.1 s, 24 cases
- `box/resample/rgba8unorm/rgba16float` 58.9 s, 16 cases
- `downsampler/rg11b10ufloat/storage/L4/spd-average` 58.1 s, 24 cases
- `downsampler/rg11b10ufloat/storage/L4/spd-average-load` 57.5 s, 24 cases
- `region/exact/rgba16float/max` 55.4 s, 12 cases
- `integer/rgba32uint/storage/L5/box-average` 54.2 s, 24 cases
