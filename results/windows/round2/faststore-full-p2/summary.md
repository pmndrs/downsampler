# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 2/3 (shards 17–32) of 48 shards, 6 at a time, cap 15 min
- 12213 cases, 26848 pattern runs, 1089 of 3234 case groups; wall 23 min 6 s

## Failures (3)

- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/depth32float/rg32float/half` 125.6 s, 13 cases
- `source/r32float/rg32float/same` 77.8 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 67.7 s, 15 cases
- `downsampler/rg11b10ufloat/storage/L4/spd-average` 67.0 s, 24 cases
- `downsampler/rg11b10ufloat/storage/L4/spd-average-load` 61.0 s, 24 cases
- `region/tail/rgba8unorm-srgb/max` 60.7 s, 12 cases
- `integer/rgba32uint/storage/L1/box-average` 60.7 s, 24 cases
- `box/resample/r32float/r32float` 57.0 s, 16 cases
- `integer/rgba32sint/storage/L4/average` 56.0 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L5/spd-average-load` 55.0 s, 24 cases
