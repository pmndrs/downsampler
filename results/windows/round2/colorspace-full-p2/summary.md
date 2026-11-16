# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 2/3 (shards 17–32) of 48 shards, 6 at a time, cap 15 min
- 12121 cases, 26577 pattern runs, 1090 of 3235 case groups; wall 16 min 56 s

## Failures (3)

- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `srgb/alias` 91.9 s, 1 cases
- `source/depth32float/rg32float/half` 73.0 s, 13 cases
- `source/r32float/rg32float/same` 61.1 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 56.4 s, 15 cases
- `integer/rgba32sint/storage/L4/average` 48.7 s, 24 cases
- `integer/rgba32uint/storage/L5/average` 46.6 s, 24 cases
- `integer/rgba32uint/storage/L1/box-average` 43.6 s, 24 cases
- `integer/rgba32sint/storage/L5/box-average` 43.5 s, 24 cases
- `region/tail/rgba8unorm-srgb/max` 43.4 s, 12 cases
- `integer/rgba32uint/storage/L1/average` 41.9 s, 24 cases
