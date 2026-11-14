# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high`; 36 shards, 7 at a time, cap 25 min
- 6782 cases, 15551 pattern runs, 619 of 3225 case groups; wall 12 min 31 s

## Failures (3)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/depth32float/rg32float/same` 100.1 s, 13 cases
- `integer/sources` 88.9 s, 22 cases
- `region/source/r32float/r32float/pow2-up/box` 78.0 s, 15 cases
- `integer/rgba32sint/storage/L4/average` 72.0 s, 24 cases
- `box/resample/r32float/r32float` 58.4 s, 16 cases
- `region/exact/rgba16float/average` 56.5 s, 12 cases
- `integer/rgba8uint/storage/L4/average` 55.9 s, 24 cases
- `downsampler/rg11b10ufloat/storage/L4/spd-average` 54.4 s, 24 cases
- `region/box/rgba8unorm/max` 53.7 s, 12 cases
- `integer/rgba8uint/storage/L6/box-average` 53.6 s, 24 cases
