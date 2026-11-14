# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high`; 36 shards, 7 at a time, cap 25 min
- 5722 cases, 12702 pattern runs, 538 of 3225 case groups; wall 12 min 2 s

## Failures (1)

- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 285.3 s, 13 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 149.0 s, 13 cases
- `source/rgba16float/rgba8unorm-srgb/half` 130.7 s, 13 cases
- `source/r32float/rg32float/same` 59.2 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 56.2 s, 15 cases
- `box/resample/depth32float/rg32float` 46.2 s, 16 cases
- `integer/rgba32sint/storage/L5/box-average` 40.2 s, 24 cases
- `integer/rgba32uint/storage/L5/box-average` 39.6 s, 24 cases
- `box/resample/rgba8unorm/rgba16float` 39.6 s, 16 cases
- `box/resample/depth16unorm/r32float` 38.7 s, 16 cases
