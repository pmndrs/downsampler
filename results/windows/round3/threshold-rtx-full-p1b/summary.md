# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; 48 shards, 6 at a time, cap 15 min
- 2943 cases, 6741 pattern runs, 271 of 3238 case groups; wall 6 min 49 s

## Failures (1)

- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba8unorm-srgb/rgba16float/half` 74.3 s, 13 cases
- `source/depth16unorm/r32float/half` 62.3 s, 13 cases
- `box/resample/rgba16float/rgba8unorm` 55.4 s, 16 cases
- `region/source/r32float/r32float/pow2-up/box` 49.3 s, 15 cases
- `downsampler/rgba32float/storage/L6/spd-average` 39.9 s, 24 cases
- `region/tail/rgba8unorm-srgb/average` 33.2 s, 12 cases
- `downsampler/rgb9e5ufloat/buffer/L5/box-average` 31.2 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L1/box-average` 30.8 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L4/box-average` 29.9 s, 24 cases
- `downsampler/rgb10a2unorm/storage/L6/spd-average-load` 29.3 s, 24 cases
