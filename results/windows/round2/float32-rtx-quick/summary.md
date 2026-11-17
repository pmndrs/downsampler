# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --power=high`; 18 shards, 6 at a time, cap 15 min
- 21784 cases, 33288 pattern runs, 1863 of 1863 case groups; wall 12 min 19 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `srgb/alias` 106.0 s, 1 cases
- `halo/rgba8unorm-srgb/auto` 99.3 s, 21 cases
- `region/exact/rgba8unorm-srgb/max` 44.5 s, 8 cases
- `downsampler/options` 43.3 s, 9 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 37.9 s, 11 cases
- `source/rgba16float/rg11b10ufloat/half` 34.7 s, 11 cases
- `halo/rgba16float/buffer` 33.4 s, 26 cases
- `downsampler/rgba8unorm-srgb/buffer/L6/box-max` 28.5 s, 11 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 26.5 s, 11 cases
- `downsampler/bgra8unorm-srgb/buffer/L6/min` 26.0 s, 11 cases
