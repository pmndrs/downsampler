# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 20 min
- 20635 cases, 31557 pattern runs, 1757 of 1859 case groups; wall 13 min 56 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 84.7 s, 21 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 71.5 s, 11 cases
- `downsampler/options` 71.2 s, 9 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 55.0 s, 11 cases
- `halo/rgba16float/buffer` 47.8 s, 26 cases
- `source/rgba16float/rgba8unorm-srgb/half` 36.9 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 30.7 s, 8 cases
- `source/rgba16float/rg11b10ufloat/half` 26.2 s, 11 cases
- `source/rgba32float/rgb9e5ufloat/same` 23.6 s, 11 cases
- `integer/sources` 22.6 s, 10 cases
