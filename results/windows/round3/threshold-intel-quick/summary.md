# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low --isolate-devices`; 18 shards, 6 at a time, cap 15 min
- 21784 cases, 33288 pattern runs, 1863 of 1863 case groups; wall 14 min 33 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 89.8 s, 21 cases
- `srgb/alias` 84.4 s, 1 cases
- `downsampler/options` 48.1 s, 9 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 44.1 s, 11 cases
- `halo/rgba16float/buffer` 39.2 s, 26 cases
- `region/exact/rgba8unorm-srgb/max` 28.1 s, 8 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 23.3 s, 11 cases
- `source/rgba16float/rg11b10ufloat/half` 22.3 s, 11 cases
- `integer/sources` 21.9 s, 10 cases
- `source/rgba16float/rgb9e5ufloat/half` 21.9 s, 11 cases
