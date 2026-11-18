# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low --isolate-devices`; 18 shards, 6 at a time, cap 15 min
- 21791 cases, 33288 pattern runs, 1864 of 1864 case groups; wall 11 min 31 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 66.6 s, 21 cases
- `srgb/alias` 61.6 s, 1 cases
- `downsampler/options` 40.4 s, 10 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 37.0 s, 11 cases
- `halo/rgba16float/buffer` 32.6 s, 26 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 23.0 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 20.4 s, 8 cases
- `integer/sources` 19.1 s, 10 cases
- `source/rgba16float/rg11b10ufloat/half` 17.8 s, 11 cases
- `source/rgba16float/rgb9e5ufloat/half` 15.1 s, 11 cases
