# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low --isolate-devices`; 18 shards, 6 at a time, cap 15 min
- 21784 cases, 33288 pattern runs, 1863 of 1863 case groups; wall 12 min 34 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 80.4 s, 21 cases
- `srgb/alias` 66.7 s, 1 cases
- `downsampler/options` 42.7 s, 9 cases
- `halo/rgba16float/buffer` 36.5 s, 26 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 32.5 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 20.1 s, 8 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 19.8 s, 11 cases
- `srgb/render` 18.9 s, 1 cases
- `integer/sources` 18.2 s, 10 cases
- `source/rgba16float/rgb9e5ufloat/half` 17.0 s, 11 cases
