# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 15 min
- 20613 cases, 31406 pattern runs, 1758 of 1860 case groups; wall 10 min 17 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `srgb/alias` 68.2 s, 1 cases
- `halo/rgba8unorm-srgb/auto` 61.0 s, 21 cases
- `downsampler/options` 49.3 s, 9 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 47.8 s, 11 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 34.5 s, 11 cases
- `halo/rgba16float/buffer` 30.5 s, 26 cases
- `source/rgba16float/rgba8unorm-srgb/half` 23.9 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 18.2 s, 8 cases
- `integer/sources` 17.7 s, 10 cases
- `source/rgba16float/rg11b10ufloat/half` 16.0 s, 11 cases
