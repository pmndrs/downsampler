# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 15 min
- 20617 cases, 31313 pattern runs, 1760 of 1862 case groups; wall 13 min 9 s

## Failures (3)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- srgb/render threw: render default 255x257x1 r32float: level 1 value 28: 5.951111793518066, the compute path's 5.951112270355225

## Slowest groups

- `srgb/alias` 91.0 s, 1 cases
- `halo/rgba8unorm-srgb/auto` 84.5 s, 21 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 68.0 s, 11 cases
- `downsampler/options` 55.5 s, 9 cases
- `halo/rgba16float/buffer` 44.8 s, 26 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 39.5 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 35.1 s, 8 cases
- `source/rgba16float/rgba8unorm-srgb/half` 29.0 s, 11 cases
- `source/rgba16float/rg11b10ufloat/half` 28.6 s, 11 cases
- `srgb/render` 28.3 s, 0 cases
