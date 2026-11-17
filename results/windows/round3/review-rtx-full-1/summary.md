# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 1/4 (shards 1–12) of 48 shards, 6 at a time, cap 15 min
- 8466 cases, 18667 pattern runs, 786 of 3239 case groups; wall 14 min 20 s

## Failures (1)

- fixtures/1920x1080 threw: tool fixtures 1920x1080: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 274.6 s, 13 cases
- `source/rgba16float/rgba8unorm/same` 141.2 s, 13 cases
- `source/rgba32float/rgb9e5ufloat/same` 136.9 s, 13 cases
- `source/rgba16float/rgba8unorm/half` 135.7 s, 13 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 133.7 s, 13 cases
- `integer/sources` 119.7 s, 22 cases
- `source/rgba16float/rgba8unorm-srgb/half` 118.8 s, 13 cases
- `source/depth32float/r32float/same` 108.9 s, 13 cases
- `source/rgba16float/rgb9e5ufloat/half` 105.6 s, 13 cases
- `source/rgba8unorm/rgba16float/half` 94.8 s, 13 cases
