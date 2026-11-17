# GPU suite, passed

- Device: apple metal-3
- Flags: `--durations=results/local/m5-full-defaults/durations.json`; part 1/2 (shards 1–12) of 24 shards, 6 at a time, cap 14 min
- 17709 cases, 38579 pattern runs, 1608 of 3238 case groups; wall 10 min 32 s

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 130.9 s, 13 cases
- `source/rgba16float/rgba8unorm/same` 75.8 s, 13 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 54.6 s, 13 cases
- `source/rgba32float/rgb9e5ufloat/same` 51.5 s, 13 cases
- `source/depth32float/r32float/same` 46.1 s, 13 cases
- `source/rgba16float/rgba8unorm/half` 46.0 s, 13 cases
- `source/rgba8unorm/rgba16float/half` 42.5 s, 13 cases
- `source/rgba16float/rgb9e5ufloat/half` 33.3 s, 13 cases
- `source/rgba16float/rg11b10ufloat/half` 33.2 s, 13 cases
- `source/depth32float/rg32float/half` 31.0 s, 13 cases
