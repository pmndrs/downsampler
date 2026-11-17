# GPU suite, passed

- Device: apple metal-3
- Flags: `--durations=results/local/m5-full-defaults/durations.json`; part 1/2 (shards 1–12) of 24 shards, 6 at a time, cap 14 min
- 17709 cases, 38579 pattern runs, 1608 of 3238 case groups; wall 12 min 54 s

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 142.8 s, 13 cases
- `source/rgba16float/rgba8unorm/same` 80.3 s, 13 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 55.7 s, 13 cases
- `source/rgba32float/rgb9e5ufloat/same` 55.6 s, 13 cases
- `source/rgba16float/rgba8unorm/half` 50.4 s, 13 cases
- `source/depth32float/r32float/same` 46.9 s, 13 cases
- `source/rgba8unorm/rgba16float/half` 44.3 s, 13 cases
- `source/rgba16float/rg11b10ufloat/half` 34.7 s, 13 cases
- `source/rgba16float/rgb9e5ufloat/half` 34.2 s, 13 cases
- `source/depth32float/rg32float/half` 34.2 s, 13 cases
