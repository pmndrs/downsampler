# GPU suite, passed

- Device: apple metal-3
- Flags: `--durations=results/local/m5-full-defaults/durations.json`; part 2/2 (shards 13–24) of 24 shards, 6 at a time, cap 14 min
- 18199 cases, 40553 pattern runs, 1630 of 3238 case groups; wall 10 min 42 s

## Slowest groups

- `source/rgba16float/rgba8unorm-srgb/half` 58.9 s, 13 cases
- `integer/sources` 43.1 s, 22 cases
- `source/r32float/rg32float/same` 33.8 s, 13 cases
- `source/depth32float/rg32float/same` 30.7 s, 13 cases
- `source/rgba8unorm-srgb/rgba16float/half` 28.5 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 25.5 s, 15 cases
- `srgb/render` 25.0 s, 1 cases
- `box/resample/r32float/r32float` 23.0 s, 16 cases
- `integer/rgba32sint/storage/L5/box-average` 21.5 s, 24 cases
- `box/resample/rgba8unorm/rgba16float` 20.2 s, 16 cases
