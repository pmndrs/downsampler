# GPU suite, passed

- Device: apple metal-3
- Flags: `--durations=results/local/m5-full-defaults/durations.json`; part 2/2 (shards 13–24) of 24 shards, 6 at a time, cap 14 min
- 18199 cases, 40553 pattern runs, 1630 of 3238 case groups; wall 9 min 39 s

## Slowest groups

- `source/rgba16float/rgba8unorm-srgb/half` 59.0 s, 13 cases
- `integer/sources` 50.0 s, 22 cases
- `source/rgba8unorm-srgb/rgba16float/half` 32.5 s, 13 cases
- `source/depth32float/rg32float/same` 31.0 s, 13 cases
- `source/r32float/rg32float/same` 30.4 s, 13 cases
- `region/source/depth32float/rg32float/pow2-up` 28.8 s, 15 cases
- `srgb/render` 24.9 s, 1 cases
- `region/spd/rgba16float/average` 21.4 s, 12 cases
- `integer/rgba32sint/storage/L5/box-average` 20.2 s, 24 cases
- `box/resample/rgba8unorm/rgba16float` 19.9 s, 16 cases
