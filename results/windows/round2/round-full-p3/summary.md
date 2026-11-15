# GPU suite, passed

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 3/3 (shards 33–48) of 48 shards, 6 at a time, cap 15 min
- 12438 cases, 27593 pattern runs, 1089 of 3234 case groups; wall 17 min 49 s

## Slowest groups

- `integer/rgba32sint/storage/L6/box-average` 49.0 s, 24 cases
- `region/spd/rgba16float/average` 47.0 s, 12 cases
- `region/box/rgba16float/average` 44.9 s, 12 cases
- `region/tail/rgba16float/average` 44.0 s, 12 cases
- `box/resample/depth32float/rg32float` 43.9 s, 16 cases
- `integer/rgba32sint/storage/L1/average` 43.5 s, 24 cases
- `region/source/rgba16float/rgba16float/pow2-down/spd` 43.2 s, 15 cases
- `region/exact/rgba8unorm-srgb/max` 42.3 s, 12 cases
- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 41.3 s, 24 cases
- `integer/rgba32uint/storage/L6/box-average` 40.3 s, 24 cases
