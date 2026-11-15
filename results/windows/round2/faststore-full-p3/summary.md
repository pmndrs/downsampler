# GPU suite, passed

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 3/3 (shards 33–48) of 48 shards, 6 at a time, cap 15 min
- 12438 cases, 27593 pattern runs, 1089 of 3234 case groups; wall 20 min 11 s

## Slowest groups

- `region/tail/rgba16float/average` 57.6 s, 12 cases
- `integer/rgba32sint/storage/L1/average` 55.6 s, 24 cases
- `region/box/rgba16float/average` 54.7 s, 12 cases
- `region/spd/rgba16float/average` 54.5 s, 12 cases
- `integer/rgba32sint/storage/L6/box-average` 53.6 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 52.5 s, 24 cases
- `box/resample/depth32float/rg32float` 50.1 s, 16 cases
- `region/exact/rgba8unorm-srgb/max` 46.2 s, 12 cases
- `integer/rgba32uint/storage/L6/box-average` 44.9 s, 24 cases
- `region/source/rgba16float/rgba16float/pow2-down/spd` 44.8 s, 15 cases
