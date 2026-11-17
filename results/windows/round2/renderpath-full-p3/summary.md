# GPU suite, passed

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 3/3 (shards 33–48) of 48 shards, 6 at a time, cap 15 min
- 12243 cases, 26821 pattern runs, 1088 of 3237 case groups; wall 19 min 6 s

## Slowest groups

- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 53.7 s, 24 cases
- `region/exact/rgba8unorm-srgb/max` 52.3 s, 12 cases
- `integer/rgba32sint/storage/L1/average` 50.1 s, 24 cases
- `region/box/rgba16float/average` 49.7 s, 12 cases
- `srgb/render` 49.6 s, 1 cases
- `region/tail/rgba16float/average` 46.1 s, 12 cases
- `integer/rgba8uint/storage/L4/box-average` 45.1 s, 24 cases
- `region/spd/rgba16float/average` 45.0 s, 12 cases
- `integer/rgba32uint/storage/L6/box-average` 44.6 s, 24 cases
- `region/box/rgba8unorm/average` 44.2 s, 12 cases
