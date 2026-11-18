# GPU suite, passed

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 4/4 (shards 37–48) of 48 shards, 6 at a time, cap 15 min
- 9164 cases, 20471 pattern runs, 819 of 3239 case groups; wall 14 min 12 s

## Slowest groups

- `srgb/alias` 104.4 s, 1 cases
- `box/resample/depth32float/rg32float` 52.7 s, 16 cases
- `integer/rgba32sint/storage/L4/box-average` 50.4 s, 24 cases
- `integer/rgba32sint/storage/L5/average` 48.0 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 47.1 s, 24 cases
- `region/spd/rgba16float/average` 46.6 s, 12 cases
- `integer/rgba32sint/storage/L6/box-average` 45.7 s, 24 cases
- `integer/rgba32uint/storage/L6/box-average` 44.2 s, 24 cases
- `region/tail/rgba16float/average` 43.2 s, 12 cases
- `region/exact/rgba8unorm-srgb/max` 42.5 s, 12 cases
