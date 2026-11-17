# GPU suite, passed

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 3/3 (shards 33–48) of 48 shards, 6 at a time, cap 15 min
- 12226 cases, 27138 pattern runs, 1091 of 3238 case groups; wall 18 min 33 s

## Slowest groups

- `srgb/alias` 97.5 s, 1 cases
- `region/spd/rgba16float/average` 47.3 s, 12 cases
- `box/resample/depth32float/rg32float` 45.4 s, 16 cases
- `region/box/rgba16float/average` 44.7 s, 12 cases
- `integer/rgba32sint/storage/L4/box-average` 43.4 s, 24 cases
- `integer/rgba32uint/storage/L4/box-average` 43.3 s, 24 cases
- `integer/rgba32uint/storage/L6/box-average` 42.9 s, 24 cases
- `integer/rgba32sint/storage/L6/box-average` 42.8 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 41.9 s, 24 cases
- `srgb/render` 41.9 s, 1 cases
