# GPU suite, passed

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 3/3 (shards 33–48) of 48 shards, 6 at a time, cap 15 min
- 12295 cases, 26960 pattern runs, 1091 of 3235 case groups; wall 16 min 36 s

## Slowest groups

- `region/spd/rgba16float/average` 45.4 s, 12 cases
- `box/resample/depth32float/rg32float` 42.6 s, 16 cases
- `region/box/rgba16float/average` 41.4 s, 12 cases
- `integer/rgba32sint/storage/L1/average` 41.0 s, 24 cases
- `integer/rgba32uint/storage/L6/box-average` 40.6 s, 24 cases
- `integer/rgba32sint/storage/L6/box-average` 40.3 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 40.0 s, 24 cases
- `region/tail/rgba16float/average` 39.7 s, 12 cases
- `integer/rgba32sint/storage/L4/box-average` 38.7 s, 24 cases
- `region/source/rgba16float/rgba16float/pow2-down/spd` 38.4 s, 15 cases
