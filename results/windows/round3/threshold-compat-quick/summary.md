# GPU suite, passed

- Device: nvidia ampere
- Flags: `--quick --compat --power=high`; 18 shards, 6 at a time, cap 15 min
- 21068 cases, 32452 pattern runs, 1799 of 1799 case groups; wall 10 min 54 s

## Skipped (feature unavailable)

- rg32float storage path (no storage in compatibility mode)
- rg32uint storage path (no storage in compatibility mode)
- rg32sint storage path (no storage in compatibility mode)
- bgra8unorm-srgb (no such texture in compatibility mode)

## Slowest groups

- `downsampler/options` 38.9 s, 8 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 33.6 s, 11 cases
- `integer/sources` 22.9 s, 10 cases
- `region/exact/rgba8unorm-srgb/max` 21.2 s, 8 cases
- `downsampler/rgba8unorm-srgb/buffer/L6/custom` 17.6 s, 11 cases
- `source/rgba16float/rgba8unorm-srgb/half` 16.8 s, 11 cases
- `source/rgba16float/rg11b10ufloat/half` 16.7 s, 11 cases
- `region/tail/rgba8unorm-srgb/max` 15.7 s, 7 cases
- `integer/api` 15.2 s, 5 cases
- `source/rgba32float/rgb9e5ufloat/same` 14.4 s, 11 cases
