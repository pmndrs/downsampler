# GPU suite, passed

- Device: nvidia ampere
- Flags: `--quick --compat --power=high`; 18 shards, 6 at a time, cap 15 min
- 21068 cases, 32452 pattern runs, 1799 of 1799 case groups; wall 9 min 12 s

## Skipped (feature unavailable)

- rg32float storage path (no storage in compatibility mode)
- rg32uint storage path (no storage in compatibility mode)
- rg32sint storage path (no storage in compatibility mode)
- bgra8unorm-srgb (no such texture in compatibility mode)

## Slowest groups

- `downsampler/options` 31.5 s, 8 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 26.1 s, 11 cases
- `integer/sources` 18.8 s, 10 cases
- `region/exact/rgba8unorm-srgb/max` 15.7 s, 8 cases
- `source/rgba16float/rg11b10ufloat/half` 15.1 s, 11 cases
- `source/rgba16float/rgba8unorm-srgb/half` 14.9 s, 11 cases
- `integer/api` 13.0 s, 5 cases
- `source/rgba32float/rgb9e5ufloat/same` 12.9 s, 11 cases
- `downsampler/rgba8unorm-srgb/buffer/L6/custom` 12.5 s, 11 cases
- `source/rgba16float/rgb9e5ufloat/half` 12.5 s, 11 cases
