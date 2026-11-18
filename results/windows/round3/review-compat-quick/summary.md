# GPU suite, passed

- Device: nvidia ampere
- Flags: `--quick --power=high --compat`; 18 shards, 6 at a time, cap 15 min
- 21074 cases, 32452 pattern runs, 1800 of 1800 case groups; wall 9 min 10 s

## Skipped (feature unavailable)

- rg32float storage path (no storage in compatibility mode)
- rg32uint storage path (no storage in compatibility mode)
- rg32sint storage path (no storage in compatibility mode)
- bgra8unorm-srgb (no such texture in compatibility mode)

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 38.2 s, 11 cases
- `downsampler/options` 32.3 s, 9 cases
- `source/rgba16float/rgba8unorm-srgb/half` 20.1 s, 11 cases
- `integer/sources` 17.5 s, 10 cases
- `region/exact/rgba8unorm-srgb/max` 15.0 s, 8 cases
- `source/rgba16float/rg11b10ufloat/half` 14.1 s, 11 cases
- `source/rgba16float/rgb9e5ufloat/half` 12.3 s, 11 cases
- `srgb/render` 11.3 s, 1 cases
- `region/box/rgba8unorm-srgb/average` 10.8 s, 8 cases
- `downsampler/rgba8unorm-srgb/buffer/L6/custom` 10.2 s, 11 cases
