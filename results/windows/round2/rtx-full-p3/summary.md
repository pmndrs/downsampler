# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high`; part 3/3 (shards 25–36) of 36 shards, 6 at a time, cap 10 min
- 7054 cases, 15880 pattern runs, 631 of 3225 case groups; wall 19 min 12 s

## Shards that did not finish

- shard 25: timed out after 10 min (log results\windows\round2\rtx-full-p3\shard-25.log)
- shard 26: timed out after 10 min (log results\windows\round2\rtx-full-p3\shard-26.log)
- shard 28: timed out after 10 min (log results\windows\round2\rtx-full-p3\shard-28.log)
- shard 32: timed out after 10 min (log results\windows\round2\rtx-full-p3\shard-32.log)
- shard 33: timed out after 10 min (log results\windows\round2\rtx-full-p3\shard-33.log)

## Slowest groups

- `source/rgba16float/rgba8unorm/half` 93.1 s, 13 cases
- `source/depth32float/rg32float/half` 70.6 s, 13 cases
- `integer/rgba32uint/storage/L6/box-average` 48.9 s, 24 cases
- `integer/rgba32uint/storage/L4/box-average` 46.0 s, 24 cases
- `downsampler/rgba8unorm-srgb/buffer/L5/spd-average` 44.7 s, 24 cases
- `integer/rgba32sint/storage/L1/box-average` 42.8 s, 24 cases
- `region/box/rgba16float/average` 42.5 s, 12 cases
- `box/resample/rgba16float/rg11b10ufloat` 41.3 s, 16 cases
- `downsampler/bgra8unorm-srgb/buffer/L5/max` 38.8 s, 24 cases
- `downsampler/rgba8unorm/storage/L6/spd-average` 38.2 s, 24 cases
