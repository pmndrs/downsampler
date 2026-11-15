# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high`; part 2/3 (shards 13–24) of 36 shards, 6 at a time, cap 10 min
- 6137 cases, 13598 pattern runs, 544 of 3225 case groups; wall 20 min 2 s

## Shards that did not finish

- shard 13: timed out after 10 min (log results\windows\round2\rtx-full-p2\shard-13.log)
- shard 15: timed out after 10 min (log results\windows\round2\rtx-full-p2\shard-15.log)
- shard 16: timed out after 10 min (log results\windows\round2\rtx-full-p2\shard-16.log)
- shard 17: timed out after 10 min (log results\windows\round2\rtx-full-p2\shard-17.log)
- shard 20: timed out after 10 min (log results\windows\round2\rtx-full-p2\shard-20.log)
- shard 24: timed out after 10 min (log results\windows\round2\rtx-full-p2\shard-24.log)

## Slowest groups

- `box/resample/rgba16float/rgba8unorm` 76.9 s, 16 cases
- `integer/rgba32uint/storage/L1/box-average` 66.7 s, 24 cases
- `region/exact/rgba16float/max` 57.2 s, 12 cases
- `downsampler/rgb9e5ufloat/buffer/L5/box-average-load` 53.0 s, 24 cases
- `region/box/rgba8unorm/average` 51.2 s, 12 cases
- `integer/rgba32uint/storage/L6/average` 45.5 s, 24 cases
- `downsampler/rg11b10ufloat/storage/L1/spd-average` 42.9 s, 24 cases
- `integer/rgba8uint/storage/L1/average` 42.6 s, 24 cases
- `downsampler/rgba16float/storage/L5/spd-average-load` 39.9 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L5/spd-average-load` 39.3 s, 24 cases
