# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high`; part 1/3 (shards 1–12) of 36 shards, 6 at a time, cap 10 min
- 3564 cases, 7827 pattern runs, 351 of 3225 case groups; wall 20 min 3 s

## Shards that did not finish

- shard 3: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-3.log)
- shard 4: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-4.log)
- shard 6: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-6.log)
- shard 7: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-7.log)
- shard 8: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-8.log)
- shard 9: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-9.log)
- shard 10: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-10.log)
- shard 12: timed out after 10 min (log results\windows\round2\rtx-full-p1\shard-12.log)

## Failures (1)

- fixtures/1920x1080 threw: tool fixtures 1920x1080: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `region/tail/rgba8unorm-srgb/max` 62.3 s, 12 cases
- `integer/rgba32uint/storage/L5/average` 53.9 s, 24 cases
- `integer/rgba32uint/storage/L4/average` 48.5 s, 24 cases
- `integer/rgba32sint/storage/L4/box-average` 47.5 s, 24 cases
- `srgb/near-exact` 46.1 s, 1 cases
- `downsampler/rgba8unorm-srgb/buffer/L5/spd-average-load` 43.4 s, 24 cases
- `region/source/rgba16float/rgba16float/pow2-down/spd` 43.3 s, 15 cases
- `downsampler/rgb10a2unorm/storage/L1/spd-average` 39.2 s, 24 cases
- `region/source/depth32float/r32float/pow2-down` 39.2 s, 15 cases
- `integer/rgba32uint/storage/L5/max` 34.1 s, 24 cases
