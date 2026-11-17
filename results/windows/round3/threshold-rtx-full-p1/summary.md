# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=results/windows/durations-rtx3060-full.json`; part 1/3 (shards 1–16) of 48 shards, 6 at a time, cap 15 min
- 8451 cases, 18242 pattern runs, 787 of 3238 case groups; wall 28 min 2 s

## Shards that did not finish

- shard 13: killed when the 28 min budget ran out (log results\windows\round3\threshold-rtx-full-p1\shard-13.log)
- shard 14: killed when the 28 min budget ran out (log results\windows\round3\threshold-rtx-full-p1\shard-14.log)
- shard 15: killed when the 28 min budget ran out (log results\windows\round3\threshold-rtx-full-p1\shard-15.log)
- shard 16: killed when the 28 min budget ran out (log results\windows\round3\threshold-rtx-full-p1\shard-16.log)

## Failures (1)

- fixtures/1920x1080 threw: tool fixtures 1920x1080: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 533.1 s, 13 cases
- `source/rgba16float/rgba8unorm/same` 277.8 s, 13 cases
- `source/rgba32float/rgb9e5ufloat/same` 274.8 s, 13 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 266.9 s, 13 cases
- `source/rgba16float/rgba8unorm-srgb/half` 234.1 s, 13 cases
- `source/rgba16float/rgba8unorm/half` 215.2 s, 13 cases
- `integer/sources` 190.1 s, 22 cases
- `source/rgba8unorm/rgba16float/half` 183.2 s, 13 cases
- `source/depth32float/r32float/same` 178.9 s, 13 cases
- `source/rgba16float/rgb9e5ufloat/half` 170.5 s, 13 cases
