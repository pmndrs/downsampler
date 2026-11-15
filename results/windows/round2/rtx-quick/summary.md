# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --power=high`; 18 shards, 6 at a time, cap 10 min
- 10599 cases, 16755 pattern runs, 920 of 1850 case groups; wall 14 min 1 s

## Shards that did not finish

- shard 10: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 11: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 12: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 13: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 14: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 15: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 16: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 17: exit 1, no report:   stderr: '' | } | Node.js v24.13.1
- shard 18: exit 1, no report:   stderr: '' | } | Node.js v24.13.1

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `region/box/rgba8unorm/average` 81.3 s, 8 cases
- `source/depth32float/rg32float/half` 45.5 s, 11 cases
- `fixtures/1129x635` 45.2 s, 0 cases
- `source/rgba16float/rgb9e5ufloat/half` 35.5 s, 11 cases
- `region/box/rgba8unorm-srgb/average` 31.2 s, 8 cases
- `integer/sources` 28.1 s, 10 cases
- `srgb/near-exact` 27.6 s, 1 cases
- `fixtures/960x540` 25.8 s, 0 cases
- `region/spd/rgba8unorm/average` 24.4 s, 8 cases
- `source/rgba16float/rg11b10ufloat/half` 22.0 s, 11 cases
