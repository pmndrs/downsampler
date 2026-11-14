# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 10 min
- 20613 cases, 31357 pattern runs, 1755 of 1857 case groups; wall 10 min 19 s

## Shards that did not finish

- shard 1: exit 1, no report: Backend messages: |  * Device removed reason: DXGI_ERROR_DEVICE_HUNG (0x887A0006) | Node.js v24.13.1

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 74.9 s, 21 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 55.0 s, 11 cases
- `halo/rgba16float/buffer` 37.6 s, 26 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 37.1 s, 11 cases
- `source/rgba16float/rgba8unorm-srgb/half` 31.7 s, 11 cases
- `region/tail/rgba8unorm-srgb/max` 25.4 s, 7 cases
- `region/exact/rgba8unorm-srgb/max` 20.5 s, 8 cases
- `region/exact/rgba16float/max` 19.6 s, 8 cases
- `integer/sources` 16.6 s, 10 cases
- `halo/rgb9e5ufloat/auto` 16.5 s, 18 cases
