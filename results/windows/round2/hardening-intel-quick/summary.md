# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 20 min
- 20635 cases, 31557 pattern runs, 1757 of 1859 case groups; wall 11 min 26 s

## Shards that did not finish

- shard 1: exit 1, no report: Backend messages: |  * Device removed reason: DXGI_ERROR_DEVICE_HUNG (0x887A0006) | Node.js v24.13.1

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `halo/rgba8unorm-srgb/auto` 86.4 s, 21 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 61.8 s, 11 cases
- `downsampler/options` 56.2 s, 9 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 44.7 s, 11 cases
- `halo/rgba16float/buffer` 41.0 s, 26 cases
- `source/rgba16float/rgba8unorm-srgb/half` 35.8 s, 11 cases
- `halo/rgb9e5ufloat/auto` 25.4 s, 18 cases
- `region/exact/rgba8unorm-srgb/max` 23.4 s, 8 cases
- `region/exact/rgba16float/max` 21.0 s, 8 cases
- `region/tail/rgba8unorm-srgb/max` 20.0 s, 7 cases
