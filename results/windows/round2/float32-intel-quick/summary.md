# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 15 min
- 19454 cases, 29569 pattern runs, 1657 of 1863 case groups; wall 11 min 20 s

## Shards that did not finish

- shard 17: exit 1, no report: Backend messages: |  * Device removed reason: DXGI_ERROR_DEVICE_HUNG (0x887A0006) | Node.js v24.13.1

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `srgb/alias` 89.2 s, 1 cases
- `halo/rgba8unorm-srgb/auto` 67.4 s, 21 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 54.8 s, 11 cases
- `downsampler/options` 52.6 s, 9 cases
- `halo/rgba16float/buffer` 37.6 s, 26 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 35.3 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 23.4 s, 8 cases
- `srgb/render` 21.5 s, 1 cases
- `integer/sources` 20.8 s, 10 cases
- `source/rgba16float/rg11b10ufloat/half` 20.5 s, 11 cases
