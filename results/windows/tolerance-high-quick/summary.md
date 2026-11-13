# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --power=high`; 18 shards, 6 at a time, cap 5 min
- 10446 cases, 25548 pattern runs, 731 of 731 case groups; wall 1 min 58 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba16float/rg11b10ufloat/half` 9.6 s, 11 cases
- `fixtures/1129x635` 8.3 s, 0 cases
- `fixtures/960x540` 7.9 s, 0 cases
- `source/rgba16float/rgb9e5ufloat/half` 7.5 s, 11 cases
- `source/rgba32float/rgb9e5ufloat/same` 7.1 s, 11 cases
- `source/depth32float/rg32float/half` 5.7 s, 11 cases
- `packed/buffer-path` 5.1 s, 7 cases
- `source/rgba16float/rgba8unorm/half` 5.0 s, 11 cases
- `source/rgba8unorm/rgba16float/half` 5.0 s, 11 cases
- `source/depth16unorm/r32float/half` 4.5 s, 11 cases
