# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 12 shards, 4 at a time, cap 5 min
- 10446 cases, 25548 pattern runs, 731 of 731 case groups; wall 2 min 49 s

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba16float/rg11b10ufloat/half` 9.4 s, 11 cases
- `fixtures/960x540` 8.0 s, 0 cases
- `fixtures/1129x635` 7.7 s, 0 cases
- `source/rgba16float/rgb9e5ufloat/half` 7.4 s, 11 cases
- `source/rgba32float/rgb9e5ufloat/same` 7.0 s, 11 cases
- `source/rgba16float/rgba8unorm/half` 5.8 s, 11 cases
- `source/rgba16float/rgba8unorm/same` 5.7 s, 11 cases
- `source/rgba8unorm/rgba16float/half` 5.7 s, 11 cases
- `source/depth32float/rg32float/half` 5.5 s, 11 cases
- `source/depth16unorm/r32float/half` 5.4 s, 11 cases
