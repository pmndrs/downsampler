# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--power=high --durations=C:/Users/denni/Documents/GitHub/pmndrs/denoiser/downsampler/results/windows/f16fast-high-full/durations.json`; 36 shards, 6 at a time, cap 10 min
- 18291 cases, 53600 pattern runs, 1674 of 1674 case groups; wall 16 min 12 s

## Failures (12)

- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- 255x257x1 rgb9e5ufloat buffer average random L6 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- 255x257x1 rgb9e5ufloat buffer tail average random L4 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- 255x257x1 rgb9e5ufloat buffer tail average random L6 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- fixtures/1920x1080 threw: tool fixtures 1920x1080: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- 255x257x1 rgb9e5ufloat buffer average random L1 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/1707x960 threw: tool fixtures 1707x960: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- 255x257x1 rgb9e5ufloat buffer average random L4 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- 255x257x1 rgb9e5ufloat buffer tail average random L5 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- 255x257x1 rgb9e5ufloat buffer average random L5 mip 1: value 12945, actual 0.015625, expected 0.0625, error 0.046875 > 0.0467529296875 (1/65024 values outside, max error 8)
- fixtures/1280x720 threw: tool fixtures 1280x720: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `source/rgba32float/rgb9e5ufloat/same` 94.5 s, 13 cases
- `source/rgba16float/rgba8unorm/same` 84.3 s, 13 cases
- `source/rgba8unorm/rgba16float/half` 56.8 s, 13 cases
- `source/rgba16float/rgb9e5ufloat/half` 55.0 s, 13 cases
- `source/rgba16float/rg11b10ufloat/half` 54.9 s, 13 cases
- `source/rgba16float/rgba8unorm/half` 54.2 s, 13 cases
- `source/depth32float/rg32float/half` 44.4 s, 13 cases
- `downsampler/rg11b10ufloat/storage/L4/spd-average` 42.0 s, 24 cases
- `downsampler/rgb9e5ufloat/buffer/L6/spd-average` 41.7 s, 24 cases
- `source/depth16unorm/r32float/half` 41.5 s, 13 cases
