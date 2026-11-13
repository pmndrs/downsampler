# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --power=high`; 18 shards, 6 at a time, cap 5 min
- 10446 cases, 25548 pattern runs, 731 of 731 case groups; wall 2 min 16 s

## Failures (56)

- 65x129x1 rgba16float storage average random L1 mip 1: value 18, actual 4.82421875, expected 4.8046875, error 0.01953125 > 0.0144140625 (2356/8192 values outside, max error 0.05078125)
- 255x257x1 rgba16float storage average random L1 mip 1: value 12327, actual 0.92724609375, expected 0.96435546875, error 0.037109375 > 0.03560791015625 (16/65024 values outside, max error 0.0703125)
- 65x129x1 rg16float storage average random L6 mip 1: value 16, actual 6.26171875, expected 6.28125, error 0.01953125 > 0.01884375 (1248/8192 values outside, max error 0.05078125)
- 65x129x1 rgb9e5ufloat buffer average random L1 mip 1: value 20, actual 180, expected 181.5, error 1.5 > 1.0001220703125 (570/8192 values outside, max error 6.5)
- 65x129x1 rg11b10ufloat storage average random L6 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- 65x129x1 r16float storage average random L5 mip 1: value 28, actual 7.5703125, expected 7.546875, error 0.0234375 > 0.022640625 (595/8192 values outside, max error 0.046875)
- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- 65x129x1 r16float storage average random L4 mip 1: value 28, actual 7.5703125, expected 7.546875, error 0.0234375 > 0.022640625 (595/8192 values outside, max error 0.046875)
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- 65x129x1 rg16float storage average random L5 mip 1: value 16, actual 6.26171875, expected 6.28125, error 0.01953125 > 0.01884375 (1248/8192 values outside, max error 0.05078125)
- rgba16float 65x129x1 -> rg11b10ufloat half average random L1 exact mip 0: value 80, actual 0.390625, expected 0.37890625, error 0.01171875 > 0.0079345703125 (99/8192 values outside, max error 0.25)
- rgba16float 65x129x1 -> rg11b10ufloat half average random L4 exact mip 0: value 80, actual 0.390625, expected 0.37890625, error 0.01171875 > 0.0079345703125 (99/8192 values outside, max error 0.25)
- rgba16float 65x129x1 -> rg11b10ufloat half average random L5 exact mip 0: value 80, actual 0.390625, expected 0.37890625, error 0.01171875 > 0.0079345703125 (99/8192 values outside, max error 0.25)
- rgba16float 65x129x1 -> rg11b10ufloat half average random L6 exact mip 0: value 80, actual 0.390625, expected 0.37890625, error 0.01171875 > 0.0079345703125 (99/8192 values outside, max error 0.25)
- 65x129x1 rg11b10ufloat storage average random L1 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- 65x129x1 rg11b10ufloat buffer average random L4 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- rgba32float 65x129x1 -> rgb9e5ufloat same average random L1 exact mip 1: value 460, actual 0.4296875, expected 0.4140625, error 0.015625 > 0.0079345703125 (50/8192 values outside, max error 0.046875)
- rgba32float 65x129x1 -> rgb9e5ufloat same average random L4 exact mip 1: value 460, actual 0.4296875, expected 0.4140625, error 0.015625 > 0.0079345703125 (50/8192 values outside, max error 0.046875)
- rgba32float 65x129x1 -> rgb9e5ufloat same average random L5 exact mip 1: value 460, actual 0.4296875, expected 0.4140625, error 0.015625 > 0.0079345703125 (50/8192 values outside, max error 0.046875)
- rgba32float 65x129x1 -> rgb9e5ufloat same average random L6 exact mip 1: value 460, actual 0.4296875, expected 0.4140625, error 0.015625 > 0.0079345703125 (50/8192 values outside, max error 0.046875)
- 65x129x1 rg16float storage average random L1 mip 1: value 16, actual 6.26171875, expected 6.28125, error 0.01953125 > 0.01884375 (1248/8192 values outside, max error 0.05078125)
- rgba16float 65x129x1 -> rgba8unorm half average random L1 exact mip 0: value 35, actual 0.6901960968971252, expected 0.7019608020782471, error 0.011764705181121826 > 0.00788235294117647 (176/8192 values outside, max error 0.039215683937072754)
- rgba16float 65x129x1 -> rgba8unorm half average random L4 exact mip 0: value 35, actual 0.6901960968971252, expected 0.7019608020782471, error 0.011764705181121826 > 0.00788235294117647 (176/8192 values outside, max error 0.039215683937072754)
- rgba16float 65x129x1 -> rgba8unorm half average random L5 exact mip 0: value 35, actual 0.6901960968971252, expected 0.7019608020782471, error 0.011764705181121826 > 0.00788235294117647 (176/8192 values outside, max error 0.039215683937072754)
- rgba16float 65x129x1 -> rgba8unorm half average random L6 exact mip 0: value 35, actual 0.6901960968971252, expected 0.7019608020782471, error 0.011764705181121826 > 0.00788235294117647 (176/8192 values outside, max error 0.039215683937072754)
- rgba16float 255x257x1 -> rgba8unorm half average random L1 exact mip 0: value 31650, actual 0.4901960790157318, expected 0.5411764979362488, error 0.05098041892051697 > 0.05060696231617647 (2/65024 values outside, max error 0.05490196496248245)
- rgba16float 255x257x1 -> rgba8unorm half average random L4 exact mip 0: value 31650, actual 0.4901960790157318, expected 0.5411764979362488, error 0.05098041892051697 > 0.05060696231617647 (2/65024 values outside, max error 0.05490196496248245)
- rgba16float 255x257x1 -> rgba8unorm half average random L5 exact mip 0: value 31650, actual 0.4901960790157318, expected 0.5411764979362488, error 0.05098041892051697 > 0.05060696231617647 (2/65024 values outside, max error 0.05490196496248245)
- rgba16float 255x257x1 -> rgba8unorm half average random L6 exact mip 0: value 31650, actual 0.4901960790157318, expected 0.5411764979362488, error 0.05098041892051697 > 0.05060696231617647 (2/65024 values outside, max error 0.05490196496248245)
- 65x129x1 rgba16float storage average random L6 mip 1: value 18, actual 4.82421875, expected 4.8046875, error 0.01953125 > 0.0144140625 (2356/8192 values outside, max error 0.05078125)
- 255x257x1 rgba16float storage average random L6 mip 1: value 12327, actual 0.92724609375, expected 0.96435546875, error 0.037109375 > 0.03560791015625 (16/65024 values outside, max error 0.0703125)
- 65x129x1 rg11b10ufloat storage average random L4 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- rgba8unorm 65x129x1 -> rgba16float half average random L1 exact mip 0: value 37, actual 0.4755859375, expected 0.47314453125, error 0.00244140625 > 0.0020002256128937006 (54/8192 values outside, max error 0.0029296875)
- rgba8unorm 65x129x1 -> rgba16float half average random L4 exact mip 0: value 37, actual 0.4755859375, expected 0.47314453125, error 0.00244140625 > 0.0020002256128937006 (54/8192 values outside, max error 0.0029296875)
- rgba8unorm 65x129x1 -> rgba16float half average random L5 exact mip 0: value 37, actual 0.4755859375, expected 0.47314453125, error 0.00244140625 > 0.0020002256128937006 (54/8192 values outside, max error 0.0029296875)
- rgba8unorm 65x129x1 -> rgba16float half average random L6 exact mip 0: value 37, actual 0.4755859375, expected 0.47314453125, error 0.00244140625 > 0.0020002256128937006 (54/8192 values outside, max error 0.0029296875)
- 65x129x1 rg11b10ufloat buffer average random L1 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- 65x129x1 rgba16float storage average random L5 mip 1: value 18, actual 4.82421875, expected 4.8046875, error 0.01953125 > 0.0144140625 (2356/8192 values outside, max error 0.05078125)
- 255x257x1 rgba16float storage average random L5 mip 1: value 12327, actual 0.92724609375, expected 0.96435546875, error 0.037109375 > 0.03560791015625 (16/65024 values outside, max error 0.0703125)
- 65x129x1 rg11b10ufloat buffer average random L6 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- rgba16float 65x129x1 -> rgb9e5ufloat half average random L1 exact mip 0: value 100, actual 3.109375, expected 3.078125, error 0.03125 > 0.0157470703125 (104/8192 values outside, max error 0.0625)
- rgba16float 65x129x1 -> rgb9e5ufloat half average random L4 exact mip 0: value 100, actual 3.109375, expected 3.078125, error 0.03125 > 0.0157470703125 (104/8192 values outside, max error 0.0625)
- rgba16float 65x129x1 -> rgb9e5ufloat half average random L5 exact mip 0: value 100, actual 3.109375, expected 3.078125, error 0.03125 > 0.0157470703125 (104/8192 values outside, max error 0.0625)
- rgba16float 65x129x1 -> rgb9e5ufloat half average random L6 exact mip 0: value 100, actual 3.109375, expected 3.078125, error 0.03125 > 0.0157470703125 (104/8192 values outside, max error 0.0625)
- rgba16float 255x257x1 -> rgb9e5ufloat half average random L1 exact mip 0: value 34994, actual 0.24365234375, expected 0.2265625, error 0.01708984375 > 0.0169677734375 (1/65024 values outside, max error 0.09375)
- rgba16float 255x257x1 -> rgb9e5ufloat half average random L4 exact mip 0: value 34994, actual 0.24365234375, expected 0.2265625, error 0.01708984375 > 0.0169677734375 (1/65024 values outside, max error 0.09375)
- rgba16float 255x257x1 -> rgb9e5ufloat half average random L5 exact mip 0: value 34994, actual 0.24365234375, expected 0.2265625, error 0.01708984375 > 0.0169677734375 (1/65024 values outside, max error 0.09375)
- rgba16float 255x257x1 -> rgb9e5ufloat half average random L6 exact mip 0: value 34994, actual 0.24365234375, expected 0.2265625, error 0.01708984375 > 0.0169677734375 (1/65024 values outside, max error 0.09375)
- 65x129x1 rgb9e5ufloat buffer average random L6 mip 1: value 20, actual 180, expected 181.5, error 1.5 > 1.0001220703125 (570/8192 values outside, max error 6.5)
- 65x129x1 rg11b10ufloat storage average random L5 mip 1: value 182, actual 0.05859375, expected 0.0556640625, error 0.0029296875 > 0.0020751953125 (264/8192 values outside, max error 32)
- 65x129x1 rgba16float storage average random L4 mip 1: value 18, actual 4.82421875, expected 4.8046875, error 0.01953125 > 0.0144140625 (2356/8192 values outside, max error 0.05078125)
- 255x257x1 rgba16float storage average random L4 mip 1: value 12327, actual 0.92724609375, expected 0.96435546875, error 0.037109375 > 0.03560791015625 (16/65024 values outside, max error 0.0703125)
- 65x129x1 rg16float storage average random L4 mip 1: value 16, actual 6.26171875, expected 6.28125, error 0.01953125 > 0.01884375 (1248/8192 values outside, max error 0.05078125)
- 65x129x1 r16float storage average random L6 mip 1: value 28, actual 7.5703125, expected 7.546875, error 0.0234375 > 0.022640625 (595/8192 values outside, max error 0.046875)
- 65x129x1 rgb9e5ufloat buffer average random L4 mip 1: value 20, actual 180, expected 181.5, error 1.5 > 1.0001220703125 (570/8192 values outside, max error 6.5)
- 65x129x1 r16float storage average random L1 mip 1: value 28, actual 7.5703125, expected 7.546875, error 0.0234375 > 0.022640625 (595/8192 values outside, max error 0.046875)

## Slowest groups

- `source/rgba16float/rg11b10ufloat/half` 10.5 s, 11 cases
- `fixtures/1129x635` 10.0 s, 0 cases
- `fixtures/960x540` 9.5 s, 0 cases
- `source/rgba32float/rgb9e5ufloat/same` 8.3 s, 11 cases
- `source/rgba16float/rgb9e5ufloat/half` 7.8 s, 11 cases
- `source/rgba8unorm/rgba16float/half` 7.1 s, 11 cases
- `packed/buffer-path` 5.9 s, 7 cases
- `source/depth32float/rg32float/half` 5.9 s, 11 cases
- `downsampler/options` 5.6 s, 6 cases
- `downsampler/rgba16float/storage/L4/max` 5.3 s, 21 cases
