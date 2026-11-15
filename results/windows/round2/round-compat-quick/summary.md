# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --compat --power=high`; 18 shards, 6 at a time, cap 15 min
- 21064 cases, 32452 pattern runs, 1795 of 1795 case groups; wall 9 min 5 s

## Failures (35)

- rgba8unorm-srgb 255x257x1 -> rgba16float half average random L1 spd mip 0: value 12494, actual 0.912109375, expected 0.91552734375, error 0.00341796875 > 0.0033509372733533382 (2/65024 values outside, max error 0.005859375)
- rgba8unorm-srgb 255x257x1 -> rgba16float half average random L4 spd mip 0: value 12494, actual 0.912109375, expected 0.91552734375, error 0.00341796875 > 0.0033509372733533382 (2/65024 values outside, max error 0.005859375)
- rgba8unorm-srgb 255x257x1 -> rgba16float half average random L5 spd mip 0: value 12494, actual 0.912109375, expected 0.91552734375, error 0.00341796875 > 0.0033509372733533382 (2/65024 values outside, max error 0.005859375)
- rgba8unorm-srgb 255x257x1 -> rgba16float half average random L6 spd mip 0: value 12494, actual 0.912109375, expected 0.91552734375, error 0.00341796875 > 0.0033509372733533382 (2/65024 values outside, max error 0.005859375)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L1 exact mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L1 spd mip 0: value 12546, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (704/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L1 box mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L4 exact mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L4 spd mip 0: value 12546, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (704/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L4 spd mip 1: value 3202, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (176/5184 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L4 spd mip 2: value 834, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (40/1248 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L4 spd mip 3: value 226, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (8/288 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L4 box mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 exact mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 spd mip 0: value 12546, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (704/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 spd mip 1: value 3202, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (176/5184 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 spd mip 2: value 834, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (40/1248 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 spd mip 3: value 226, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (8/288 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 spd mip 4: value 66, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (2/72 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L5 box mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 exact mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 spd mip 0: value 12546, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (704/20736 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 spd mip 1: value 3202, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (176/5184 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 spd mip 2: value 834, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (40/1248 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 spd mip 3: value 226, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (8/288 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 spd mip 4: value 66, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.0029252564828166215 (2/72 values outside, max error 0.0029296875)
- rgba8unorm-srgb 192x108x1 -> rgba16float half average tiles L6 box mip 0: value 12934, actual 0.9765625, expected 0.9736328125, error 0.0029296875 > 0.002921362701803446 (651/20736 values outside, max error 0.0029296875)
- f16 ties spd mip 1: value 0, actual 1.90625, expected 4.56640625, error 2.66015625 > 0.013717484375840286 (391794/393216 values outside, max error 9.83203125)
- f16 ties spd mip 2: value 0, actual 4.56640625, expected 3.9375, error 0.62890625 > 0.01182824609450472 (97554/98304 values outside, max error 5.921875)
- f16 ties spd mip 3: value 0, actual 3.9375, expected 3.544921875, error 0.392578125 > 0.010648945312241268 (24189/24576 values outside, max error 2.6953125)
- f16 ties spd mip 4: value 0, actual 3.544921875, expected 3.603515625, error 0.05859375 > 0.010824962890801544 (5931/6144 values outside, max error 1.39453125)
- f16 ties spd mip 5: value 0, actual 3.603515625, expected 3.9296875, error 0.326171875 > 0.011804777344266769 (1438/1536 values outside, max error 0.66796875)
- f16 ties spd mip 6: value 1, actual 3.80078125, expected 3.96484375, error 0.1640625 > 0.011910388672113185 (338/384 values outside, max error 0.29296875)
- f16 ties spd mip 7: value 0, actual 3.9296875, expected 3.98046875, error 0.05078125 > 0.011957326172589091 (70/96 values outside, max error 0.1640625)
- f16 ties spd mip 8: value 2, actual 3.94921875, expected 3.96875, error 0.01953125 > 0.011922123047232162 (14/24 values outside, max error 0.052734375)

## Skipped (feature unavailable)

- rg32float storage path (no storage in compatibility mode)
- rg32uint storage path (no storage in compatibility mode)
- rg32sint storage path (no storage in compatibility mode)
- bgra8unorm-srgb (no such texture in compatibility mode)

## Slowest groups

- `downsampler/options` 31.1 s, 8 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 28.4 s, 11 cases
- `source/rgba16float/rgba8unorm-srgb/half` 15.6 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 15.4 s, 8 cases
- `integer/sources` 15.1 s, 10 cases
- `source/rgba16float/rg11b10ufloat/half` 14.6 s, 11 cases
- `source/rgba32float/rgb9e5ufloat/same` 12.8 s, 11 cases
- `source/rgba16float/rgb9e5ufloat/half` 12.4 s, 11 cases
- `region/tail/rgba8unorm-srgb/max` 12.3 s, 7 cases
- `downsampler/rgba8unorm-srgb/buffer/L6/custom` 11.7 s, 11 cases
