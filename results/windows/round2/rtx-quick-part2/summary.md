# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --power=high`; part 2/2 (shards 10–18) of 18 shards, 6 at a time, cap 10 min
- 11022 cases, 16533 pattern runs, 930 of 1850 case groups; wall 11 min 46 s

## Failures (27)

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

## Slowest groups

- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 61.8 s, 11 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 46.5 s, 11 cases
- `source/rgba16float/rgba8unorm-srgb/half` 46.0 s, 11 cases
- `downsampler/rgba8unorm-srgb/buffer/L4/custom` 32.7 s, 11 cases
- `downsampler/bgra8unorm-srgb/buffer/L6/max` 31.8 s, 11 cases
- `region/spd/rgba16float/average` 31.5 s, 8 cases
- `downsampler/bgra8unorm-srgb/buffer/L4/min` 28.0 s, 11 cases
- `region/exact/rgba8unorm-srgb/average` 26.8 s, 8 cases
- `downsampler/rgba8unorm-srgb/buffer/L6/box-max` 25.8 s, 11 cases
- `integer/rgba32sint/storage/L6/box-average` 23.8 s, 21 cases
