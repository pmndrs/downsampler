# GPU suite, FAILED

- Device: nvidia ampere
- Flags: `--quick --compat --power=high`; 18 shards, 6 at a time, cap 15 min
- 21064 cases, 32452 pattern runs, 1795 of 1795 case groups; wall 14 min 37 s

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

## Skipped (feature unavailable)

- rg32float storage path (no storage in compatibility mode)
- rg32uint storage path (no storage in compatibility mode)
- rg32sint storage path (no storage in compatibility mode)
- bgra8unorm-srgb (no such texture in compatibility mode)

## Slowest groups

- `downsampler/options` 52.5 s, 8 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 47.1 s, 11 cases
- `source/rgba16float/rgba8unorm-srgb/half` 34.7 s, 11 cases
- `region/exact/rgba8unorm-srgb/max` 27.3 s, 8 cases
- `source/rgba16float/rg11b10ufloat/half` 25.3 s, 11 cases
- `source/rgba32float/rgb9e5ufloat/same` 25.3 s, 11 cases
- `integer/sources` 24.5 s, 10 cases
- `source/rgba8unorm/rgba16float/half` 21.2 s, 11 cases
- `integer/api` 21.1 s, 5 cases
- `source/rgba16float/rgb9e5ufloat/half` 21.0 s, 11 cases
