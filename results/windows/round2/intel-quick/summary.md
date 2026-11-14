# GPU suite, FAILED

- Device: intel gen-12lp
- Flags: `--quick --power=low`; 18 shards, 6 at a time, cap 10 min
- 18097 cases, 27118 pattern runs, 1544 of 1850 case groups; wall 25 min 34 s

## Shards that did not finish

- shard 1: exit 1, no report: Backend messages: |  * Device removed reason: DXGI_ERROR_DEVICE_HUNG (0x887A0006) | Node.js v24.13.1
- shard 5: exit 1, no report:                       ^ | Error: The browser is already running for C:\Users\denni\AppData\Local\Temp\puppeteer_dev_chrome_profile-oKO0Mq. Use a different `userDataDir` or stop the running browser first. | Node.js v24.13.1
- shard 7: exit 1, no report:                       ^ | TimeoutError: Timed out after 30000 ms while waiting for the WS endpoint URL to appear in stdout! | Node.js v24.13.1

## Failures (2)

- fixtures/1129x635 threw: tool fixtures 1129x635: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp
- fixtures/960x540 threw: tool fixtures 960x540: oracle checks failed: fsr/luma-pyramid, generator/luma-pyramid-clamp

## Slowest groups

- `region/box/rgba16float/average` 86.7 s, 8 cases
- `source/rgba8unorm-srgb/rgba8unorm-srgb/same` 74.6 s, 11 cases
- `source/rgba32float/rgb9e5ufloat/same` 67.5 s, 11 cases
- `region/exact/rgba16float/max` 59.9 s, 8 cases
- `source/bgra8unorm-srgb/bgra8unorm-srgb/half` 57.9 s, 11 cases
- `region/box/rgba8unorm/max` 57.9 s, 8 cases
- `source/rgba16float/rgba8unorm-srgb/half` 55.8 s, 11 cases
- `source/depth32float/r32float/same` 54.1 s, 11 cases
- `source/rgba16float/rg11b10ufloat/half` 50.5 s, 11 cases
- `region/spd/rgba16float/average` 43.1 s, 8 cases
