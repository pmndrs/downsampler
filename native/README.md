# Native harnesses

Native D3D12, D3D11 and Vulkan mip-chain timings on the same GPU as the WebGPU runs: a ceiling and a baseline for the WebGPU numbers (docs/WINDOWS-HANDOFF.md §3). They are measurement tools only. Nothing here is part of the library, and no upstream code is vendored: `fetch-deps.ps1` checks out the pinned commits into `native/.deps/` (gitignored).

| Harness | API | What it times | Upstream |
|---|---|---|---|
| [`d3d12-mips/`](d3d12-mips/) | D3D12, D3D11 | AMD FidelityFX SPD (its own `ffx_spd.h`): atomic single dispatch and 6 levels per dispatch, quad-wave or LDS, load or linear sampler; a render-pass chain; D3D11 `GenerateMips` | FidelityFX SDK `7539ae8` (FSR 3.1.6) |
| [`vk-mips/`](vk-mips/) | Vulkan | NVIDIA nvpro_pyramid (fast + general pipelines); `vkCmdBlitImage` chain | vk_compute_mipmaps `6c248f5` |

Both use the same input (uniform random, xorshift32 seed 1337), the same sizes (960×540, 1129×635, 1280×720, 1707×960, 1920×1080, 2560×1440, 1024², 4096²), the same formats (rgba8unorm and rgba16float average; r32float max for Hi-Z) and the same timing: GPU timestamps around 16 back-to-back chains per submit, variants rotated each round, median over 128 rounds, after 8 warm-up rounds. That is test/bench/timing.js's method without its marker passes. Shared code is in [`common/harness.hpp`](common/harness.hpp).

## Build (Windows)

Needs Visual Studio 2022 (C++), CMake ≥ 3.20, the Windows SDK (its `dxc.exe` signs DXIL; the Vulkan SDK's doesn't) and the Vulkan SDK (1.3.283 used).

```powershell
pwsh native/fetch-deps.ps1
cmake -S native/d3d12-mips -B native/d3d12-mips/build -G "Visual Studio 17 2022" -A x64
cmake --build native/d3d12-mips/build --config Release
cmake -S native/vk-mips -B native/vk-mips/build -G "Visual Studio 17 2022" -A x64
cmake --build native/vk-mips/build --config Release
```

If configure fails with "Could not find toolchain file … vcpkg.cmake", a global `CMAKE_TOOLCHAIN_FILE` points at a missing vcpkg: clear it for the build (`env -u CMAKE_TOOLCHAIN_FILE cmake …` in Git Bash).

## Run

```
native/d3d12-mips/build/Release/d3d12-mips.exe --power=high --output=results/windows/native-d3d12-rtx3060.json
native/vk-mips/build/Release/vk-mips.exe --power=high --output=results/windows/native-vk-rtx3060.json
```

Options (both): `--power=high|low` (D3D12: DXGI GPU preference; Vulkan: discrete or integrated; `vk-mips --gpu=<name part>` overrides), `--sizes=WxH,…`, `--formats=rgba8unorm,rgba16float,r32float`, `--rounds=128`, `--repeats=16`, `--warmup=8`, `--only=<name parts>`, `--no-timing`, `--dump=<folder> [--dump-sizes=1129x635]`. Debugging: `d3d12-mips --debug` (D3D12 debug layer, messages on stderr), `vk-mips --validation`.

`results/windows/run-benches.sh` runs both with the WebGPU benchmarks in one session under the shared GPU lock.

## Semantics check

`--dump` writes every level of every variant as float32. `d3d12-mips/check-dump.mjs` diffs them against this library's CPU oracles, starting from the dumped level 0:

```
npm run build
native/d3d12-mips/build/Release/d3d12-mips.exe --no-timing --dump=tmp/dump
node native/d3d12-mips/check-dump.mjs tmp/dump --json=results/windows/native-semantics.json
```

| Variant | Oracle |
|---|---|
| `spd-*` | `spdChainReference( …, levelsPerDispatch 6 )`: our `oddSizes: 'spd'` |
| `render*`, `d3d11-generatemips`, `blit` | `'exact'` level by level (`reduceReference`) |
| `nvpro-*` | `'box'` level by level: nvpro's area weights, defined in the script (the library's `'box'` is planned for M2.5) |

It reports the largest difference in storage steps and the values outside the library's `tolerance()`.
