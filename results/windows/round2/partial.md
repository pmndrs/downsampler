# Partial updates on the RTX 3060 (round 2, 2026-11-15)

`npm run bench -- --compare=partial --skip-validation --power=high --rounds=21 --repeats=64`
([rtx3060-partial-2.json](rtx3060-partial-2.json), log [rtx3060-partial-2.txt](rtx3060-partial-2.txt); VALID, SM median
1702 MHz under load). The first run ([rtx3060-partial.json](rtx3060-partial.json), 11 rounds × 16) got no clock
verdict (the GPU stayed near idle clocks) and its rgba16float numbers are up to 4× off; don't use it.

GPU ms, and the ratio to regenerating the whole chain (paired per round). `rect32`/`rect256`: a 32² or 256² rectangle
of level 0 changed, at a third of the width and height. `layer`: one layer of six (`layers: [ 2, 1 ]`).
`layerRect32`: a 32² rectangle on that one layer. Dispatches are the same as the full chain's in every case.

| Size | Format | full | rect32 | rect256 | layer | layerRect32 |
|---|---|---:|---:|---:|---:|---:|
| 1920×1080 | rgba8unorm | 0.0568 | 0.0174 (0.31) | 0.0180 (0.32) | | |
| 2560×1440 | rgba8unorm | 0.0788 | 0.0138 (0.17) | 0.0141 (0.18) | | |
| 1024² | rgba8unorm | 0.0334 | 0.0098 (0.29) | 0.0102 (0.31) | | |
| 4096² | rgba8unorm | 0.2838 | 0.0089 (0.03) | 0.0094 (0.03) | | |
| 512² × 6 | rgba8unorm | 0.0355 | 0.0072 (0.20) | 0.0168 (0.47) | 0.0094 (0.26) | 0.0069 (0.19) |
| 1024² × 6 | rgba8unorm | 0.1316 | 0.0160 (0.12) | 0.0342 (0.26) | 0.0456 (0.35) | 0.0151 (0.12) |
| 1920×1080 | rgba16float | 0.1031 | 0.0183 (0.18) | 0.0196 (0.19) | | |
| 2560×1440 | rgba16float | 0.1309 | 0.0099 (0.08) | 0.0104 (0.08) | | |
| 1024² | rgba16float | 0.0435 | 0.0072 (0.17) | 0.0077 (0.18) | | |
| 4096² | rgba16float | 0.5524 | 0.0094 (0.02) | 0.0101 (0.02) | | |
| 512² × 6 | rgba16float | 0.0605 | 0.0081 (0.13) | 0.0270 (0.45) | 0.0110 (0.18) | 0.0076 (0.13) |
| 1024² × 6 | rgba16float | 0.2172 | 0.0112 (0.05) | 0.0312 (0.14) | 0.0512 (0.24) | 0.0105 (0.05) |

(Without `layers`, the rectangles cover every layer, hence `rect256` on the 6-layer textures.)

- **A rectangle costs about 7–18 µs, whatever its size up to 256².** That is the floor of the chain's dispatches
  (2–4 here) on this GPU, not the texels written: `rect32` and `rect256` differ by under 1 µs on single-layer
  textures. So the saving grows with the texture: 3 % of a full chain at 4096², 17–31 % at 1080p and 1024², where
  the full chain is already short.
- **One layer of six costs 0.18–0.35 of the cube** (ideal 1/6 = 0.17); a rectangle on one face, 0.05–0.19.
- Use partial updates for large textures, cube faces and texture-painting tools. On a small single texture
  the full chain is already close to the dispatch floor, so a rectangle saves about 2–4×, not 30×.
- Not measured on the M5 or Intel. No change to the library from this.
