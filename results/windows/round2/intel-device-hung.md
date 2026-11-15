# Intel UHD: DXGI_ERROR_DEVICE_HUNG with two devices writing rgb9e5ufloat (round 2, 2026-11-14)

Intel UHD Graphics (gen-12lp, driver 31.0.101.5333), Chrome 148 headless, D3D12, `--power=low`.

## Symptom

Intel quick suite, shard 1/18: the suite's device is lost with `DXGI_ERROR_DEVICE_HUNG`, also with the shard
running alone (twice). Dawn reports it at the next resource creation, so the error names a later group; the
suite page now logs the moment the device is lost (`device lost after: …`).

## Minimal reproduction (reproducible every time, 5 runs)

```sh
node test/run.js --quick --power=low --keep-going --verbose --groups=@results/windows/round2/intel-hang-groups.txt
```

The two case groups: `source/rgba16float/rgb9e5ufloat/half` (the suite's device writes rgb9e5ufloat levels
through the buffer path), then `packed/buffer-path`, which opens a **second device** with default limits and
runs a 256×128 rgb9e5ufloat chain on it (buffer path, 6 + 2 levels).

| Variant | Device lost |
|---|---|
| Both groups | yes (5/5) |
| `packed/buffer-path` alone, or the whole `--only=packed` part (948 cases) | no |
| Shard 1's other 25 preceding groups + `packed/buffer-path` | no |
| Without the second device's block | no |
| Second device: rg11b10ufloat only | no |
| Second device: rgb9e5ufloat only | yes |
| The same rgb9e5ufloat block on the suite's own device | no |
| `results/probes/intel-buffer-path-hang.mjs` (two devices, rgb9e5ufloat and others on the buffer path, no separate source) | no |

So it takes two live devices on the Intel GPU, both having written rgb9e5ufloat levels through
`copyBufferToTexture` (rgb9e5ufloat is never a storage texture, so copies are its only GPU route). The kernels
have no loops (`ds_pack9e5` is straight-line), the same work on one device is fine, and the RTX runs the same
groups without trouble. Most likely an Intel D3D12 driver (or Dawn) issue with a second device; not a defect
in our kernels. One device, as applications use, isn't affected.

## What we do

- The suite keeps the check (it tests the default-limit path). Intel runs cover shard 1 without
  `packed/buffer-path`, then that group alone (`--groups=`).
- Not reported upstream (drafts stay local, owner rule). Worth re-testing on a newer Intel driver.
