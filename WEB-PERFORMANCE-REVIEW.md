# Downsampler performance and code review

November 17, 2026 · Reviewed `d8368d0` · [Read in Pages](https://chatgpt.com/space/page_31c5c354d25c81918908d3903f124634)

Fix the scratch-buffer retry before release. For speed, the clearest opportunity is a moving dirty rectangle: the GPU already does little work, but the host rebuilds bindings for every new brush position. For startup, measure first use of the actual format and path before changing kernels.

## DS1  Preserve a working plan when a replacement fails

Reproduced on Windows, Chrome 148, NVIDIA Ampere. Generate a valid chain with `output: 'buffer'`, then call `prepare` on the same texture with a four-byte scratch buffer. It throws the expected size error. After the queue settles, calling `generate` again with the original options produces a WebGPU validation error for a destroyed output-layout buffer. A caller that handles the initial exception loses its previously working chain.

`src/webgpu/downsampler.ts` line 804 schedules the old plan’s buffers for destruction before line 805 builds the replacement. On failure, the cache still points to the old plan. Build the replacement, publish it, then retire the previous buffers. Apply the same successful-replacement ordering to the region cache and release newly owned resources on a partial construction failure. A regression test should verify a successful retry with the original options using a validation scope and compare mip contents to a clean run. Queue completion covers previously submitted work. [WebGPU queue specification](https://www.w3.org/TR/webgpu/#dom-gpuqueue-onsubmittedworkdone)

## DS2  Reuse bindings when a brush moves

In a fresh Chrome 148 run on NVIDIA Ampere, a warmed 1920×1080 `rgba8unorm` texture received 64 updates with a 32×32 brush. Moving the brush by one pixel each time created 64 uniform buffers, 256 bind groups and zero compute pipelines. Repeating one rectangle created zero of all three. Both runs of each case gave the same counts. CPU encode p95 was roughly 0.2 ms moving and 0.1 ms fixed, close to the timer resolution. The object counts are the stronger result.

The per-texture cache key includes the rectangle’s exact coordinates. A new position goes through planning, texture-view creation and a uniform upload. The existing partial benchmark keeps its rectangle fixed. Its clock-validated RTX run measured 17.4 µs of GPU time for the 1080p 32×32 update versus 56.8 µs full-chain. That answers how much GPU work is saved. The new probe measured host work with a submit and queue completion between updates. See `src/webgpu/downsampler.ts` lines 823, 987 and 1044, `test/bench/partial.js` line 13 and `results/windows/round2/partial.md`.

First cache invariant views and batch topology per texture and option set. Then test an internal encode path with immutable per-dispatch uniform slots and dynamic offsets. Preserve public `prepare()` batches and their existing binding contract. An encoder containing multiple brush strokes must see the correct coordinates for each dispatch. Check moving, repeated, disjoint and clipped rectangles, multiple encoders, layers and empty updates against full recomputation. Track creations, CPU encode p95 and GPU time separately.

## DS3  Move first use of pipelines off the interactive frame

Compute and render pipelines are created synchronously on cache misses. `prepare()` can populate the compute cache ahead of time, but it is synchronous and throws for a render plan. A texture first encountered during a resize or material swap can therefore expose compilation in an interactive frame. See `src/webgpu/downsampler.ts` lines 761, 1106 and 1272. The repository already handled an expensive sRGB lookup-table compiler problem, so use the current shader as the baseline.

Prototype a narrowly scoped async warm-up accepting the real texture and options. Share descriptor construction and in-flight pipeline promises across identical requests and remove failed promises so retry works. Keep encoding synchronous once warm. Measure cold first frame, warm revisit and actual compiler events for storage, sRGB render and buffer-copy chains. Include an integrated GPU and a software adapter before expanding the API.

## DS4  Keep kernel choices tied to the workload

The archived halo runs show why dispatch count is an incomplete score. At 1920×1080 the capped halo reduced four dispatches to three. For rgba8 it took 0.92× the default time on the RTX and 1.00–1.01× on Intel. Uncapped halos took up to 1.52× default time on Intel at one tested odd size. Keep the cap and the opt-in setting pending the M5 comparison already requested by the project. A fixed-size win can be exposed to callers independently of a global default. See `results/windows/round2/halo.md` and `results/windows/round2/defaults.md`.

Likewise, the Windows dispatch benchmark found essentially equal GPU times for separate compute passes and one shared pass at a fixed dispatch count. AMD’s native SPD algorithm coordinates the last group using a globally coherent counter and mip. Retain the portable multi-dispatch implementation as the reference. Compare odd-size color averages and conservative depth folds against their own oracles. A faster chain that drops an edge texel answers a different question for occlusion culling. [FidelityFX SPD manual](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/)

## DS5  Put the tradeoff in the existing examples

Give `examples/partial-update.html` a benchmark mode that replays the same brush path in partial and full modes. Collect CPU encode time and GPU timestamps with the reference recompute and readback outside the timed interval. Keep the existing byte comparison available after replay. Extend the Hi-Z page with objects near a thin occluder and show how many draws are culled and whether any visible object disappears. That gives the min/max guarantee a visible consequence.

The README says the package is unpublished and immediately offers `npm install downsampler`. Until publication is confirmed, lead with a tested build-from-checkout path. Keep one setup example and a short table for ordinary color, exact sRGB, Hi-Z and dirty regions. Point historical findings to dated reports. For example, `docs/WINDOWS-REPORT.md` still lists the NVIDIA sRGB tolerance as open, while `windows-3` records the decision in `f7248a3`.

## Verification

Reviewed `windows-3` at `d8368d0`, including its opt-in halo implementation. Consulted the separate halo checkout at `f8f4064` and the Windows result files. The active Windows checkout has local harness and results work. The browser probe used its freshly built library, whose tracked `src` matched the reviewed commit. GPU measurements from the existing reports are identified above. The new probe covers one browser and adapter. The pinned worktree built successfully and passed all 129 core tests. The active checkout passed 130, including its additional local harness test.

## Browser regression case for DS1

Run on a WebGPU device with a mipmapped rgba8unorm texture that has TEXTURE_BINDING and COPY_DST usage. The initial call uses library-owned scratch. The expected result after a fix is a null validation error and the same mip contents as the initial chain.

```js
const ds = new Downsampler(device, { output: 'buffer' });
ds.generate(texture);
await device.queue.onSubmittedWorkDone();
const small = device.createBuffer({
  size: 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
try {
  ds.prepare(texture, { scratch: small });
} catch (error) {
  console.log(error.message); // Expected size error on a 1920 × 1080 texture.
}
await device.queue.onSubmittedWorkDone();
await new Promise(resolve => setTimeout(resolve, 0));
device.pushErrorScope('validation');
ds.generate(texture);
await device.queue.onSubmittedWorkDone();
console.log(await device.popErrorScope());
small.destroy();
ds.release(texture);
```
