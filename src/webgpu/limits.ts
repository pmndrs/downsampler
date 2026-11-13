/**
 * Device limits worth requesting for every class in `downsampler/webgpu`, as
 * `adapter.requestDevice( { requiredLimits: requiredLimits( adapter ) } )`:
 * - `maxStorageTexturesPerShaderStage`: up to 6 (default 4), so a `Downsampler`
 *   writes six levels per dispatch and a `Reducer` emits five or six levels to
 *   storage textures.
 * - `maxComputeInvocationsPerWorkgroup` and `maxComputeWorkgroupSizeX`: 256 where
 *   the adapter has them (compatibility mode defaults to 128). The 5- and
 *   6-level kernels of `Downsampler` and `Reducer` use 16 × 16 workgroups, and
 *   `ScalarReduction` needs 256 invocations, 256 wide (its combine workgroup).
 *
 * Without them everything still works with fewer levels per dispatch, except
 * `ScalarReduction` and 5- or 6-level `Reducer`s, which say so in their error.
 * Not included: a 6-level `Reducer` with 8 channels needs 32 KiB of workgroup
 * memory (`maxComputeWorkgroupStorageSize: 32768`), which you request yourself.
 */
export function requiredLimits( adapter: GPUAdapter ): Record<string, number> {

	const limits: Record<string, number> = { maxStorageTexturesPerShaderStage: Math.max( 4, Math.min( 6, adapter.limits.maxStorageTexturesPerShaderStage ) ) };
	if ( adapter.limits.maxComputeInvocationsPerWorkgroup >= 256 ) limits.maxComputeInvocationsPerWorkgroup = 256;
	if ( adapter.limits.maxComputeWorkgroupSizeX >= 256 ) limits.maxComputeWorkgroupSizeX = 256;
	return limits;

}
