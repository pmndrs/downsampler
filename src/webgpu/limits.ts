/**
 * Device limits worth requesting for every class in `@pmndrs/downsampler/webgpu`, as
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

/** The optional features {@link requiredFeatures} asks for, where the adapter has them. */
export const optionalFeatures = [ 'float32-filterable', 'texture-formats-tier1', 'bgra8unorm-storage' ] as const;

/**
 * Optional device features the library uses, as
 * `adapter.requestDevice( { requiredFeatures: requiredFeatures( adapter ) } )`: the ones the
 * adapter has, of
 * - `float32-filterable`: 'average' chains of r32float, rg32float and rgba32float sample instead of
 *   loading, like every other filterable format: 0.74–1.00× the time on an RTX 3060 and 0.69–1.00×
 *   on Intel UHD (DIVERGES.md #13);
 * - `texture-formats-tier1`: storage for r8unorm, rg8unorm, r16float, rg16float, the 16-bit
 *   normalized formats and more, so they don't need the buffer path or aren't rejected;
 * - `bgra8unorm-storage`: storage for bgra8unorm.
 * Everything works without them. Add your own features to the list.
 */
export function requiredFeatures( adapter: GPUAdapter ): GPUFeatureName[] {

	return optionalFeatures.filter( feature => adapter.features.has( feature ) ) as GPUFeatureName[];

}
