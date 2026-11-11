/**
 * Compatibility mode (WebGPU devices without 'core-features-and-limits'; SPEC.md
 * §4 Compatibility mode). Its rules that matter here: every view of a texture
 * bound in one dispatch must have the same mip range and aspect, views must
 * cover all array layers and have the texture's `textureBindingViewDimension`,
 * and rg32float has no storage.
 */

/**
 * Whether `texture` is under compatibility-mode validation: its device lacks
 * 'core-features-and-limits', and the texture has a `textureBindingViewDimension`
 * (defined only on compatibility-mode devices, so implementations that predate
 * the feature name aren't mistaken for compatibility mode).
 */
export function isCompatibilityTexture( device: GPUDevice, texture: GPUTexture ): boolean {

	const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: string } ).textureBindingViewDimension;
	return ! device.features.has( 'core-features-and-limits' as GPUFeatureName ) && viewDimension !== undefined;

}
