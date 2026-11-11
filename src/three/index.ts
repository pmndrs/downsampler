/**
 * three.js adapter for `WebGPURenderer`.
 *
 * Typed structurally so `three` stays an optional peer dependency. The texture
 * must already be allocated on the GPU (rendered to, uploaded or initialized via
 * `renderer.initTexture`) with STORAGE_BINDING usage — in current three.js that
 * means a `StorageTexture` (or `StorageArrayTexture`) — or with COPY_DST usage
 * in a format the buffer output path writes (SPEC.md §6.1), which is how
 * rgb9e5ufloat textures are supported. Native integration for all textures is
 * planned as an in-tree three.js option; see SPEC.md.
 */
import { Downsampler, type EncodeOptions } from '../webgpu/downsampler.js';

interface ThreeBackendLike {
	isWebGPUBackend?: boolean;
	device: GPUDevice;
	get( object: object ): { texture?: GPUTexture };
}

export interface ThreeRendererLike {
	backend: ThreeBackendLike;
}

const downsamplers = new WeakMap<GPUDevice, Downsampler>();

function backendOf( renderer: ThreeRendererLike ): ThreeBackendLike {

	const backend = renderer.backend;
	if ( backend?.isWebGPUBackend !== true || ! backend.device ) throw new Error( 'downsampler/three: requires an initialized WebGPURenderer with the WebGPU backend (await renderer.init()).' );
	return backend;

}

/** The shared Downsampler for a renderer's device. */
export function getDownsampler( renderer: ThreeRendererLike ): Downsampler {

	const { device } = backendOf( renderer );
	let downsampler = downsamplers.get( device );
	if ( downsampler === undefined ) downsamplers.set( device, downsampler = new Downsampler( device ) );
	return downsampler;

}

/** The GPUTexture behind a three.js texture, or undefined if not yet allocated. */
export function getGPUTexture( renderer: ThreeRendererLike, texture: object ): GPUTexture | undefined {

	return backendOf( renderer ).get( texture ).texture;

}

/**
 * Generate the mip chain of a three.js texture with compute shaders. Returns
 * false (doing nothing) when the texture is not allocated or not supported, so
 * callers can fall back to three's own mipmap generation.
 */
export function generateMipmaps( renderer: ThreeRendererLike, texture: object, options: EncodeOptions = {} ): boolean {

	const gpuTexture = getGPUTexture( renderer, texture );
	if ( gpuTexture === undefined || gpuTexture.mipLevelCount < 2 ) return false;
	const downsampler = getDownsampler( renderer );
	if ( ! downsampler.supports( gpuTexture ) ) return false;
	downsampler.generate( gpuTexture, options );
	return true;

}

export { Downsampler };

/**
 * Device limits to pass as `new WebGPURenderer( { requiredLimits } )` so six
 * levels fit in one dispatch. Requires an adapter, e.g. from
 * `await navigator.gpu.requestAdapter()`.
 */
export const requiredLimits = ( adapter: GPUAdapter ) => Downsampler.requiredLimits( adapter );
