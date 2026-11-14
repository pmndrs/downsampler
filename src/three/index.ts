/**
 * three.js adapter for `WebGPURenderer`.
 *
 * Typed structurally so `three` stays an optional peer dependency. The texture
 * must already be allocated on the GPU (rendered to, uploaded or initialized via
 * `renderer.initTexture`) with STORAGE_BINDING usage — in current three.js that
 * means a `StorageTexture` (or `StorageArrayTexture`) — or, for formats that
 * can't be storage textures (rgb9e5ufloat; rgba8unorm-srgb and bgra8unorm-srgb, so
 * regular sRGB colour textures; rg11b10ufloat without tier1), with COPY_DST usage:
 * they take the buffer output path (SPEC.md §6.1). Native integration for all textures is
 * planned as an in-tree three.js option; see SPEC.md.
 */
import { Downsampler, type DownsamplerOptions, type EncodeOptions } from '../webgpu/downsampler.js';
import { requiredLimits } from '../webgpu/limits.js';

interface ThreeBackendLike {
	isWebGPUBackend?: boolean;
	device: GPUDevice;
	get( object: object ): { texture?: GPUTexture };
}

/** The part of a three.js `WebGPURenderer` the adapter uses (typed structurally, so `three` stays optional). */
export interface ThreeRendererLike {
	backend: ThreeBackendLike;
}

/** Options of {@link generateMipmaps}: a chain's `EncodeOptions`, plus where to record it and which shared Downsampler records it. */
export interface GenerateMipmapsOptions extends EncodeOptions {
	/**
	 * Record into this encoder and don't submit: the caller submits it (e.g.
	 * with its own passes). Default: a new encoder, submitted at once.
	 */
	encoder?: GPUCommandEncoder;
	/**
	 * Options of the shared Downsampler that records the chain (e.g.
	 * `{ averageAccess: 'load', f16Rounding: 'device' }`); see {@link getDownsampler}.
	 */
	downsampler?: DownsamplerOptions;
}

const downsamplers = new WeakMap<GPUDevice, Map<string, Downsampler>>();

function backendOf( renderer: ThreeRendererLike ): ThreeBackendLike {

	const backend = renderer.backend;
	if ( backend?.isWebGPUBackend !== true || ! backend.device ) throw new Error( 'downsampler/three: requires an initialized WebGPURenderer with the WebGPU backend (await renderer.init()).' );
	return backend;

}

/** Equal option sets give the same key, whatever their key order (undefined values are left out). */
function optionsKey( options: DownsamplerOptions ): string {

	return JSON.stringify( Object.entries( options ).filter( ( [ , value ] ) => value !== undefined ).sort( ( [ a ], [ b ] ) => a < b ? - 1 : 1 ) );

}

/**
 * The shared Downsampler for a renderer's device and `options`: one per device
 * and distinct option set (`averageAccess`, `f16Rounding`, `output`, `tail`,
 * …), created on first use. Pipelines and plans are cached per Downsampler, so
 * reuse one option set where you can.
 */
export function getDownsampler( renderer: ThreeRendererLike, options: DownsamplerOptions = {} ): Downsampler {

	const { device } = backendOf( renderer );
	let shared = downsamplers.get( device );
	if ( shared === undefined ) downsamplers.set( device, shared = new Map() );
	const key = optionsKey( options );
	let downsampler = shared.get( key );
	if ( downsampler === undefined ) shared.set( key, downsampler = new Downsampler( device, options ) );
	return downsampler;

}

/** The GPUTexture behind a three.js texture, or undefined if not yet allocated. */
export function getGPUTexture( renderer: ThreeRendererLike, texture: object ): GPUTexture | undefined {

	return backendOf( renderer ).get( texture ).texture;

}

/**
 * Why {@link generateMipmaps} would decline a texture with these options, or
 * null if it would generate: not allocated yet, a single level (in place), or
 * a reason from the shared Downsampler (`unsupportedReason`, and
 * `sourceUnsupportedReason` with a separate `source`).
 */
export function unsupportedReason( renderer: ThreeRendererLike, texture: object, options: GenerateMipmapsOptions = {} ): string | null {

	const gpuTexture = getGPUTexture( renderer, texture );
	if ( gpuTexture === undefined ) return 'the texture is not allocated on the GPU yet (render to it, upload it or call renderer.initTexture first)';
	if ( options.source === undefined && gpuTexture.mipLevelCount < 2 ) return 'the texture has a single mip level (set generateMipmaps = true before it is allocated)';
	const downsampler = getDownsampler( renderer, options.downsampler );
	return downsampler.unsupportedReason( gpuTexture ) ?? ( options.source === undefined ? null : downsampler.sourceUnsupportedReason( gpuTexture, options.source, options.baseMipLevel ) );

}

/**
 * Generate the mip chain of a three.js texture with compute shaders, using the
 * renderer's shared {@link getDownsampler} for `options.downsampler`. `options`
 * are a Downsampler's `EncodeOptions` (e.g. `{ reduction: 'max' }`), plus
 * `encoder` to record into your own encoder instead of submitting at once.
 * Returns false, doing nothing, when the texture can't be generated, so callers
 * can fall back to three's own mipmap generation; {@link unsupportedReason}
 * says why. Set `texture.mipmapsAutoUpdate = false` so three doesn't
 * regenerate the levels with render passes after a compute write.
 */
export function generateMipmaps( renderer: ThreeRendererLike, texture: object, options: GenerateMipmapsOptions = {} ): boolean {

	if ( unsupportedReason( renderer, texture, options ) !== null ) return false;
	const { encoder, downsampler: downsamplerOptions, ...encodeOptions } = options;
	const gpuTexture = getGPUTexture( renderer, texture )!;
	const downsampler = getDownsampler( renderer, downsamplerOptions );
	if ( encoder !== undefined ) downsampler.encode( encoder, gpuTexture, encodeOptions );
	else downsampler.generate( gpuTexture, encodeOptions );
	return true;

}

export { Downsampler };

/**
 * Device limits to pass as `new WebGPURenderer( { requiredLimits } )`, so six
 * levels fit in one dispatch (and 5- or 6-level `Reducer`s and `ScalarReduction`
 * work in compatibility mode): `requiredLimits` from `downsampler/webgpu`.
 * Requires an adapter, e.g. from `await navigator.gpu.requestAdapter()`.
 */
export { requiredLimits };
