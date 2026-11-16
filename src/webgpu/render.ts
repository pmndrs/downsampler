import { formats, isIntegerDomain, sampledChannels, type SupportedFormat } from '../core/formats.js';
import { firstWgsl, float16Wgsl, quantizeWgsl, srgbTableBinding, srgbTableBytes, srgbWgsl, type F16Rounding, type SourceAccess, type ViewDimension } from './kernel.js';

/**
 * The render path (`output: 'render'`, and 'auto' for sRGB; DIVERGES.md #16): one render pass
 * per level and layer, each drawing one triangle over the level. Its fragment shader is the
 * exact compute kernel's first output (`firstWgsl`): the same 'exact' or 'box' rule, the same
 * reads ('sample' or 'load'), and the same rounding (`quantizeWgsl`). It returns the value as
 * stored, so the hardware's conversion into the attachment (sRGB encode, unorm, float16) meets a
 * value it holds exactly and writes the same bits the compute path stores.
 *
 * Formats: color-renderable float formats whose exact kernel has a first output. sRGB reads the
 * decode table from a storage buffer in the fragment stage.
 */
export const renderFormats: ReadonlySet<string> = new Set( [
	'rgba8unorm-srgb', 'bgra8unorm-srgb', 'rgba8unorm', 'bgra8unorm', 'r8unorm', 'rg8unorm',
	'rgba16float', 'rg16float', 'r16float', 'rgb10a2unorm', 'r32float', 'rg32float', 'rgba32float'
] );

export interface RenderKernelOptions {
	/** The attachment's format (for `colorSpace: 'srgb'`, the sRGB view format). */
	format: SupportedFormat;
	/** The source view's dimension: '2d', or '2d-array' (the layer is the draw's instance). */
	dimension: ViewDimension;
	/** 'sample': one bilinear sample at the texel center; 'load': shader weights. */
	access: SourceAccess;
	/** Area weights ('box' odd steps), with 'load' access. */
	area?: boolean;
	f16Rounding?: F16Rounding;
	/**
	 * sRGB formats: 'exact' (default) rounds each level to its code in the shader, as the compute
	 * paths do; 'device' returns the linear value and lets the attachment encode it, and reads
	 * through the hardware's decode (no tables), as plain render-pass mipmaps do.
	 */
	srgbRounding?: SrgbRounding;
}

/** How the render path rounds sRGB levels: see {@link RenderKernelOptions.srgbRounding}. */
export type SrgbRounding = 'exact' | 'device';

/**
 * WGSL of a render-path pipeline: `ds_vertex` (a full-screen triangle; the instance index is the
 * layer) and `ds_fragment`, the level's texel from the level above, bound at group 0 as the
 * compute kernels bind them (0 sampler, 1 source level, the sRGB tables at {@link srgbTableBinding}).
 */
export function createRenderKernel( { format, dimension, access, area = false, f16Rounding = 'nearest', srgbRounding = 'exact' }: RenderKernelOptions ): string {

	if ( ! renderFormats.has( format ) ) throw new Error( `downsampler: ${ format } isn't a render-path format.` );
	if ( area && access !== 'load' ) throw new Error( 'downsampler: area-weighted (box) steps read their source with textureLoad (access "load").' );
	if ( srgbRounding !== 'exact' && srgbRounding !== 'device' ) throw new Error( `downsampler: srgbRounding must be 'exact' or 'device', got ${ JSON.stringify( srgbRounding ) }.` );
	// 'device' sRGB: linear values throughout, decoded and encoded by the hardware.
	const quantization = usesTables( format, srgbRounding ) || formats[ format ].quantization !== 'srgb8' ? formats[ format ].quantization : 'float32';
	const array = dimension === '2d-array';
	const srgb = quantization === 'srgb8';
	let code = access === 'sample' ? '@group(0) @binding(0) var imageSampler: sampler;\n' : '';
	code += `@group(0) @binding(1) var source: texture_2d${ array ? '_array' : '' }<f32>;\n`;
	if ( quantization === 'float16' ) code += float16Wgsl( f16Rounding, false );
	if ( srgb ) code += srgbWgsl();
	code += quantizeWgsl( quantization );
	code += firstWgsl( { access, array, splat: sampledChannels( format ) === 1 ? '.xxxx' : '', srgbSource: srgb, exactLoads: isIntegerDomain( quantization ), srgb, average: true, area, areaHelpers: area } );
	code += `
struct DsVaryings {
	@builtin(position) position: vec4f,
	@location(0) @interpolate(flat, either) layer: u32,
};
@vertex fn ds_vertex(@builtin(vertex_index) index: u32, @builtin(instance_index) layer: u32) -> DsVaryings {
	let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0))[index];
	return DsVaryings(vec4f(p, 0.0, 1.0), layer);
}
@fragment fn ds_fragment(in: DsVaryings) -> @location(0) vec4f {
	let size = max(textureDimensions(source) >> vec2u(1u), vec2u(1u));
	return first(vec2u(in.position.xy), size, in.layer) / SCALE;
}
`;
	return code;

}

/** Group 0 of a render-path pipeline. */
export function renderLayoutEntries( options: RenderKernelOptions ): GPUBindGroupLayoutEntry[] {

	const { format, dimension, access } = options;
	const visibility = GPUShaderStage.FRAGMENT;
	const entries: GPUBindGroupLayoutEntry[] = access === 'sample' ? [ { binding: 0, visibility, sampler: { type: 'filtering' } } ] : [];
	entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
	if ( usesTables( format, options.srgbRounding ) ) entries.push( { binding: srgbTableBinding, visibility, buffer: { type: 'read-only-storage', minBindingSize: srgbTableBytes } } );
	return entries;

}

/** Whether a render-path pipeline reads the sRGB tables (bound at {@link srgbTableBinding}): sRGB with 'exact' rounding. */
export function usesTables( format: string, srgbRounding: SrgbRounding = 'exact' ): boolean {

	return srgbRounding === 'exact' && formats[ format as SupportedFormat ]?.quantization === 'srgb8';

}
