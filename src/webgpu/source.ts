/**
 * Sources, one shape for every class (SPEC.md §5 Sources): `Downsampler`'s
 * `source`, `Reducer`'s and `ScalarReduction`'s take a `GPUTexture` (its level
 * 0) or a {@link TextureLevel}, and pass the same checks here.
 */
import type { ViewDimension } from './kernel.js';

/** A texture level: level `mipLevel` (default 0) of `texture`. */
export interface TextureLevel {
	texture: GPUTexture;
	/** Default 0. */
	mipLevel?: number;
}

/** A texture, read at its level 0, or a {@link TextureLevel}. */
export type TextureSource = GPUTexture | TextureLevel;

/** `source` as `{ texture, mipLevel }`, the level filled in (0 for a bare texture). */
export function asTextureLevel( source: TextureSource ): Required<TextureLevel> {

	return 'texture' in source ? { texture: source.texture, mipLevel: source.mipLevel ?? 0 } : { texture: source, mipLevel: 0 };

}

/** Why `source` is neither a GPUTexture nor `{ texture, mipLevel? }`, or null. */
export function sourceShapeReason( source: unknown, name = 'source' ): string | null {

	if ( source === null || typeof source !== 'object' ) return `${ name } must be a GPUTexture or { texture, mipLevel }`;
	if ( 'texture' in source && ( source.texture === null || typeof source.texture !== 'object' ) ) return `${ name }.texture must be a GPUTexture`;
	return null;

}

/**
 * Why level `mipLevel` of `texture` can't be read as a source, or null: a 2D,
 * single-sampled, float-sampleable texture (depth too, through its depth
 * aspect) with TEXTURE_BINDING usage, a 2d or 2d-array binding view dimension
 * (compatibility mode), and the level inside the texture. `name` starts the
 * reason ('source').
 */
export function sourceTextureReason( { texture, mipLevel }: Required<TextureLevel>, name = 'source' ): string | null {

	if ( texture.dimension !== '2d' ) return `${ name } dimension ${ texture.dimension } is not supported`;
	const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: string } ).textureBindingViewDimension;
	if ( viewDimension !== undefined && viewDimension !== '2d' && viewDimension !== '2d-array' ) return `${ name } textureBindingViewDimension ${ viewDimension } is not supported`;
	if ( texture.sampleCount !== 1 ) return `multisampled ${ name }s are not supported (resolve first)`;
	if ( ( texture.usage & GPUTextureUsage.TEXTURE_BINDING ) === 0 ) return `${ name } needs TEXTURE_BINDING usage`;
	if ( /int$/.test( texture.format ) || texture.format === 'stencil8' ) return `${ name } format ${ texture.format } is not float-sampleable`;
	if ( ! Number.isInteger( mipLevel ) || mipLevel < 0 || mipLevel >= texture.mipLevelCount ) return `${ name } mipLevel ${ mipLevel } is outside the ${ name } texture (${ texture.mipLevelCount } level${ texture.mipLevelCount === 1 ? '' : 's' })`;
	return null;

}

/** The size of level `mipLevel` of `texture`. */
export function levelSizeOf( { texture, mipLevel }: Required<TextureLevel> ): [ number, number ] {

	return [ Math.max( 1, texture.width >> mipLevel ), Math.max( 1, texture.height >> mipLevel ) ];

}

/**
 * The view dimension `texture` is bound with: its `textureBindingViewDimension`
 * where defined (compatibility mode), else '2d-array' for several layers.
 */
export function bindingViewDimension( texture: GPUTexture ): ViewDimension {

	const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: ViewDimension } ).textureBindingViewDimension;
	return viewDimension ?? ( texture.depthOrArrayLayers > 1 ? '2d-array' : '2d' );

}

/** A one-level view of a source for `textureLoad`/sampling; depth-stencil formats through their depth aspect. */
export function sourceView( { texture, mipLevel }: Required<TextureLevel>, dimension: ViewDimension, label: string ): GPUTextureView {

	const aspect = texture.format.includes( 'stencil' ) ? 'depth-only' : 'all';
	return texture.createView( { label, dimension, baseMipLevel: mipLevel, mipLevelCount: 1, aspect } );

}
