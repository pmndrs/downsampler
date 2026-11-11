import { formats, isSupportedFormat, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { levelSize, tileDispatch } from '../core/pyramid.js';
import { resolveReduction, type Reduction, type TileEdgePolicy } from '../core/reduction.js';
import { createKernel, kernelLayoutEntries, sizeBinding, type KernelOptions, type ViewDimension } from './kernel.js';
import { checkToolBindings, setToolBindGroup, toolBindGroupLayout, toolBindGroupReason, toolGroupLayouts, toolPipelineLayout, type ToolBindings } from './tool.js';

export type { ToolBindings } from './tool.js';

export interface ReducerOptions {
	/**
	 * Per-channel ops (`[ 'average', 'average', 'sum', 'max' ]`, with 'weight' and
	 * 'weighted' for weighted means), a builtin ('average', 'min', 'max',
	 * 'minmax') or a custom `{ wgsl, kind: 'fold' | 'sum' }`. Default 'average'.
	 */
	reduction?: Reduction;
	/**
	 * Source-relative levels to emit (1 = half size), strictly increasing, at
	 * most 6. Every level up to the last is reduced in workgroup memory; only
	 * these are stored, and only these get an output binding.
	 */
	levels: readonly number[];
	/** 'zero-weight' (default, ⌈n/2⌉ sizes) or 'clamp' (SPD/FSR, ⌊n/2⌋ sizes). */
	edges?: TileEdgePolicy;
	/** WGSL `fn load(texel: vec2i, layer: u32) -> vec4f`. Without it the Reducer reads a source texture. */
	load?: string;
	/** WGSL `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f)`. Without it the Reducer writes output textures of `format`. */
	store?: string;
	/** WGSL `fn size() -> vec2u`: the source size, with a load hook. Without it the size comes from a uniform the Reducer manages. */
	size?: string;
	/** Storage format of the output textures (no store hook). */
	format?: SupportedFormat;
	/** View dimension of the source and output textures. Default '2d'; '2d-array' reduces every layer. */
	dimension?: ViewDimension;
	/** Format of the source texture (no load hook), so single-channel formats (depth, r*) are splatted. Default 4 channels. */
	sourceFormat?: string;
	/** The tool's group-1 bindings, used by its hooks. */
	bindings?: ToolBindings;
	label?: string;
}

/** A texture level: a texture's level `mipLevel` (default 0). */
export interface TextureLevel {
	texture: GPUTexture;
	mipLevel?: number;
}

export interface ReducerDispatch {
	/** Source size in texels. Required with a load hook; otherwise the source level's size. */
	width?: number;
	height?: number;
	/** Array layers (workgroup z). Default: the source's or first output's layers with a '2d-array' dimension, else 1. */
	layers?: number;
	/** The source texture (no load hook). */
	source?: GPUTexture | TextureLevel;
	/** One output per emitted level, in `levels` order (no store hook). Each must be at least the level's size. */
	outputs?: readonly ( GPUTexture | TextureLevel )[];
	/** The tool's group-1 bind group, created from `bindGroupLayouts[ 1 ]` (or an equivalent layout). */
	bindGroup?: GPUBindGroup;
}

interface Prepared {
	bindGroup: GPUBindGroup;
	workgroups: [ number, number, number ];
}

interface CachedPipeline {
	pipeline: GPUComputePipeline;
	layout: GPUPipelineLayout;
	bindGroupLayouts: GPUBindGroupLayout[];
}

const pipelines = new WeakMap<GPUDevice, Map<string, CachedPipeline>>();
const textureIds = new WeakMap<GPUTexture, number>();
let nextTextureId = 0;
const idOf = ( texture: GPUTexture ) => {

	let id = textureIds.get( texture );
	if ( id === undefined ) textureIds.set( texture, id = nextTextureId ++ );
	return id;

};

const asLevel = ( value: GPUTexture | TextureLevel ): Required<TextureLevel> => 'texture' in value ? { texture: value.texture, mipLevel: value.mipLevel ?? 0 } : { texture: value, mipLevel: 0 };
const mipSize = ( size: number, level: number ) => Math.max( 1, size >> level );

/** Bind groups (and size uniforms) kept per Reducer, most recent last. */
const cacheSize = 16;

/**
 * A tile reduction kernel for tools (SPEC.md §14.1): a custom or texture source,
 * per-channel reductions including sums and weighted means, a choice of emitted
 * levels, the 'zero-weight' or 'clamp' edge policy, and a store hook or output
 * textures. All levels (at most 6) are reduced in one dispatch at any size.
 *
 * Record it with `encode` or `encodePass`, or dispatch it yourself with `wgsl`
 * (or `pipeline`), `layout`, `bindGroup()` and `workgroups()`. Group 0 is the
 * Reducer's; group 1 is the tool's.
 */
export class Reducer {

	readonly device: GPUDevice;
	/** The kernel's WGSL. Entry point {@link Reducer.entryPoint}. */
	readonly wgsl: string;
	readonly entryPoint = 'main';
	/** Pipeline layout: [ group 0 (ours), group 1 (the tool's, if it has bindings) ]. */
	readonly layout: GPUPipelineLayout;
	readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
	readonly pipeline: GPUComputePipeline;
	readonly levels: readonly number[];
	readonly edges: TileEdgePolicy;
	readonly label: string;
	private readonly options: ReducerOptions;
	private readonly cache = new Map<string, Prepared>();

	constructor( device: GPUDevice, options: ReducerOptions ) {

		const edges = options.edges ?? 'zero-weight';
		if ( edges !== 'zero-weight' && edges !== 'clamp' ) throw new Error( `downsampler: a Reducer takes edges 'zero-weight' or 'clamp', got ${ JSON.stringify( edges ) }; 'exact' mip chains (a batch per odd size) are Downsampler's.` );
		const reduction = options.reduction ?? 'average';
		resolveReduction( reduction );
		checkToolBindings( options.bindings );
		const kernel: KernelOptions = {
			edges, levels: options.levels, reduction, format: options.format, dimension: options.dimension ?? '2d', sourceFormat: options.sourceFormat,
			load: options.load, store: options.store, size: options.size, bindings: options.bindings?.wgsl
		};

		if ( options.store === undefined ) {

			const format = options.format;
			if ( format !== undefined && isSupportedFormat( format ) ) {

				const feature: string | undefined = ( formats[ format ] as FormatInfo ).feature;
				if ( feature !== undefined && ! device.features.has( feature ) ) throw new Error( `downsampler: output format ${ format } needs the "${ feature }" device feature.` );

			}

			if ( options.levels.length > device.limits.maxStorageTexturesPerShaderStage ) throw new Error( `downsampler: ${ options.levels.length } output textures exceed this device's maxStorageTexturesPerShaderStage (${ device.limits.maxStorageTexturesPerShaderStage }); emit fewer levels or use a store hook.` );

		}

		const wgsl = createKernel( kernel );
		const label = options.label ?? `downsampler reducer ${ edges } ${ options.levels.join( ',' ) }`;
		const toolEntries = options.bindings?.entries;
		const key = `${ wgsl }\n${ JSON.stringify( kernelLayoutEntries( kernel ) ) }\n${ JSON.stringify( toolEntries ?? null ) }`;
		let cache = pipelines.get( device );
		if ( cache === undefined ) pipelines.set( device, cache = new Map() );
		let cached = cache.get( key );

		if ( cached === undefined ) {

			const group0 = device.createBindGroupLayout( { label: `${ label } group 0`, entries: kernelLayoutEntries( kernel ) } );
			const tool = toolBindGroupLayout( device, `${ label } group 1`, options.bindings );
			const layout = toolPipelineLayout( device, label, group0, tool );
			const bindGroupLayouts = toolGroupLayouts( group0, tool );
			const module = device.createShaderModule( { label, code: wgsl } );
			cached = { pipeline: device.createComputePipeline( { label, layout, compute: { module, entryPoint: 'main' } } ), layout, bindGroupLayouts };
			cache.set( key, cached );

		}

		this.device = device;
		this.wgsl = wgsl;
		this.layout = cached.layout;
		this.bindGroupLayouts = cached.bindGroupLayouts;
		this.pipeline = cached.pipeline;
		this.levels = [ ...options.levels ];
		this.edges = edges;
		this.label = label;
		this.options = { ...options, reduction };

	}

	/** Size of emitted level `level` (1 = half) for a `width` × `height` source. */
	levelSize( level: number, width: number, height: number ): [ number, number ] {

		return [ levelSize( width, level, this.edges ), levelSize( height, level, this.edges ) ];

	}

	/** Workgroup counts for a `width` × `height` source with `layers` array layers. */
	workgroups( width: number, height: number, layers = 1 ): [ number, number, number ] {

		const { groupsX, groupsY } = tileDispatch( width, height, this.levels[ this.levels.length - 1 ], this.edges );
		return [ groupsX, groupsY, layers ];

	}

	/**
	 * The group-0 bind group for a dispatch: the source texture, the output
	 * textures and the size uniform, as this kernel needs them. Cached per size
	 * and textures (the most recent 16).
	 */
	bindGroup( dispatch: ReducerDispatch = {} ): GPUBindGroup {

		return this.prepare( dispatch ).bindGroup;

	}

	/** Record the dispatch into a compute pass the caller owns. */
	encodePass( pass: GPUComputePassEncoder, dispatch: ReducerDispatch = {} ): void {

		const { bindGroup, workgroups } = this.prepare( dispatch );
		pass.setPipeline( this.pipeline );
		pass.setBindGroup( 0, bindGroup );
		const reason = toolBindGroupReason( this.bindGroupLayouts.length > 1, dispatch.bindGroup, 'this Reducer' );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		setToolBindGroup( pass, dispatch.bindGroup );
		pass.dispatchWorkgroups( ...workgroups );

	}

	/** Record the dispatch in its own compute pass. */
	encode( encoder: GPUCommandEncoder, dispatch: ReducerDispatch = {} ): void {

		const pass = encoder.beginComputePass( { label: this.label } );
		this.encodePass( pass, dispatch );
		pass.end();

	}

	private prepare( dispatch: ReducerDispatch ): Prepared {

		const { load, store, size, format, dimension = '2d' } = this.options;
		const source = dispatch.source === undefined ? undefined : asLevel( dispatch.source );
		const outputs = dispatch.outputs?.map( asLevel );
		const array = dimension === '2d-array';

		if ( load === undefined && source === undefined ) throw new Error( 'downsampler: this Reducer reads a source texture; pass `source`.' );
		if ( load !== undefined && source !== undefined ) throw new Error( 'downsampler: this Reducer has a load hook; it takes no source texture.' );
		if ( store === undefined && outputs?.length !== this.levels.length ) throw new Error( `downsampler: pass one output per emitted level (${ this.levels.length }).` );
		if ( store !== undefined && outputs !== undefined ) throw new Error( 'downsampler: this Reducer has a store hook; it takes no output textures.' );

		let width = dispatch.width, height = dispatch.height;
		if ( source !== undefined ) {

			const { texture, mipLevel } = source;
			if ( ! Number.isInteger( mipLevel ) || mipLevel < 0 || mipLevel >= texture.mipLevelCount ) throw new Error( 'downsampler: source mipLevel is outside the source texture.' );
			if ( ( texture.usage & GPUTextureUsage.TEXTURE_BINDING ) === 0 ) throw new Error( 'downsampler: source needs TEXTURE_BINDING usage.' );
			const sw = mipSize( texture.width, mipLevel ), sh = mipSize( texture.height, mipLevel );
			if ( ( width !== undefined && width !== sw ) || ( height !== undefined && height !== sh ) ) throw new Error( `downsampler: width and height (${ width }×${ height }) must match the source level (${ sw }×${ sh }), or be left out.` );
			width = sw; height = sh;

		}

		if ( width === undefined || height === undefined ) throw new Error( 'downsampler: pass the source width and height.' );
		const layers = dispatch.layers ?? ( array ? ( source ?? outputs?.[ 0 ] )?.texture.depthOrArrayLayers ?? 1 : 1 );
		if ( ! array && layers !== 1 ) throw new Error( 'downsampler: several layers need dimension \'2d-array\'.' );
		const workgroups = this.workgroups( width, height, layers );

		const key = [ width, height, layers, source ? `${ idOf( source.texture ) }/${ source.mipLevel }` : '', ...( outputs ?? [] ).map( output => `${ idOf( output.texture ) }/${ output.mipLevel }` ) ].join( ':' );
		const cached = this.cache.get( key );
		if ( cached !== undefined ) {

			this.cache.delete( key );
			this.cache.set( key, cached );
			return cached;

		}

		const entries: GPUBindGroupEntry[] = [];
		if ( source !== undefined ) {

			if ( source.texture.depthOrArrayLayers < layers ) throw new Error( `downsampler: source has ${ source.texture.depthOrArrayLayers } layers, the dispatch ${ layers }.` );
			const aspect = source.texture.format.includes( 'stencil' ) ? 'depth-only' : 'all';
			entries.push( { binding: 1, resource: source.texture.createView( { label: `${ this.label } source`, dimension, baseMipLevel: source.mipLevel, mipLevelCount: 1, aspect } ) } );

		}

		outputs?.forEach( ( { texture, mipLevel }, i ) => {

			const level = this.levels[ i ];
			const [ w, h ] = this.levelSize( level, width!, height! );
			const ow = mipSize( texture.width, mipLevel ), oh = mipSize( texture.height, mipLevel );
			if ( texture.format !== format ) throw new Error( `downsampler: output for level ${ level } is ${ texture.format }; this Reducer writes ${ format }.` );
			if ( ( texture.usage & GPUTextureUsage.STORAGE_BINDING ) === 0 ) throw new Error( `downsampler: output for level ${ level } needs STORAGE_BINDING usage.` );
			if ( ! Number.isInteger( mipLevel ) || mipLevel < 0 || mipLevel >= texture.mipLevelCount ) throw new Error( `downsampler: output mipLevel for level ${ level } is outside its texture.` );
			if ( ow < w || oh < h ) throw new Error( `downsampler: output for level ${ level } is ${ ow }×${ oh }, smaller than the level (${ w }×${ h }).${ this.edges === 'zero-weight' ? ' \'zero-weight\' levels are ⌈n/2⌉, so a mip of the source size is one texel short at odd sizes: use separate textures or a store hook.' : '' }` );
			if ( texture.depthOrArrayLayers < layers ) throw new Error( `downsampler: output for level ${ level } has ${ texture.depthOrArrayLayers } layers, the dispatch ${ layers }.` );
			entries.push( { binding: i + 2, resource: texture.createView( { label: `${ this.label } level ${ level }`, dimension, baseMipLevel: mipLevel, mipLevelCount: 1 } ) } );

		} );

		if ( load !== undefined && size === undefined ) {

			const buffer = this.device.createBuffer( { label: `${ this.label } size`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			this.device.queue.writeBuffer( buffer, 0, Uint32Array.of( width, height, 0, 0 ) );
			entries.push( { binding: sizeBinding, resource: { buffer } } );

		}

		const prepared = { bindGroup: this.device.createBindGroup( { label: this.label, layout: this.bindGroupLayouts[ 0 ], entries } ), workgroups };
		this.cache.set( key, prepared );
		if ( this.cache.size > cacheSize ) this.cache.delete( this.cache.keys().next().value! );
		return prepared;

	}

}
