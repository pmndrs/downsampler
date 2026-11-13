import { bufferWritable, formats, isIntFormat, isSupportedFormat, storageWritable, type FormatInfo, type SupportedFormat } from '../core/formats.js';
import { outputBufferLayout, tileSize, workgroupSide } from '../core/plan.js';
import { levelSize, tileDispatch } from '../core/pyramid.js';
import { resolveReduction, type Channels, type Reduction, type TileEdgePolicy } from '../core/reduction.js';
import { isCompatibilityTexture } from './compat.js';
import { createKernel, kernelLayoutEntries, outputBufferBinding, outputLayoutBinding, outputLayoutBytes, sizeBinding, type F16Rounding, type KernelOptions, type OutputPath, type ViewDimension } from './kernel.js';
import { asTextureLevel, sourceShapeReason, sourceTextureReason, sourceView, type TextureLevel } from './source.js';
import { checkToolBindings, retireBuffers, setToolBindGroup, toolBindGroupLayout, toolBindGroupReason, toolGroupLayouts, toolPipelineLayout, type ToolBindings } from './tool.js';

export type { ToolBindings } from './tool.js';

/** Options of `new Reducer( device, options )`. The kernel is generated and compiled in the constructor. See SPEC.md §14.1. */
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
	/**
	 * How output textures are written (no store hook): 'storage' textures, or
	 * 'buffer': packed texels into a buffer the Reducer owns, then a
	 * `copyBufferToTexture` per level (SPEC.md §6.1). Default 'storage' when the
	 * device can write `format` as a storage texture, else 'buffer' (rgb9e5ufloat;
	 * rg11b10ufloat without texture-formats-tier1). float16 outputs are rounded
	 * before the store on both paths (`f16Rounding`). Buffer outputs need COPY_DST
	 * usage instead of STORAGE_BINDING; `encode` records the copies, and after
	 * `encodePass` or a self-dispatch call `encodeCopies` once the pass has ended.
	 */
	output?: OutputPath;
	/**
	 * float16 output formats (no store hook): 'nearest' (default) rounds to nearest even
	 * in the shader, the same on every GPU; 'device' uses `pack2x16float`, which
	 * truncates on D3D12. See {@link F16Rounding}.
	 */
	f16Rounding?: F16Rounding;
	/**
	 * Accumulator channels per texel: 4 (a `vec4f`, default) or 8. With 8, the
	 * load hook returns, the store hook takes, and a custom reduce combines
	 * `DsWide { lo: vec4f, hi: vec4f }` (channels 0–3, 4–7), and both hooks are
	 * required. Eight channel ops imply it. Workgroup memory doubles: 8 KiB
	 * for 5 levels, 32 KiB for 6 (above the default limit).
	 */
	channels?: Channels;
	label?: string;
}

/** One dispatch of a Reducer: the source size, textures and the tool's group 1. */
export interface ReducerDispatch {
	/** Source size in texels. Required with a load hook; otherwise the source level's size. */
	width?: number;
	height?: number;
	/**
	 * Array layers to reduce, from layer 0 (workgroup z). Default: the source's or first
	 * output's layers with a '2d-array' dimension, else 1. (A count, not `[ first, count ]`:
	 * the tile kernel has no layer offset, and hooks see the workgroup's layer.)
	 */
	layerCount?: number;
	/** The source (no load hook): a texture (its level 0) or `{ texture, mipLevel }`. */
	source?: GPUTexture | TextureLevel;
	/** One output per emitted level, in `levels` order (no store hook). Each must be at least the level's size. */
	outputs?: readonly ( GPUTexture | TextureLevel )[];
	/** The tool's group-1 bind group, created from `bindGroupLayouts[ 1 ]` (or an equivalent layout). */
	bindGroup?: GPUBindGroup;
}

/** A `copyBufferToTexture` of one emitted level (output 'buffer'), recorded by `encodeCopies`. */
export interface ReducerCopy {
	texture: GPUTexture;
	mipLevel: number;
	/** Byte offset in the Reducer's buffer, a multiple of 256. */
	offset: number;
	bytesPerRow: number;
	rowsPerImage: number;
	width: number;
	height: number;
	/** Array layers copied, from layer 0. */
	layerCount: number;
}

/**
 * One dispatch of a Reducer, as `prepare( dispatch )` returns it: set `pipeline`,
 * `bindGroup` at group 0 and your group 1, `dispatchWorkgroups( ...workgroups )`,
 * then (output 'buffer', after the pass) record `copies` from `buffer`.
 */
export interface PreparedReducer {
	pipeline: GPUComputePipeline;
	/** Group 0, owned and cached by the Reducer. */
	bindGroup: GPUBindGroup;
	/** The tool's group 1 from the dispatch, if any. */
	toolBindGroup: GPUBindGroup | undefined;
	workgroups: [ number, number, number ];
	/** Output 'buffer': the buffer the dispatch writes (owned by the Reducer), and one copy per emitted level. */
	buffer?: GPUBuffer;
	copies?: readonly ReducerCopy[];
}

/** A cached group-0 entry, per size and textures. */
interface Prepared {
	bindGroup: GPUBindGroup;
	workgroups: [ number, number, number ];
	buffer?: GPUBuffer;
	copies?: readonly ReducerCopy[];
	/** Textures the entry binds or copies into, for `release( texture )`. */
	textures: GPUTexture[];
	/** Buffers the Reducer created for the entry (size uniform, output buffer and layout): destroyed on eviction and release. */
	owned: GPUBuffer[];
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

const mipSize = ( size: number, level: number ) => Math.max( 1, size >> level );

/** Bind groups (and their size uniforms and output buffers) kept per Reducer, most recent last. */
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
	/** The compute entry point of {@link Reducer.wgsl}. */
	readonly entryPoint = 'main';
	/** Pipeline layout: [ group 0 (ours), group 1 (the tool's, if it has bindings) ]. */
	readonly layout: GPUPipelineLayout;
	/** [ group 0 (the Reducer's), group 1 (the tool's, from `bindings`, if any) ]. Create the tool's bind group from index 1. */
	readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
	/** The compiled pipeline, shared by equal Reducers on a device. */
	readonly pipeline: GPUComputePipeline;
	/** Emitted levels, source-relative (1 = half size). */
	readonly levels: readonly number[];
	/** Edge policy: 'zero-weight' (⌈n/2⌉ sizes) or 'clamp' (SPD, ⌊n/2⌋). */
	readonly edges: TileEdgePolicy;
	/** Accumulator channels per texel (4 or 8). */
	readonly channels: Channels;
	readonly label: string;
	/** How output textures are written; undefined with a store hook. */
	readonly output: OutputPath | undefined;
	private readonly options: ReducerOptions;
	private readonly cache = new Map<string, Prepared>();

	constructor( device: GPUDevice, options: ReducerOptions ) {

		const edges = options.edges ?? 'zero-weight';
		if ( edges !== 'zero-weight' && edges !== 'clamp' ) throw new Error( `downsampler: a Reducer takes edges 'zero-weight' or 'clamp', got ${ JSON.stringify( edges ) }; 'exact' mip chains (a batch per odd size) are Downsampler's.` );
		const reduction = options.reduction ?? 'average';
		const { channels } = resolveReduction( reduction, options.channels );
		checkToolBindings( options.bindings );
		const format = options.format;
		let output: OutputPath | undefined;

		if ( options.store === undefined && format !== undefined && isSupportedFormat( format ) && isIntFormat( format ) ) throw new Error( `downsampler: a Reducer accumulates floats, so it doesn't write integer formats (${ format }); use a store hook, or a Downsampler (exact integer chains).` );
		if ( options.store === undefined && format !== undefined && isSupportedFormat( format ) ) {

			output = options.output ?? ( storageWritable( format, device.features ) || ! bufferWritable( format ) ? 'storage' : 'buffer' );
			if ( output === 'storage' ) {

				const { feature, storage }: FormatInfo = formats[ format ];
				if ( storage === false ) throw new Error( `downsampler: ${ format } can't be a storage texture; use output 'buffer'.` );
				if ( feature !== undefined && ! device.features.has( feature ) ) throw new Error( `downsampler: output format ${ format } needs the "${ feature }" device feature${ bufferWritable( format ) ? ', or output \'buffer\'' : '' }.` );
				if ( options.levels.length > device.limits.maxStorageTexturesPerShaderStage ) throw new Error( `downsampler: ${ options.levels.length } output textures exceed this device's maxStorageTexturesPerShaderStage (${ device.limits.maxStorageTexturesPerShaderStage }); emit fewer levels, use output 'buffer' or a store hook.` );

			} else if ( output === 'buffer' ) {

				if ( ! bufferWritable( format ) ) throw new Error( `downsampler: format ${ format } has ${ formats[ format ].bytesPerTexel }-byte texels, and the buffer output path writes 4-byte words.` );

			} else throw new Error( `downsampler: output must be 'storage' or 'buffer', got ${ JSON.stringify( output ) }.` );

		} else if ( options.output !== undefined && options.store !== undefined ) throw new Error( 'downsampler: a store hook replaces the output textures, so it takes no output path.' );

		const kernel: KernelOptions = {
			edges, levels: options.levels, reduction, format, dimension: options.dimension ?? '2d', sourceFormat: options.sourceFormat,
			load: options.load, store: options.store, size: options.size, bindings: options.bindings?.wgsl, output, f16Rounding: options.f16Rounding, ...( channels === 8 ? { channels } : {} )
		};
		const deepest = options.levels[ options.levels.length - 1 ];
		const invocations = workgroupSide( tileSize( deepest ) ) ** 2;
		if ( Number.isInteger( deepest ) && invocations > device.limits.maxComputeInvocationsPerWorkgroup ) throw new Error( `downsampler: ${ deepest } levels take workgroups of ${ invocations } invocations; this device allows ${ device.limits.maxComputeInvocationsPerWorkgroup } (maxComputeInvocationsPerWorkgroup; compatibility mode defaults to 128). Emit at most 4 levels or request a higher limit (requiredLimits( adapter ) from downsampler/webgpu).` );
		const tileBytes = tileSize( deepest ) ** 2 * channels * 4;
		if ( Number.isInteger( deepest ) && tileBytes > device.limits.maxComputeWorkgroupStorageSize ) throw new Error( `downsampler: ${ deepest } levels of ${ channels }-channel accumulators take ${ tileBytes } bytes of workgroup memory; this device allows ${ device.limits.maxComputeWorkgroupStorageSize } (maxComputeWorkgroupStorageSize). Emit fewer levels or request a higher limit.` );

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
		this.channels = channels;
		this.edges = edges;
		this.label = label;
		this.output = output;
		this.options = { ...options, reduction };

	}

	/** Size of emitted level `level` (1 = half) for a `width` × `height` source. */
	levelSize( level: number, width: number, height: number ): [ number, number ] {

		return [ levelSize( width, level, this.edges ), levelSize( height, level, this.edges ) ];

	}

	/** Workgroup counts for a `width` × `height` source with `layerCount` array layers. */
	workgroups( width: number, height: number, layerCount = 1 ): [ number, number, number ] {

		const { groupsX, groupsY } = tileDispatch( width, height, this.levels[ this.levels.length - 1 ], this.edges );
		return [ groupsX, groupsY, layerCount ];

	}

	/**
	 * The group-0 bind group for a dispatch: the source texture, the output
	 * textures and the size uniform, as this kernel needs them. Cached per size
	 * and textures (the most recent 16). Same as `prepare( dispatch ).bindGroup`.
	 */
	bindGroup( dispatch: ReducerDispatch = {} ): GPUBindGroup {

		return this.entry( dispatch ).bindGroup;

	}

	/**
	 * Record the dispatch into a compute pass the caller owns. With output
	 * 'buffer', call {@link Reducer.encodeCopies} after the pass ends.
	 */
	encodePass( pass: GPUComputePassEncoder, dispatch: ReducerDispatch = {} ): void {

		const { pipeline, bindGroup, toolBindGroup, workgroups } = this.prepare( dispatch );
		pass.setPipeline( pipeline );
		pass.setBindGroup( 0, bindGroup );
		setToolBindGroup( pass, toolBindGroup );
		pass.dispatchWorkgroups( ...workgroups );

	}

	/** Record the dispatch in its own compute pass (and, with output 'buffer', the copies into the output textures). */
	encode( encoder: GPUCommandEncoder, dispatch: ReducerDispatch = {} ): void {

		const pass = encoder.beginComputePass( { label: this.label } );
		this.encodePass( pass, dispatch );
		pass.end();
		this.encodeCopies( encoder, dispatch );

	}

	/**
	 * Output 'buffer': copy each emitted level from the Reducer's buffer into its
	 * output texture. Record it after the pass holding the dispatch has ended
	 * (`encodePass`, or a self-dispatch with `bindGroup( dispatch )`), before
	 * anything reads the outputs. Does nothing on the storage path.
	 */
	encodeCopies( encoder: GPUCommandEncoder, dispatch: ReducerDispatch = {} ): void {

		const { buffer, copies = [] } = this.entry( dispatch );
		for ( const { texture, mipLevel, offset, bytesPerRow, rowsPerImage, width, height, layerCount } of copies ) {

			encoder.copyBufferToTexture( { buffer: buffer!, offset, bytesPerRow, rowsPerImage }, { texture, mipLevel }, [ width, height, layerCount ] );

		}

	}

	/**
	 * What `encodePass` records for a dispatch, for tools that dispatch the
	 * kernel themselves: the pipeline, the group-0 bind group, the dispatch's
	 * group 1, the workgroup counts and (output 'buffer') the copies that follow
	 * the pass. Validates like `encode` and throws with the reason. Cached per
	 * size and textures, the most recent 16 (see {@link Reducer.release}).
	 */
	prepare( dispatch: ReducerDispatch = {} ): PreparedReducer {

		const { bindGroup, workgroups, buffer, copies } = this.entry( dispatch );
		const reason = toolBindGroupReason( this.bindGroupLayouts.length > 1, dispatch.bindGroup, 'this Reducer' );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		return { pipeline: this.pipeline, bindGroup, toolBindGroup: dispatch.bindGroup, workgroups, ...( buffer !== undefined ? { buffer, copies } : {} ) };

	}

	/**
	 * Drop cached bind groups and destroy the buffers the Reducer created for
	 * them (size uniforms, buffer-output buffers): every entry, or only those
	 * that bind or copy into `texture`. Call it after submitting the last work
	 * that uses them. Pipelines stay cached per device; your textures, buffers
	 * and bind groups are never destroyed. Entries evicted from the 16-entry
	 * cache have their buffers destroyed once the queue has finished the work
	 * submitted so far.
	 */
	release( texture?: GPUTexture ): void {

		for ( const [ key, entry ] of this.cache ) {

			if ( texture !== undefined && ! entry.textures.includes( texture ) ) continue;
			for ( const buffer of entry.owned ) buffer.destroy();
			this.cache.delete( key );

		}

	}

	private entry( dispatch: ReducerDispatch ): Prepared {

		const { load, store, size, format, dimension = '2d' } = this.options;
		// Renamed before 1.0: a count is `layerCount` (a `layers` field is always [ first, count ]).
		if ( ( dispatch as { layers?: unknown } ).layers !== undefined ) throw new Error( 'downsampler: a Reducer dispatch takes `layerCount` (the layers from 0), not `layers`.' );
		const shape = dispatch.source === undefined ? null : sourceShapeReason( dispatch.source );
		if ( shape !== null ) throw new Error( `downsampler: ${ shape }.` );
		const source = dispatch.source === undefined ? undefined : asTextureLevel( dispatch.source );
		const outputs = dispatch.outputs?.map( output => {

			const reason = sourceShapeReason( output, 'each output' );
			if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
			return asTextureLevel( output );

		} );
		const array = dimension === '2d-array';

		if ( load === undefined && source === undefined ) throw new Error( 'downsampler: this Reducer reads a source texture; pass `source`.' );
		if ( load !== undefined && source !== undefined ) throw new Error( 'downsampler: this Reducer has a load hook; it takes no source texture.' );
		if ( store === undefined && outputs?.length !== this.levels.length ) throw new Error( `downsampler: pass one output per emitted level (${ this.levels.length }).` );
		if ( store !== undefined && outputs !== undefined ) throw new Error( 'downsampler: this Reducer has a store hook; it takes no output textures.' );

		let width = dispatch.width, height = dispatch.height;
		if ( source !== undefined ) {

			const { texture, mipLevel } = source;
			const reason = sourceTextureReason( source, 'source', { mismatch: format => `a Reducer reads float-sampleable sources, not ${ format }; read integer textures in a load hook` } );
			if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
			const sw = mipSize( texture.width, mipLevel ), sh = mipSize( texture.height, mipLevel );
			if ( ( width !== undefined && width !== sw ) || ( height !== undefined && height !== sh ) ) throw new Error( `downsampler: width and height (${ width }×${ height }) must match the source level (${ sw }×${ sh }), or be left out.` );
			width = sw; height = sh;

		}

		if ( width === undefined || height === undefined ) throw new Error( 'downsampler: pass the source width and height.' );
		const layers = dispatch.layerCount ?? ( array ? ( source ?? outputs?.[ 0 ] )?.texture.depthOrArrayLayers ?? 1 : 1 );
		if ( ! Number.isInteger( layers ) || layers < 1 ) throw new Error( `downsampler: layerCount must be a positive integer, got ${ layers }.` );
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
		const owned: GPUBuffer[] = [];
		if ( source !== undefined ) {

			if ( source.texture.depthOrArrayLayers < layers ) throw new Error( `downsampler: source has ${ source.texture.depthOrArrayLayers } layers, the dispatch ${ layers }.` );
			entries.push( { binding: 1, resource: sourceView( source, dimension, `${ this.label } source` ) } );

		}

		outputs?.forEach( ( { texture, mipLevel }, i ) => {

			const level = this.levels[ i ];
			const [ w, h ] = this.levelSize( level, width!, height! );
			const ow = mipSize( texture.width, mipLevel ), oh = mipSize( texture.height, mipLevel );
			if ( texture.format !== format ) throw new Error( `downsampler: output for level ${ level } is ${ texture.format }; this Reducer writes ${ format }.` );
			if ( this.output === 'storage' && ( texture.usage & GPUTextureUsage.STORAGE_BINDING ) === 0 ) throw new Error( `downsampler: output for level ${ level } needs STORAGE_BINDING usage.` );
			if ( this.output === 'buffer' && ( texture.usage & GPUTextureUsage.COPY_DST ) === 0 ) throw new Error( `downsampler: output for level ${ level } needs COPY_DST usage (output 'buffer').` );
			if ( ! Number.isInteger( mipLevel ) || mipLevel < 0 || mipLevel >= texture.mipLevelCount ) throw new Error( `downsampler: output mipLevel for level ${ level } is outside its texture.` );
			if ( ow < w || oh < h ) throw new Error( `downsampler: output for level ${ level } is ${ ow }×${ oh }, smaller than the level (${ w }×${ h }).${ this.edges === 'zero-weight' ? ' \'zero-weight\' levels are ⌈n/2⌉, so a mip of the source size is one texel short at odd sizes: use separate textures or a store hook.' : '' }` );
			if ( texture.depthOrArrayLayers < layers ) throw new Error( `downsampler: output for level ${ level } has ${ texture.depthOrArrayLayers } layers, the dispatch ${ layers }.` );
			if ( this.output === 'storage' ) entries.push( { binding: i + 2, resource: texture.createView( { label: `${ this.label } level ${ level }`, dimension, baseMipLevel: mipLevel, mipLevelCount: 1 } ) } );

		} );

		// Compatibility mode binds one view (mip range) of a texture per dispatch. Buffer
		// outputs aren't bound (copies write them), so they may share the source's texture.
		const bound = [ ...( source ? [ source.texture ] : [] ), ...( this.output === 'storage' ? outputs ?? [] : [] ).map( output => output.texture ) ];
		if ( new Set( bound ).size < bound.length && bound.some( texture => isCompatibilityTexture( this.device, texture ) ) ) throw new Error( 'downsampler: in compatibility mode a dispatch binds one view of a texture, so the source and each output level must be separate textures (or use output \'buffer\' or a store hook).' );
		const layout = this.output === 'buffer' && outputs !== undefined ? outputBufferLayout( this.levels.map( level => this.levelSize( level, width!, height! ) ), layers, formats[ format! ].bytesPerTexel ) : undefined;
		const limit = Math.min( this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize );
		if ( layout !== undefined && layout.size > limit ) throw new Error( `downsampler: the buffer output needs a ${ layout.size }-byte buffer, above this device's maxStorageBufferBindingSize / maxBufferSize (${ limit }).` );

		// Validated: from here on the entry creates buffers, which it owns.
		if ( load !== undefined && size === undefined ) {

			const buffer = this.device.createBuffer( { label: `${ this.label } size`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			this.device.queue.writeBuffer( buffer, 0, Uint32Array.of( width, height, 0, 0 ) );
			owned.push( buffer );
			entries.push( { binding: sizeBinding, resource: { buffer } } );

		}

		// Buffer output: each emitted level at its own offset of one buffer, rows padded
		// to 256 bytes, copied into its output texture by encodeCopies.
		let buffer: GPUBuffer | undefined, copies: ReducerCopy[] | undefined;
		if ( layout !== undefined && outputs !== undefined ) {

			buffer = this.device.createBuffer( { label: `${ this.label } output`, size: layout.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
			const words = new Uint32Array( outputLayoutBytes / 4 );
			words.set( [ width, height ] );
			layout.levels.forEach( ( level, i ) => { words[ 4 + i ] = level.offset / 4; words[ 12 + i ] = level.bytesPerRow / 4; words[ 20 + i ] = level.rowsPerImage; } );
			const uniform = this.device.createBuffer( { label: `${ this.label } output layout`, size: outputLayoutBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			this.device.queue.writeBuffer( uniform, 0, words );
			owned.push( buffer, uniform );
			entries.push( { binding: outputLayoutBinding, resource: { buffer: uniform } }, { binding: outputBufferBinding, resource: { buffer } } );
			copies = layout.levels.map( ( level, i ) => ( { texture: outputs[ i ].texture, mipLevel: outputs[ i ].mipLevel, offset: level.offset, bytesPerRow: level.bytesPerRow, rowsPerImage: level.rowsPerImage, width: level.width, height: level.height, layerCount: layers } ) );

		}

		const textures = [ ...( source ? [ source.texture ] : [] ), ...( outputs ?? [] ).map( output => output.texture ) ];
		const prepared: Prepared = { bindGroup: this.device.createBindGroup( { label: this.label, layout: this.bindGroupLayouts[ 0 ], entries } ), workgroups, buffer, copies, textures, owned };
		this.cache.set( key, prepared );
		if ( this.cache.size > cacheSize ) {

			// The evicted entry may still be recorded in an unsubmitted encoder: destroy its buffers later.
			const [ oldest, evicted ] = this.cache.entries().next().value!;
			this.cache.delete( oldest );
			retireBuffers( this.device, evicted.owned );

		}

		return prepared;

	}

}
