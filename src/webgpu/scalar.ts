import { sampledChannels } from '../core/formats.js';
import { isSumOp, scalarBlock, scalarInvocations, scalarOps, scalarSide, scalarWeightChannel, scalarWorkgroups, type ScalarOp, type ScalarReduce } from '../core/scalar.js';
import type { ViewDimension } from './kernel.js';
import { lanes, payloadCombine } from './lanes.js';
import { asTextureLevel, bindingViewDimension, levelSizeOf, sourceShapeReason, sourceTextureReason, sourceView, type TextureLevel } from './source.js';
import { checkHooks, checkToolBindings, hookArity, retireBuffers, setToolBindGroup, toolBindGroupLayout, toolBindGroupReason, toolGroupLayouts, toolPipelineLayout, type ToolBindings } from './tool.js';

/** Options of `new ScalarReduction( device, options )`. See SPEC.md §4 Scalar reductions. */
export interface ScalarReductionOptions {
	/**
	 * One op for all channels ('sum', 'average', 'min', 'max', …), 'minmax'
	 * (even channels min, odd channels max), or one op per channel. Default
	 * 'average'. Named as `Downsampler`'s and `Reducer`'s `reduction`.
	 */
	reduction?: ScalarReduce;
	/**
	 * WGSL defining `fn load( texel: vec2i, layer: u32 ) -> vec4f`, the value of
	 * one in-image texel. It reads the tool's own `@group(1)` resources. Without
	 * it, runs read a source texture (`ScalarRun.source`) with `textureLoad`.
	 */
	load?: string;
	/**
	 * WGSL defining `fn finish( value: vec4f, weight: f32 )`, run by one
	 * invocation per layer with the emitted value (averages and weighted means
	 * already divided) and the total weight (Σ w with a weight channel, else the
	 * texel count). A third parameter `layer: u32` is optional. Without it, runs
	 * write `{ value: vec4f, weight: f32 }` (32 bytes per layer) to `ScalarRun.output`.
	 */
	finish?: string;
	/** Declarations and layout of the tool's `@group(1)`, used by `load` and `finish`. */
	bindings?: ToolBindings;
	/** Texels per invocation along each axis (1, 2, 4 or 8). Default 4: a 64 × 64 tile per workgroup. */
	block?: number;
	/**
	 * Accumulator channels: 4 (a `vec4f`, default) or 8. With 8, `load`
	 * returns and `finish` takes `DsWide { lo: vec4f, hi: vec4f }` (channels
	 * 0–3, 4–7), partials are 32 bytes and output results 48
	 * (`{ value: DsWide, weight: f32 }`). Eight ops imply it. Without a load
	 * hook, an 8-channel reduction only combines partials (`ScalarRun.partials`).
	 */
	channels?: 4 | 8;
}

/** Options of the pure generator {@link createScalarKernel}: a reduction plus the source shape. */
export interface ScalarKernelOptions extends ScalarReductionOptions {
	/** Source texture view dimension when there is no `load` hook. Default '2d'. */
	dimension?: ViewDimension;
	/** Read single-channel sources (depth, r*) as (v, v, v, v). Default false. */
	splat?: boolean;
}

/**
 * One run of a ScalarReduction: a source texture, a size for a load hook, or
 * (combine only) partials a tool wrote; plus where results go without `finish`.
 */
export interface ScalarRun {
	/**
	 * Source, without a `load` hook: a texture (its level 0) or `{ texture, mipLevel }`,
	 * any float-sampleable 2D format, including depth (its depth aspect).
	 */
	source?: GPUTexture | TextureLevel;
	/** Image size in texels: required with a `load` hook; with a source, its level's size (if given, it must match). */
	width?: number;
	height?: number;
	/**
	 * Layers to reduce, each to its own result: `[ first, count ]`. Default all
	 * of the source's layers, or `[ 0, 1 ]` with a `load` hook.
	 */
	layers?: readonly [ number, number ];
	/** The tool's `@group(1)` bind group, required with `bindings`. */
	bindGroup?: GPUBindGroup;
	/**
	 * Where results go without a `finish` hook (`resultBytes` per layer).
	 * `offset` must be a multiple of `minStorageBufferOffsetAlignment` (256 by default).
	 */
	output?: { buffer: GPUBuffer, offset?: number };
	/**
	 * Combine only: `count` accumulations per layer that a tool (e.g. a
	 * Reducer's store hook) wrote to `buffer`, one vec4f each (a DsWide with 8 channels), layer after
	 * layer from `offset` (a multiple of `minStorageBufferOffsetAlignment`).
	 * Sum-like channels hold sums ('weighted' channels Σ w·v, 'weight' Σ w),
	 * min/max channels the min/max. The run records the combine dispatch alone.
	 * The tool's group 1 for this run must not bind `buffer` as well (it would
	 * alias a writable binding).
	 */
	partials?: { buffer: GPUBuffer, offset?: number, count: number };
	/**
	 * Combine only: the texels behind the partials, the divisor of 'average'
	 * channels and the weight passed on without a 'weight' channel. Required
	 * then; otherwise optional.
	 */
	texels?: number;
}

/** What `prepare( run )` returns, for tools that dispatch the two passes themselves: `partials` (unless combine-only), then `combine`, each with `bindGroup` at group 0 and `toolBindGroup` at group 1. */
export interface PreparedScalar {
	/** Null for a combine-only run (`ScalarRun.partials`). */
	partials: GPUComputePipeline | null;
	combine: GPUComputePipeline;
	/** `@group(0)`, owned by the reduction. */
	bindGroup: GPUBindGroup;
	/** The tool's `@group(1)`, if any. */
	toolBindGroup: GPUBindGroup | undefined;
	/** Workgroups of the partials dispatch: [ x, y, layers ]. */
	workgroups: [ number, number, number ];
	/** Workgroups of the combine dispatch: one per layer. */
	combineWorkgroups: [ number, number, number ];
}

/** Bytes per result written to `ScalarRun.output`: `struct { value: vec4f, weight: f32 }`. */
export const scalarResultBytes = 32;

/** Bytes per output result for `channels` (4: 32, 8: `{ value: DsWide, weight: f32 }`, 48). */
export function scalarResultSize( channels: number ): number {

	return channels === 8 ? 48 : scalarResultBytes;

}

/** Bindings of the reduction's own `@group(0)`. */
export function scalarLayoutEntries( { load, finish, dimension = '2d', combineOnly = false }: Pick<ScalarKernelOptions, 'load' | 'finish' | 'dimension'> & { combineOnly?: boolean } ): GPUBindGroupLayoutEntry[] {

	const visibility = GPUShaderStage.COMPUTE;
	const entries: GPUBindGroupLayoutEntry[] = [
		{ binding: 0, visibility, buffer: { type: 'uniform' } },
		{ binding: 1, visibility, buffer: { type: 'storage' } }
	];
	if ( load === undefined && ! combineOnly ) entries.push( { binding: 2, visibility, texture: { sampleType: 'unfilterable-float', viewDimension: dimension } } );
	if ( finish === undefined ) entries.push( { binding: 3, visibility, buffer: { type: 'storage' } } );
	return entries;

}

/**
 * WGSL for both dispatches, entry points `partials` (16 × 16 invocations, one
 * workgroup per tile of 16·block texels square, z = layer) and `combine` (256
 * invocations, one workgroup per layer). See SPEC.md §4 Scalar reductions and
 * the combination order in core/scalar.ts, which the CPU oracle reproduces.
 *
 * Generator identifiers start with `ds_`; tool code should avoid that prefix.
 */
export function createScalarKernel( options: ScalarKernelOptions = {} ): string {

	const { load, finish, bindings, dimension = '2d', splat = false, block = scalarBlock } = options;
	const ops = scalarOps( options.reduction, options.channels );
	if ( ! [ 1, 2, 4, 8 ].includes( block ) ) throw new Error( 'downsampler: block must be 1, 2, 4 or 8.' );
	checkHooks( { load, finish } );
	checkToolBindings( bindings );
	const weightChannel = scalarWeightChannel( ops );
	const tile = scalarSide * block;
	const array = dimension === '2d-array';
	// The value type (vec4f, or DsWide for 8 channels). An 8-channel reduction
	// without a load hook has no source to read: it only combines partials.
	const lane = lanes( ops.length as 4 | 8 );
	const { type: T, channel: ch, build, halves, join, declarations } = lane;
	const combineOnly = ops.length === 8 && load === undefined;
	const sumsOf = ( from: number ) => `vec4<bool>(${ ops.slice( from, from + 4 ).map( op => isSumOp( op ) ).join( ', ' ) })`;

	let code = `${ declarations }struct DsParams {
	size: vec2u,
	firstLayer: u32,
	groupsX: u32,
	partialCount: u32,
	count: u32,
}
struct DsResult {
	value: ${ T },
	weight: f32,
}
@group(0) @binding(0) var<uniform> ds_params: DsParams;
@group(0) @binding(1) var<storage, read_write> ds_partials: array<${ T }>;
var<workgroup> ds_slots: array<${ T }, ${ scalarInvocations }>;
`;

	if ( load === undefined && ! combineOnly ) {

		code += `@group(0) @binding(2) var ds_source: texture_2d${ array ? '_array' : '' }<f32>;
fn load(texel: vec2i, layer: u32) -> vec4f {
	return textureLoad(ds_source, texel${ array ? ', i32(layer)' : '' }, 0)${ splat ? '.xxxx' : '' };
}
`;

	}

	if ( finish === undefined ) code += '@group(0) @binding(3) var<storage, read_write> ds_output: array<DsResult>;\n';
	if ( bindings !== undefined ) code += `\n${ bindings.wgsl }\n`;
	if ( load !== undefined ) code += `\n${ load }\n`;
	if ( finish !== undefined ) code += `\n${ finish }\n`;

	const combine = ops.map( ( op, c ) => op === 'min' ? `min(${ ch( 'a', c ) }, ${ ch( 'b', c ) })` : op === 'max' ? `max(${ ch( 'a', c ) }, ${ ch( 'b', c ) })` : `${ ch( 'a', c ) } + ${ ch( 'b', c ) }` );
	const sums = sumsOf( 0 );
	const term = ops.map( ( op, c ) => op === 'weighted' ? `${ ch( 'v', c ) } * ${ ch( 'v', weightChannel ) }` : ch( 'v', c ) );
	// Zero the sum-like channels of a value where `condition` holds (per vec4f half).
	const zeroSums = ( value: string[], condition: string ) => join( value.map( ( half, i ) => `select(${ half }, vec4f(0), ${ sumsOf( 4 * i ) } & vec4<bool>(${ condition }))` ) );

	// Payload folds (argmin/argmax with payload channels) need a comparison first.
	code += `\n${ payloadCombine( ops, lane, c => combine[ c ] ) ?? `fn ds_combine(a: ${ T }, b: ${ T }) -> ${ T } {
	return ${ build( combine ) };
}
` }`;

	if ( ! combineOnly ) code += ops.length === 4 ? `
// Sum-like channels of off-image texels contribute 0; min and max see the
// clamped (in-image) texel, a harmless duplicate.
fn ds_term(texel: vec2u, layer: u32) -> vec4f {
	let inside = all(texel < ds_params.size);
	let v = load(vec2i(min(texel, ds_params.size - vec2u(1))), layer);
	return select(vec4f(${ term.join( ', ' ) }), vec4f(0), ${ sums } & vec4<bool>(!inside));
}
` : `
// Sum-like channels of off-image texels contribute 0; min and max see the
// clamped (in-image) texel, a harmless duplicate.
fn ds_term(texel: vec2u, layer: u32) -> DsWide {
	let inside = all(texel < ds_params.size);
	let v = load(vec2i(min(texel, ds_params.size - vec2u(1))), layer);
	return ${ zeroSums( [ `vec4f(${ term.slice( 0, 4 ).join( ', ' ) })`, `vec4f(${ term.slice( 4 ).join( ', ' ) })` ], '!inside' ) };
}
`;

	code += `
// Stride-halving tree over the workgroup's 256 slots; the total ends in slot 0.
fn ds_tree(index: u32) {
	for (var stride = ${ scalarInvocations / 2 }u; stride > 0u; stride >>= 1u) {
		workgroupBarrier();
		if (index < stride) { ds_slots[index] = ds_combine(ds_slots[index], ds_slots[index + stride]); }
	}
	workgroupBarrier();
}
`;

	if ( ! combineOnly ) code += `
@compute @workgroup_size(${ scalarSide }, ${ scalarSide })
fn partials(@builtin(local_invocation_id) local: vec3u, @builtin(local_invocation_index) index: u32, @builtin(workgroup_id) group: vec3u) {
	let origin = group.xy * ${ tile }u + local.xy;
	let layer = ds_params.firstLayer + group.z;
`;

	// Adjacent invocations read adjacent texels; each reads block × block of them, 16 apart.
	let names: string[] = [];
	if ( ! combineOnly ) for ( let jy = 0; jy < block; jy ++ ) for ( let jx = 0; jx < block; jx ++ ) {

		const name = `v${ jy * block + jx }`;
		code += `	let ${ name } = ds_term(origin + vec2u(${ jx * scalarSide }u, ${ jy * scalarSide }u), layer);\n`;
		names.push( name );

	}

	for ( let level = 0; ! combineOnly && names.length > 1; level ++ ) {

		const half = names.length / 2;
		const next = names.slice( 0, half ).map( ( _, i ) => `t${ level }_${ i }` );
		for ( let i = 0; i < half; i ++ ) code += `	let ${ next[ i ] } = ds_combine(${ names[ i ] }, ${ names[ i + half ] });\n`;
		names = next;

	}

	const layerArgument = finish !== undefined && hookArity( finish, 'finish' ) >= 3 ? ', ds_params.firstLayer + group.z' : '';
	const normalise = ops.map( ( op, c ) => op === 'average' ? `${ ch( 'total', c ) } / f32(ds_params.count)` : op === 'weighted' ? `select(${ ch( 'total', c ) } / weight, 0.0, weight == 0.0)` : ch( 'total', c ) );

	if ( ! combineOnly ) code += `	ds_slots[index] = ${ names[ 0 ] };
	ds_tree(index);
	if (index == 0u) { ds_partials[group.z * ds_params.partialCount + group.y * ds_params.groupsX + group.x] = ds_slots[0]; }
}
`;

	code += `
@compute @workgroup_size(${ scalarInvocations })
fn combine(@builtin(local_invocation_index) index: u32, @builtin(workgroup_id) group: vec3u) {
	let n = ds_params.partialCount;
	let base = group.z * n;
${ ops.length === 4 ? `	var value = select(ds_partials[base + min(index, n - 1u)], vec4f(0), ${ sums } & vec4<bool>(index >= n));` : `	let first = ds_partials[base + min(index, n - 1u)];
	var value = ${ zeroSums( halves( 'first' ), 'index >= n' ) };` }
	for (var i = index + ${ scalarInvocations }u; i < n; i += ${ scalarInvocations }u) { value = ds_combine(value, ds_partials[base + i]); }
	ds_slots[index] = value;
	ds_tree(index);
	if (index == 0u) {
		let total = ds_slots[0];
		let weight = ${ weightChannel >= 0 ? ch( 'total', weightChannel ) : 'f32(ds_params.count)' };
		let result = ${ build( normalise ) };
		${ finish !== undefined ? `finish(result, weight${ layerArgument });` : 'ds_output[group.z] = DsResult(result, weight);' }
	}
}
`;

	return code;

}

interface Plan {
	key: string;
	texture: GPUTexture | undefined;
	output: GPUBuffer | undefined;
	/** The tool's partials (combine-only runs) or the reduction's own buffer. */
	partials: GPUBuffer;
	params: GPUBuffer;
	/** The buffers this reduction created for the plan (params, and its own partials): destroyed on eviction and release. */
	owned: GPUBuffer[];
	bindGroup: GPUBindGroup;
	/** Null for a combine-only run. */
	partialsPipeline: GPUComputePipeline | null;
	combine: GPUComputePipeline;
	workgroups: [ number, number, number ];
}

/** One compiled variant of a scalar reduction (per source view dimension and splat). */
export interface ScalarKernel {
	dimension: ViewDimension;
	splat: boolean;
	layout: GPUBindGroupLayout;
	partials: GPUComputePipeline;
	combine: GPUComputePipeline;
	wgsl: string;
}

/** The combine pass alone, for partials a tool wrote (`ScalarRun.partials`). */
export interface ScalarCombineKernel {
	/** `@group(0)` without the source texture: uniform (0), partials (1), output (3, without `finish`). */
	layout: GPUBindGroupLayout;
	combine: GPUComputePipeline;
	wgsl: string;
}

/**
 * Reduce an image to one value per layer in two dispatches: per-workgroup
 * partials into a storage buffer, then one workgroup per layer combines them
 * and calls `finish` (or writes the result to a buffer). Every texel is counted
 * once at any size. Pipelines are compiled per source view dimension and
 * channel count; bind groups and the partials buffer are cached per source
 * (or output buffer) and run shape.
 *
 * Exposure, as in FSR 3:
 * ```ts
 * const exposure = new ScalarReduction( device, {
 * 	bindings: { wgsl: '@group(1) @binding(0) var colour: texture_2d<f32>; …', entries },
 * 	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f { … log( max( luma, 6.1e-5 ) ) … }',
 * 	finish: 'fn finish( value: vec4f, weight: f32 ) { … smoothing, store … }',
 * } );
 * exposure.encodePass( pass, { width, height, bindGroup } );
 * ```
 *
 * A run with `partials` records the combine pass alone, over accumulations a
 * tool wrote (e.g. a Reducer's deepest level through its store hook): one
 * dispatch, so "Reducer levels + scalar of the same source" is two dispatches
 * with one source read.
 */
export class ScalarReduction {

	readonly device: GPUDevice;
	/** The options the reduction was created with. */
	readonly options: Readonly<ScalarReductionOptions>;
	/** Per-channel ops, resolved from `reduction` (4 or 8). */
	readonly ops: readonly ScalarOp[];
	/** Texels per invocation along each axis (the tile is 16 · block). */
	readonly block: number;
	/** Bytes per partial: one vec4f, or a DsWide (32) with 8 channels. */
	readonly partialBytes: number;
	/** Bytes per output result: 32, or 48 with 8 channels ({@link scalarResultSize}). */
	readonly resultBytes: number;
	private readonly toolLayout: GPUBindGroupLayout | undefined;
	private readonly variants = new Map<string, ScalarKernel>();
	private combineOnly: ScalarCombineKernel | undefined;
	private groupLayouts: GPUBindGroupLayout[] | undefined;
	private readonly plans = new WeakMap<object, Plan[]>();
	/** Every plan owner, held weakly (so caching never keeps a texture alive), for `release()` without a resource. */
	private owners: WeakRef<object>[] = [];

	constructor( device: GPUDevice, options: ScalarReductionOptions = {} ) {

		// Renamed before 1.0: catch the old key instead of silently averaging.
		if ( 'reduce' in options ) throw new Error( 'downsampler: ScalarReduction takes `reduction`, not `reduce`.' );
		this.device = device;
		this.options = { ...options };
		this.ops = scalarOps( options.reduction, options.channels );
		this.block = options.block ?? scalarBlock;
		const { maxComputeInvocationsPerWorkgroup: invocations, maxComputeWorkgroupSizeX: sizeX } = device.limits;
		if ( invocations < scalarInvocations || sizeX < scalarInvocations ) throw new Error( `downsampler: a ScalarReduction needs workgroups of ${ scalarInvocations } invocations (${ scalarInvocations } wide); this device allows ${ invocations } (maxComputeInvocationsPerWorkgroup) and ${ sizeX } wide (maxComputeWorkgroupSizeX). Compatibility-mode devices default to 128: request more (requiredLimits( adapter ) from downsampler/webgpu).` );
		this.partialBytes = this.ops.length * 4;
		this.resultBytes = scalarResultSize( this.ops.length );
		createScalarKernel( options ); // validates
		this.toolLayout = toolBindGroupLayout( device, 'downsampler scalar tool', options.bindings );

	}

	/** Workgroups of the partials dispatch for an image: [ x, y, layers ]. */
	workgroups( width: number, height: number, layers = 1 ): [ number, number, number ] {

		return [ ...scalarWorkgroups( width, height, this.block ), layers ];

	}

	/** Bytes the partials buffer needs: one vec4f (DsWide) per workgroup and layer. */
	partialsBytes( width: number, height: number, layers = 1 ): number {

		const [ x, y ] = scalarWorkgroups( width, height, this.block );
		return x * y * layers * this.partialBytes;

	}

	/** WGSL, `@group(0)` layout and pipelines for a source shape, for tools that dispatch themselves. */
	kernel( { dimension = '2d', splat = false }: { dimension?: ViewDimension, splat?: boolean } = {} ): Readonly<ScalarKernel> {

		if ( this.combinesOnly ) throw new Error( 'downsampler: an 8-channel reduction without a load hook only combines partials; use combineKernel().' );
		if ( this.options.load !== undefined ) dimension = '2d', splat = false;
		const key = `${ dimension }:${ splat }`;
		let variant = this.variants.get( key );
		if ( variant !== undefined ) return variant;
		const wgsl = createScalarKernel( { ...this.options, dimension, splat } );
		const label = `downsampler scalar ${ this.ops.join( ',' ) } ${ key }`;
		const layout = this.device.createBindGroupLayout( { label, entries: scalarLayoutEntries( { ...this.options, dimension } ) } );
		const pipelineLayout = toolPipelineLayout( this.device, label, layout, this.toolLayout );
		const module = this.device.createShaderModule( { label, code: wgsl } );
		variant = {
			dimension, splat, layout, wgsl,
			partials: this.device.createComputePipeline( { label: `${ label } partials`, layout: pipelineLayout, compute: { module, entryPoint: 'partials' } } ),
			combine: this.device.createComputePipeline( { label: `${ label } combine`, layout: pipelineLayout, compute: { module, entryPoint: 'combine' } } )
		};
		this.variants.set( key, variant );
		return variant;

	}

	/**
	 * The combine pass alone (entry point `combine` of the same WGSL), with a
	 * group 0 that has no source texture, for combine-only runs and tools that
	 * dispatch them themselves.
	 */
	combineKernel(): Readonly<ScalarCombineKernel> {

		if ( this.combineOnly !== undefined ) return this.combineOnly;
		// The combine entry point never reads the source, so any variant's WGSL serves.
		const wgsl = createScalarKernel( this.options );
		const label = `downsampler scalar ${ this.ops.join( ',' ) } combine-only`;
		const layout = this.device.createBindGroupLayout( { label, entries: scalarLayoutEntries( { ...this.options, combineOnly: true } ) } );
		const module = this.device.createShaderModule( { label, code: wgsl } );
		const combine = this.device.createComputePipeline( { label, layout: toolPipelineLayout( this.device, label, layout, this.toolLayout ), compute: { module, entryPoint: 'combine' } } );
		return this.combineOnly = { layout, combine, wgsl };

	}

	/**
	 * `[ group 0, group 1 ]`, as `Reducer.bindGroupLayouts`: create the tool's bind group for
	 * the hooks from index 1 (present with `bindings`; every variant's pipeline uses this very
	 * layout). Group 0 is the reduction's own: here the layout of a '2d' source (or of a load
	 * hook, or combine-only), equal to `kernel().layout`; other source shapes have their own
	 * (`kernel( { dimension } ).layout`). Creating it compiles nothing.
	 */
	get bindGroupLayouts(): readonly GPUBindGroupLayout[] {

		if ( this.groupLayouts === undefined ) {

			const group0 = this.device.createBindGroupLayout( { label: `downsampler scalar ${ this.ops.join( ',' ) } group 0`, entries: scalarLayoutEntries( { ...this.options, dimension: '2d', combineOnly: this.combinesOnly } ) } );
			this.groupLayouts = toolGroupLayouts( group0, this.toolLayout );

		}

		return this.groupLayouts;

	}

	/** An 8-channel reduction without a load hook: it has no source, so it takes combine-only runs only. */
	get combinesOnly(): boolean {

		return this.ops.length === 8 && this.options.load === undefined;

	}

	/** Why a run can't be recorded, or null. */
	unsupportedReason( run: ScalarRun ): string | null {

		const { load, finish, bindings } = this.options;
		// Renamed before 1.0 (one source shape for every class): catch the old keys instead of ignoring them.
		for ( const key of [ 'texture', 'mipLevel', 'size' ] ) if ( ( run as unknown as Record<string, unknown> )[ key ] !== undefined ) return `a ScalarRun takes \`source\` (a texture or { texture, mipLevel }) and \`width\`/\`height\`, not \`${ key }\``;
		const combineOnly = run.partials !== undefined;
		if ( ! combineOnly && this.combinesOnly ) return 'an 8-channel reduction without a load hook only combines partials (a run needs `partials`)';
		if ( combineOnly ) {

			const reason = this.partialsReason( run );
			if ( reason !== null ) return reason;

		} else if ( run.texels !== undefined ) return '`texels` belongs to a combine-only run (with `partials`)';
		else if ( load === undefined ) {

			if ( run.source === undefined ) return 'a run needs a source texture (or the reduction a load hook, or the run partials)';
			const reason = sourceShapeReason( run.source ) ?? sourceTextureReason( asTextureLevel( run.source ) );
			if ( reason !== null ) return reason;
			const [ sw, sh ] = levelSizeOf( asTextureLevel( run.source ) );
			if ( ( run.width !== undefined && run.width !== sw ) || ( run.height !== undefined && run.height !== sh ) ) return `width and height (${ run.width }×${ run.height }) must match the source level (${ sw }×${ sh }), or be left out`;

		} else {

			if ( run.source !== undefined ) return 'a reduction with a load hook reads its own bindings, not a source texture';
			if ( run.width === undefined || run.height === undefined ) return 'a reduction with a load hook needs the run\'s width and height';

		}

		const [ first, count ] = this.layers( run );
		if ( ! Number.isInteger( first ) || ! Number.isInteger( count ) || first < 0 || count < 1 ) return 'layers must be [ first >= 0, count >= 1 ]';
		const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
		if ( ! combineOnly ) {

			const [ width, height ] = this.size( run );
			if ( ! Number.isInteger( width ) || ! Number.isInteger( height ) || width < 1 || height < 1 ) return 'width and height must be positive integers';
			const texture = this.sourceTexture( run );
			if ( texture !== undefined && first + count > texture.depthOrArrayLayers ) return 'layers are outside the source texture';
			const [ x, y ] = scalarWorkgroups( width, height, this.block );
			if ( x > limit || y > limit ) return 'too many workgroups for this device';

		} else if ( run.partials!.count * count * this.partialBytes > ( run.partials!.buffer.size - ( run.partials!.offset ?? 0 ) ) ) return 'the partials buffer is too small for its count and layers';

		if ( count > limit ) return 'too many workgroups for this device';
		const toolReason = toolBindGroupReason( bindings !== undefined, run.bindGroup, 'this reduction', { strict: false } );
		if ( toolReason !== null ) return toolReason;
		if ( finish === undefined ) {

			if ( run.output === undefined ) return 'without a finish hook, a run needs an output buffer';
			const offset = run.output.offset ?? 0;
			const alignment = this.device.limits.minStorageBufferOffsetAlignment;
			if ( ! Number.isInteger( offset ) || offset < 0 || offset % alignment !== 0 ) return `output offset must be a multiple of ${ alignment } (minStorageBufferOffsetAlignment)`;
			if ( ( run.output.buffer.usage & GPUBufferUsage.STORAGE ) === 0 ) return 'output buffer needs STORAGE usage';
			if ( offset + count * this.resultBytes > run.output.buffer.size ) return 'output buffer is too small';

		}

		return null;

	}

	/** Record the dispatches (both, or the combine alone for a run with `partials`), each in its own compute pass. */
	encode( encoder: GPUCommandEncoder, run: ScalarRun ): void {

		const prepared = this.prepare( run );
		for ( const step of this.steps( prepared ) ) {

			const pass = encoder.beginComputePass( { label: `downsampler scalar ${ step }` } );
			this.dispatch( pass, prepared, step );
			pass.end();

		}

	}

	/**
	 * Record the dispatches into a compute pass the caller owns. Dispatches are
	 * separate usage scopes, so the combine reads what the partials (or the
	 * tool's dispatch before it) wrote. Leaves the pass's pipeline and bind
	 * groups changed.
	 */
	encodePass( pass: GPUComputePassEncoder, run: ScalarRun ): void {

		const prepared = this.prepare( run );
		for ( const step of this.steps( prepared ) ) this.dispatch( pass, prepared, step );

	}

	/**
	 * What `encodePass` records for a run, for tools that dispatch it
	 * themselves: both pipelines (`partials` is null for a combine-only run), the
	 * group-0 bind group, the run's group 1 and both workgroup counts. Validates
	 * like `encode` and throws with {@link ScalarReduction.unsupportedReason}.
	 * Cached per resource and run shape (see {@link ScalarReduction.release}).
	 */
	prepare( run: ScalarRun ): PreparedScalar {

		const reason = this.unsupportedReason( run );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		const plan = this.plan( run );
		return {
			partials: plan.partialsPipeline, combine: plan.combine, bindGroup: plan.bindGroup, toolBindGroup: run.bindGroup,
			workgroups: plan.workgroups, combineWorkgroups: [ 1, 1, plan.workgroups[ 2 ] ]
		};

	}

	/**
	 * Drop the cached plans of a source texture, partials buffer or output
	 * buffer (without one: every plan), and destroy the uniform and partials
	 * buffers the reduction created for them, never `resource` itself nor any
	 * other buffer of yours. Call it after submitting the last work that uses
	 * them. Plans are also kept for at most 4 run shapes per resource; older ones
	 * are dropped and their buffers destroyed once the queue has finished the
	 * work submitted so far.
	 */
	release( resource?: GPUTexture | GPUBuffer ): void {

		const owners = resource !== undefined ? [ resource ] : this.owners.map( owner => owner.deref() ).filter( owner => owner !== undefined );
		for ( const owner of owners ) {

			for ( const plan of this.plans.get( owner ) ?? [] ) for ( const buffer of plan.owned ) buffer.destroy();
			this.plans.delete( owner );

		}

		if ( resource === undefined ) this.owners = [];

	}

	private steps( prepared: PreparedScalar ): ( 'partials' | 'combine' )[] {

		return prepared.partials === null ? [ 'combine' ] : [ 'partials', 'combine' ];

	}

	private dispatch( pass: GPUComputePassEncoder, prepared: PreparedScalar, step: 'partials' | 'combine' ): void {

		pass.setPipeline( prepared[ step ]! );
		pass.setBindGroup( 0, prepared.bindGroup );
		setToolBindGroup( pass, prepared.toolBindGroup );
		const [ x, y, z ] = step === 'partials' ? prepared.workgroups : prepared.combineWorkgroups;
		pass.dispatchWorkgroups( x, y, z );

	}

	/** Checks the combine-only part of a run. */
	private partialsReason( run: ScalarRun ): string | null {

		const { buffer, offset = 0, count } = run.partials!;
		if ( run.source !== undefined || run.width !== undefined || run.height !== undefined ) return 'a combine-only run (with `partials`) takes no source, width or height';
		if ( ! Number.isInteger( count ) || count < 1 ) return 'partials.count must be a positive integer';
		if ( ( buffer.usage & GPUBufferUsage.STORAGE ) === 0 ) return 'the partials buffer needs STORAGE usage';
		const alignment = this.device.limits.minStorageBufferOffsetAlignment;
		if ( ! Number.isInteger( offset ) || offset < 0 || offset % alignment !== 0 ) return `partials.offset must be a multiple of ${ alignment } (minStorageBufferOffsetAlignment)`;
		const texels = run.texels;
		const needsTexels = this.ops.includes( 'average' ) || scalarWeightChannel( this.ops ) < 0;
		if ( texels === undefined ? needsTexels : ! Number.isInteger( texels ) || texels < 1 || texels > 0xffffffff ) return needsTexels ? 'a combine-only run without a \'weight\' channel, or with \'average\' channels, needs `texels` (the texel count behind the partials)' : 'texels must be a positive integer';
		return null;

	}

	private size( run: ScalarRun ): [ number, number ] {

		return run.source === undefined ? [ run.width ?? 0, run.height ?? 0 ] : levelSizeOf( asTextureLevel( run.source ) );

	}

	private sourceTexture( run: ScalarRun ): GPUTexture | undefined {

		return run.source === undefined ? undefined : asTextureLevel( run.source ).texture;

	}

	private layers( run: ScalarRun ): [ number, number ] {

		return run.layers !== undefined ? [ run.layers[ 0 ], run.layers[ 1 ] ] : [ 0, this.sourceTexture( run )?.depthOrArrayLayers ?? 1 ];

	}

	/** Bind groups and uniforms cached per owner (texture, partials or output buffer), a few shapes each. */
	private cachedPlan( owner: object, find: ( plan: Plan ) => boolean, make: () => Plan ): Plan {

		let plans = this.plans.get( owner );
		if ( plans === undefined ) {

			this.plans.set( owner, plans = [] );
			if ( this.owners.length >= 64 ) this.owners = this.owners.filter( ref => ref.deref() !== undefined && this.plans.has( ref.deref()! ) );
			this.owners.push( new WeakRef( owner ) );

		}
		const cached = plans.find( find );
		if ( cached !== undefined ) return cached;
		const plan = make();
		// A few shapes per owner (e.g. a resize); the oldest is dropped.
		plans.push( plan );
		if ( plans.length > 4 ) retireBuffers( this.device, plans.shift()!.owned );
		return plan;

	}

	private outputEntry( run: ScalarRun, layers: number ): GPUBindGroupEntry[] {

		const { output } = run;
		return output !== undefined && this.options.finish === undefined ? [ { binding: 3, resource: { buffer: output.buffer, offset: output.offset ?? 0, size: layers * this.resultBytes } } ] : [];

	}

	private params( values: number[] ): GPUBuffer {

		const params = this.device.createBuffer( { label: 'downsampler scalar params', size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true } );
		new Uint32Array( params.getMappedRange() ).set( values );
		params.unmap();
		return params;

	}

	private plan( run: ScalarRun ): Plan {

		const { output } = run;
		const source = run.source === undefined ? undefined : asTextureLevel( run.source );
		const texture = source?.texture;
		const [ first, count ] = this.layers( run );
		const offset = output?.offset ?? 0;

		if ( run.partials !== undefined ) {

			const { buffer, offset: partialsOffset = 0, count: partialCount } = run.partials;
			const texels = run.texels ?? 0;
			const key = `partials:${ partialCount }:${ partialsOffset }:${ texels }:${ first }:${ count }:${ offset }`;
			return this.cachedPlan( buffer, plan => plan.key === key && plan.output === output?.buffer, () => {

				const kernel = this.combineKernel();
				const params = this.params( [ 0, 0, first, 0, partialCount, texels ] );
				const entries: GPUBindGroupEntry[] = [ { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer, offset: partialsOffset, size: partialCount * count * this.partialBytes } }, ...this.outputEntry( run, count ) ];
				return {
					key, texture: undefined, output: output?.buffer, partials: buffer, params, owned: [ params ], partialsPipeline: null, combine: kernel.combine,
					bindGroup: this.device.createBindGroup( { label: 'downsampler scalar combine-only', layout: kernel.layout, entries } ),
					workgroups: [ 0, 0, count ]
				};

			} );

		}

		const [ width, height ] = this.size( run );
		const mipLevel = source?.mipLevel ?? 0;
		const key = `${ width }:${ height }:${ first }:${ count }:${ mipLevel }:${ offset }`;
		return this.cachedPlan( texture ?? output?.buffer ?? this, plan => plan.key === key && plan.texture === texture && plan.output === output?.buffer, () => {

			const dimension: ViewDimension = texture === undefined ? '2d' : bindingViewDimension( texture );
			const variant = this.kernel( { dimension, splat: texture !== undefined && sampledChannels( texture.format ) === 1 } );
			const [ groupsX, groupsY ] = scalarWorkgroups( width, height, this.block );
			const params = this.params( [ width, height, first, groupsX, groupsX * groupsY, width * height ] );
			const partials = this.device.createBuffer( { label: 'downsampler scalar partials', size: this.partialsBytes( width, height, count ), usage: GPUBufferUsage.STORAGE } );
			const entries: GPUBindGroupEntry[] = [ { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: partials } } ];
			if ( source !== undefined ) entries.push( { binding: 2, resource: sourceView( source, dimension, 'downsampler scalar source' ) } );

			entries.push( ...this.outputEntry( run, count ) );
			return {
				key, texture, output: output?.buffer, partials, params, owned: [ params, partials ], partialsPipeline: variant.partials, combine: variant.combine,
				bindGroup: this.device.createBindGroup( { label: 'downsampler scalar', layout: variant.layout, entries } ),
				workgroups: [ groupsX, groupsY, count ]
			};

		} );

	}

}
