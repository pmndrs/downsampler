import { sampledChannels } from '../core/formats.js';
import { isSumOp, scalarBlock, scalarInvocations, scalarOps, scalarSide, scalarWeightChannel, scalarWorkgroups, type ScalarOp, type ScalarReduce } from '../core/scalar.js';
import type { ViewDimension } from './kernel.js';
import { checkHooks, checkToolBindings, hookArity, setToolBindGroup, toolBindGroupLayout, toolBindGroupReason, toolPipelineLayout, type ToolBindings } from './tool.js';

export interface ScalarReductionOptions {
	/** One op for all channels, or one per channel. Default 'average'. */
	reduce?: ScalarReduce;
	/**
	 * WGSL defining `fn load( texel: vec2i, layer: u32 ) -> vec4f`, the value of
	 * one in-image texel. It reads the tool's own `@group(1)` resources. Without
	 * it, runs read a source texture (`ScalarRun.texture`) with `textureLoad`.
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
}

export interface ScalarKernelOptions extends ScalarReductionOptions {
	/** Source texture view dimension when there is no `load` hook. Default '2d'. */
	dimension?: ViewDimension;
	/** Read single-channel sources (depth, r*) as (v, v, v, v). Default false. */
	splat?: boolean;
}

export interface ScalarRun {
	/** Source texture, without a `load` hook: any float-sampleable 2D format, including depth (its depth aspect). */
	texture?: GPUTexture;
	/** Source level. Default 0. */
	mipLevel?: number;
	/** Image size in texels, with a `load` hook (a texture's own size otherwise). */
	size?: readonly [ number, number ];
	/**
	 * Layers to reduce, each to its own result: `[ first, count ]`. Default all
	 * of the texture's layers, or `[ 0, 1 ]` with a `load` hook.
	 */
	layers?: readonly [ number, number ];
	/** The tool's `@group(1)` bind group, required with `bindings`. */
	bindGroup?: GPUBindGroup;
	/** Where results go without a `finish` hook. `offset` must be a multiple of 256. */
	output?: { buffer: GPUBuffer, offset?: number };
}

export interface PreparedScalar {
	partials: GPUComputePipeline;
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

const channels = [ 'x', 'y', 'z', 'w' ];

/** Bindings of the reduction's own `@group(0)`. */
export function scalarLayoutEntries( { load, finish, dimension = '2d' }: Pick<ScalarKernelOptions, 'load' | 'finish' | 'dimension'> ): GPUBindGroupLayoutEntry[] {

	const visibility = GPUShaderStage.COMPUTE;
	const entries: GPUBindGroupLayoutEntry[] = [
		{ binding: 0, visibility, buffer: { type: 'uniform' } },
		{ binding: 1, visibility, buffer: { type: 'storage' } }
	];
	if ( load === undefined ) entries.push( { binding: 2, visibility, texture: { sampleType: 'unfilterable-float', viewDimension: dimension } } );
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
	const ops = scalarOps( options.reduce );
	if ( ! [ 1, 2, 4, 8 ].includes( block ) ) throw new Error( 'downsampler: block must be 1, 2, 4 or 8.' );
	checkHooks( { load, finish } );
	checkToolBindings( bindings );
	const weightChannel = scalarWeightChannel( ops );
	const tile = scalarSide * block;
	const array = dimension === '2d-array';

	let code = `struct DsParams {
	size: vec2u,
	firstLayer: u32,
	groupsX: u32,
	partialCount: u32,
	count: u32,
}
struct DsResult {
	value: vec4f,
	weight: f32,
}
@group(0) @binding(0) var<uniform> ds_params: DsParams;
@group(0) @binding(1) var<storage, read_write> ds_partials: array<vec4f>;
var<workgroup> ds_slots: array<vec4f, ${ scalarInvocations }>;
`;

	if ( load === undefined ) {

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

	const combine = ops.map( ( op, c ) => op === 'min' ? `min(a.${ channels[ c ] }, b.${ channels[ c ] })` : op === 'max' ? `max(a.${ channels[ c ] }, b.${ channels[ c ] })` : `a.${ channels[ c ] } + b.${ channels[ c ] }` );
	const sums = `vec4<bool>(${ ops.map( op => isSumOp( op ) ).join( ', ' ) })`;
	const term = ops.map( ( op, c ) => op === 'weighted' ? `v.${ channels[ c ] } * v.${ channels[ weightChannel ] }` : `v.${ channels[ c ] }` );

	code += `
fn ds_combine(a: vec4f, b: vec4f) -> vec4f {
	return vec4f(${ combine.join( ', ' ) });
}

// Sum-like channels of off-image texels contribute 0; min and max see the
// clamped (in-image) texel, a harmless duplicate.
fn ds_term(texel: vec2u, layer: u32) -> vec4f {
	let inside = all(texel < ds_params.size);
	let v = load(vec2i(min(texel, ds_params.size - vec2u(1))), layer);
	return select(vec4f(${ term.join( ', ' ) }), vec4f(0), ${ sums } & vec4<bool>(!inside));
}

// Stride-halving tree over the workgroup's 256 slots; the total ends in slot 0.
fn ds_tree(index: u32) {
	for (var stride = ${ scalarInvocations / 2 }u; stride > 0u; stride >>= 1u) {
		workgroupBarrier();
		if (index < stride) { ds_slots[index] = ds_combine(ds_slots[index], ds_slots[index + stride]); }
	}
	workgroupBarrier();
}

@compute @workgroup_size(${ scalarSide }, ${ scalarSide })
fn partials(@builtin(local_invocation_id) local: vec3u, @builtin(local_invocation_index) index: u32, @builtin(workgroup_id) group: vec3u) {
	let origin = group.xy * ${ tile }u + local.xy;
	let layer = ds_params.firstLayer + group.z;
`;

	// Adjacent invocations read adjacent texels; each reads block × block of them, 16 apart.
	let names: string[] = [];
	for ( let jy = 0; jy < block; jy ++ ) for ( let jx = 0; jx < block; jx ++ ) {

		const name = `v${ jy * block + jx }`;
		code += `	let ${ name } = ds_term(origin + vec2u(${ jx * scalarSide }u, ${ jy * scalarSide }u), layer);\n`;
		names.push( name );

	}

	for ( let level = 0; names.length > 1; level ++ ) {

		const half = names.length / 2;
		const next = names.slice( 0, half ).map( ( _, i ) => `t${ level }_${ i }` );
		for ( let i = 0; i < half; i ++ ) code += `	let ${ next[ i ] } = ds_combine(${ names[ i ] }, ${ names[ i + half ] });\n`;
		names = next;

	}

	const layerArgument = finish !== undefined && hookArity( finish, 'finish' ) >= 3 ? ', ds_params.firstLayer + group.z' : '';
	const normalise = ops.map( ( op, c ) => op === 'average' ? `total.${ channels[ c ] } / f32(ds_params.count)` : op === 'weighted' ? `select(total.${ channels[ c ] } / weight, 0.0, weight == 0.0)` : `total.${ channels[ c ] }` );

	code += `	ds_slots[index] = ${ names[ 0 ] };
	ds_tree(index);
	if (index == 0u) { ds_partials[group.z * ds_params.partialCount + group.y * ds_params.groupsX + group.x] = ds_slots[0]; }
}

@compute @workgroup_size(${ scalarInvocations })
fn combine(@builtin(local_invocation_index) index: u32, @builtin(workgroup_id) group: vec3u) {
	let n = ds_params.partialCount;
	let base = group.z * n;
	var value = select(ds_partials[base + min(index, n - 1u)], vec4f(0), ${ sums } & vec4<bool>(index >= n));
	for (var i = index + ${ scalarInvocations }u; i < n; i += ${ scalarInvocations }u) { value = ds_combine(value, ds_partials[base + i]); }
	ds_slots[index] = value;
	ds_tree(index);
	if (index == 0u) {
		let total = ds_slots[0];
		let weight = ${ weightChannel >= 0 ? `total.${ channels[ weightChannel ] }` : 'f32(ds_params.count)' };
		let result = vec4f(${ normalise.join( ', ' ) });
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
	partials: GPUBuffer;
	params: GPUBuffer;
	bindGroup: GPUBindGroup;
	variant: ScalarKernel;
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
 * exposure.encodePass( pass, { size: [ width, height ], bindGroup } );
 * ```
 */
export class ScalarReduction {

	readonly device: GPUDevice;
	readonly options: Readonly<ScalarReductionOptions>;
	readonly ops: readonly ScalarOp[];
	readonly block: number;
	private readonly toolLayout: GPUBindGroupLayout | undefined;
	private readonly variants = new Map<string, ScalarKernel>();
	private readonly plans = new WeakMap<object, Plan[]>();

	constructor( device: GPUDevice, options: ScalarReductionOptions = {} ) {

		this.device = device;
		this.options = { ...options };
		this.ops = scalarOps( options.reduce );
		this.block = options.block ?? scalarBlock;
		createScalarKernel( options ); // validates
		this.toolLayout = toolBindGroupLayout( device, 'downsampler scalar tool', options.bindings );

	}

	/** Workgroups of the partials dispatch for an image: [ x, y, layers ]. */
	workgroups( width: number, height: number, layers = 1 ): [ number, number, number ] {

		return [ ...scalarWorkgroups( width, height, this.block ), layers ];

	}

	/** Bytes the partials buffer needs: one vec4f per workgroup and layer. */
	partialsBytes( width: number, height: number, layers = 1 ): number {

		const [ x, y ] = scalarWorkgroups( width, height, this.block );
		return x * y * layers * 16;

	}

	/** WGSL, `@group(0)` layout and pipelines for a source shape, for tools that dispatch themselves. */
	kernel( { dimension = '2d', splat = false }: { dimension?: ViewDimension, splat?: boolean } = {} ): Readonly<ScalarKernel> {

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

	/** Why a run can't be recorded, or null. */
	unsupportedReason( run: ScalarRun ): string | null {

		const { load, finish, bindings } = this.options;
		if ( load === undefined ) {

			const texture = run.texture;
			if ( texture === undefined ) return 'a run needs a source texture (or the reduction a load hook)';
			if ( texture.dimension !== '2d' ) return `source dimension ${ texture.dimension } is not supported`;
			const viewDimension = ( texture as GPUTexture & { textureBindingViewDimension?: string } ).textureBindingViewDimension;
			if ( viewDimension !== undefined && viewDimension !== '2d' && viewDimension !== '2d-array' ) return `textureBindingViewDimension ${ viewDimension } is not supported`;
			if ( texture.sampleCount !== 1 ) return 'multisampled sources are not supported (resolve first)';
			if ( ( texture.usage & GPUTextureUsage.TEXTURE_BINDING ) === 0 ) return 'source needs TEXTURE_BINDING usage';
			if ( /int$/.test( texture.format ) || texture.format === 'stencil8' ) return `source format ${ texture.format } is not float-sampleable`;
			const mipLevel = run.mipLevel ?? 0;
			if ( ! Number.isInteger( mipLevel ) || mipLevel < 0 || mipLevel >= texture.mipLevelCount ) return 'mipLevel is outside the source texture';

		} else {

			if ( run.texture !== undefined ) return 'a reduction with a load hook reads its own bindings, not a source texture';
			if ( run.size === undefined ) return 'a reduction with a load hook needs a run size';

		}

		const [ width, height ] = this.size( run );
		if ( ! Number.isInteger( width ) || ! Number.isInteger( height ) || width < 1 || height < 1 ) return 'size must be positive integers';
		const [ first, count ] = this.layers( run );
		if ( ! Number.isInteger( first ) || ! Number.isInteger( count ) || first < 0 || count < 1 ) return 'layers must be [ first >= 0, count >= 1 ]';
		if ( run.texture !== undefined && first + count > run.texture.depthOrArrayLayers ) return 'layers are outside the source texture';
		const [ x, y ] = scalarWorkgroups( width, height, this.block );
		const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
		if ( x > limit || y > limit || count > limit ) return 'too many workgroups for this device';
		const toolReason = toolBindGroupReason( bindings !== undefined, run.bindGroup, 'this reduction', { strict: false } );
		if ( toolReason !== null ) return toolReason;
		if ( finish === undefined ) {

			if ( run.output === undefined ) return 'without a finish hook, a run needs an output buffer';
			const offset = run.output.offset ?? 0;
			if ( offset % 256 !== 0 ) return 'output offset must be a multiple of 256';
			if ( ( run.output.buffer.usage & GPUBufferUsage.STORAGE ) === 0 ) return 'output buffer needs STORAGE usage';
			if ( offset + count * scalarResultBytes > run.output.buffer.size ) return 'output buffer is too small';

		}

		return null;

	}

	/** Record both dispatches, each in its own compute pass. */
	encode( encoder: GPUCommandEncoder, run: ScalarRun ): void {

		const prepared = this.prepare( run );
		for ( const step of [ 'partials', 'combine' ] as const ) {

			const pass = encoder.beginComputePass( { label: `downsampler scalar ${ step }` } );
			this.dispatch( pass, prepared, step );
			pass.end();

		}

	}

	/**
	 * Record both dispatches into a compute pass the caller owns. Dispatches are
	 * separate usage scopes, so the combine reads what the partials wrote.
	 * Leaves the pass's pipeline and bind groups changed.
	 */
	encodePass( pass: GPUComputePassEncoder, run: ScalarRun ): void {

		const prepared = this.prepare( run );
		this.dispatch( pass, prepared, 'partials' );
		this.dispatch( pass, prepared, 'combine' );

	}

	/** Pipelines, bind groups and workgroup counts for a run (cached), for tools that dispatch themselves. */
	prepare( run: ScalarRun ): PreparedScalar {

		const reason = this.unsupportedReason( run );
		if ( reason !== null ) throw new Error( `downsampler: ${ reason }.` );
		const plan = this.plan( run );
		return {
			partials: plan.variant.partials, combine: plan.variant.combine, bindGroup: plan.bindGroup, toolBindGroup: run.bindGroup,
			workgroups: plan.workgroups, combineWorkgroups: [ 1, 1, plan.workgroups[ 2 ] ]
		};

	}

	/** Drop cached bind groups and buffers for a source texture or output buffer. */
	release( resource: GPUTexture | GPUBuffer ): void {

		this.plans.delete( resource );

	}

	private dispatch( pass: GPUComputePassEncoder, prepared: PreparedScalar, step: 'partials' | 'combine' ): void {

		pass.setPipeline( prepared[ step ] );
		pass.setBindGroup( 0, prepared.bindGroup );
		setToolBindGroup( pass, prepared.toolBindGroup );
		const [ x, y, z ] = step === 'partials' ? prepared.workgroups : prepared.combineWorkgroups;
		pass.dispatchWorkgroups( x, y, z );

	}

	private size( run: ScalarRun ): [ number, number ] {

		if ( run.texture === undefined ) return [ run.size?.[ 0 ] ?? 0, run.size?.[ 1 ] ?? 0 ];
		const mipLevel = run.mipLevel ?? 0;
		return [ Math.max( 1, run.texture.width >> mipLevel ), Math.max( 1, run.texture.height >> mipLevel ) ];

	}

	private layers( run: ScalarRun ): [ number, number ] {

		return run.layers !== undefined ? [ run.layers[ 0 ], run.layers[ 1 ] ] : [ 0, run.texture?.depthOrArrayLayers ?? 1 ];

	}

	private plan( run: ScalarRun ): Plan {

		const { texture, output } = run;
		const [ width, height ] = this.size( run );
		const [ first, count ] = this.layers( run );
		const mipLevel = run.mipLevel ?? 0;
		const offset = output?.offset ?? 0;
		const key = `${ width }:${ height }:${ first }:${ count }:${ mipLevel }:${ offset }`;
		const owner: object = texture ?? output?.buffer ?? this;
		let plans = this.plans.get( owner );
		if ( plans === undefined ) this.plans.set( owner, plans = [] );
		const cached = plans.find( plan => plan.key === key && plan.texture === texture && plan.output === output?.buffer );
		if ( cached !== undefined ) return cached;

		const viewDimension = ( texture as ( GPUTexture & { textureBindingViewDimension?: ViewDimension } ) | undefined )?.textureBindingViewDimension;
		const dimension: ViewDimension = viewDimension ?? ( texture !== undefined && texture.depthOrArrayLayers > 1 ? '2d-array' : '2d' );
		const variant = this.kernel( { dimension, splat: texture !== undefined && sampledChannels( texture.format ) === 1 } );
		const [ groupsX, groupsY ] = scalarWorkgroups( width, height, this.block );
		const params = this.device.createBuffer( { label: 'downsampler scalar params', size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true } );
		new Uint32Array( params.getMappedRange() ).set( [ width, height, first, groupsX, groupsX * groupsY, width * height ] );
		params.unmap();
		const partials = this.device.createBuffer( { label: 'downsampler scalar partials', size: this.partialsBytes( width, height, count ), usage: GPUBufferUsage.STORAGE } );
		const entries: GPUBindGroupEntry[] = [ { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: partials } } ];
		if ( texture !== undefined ) {

			const aspect = texture.format.includes( 'stencil' ) ? 'depth-only' : 'all';
			entries.push( { binding: 2, resource: texture.createView( { label: 'downsampler scalar source', dimension, baseMipLevel: mipLevel, mipLevelCount: 1, aspect } ) } );

		}

		if ( output !== undefined && this.options.finish === undefined ) entries.push( { binding: 3, resource: { buffer: output.buffer, offset, size: count * scalarResultBytes } } );
		const plan: Plan = {
			key, texture, output: output?.buffer, partials, params, variant,
			bindGroup: this.device.createBindGroup( { label: 'downsampler scalar', layout: variant.layout, entries } ),
			workgroups: [ groupsX, groupsY, count ]
		};

		// A few shapes per owner (e.g. a resize); the oldest is dropped.
		plans.push( plan );
		if ( plans.length > 4 ) plans.shift();
		return plan;

	}

}
