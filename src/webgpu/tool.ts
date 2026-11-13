/**
 * The tool side of the generator (SPEC.md §14.1): the tool's `@group(1)`
 * bindings, hook validation, and the pipeline layouts that put group 0 (ours)
 * next to group 1 (the tool's). Shared by `Reducer` and `ScalarReduction`.
 */

/** The tool's `@group(1)`: WGSL declarations and the matching layout entries. */
export interface ToolBindings {
	/** e.g. `@group(1) @binding(0) var luma: texture_2d<f32>;` */
	wgsl: string;
	entries: GPUBindGroupLayoutEntry[];
}

/** The hooks a tool can pass, by name. */
export type HookName = 'load' | 'store' | 'size' | 'finish';

/** Signatures of the hooks, as error messages name them. */
export const hookSignatures: Record<HookName, string> = {
	load: 'fn load( texel: vec2i, layer: u32 ) -> vec4f',
	store: 'fn store( level: u32, texel: vec2i, layer: u32, value: vec4f )',
	size: 'fn size() -> vec2u',
	finish: 'fn finish( value: vec4f, weight: f32 )'
};

/**
 * Throws unless `wgsl` is undefined or WGSL defining `fn name(`. Hooks may
 * define helpers too. `signature` overrides the one named in the error.
 */
export function checkHook( name: HookName, wgsl: unknown, signature = hookSignatures[ name ] ): void {

	if ( wgsl === undefined ) return;
	if ( typeof wgsl !== 'string' || ! new RegExp( `\\bfn\\s+${ name }\\s*\\(` ).test( wgsl ) ) throw new Error( `downsampler: the ${ name } hook must be WGSL defining \`${ signature }\`.` );

}

/** Checks every hook present in `hooks`. */
export function checkHooks( hooks: Partial<Record<HookName, unknown>>, signatures: Partial<Record<HookName, string>> = {} ): void {

	for ( const [ name, wgsl ] of Object.entries( hooks ) ) checkHook( name as HookName, wgsl, signatures[ name as HookName ] );

}

/** Number of parameters a hook declares (e.g. `finish` with an optional `layer`). */
export function hookArity( wgsl: string, name: HookName ): number {

	const parameters = new RegExp( `\\bfn\\s+${ name }\\s*\\(([^)]*)\\)` ).exec( wgsl )?.[ 1 ] ?? '';
	return parameters.split( ',' ).filter( parameter => parameter.trim() !== '' ).length;

}

/** Throws unless `bindings` is undefined or `{ wgsl: string, entries: GPUBindGroupLayoutEntry[] }`. */
export function checkToolBindings( bindings: unknown ): asserts bindings is ToolBindings | undefined {

	if ( bindings === undefined ) return;
	const { wgsl, entries } = ( bindings ?? {} ) as Partial<ToolBindings>;
	if ( typeof wgsl !== 'string' ) throw new Error( 'downsampler: bindings must be WGSL declarations for @group(1).' );
	if ( ! Array.isArray( entries ) ) throw new Error( 'downsampler: bindings need the layout entries of their @group(1) declarations.' );

}

/** The tool's group-1 layout, or undefined without bindings. */
export function toolBindGroupLayout( device: GPUDevice, label: string, bindings: ToolBindings | undefined ): GPUBindGroupLayout | undefined {

	return bindings === undefined ? undefined : device.createBindGroupLayout( { label, entries: bindings.entries } );

}

/** `[ group 0, group 1 ]`, or just group 0 without tool bindings. */
export function toolGroupLayouts( group0: GPUBindGroupLayout, tool: GPUBindGroupLayout | undefined ): GPUBindGroupLayout[] {

	return tool === undefined ? [ group0 ] : [ group0, tool ];

}

/** The pipeline layout of a generator kernel: our group 0, then the tool's group 1. */
export function toolPipelineLayout( device: GPUDevice, label: string, group0: GPUBindGroupLayout, tool: GPUBindGroupLayout | undefined ): GPUPipelineLayout {

	return device.createPipelineLayout( { label, bindGroupLayouts: toolGroupLayouts( group0, tool ) } );

}

/** Why a dispatch's tool bind group doesn't fit, or null. `owner` names the object ('this Reducer'). */
export function toolBindGroupReason( hasBindings: boolean, bindGroup: GPUBindGroup | undefined, owner: string, { strict = true }: { strict?: boolean } = {} ): string | null {

	if ( hasBindings && bindGroup === undefined ) return `${ owner } has tool bindings; pass their group-1 bindGroup`;
	if ( strict && ! hasBindings && bindGroup !== undefined ) return `a bindGroup was passed, but ${ owner } has no tool bindings`;
	return null;

}

/** Sets the tool's group 1 on a pass, if there is one. */
export function setToolBindGroup( pass: GPUComputePassEncoder, bindGroup: GPUBindGroup | undefined ): void {

	if ( bindGroup !== undefined ) pass.setBindGroup( 1, bindGroup );

}

/**
 * Destroys buffers a cache evicted, once the queue has finished the work
 * submitted so far. An evicted entry may still be recorded in a command buffer
 * the caller hasn't submitted yet (more than a cache's worth of dispatches in
 * one encoder); destroying it at once would make that submit fail. The
 * promise resolves in a later task, after a frame's usual encode-and-submit;
 * destroy() then waits for any submitted work that still uses the buffers.
 */
export function retireBuffers( device: GPUDevice, buffers: readonly GPUBuffer[] ): void {

	if ( buffers.length === 0 ) return;
	const destroy = () => { for ( const buffer of buffers ) buffer.destroy(); };
	device.queue.onSubmittedWorkDone().then( destroy, destroy );

}
