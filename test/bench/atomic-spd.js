// MEASUREMENT ONLY — NOT A SHIPPED API. A single-dispatch mip chain with SPD's
// atomic last-workgroup handoff, built to measure against the library's
// dispatch-per-batch plans (SPEC.md §2.2, DIVERGES.md #1).
//
// It relies on behaviour WGSL does not guarantee. Each workgroup reduces its
// 64×64 source tile to levels 1–6 (the library's 'clamp' tile kernel, so the
// same SPD semantics as `oddSizes: 'spd'`), publishes its level-6 texel with
// atomicStore, runs storageBarrier() and increments a global counter. The
// workgroup that draws the last ticket reads every payload with atomicLoad and
// reduces the remaining levels (≤ 6, so up to 4096²). WGSL atomics are relaxed
// and storageBarrier() is workgroup-scoped, so seeing the last ticket does not
// make other workgroups' payload stores visible: the tail can read stale data on
// hardware that reorders (WGSL memory model; gpuweb#5572; MC Mutants, ASPLOS
// 2023). Validation here is evidence for one device, not a proof.
//
// Tail levels go to storage textures while the device's storage texture limit
// allows (6 + direct ≤ maxStorageTexturesPerShaderStage); the rest go to a
// packed buffer and are copied into the texture ('texture ready'). The counter
// resets itself, as SPD's does, so no clear is recorded per chain.
import { createKernel, float16Wgsl } from '/dist/webgpu/kernel.js';

const formatWords = { rgba8unorm: 1, rgba16float: 2 };
const entry = '@compute @workgroup_size(16, 16)\nfn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {';

/** A generated tile kernel as a plain function `name( local, group )`; `prefix` renames its identifiers. */
function asFunction( wgsl, name, prefix = '' ) {

	if ( ! wgsl.includes( entry ) ) throw new Error( 'atomic probe: unexpected tile kernel entry point' );
	let code = wgsl.replace( entry, `fn ${ name }(local: vec3u, group: vec3u) {` );
	if ( prefix ) {

		// Share the phase-1 tile (two 16 KiB tiles exceed the workgroup memory limit).
		code = code.replace( /var<workgroup> ds_tile: array<vec4f, \d+>;\n/, '' );
		code = code.replace( /\b(ds_(?!tile\b)\w+|load|store|size)\b/g, `${ prefix }$1` );

	}

	return code;

}

export class AtomicSpd {

	/** `access`: 'sample' or 'load' for averages (folds always load). */
	constructor( device, { reduction = 'average', access = 'sample' } = {} ) {

		this.device = device;
		this.reduction = reduction;
		this.access = reduction === 'average' ? access : 'load';
		this.plans = new WeakMap();
		this.sampler = device.createSampler( { minFilter: 'linear', magFilter: 'linear' } );

	}

	unsupportedReason( texture ) {

		if ( ! ( texture.format in formatWords ) ) return `format ${ texture.format }`;
		if ( texture.depthOrArrayLayers !== 1 ) return 'one layer only';
		if ( texture.mipLevelCount - 1 > 12 ) return 'at most 12 output levels (4096²)';
		if ( this.device.limits.maxStorageTexturesPerShaderStage < 6 ) return 'needs 6 storage textures';
		return null;

	}

	/** Pipeline, bind group and copies for a texture (cached). */
	prepare( texture ) {

		let plan = this.plans.get( texture );
		if ( plan ) return plan;
		const reason = this.unsupportedReason( texture );
		if ( reason ) throw new Error( `atomic probe: ${ reason }` );

		const device = this.device, format = texture.format, words = formatWords[ format ];
		const levels = texture.mipLevelCount - 1;
		const tail = Math.max( 0, levels - 6 );
		const direct = Math.min( tail, device.limits.maxStorageTexturesPerShaderStage - 6 );
		const w1 = Math.max( 1, texture.width >> 1 ), h1 = Math.max( 1, texture.height >> 1 );
		const groupsX = Math.ceil( w1 / 32 ), groupsY = Math.ceil( h1 / 32 );
		const mip6 = [ Math.max( 1, texture.width >> 6 ), Math.max( 1, texture.height >> 6 ) ];

		// Packed tail levels past the storage texture limit, 256-byte rows.
		const packed = [];
		let offset = 0;
		for ( let level = 7 + direct; level <= levels; level ++ ) {

			const width = Math.max( 1, texture.width >> level ), height = Math.max( 1, texture.height >> level );
			const bytesPerRow = Math.ceil( width * words * 4 / 256 ) * 256;
			packed.push( { level, width, height, offset, bytesPerRow } );
			offset += bytesPerRow * height;

		}

		const pack = value => words === 1 ? `pack4x8unorm(${ value })` : `vec2u(ds_pack_f16(${ value }.xy), ds_pack_f16(${ value }.zw))`;
		const unpack = address => words === 1 ? `unpack4x8unorm(atomicLoad(&spd_state[${ address }]))` : `vec4f(unpack2x16float(atomicLoad(&spd_state[${ address }])), unpack2x16float(atomicLoad(&spd_state[${ address } + 1u])))`;

		let bindings = '';
		for ( let level = 1; level <= 6 + direct; level ++ ) bindings += `@group(1) @binding(${ level - 1 }) var spd_mip${ level }: texture_storage_2d<${ format }, write>;\n`;
		bindings += '// 0: completion counter; then the packed level-6 texel of each workgroup.\n@group(1) @binding(12) var<storage, read_write> spd_state: array<atomic<u32>>;\n';
		if ( packed.length ) bindings += '@group(1) @binding(13) var<storage, read_write> spd_packed: array<u32>;\n';

		// float16 stores and payloads round to nearest even first with the library's helper
		// (f16Rounding 'nearest'): textureStore truncates on Apple and D3D12, pack2x16float on D3D12.
		const half = words === 1 ? '' : float16Wgsl( 'nearest' );
		const stored = words === 1 ? 'value' : 'ds_round_f16(value)';
		const storeLevels = ( base, count ) => Array.from( { length: count }, ( _, i ) => `	if (level == ${ i + 1 }u) { textureStore(spd_mip${ base + i + 1 }, texel, ${ stored }); }\n` ).join( '' );
		const publish = words === 1 ? '		atomicStore(&spd_state[address], pack4x8unorm(value));\n' : '		let p = vec2u(ds_pack_f16(value.xy), ds_pack_f16(value.zw));\n		atomicStore(&spd_state[address], p.x);\n		atomicStore(&spd_state[address + 1u], p.y);\n';
		const phase1Store = `${ half }fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
${ storeLevels( 0, 6 ) }	if (level == 6u) {
		let address = 1u + (u32(texel.y) * ${ groupsX }u + u32(texel.x)) * ${ words }u;
${ publish }	}
}`;
		const tailStore = `${ half }fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
${ storeLevels( 6, direct ) }${ packed.map( ( { level, offset, bytesPerRow } ) => `	if (level == ${ level - 6 }u) {
		let address = ${ offset / 4 }u + u32(texel.y) * ${ bytesPerRow / 4 }u + u32(texel.x) * ${ words }u;
${ words === 1 ? '		spd_packed[address] = pack4x8unorm(value);' : '		let p = ' + pack( 'value' ) + ';\n		spd_packed[address] = p.x;\n		spd_packed[address + 1u] = p.y;' }
	}
` ).join( '' ) }}`;
		const tailLoad = `fn load(texel: vec2i, layer: u32) -> vec4f {
	let address = 1u + (u32(texel.y) * ${ groupsX }u + u32(texel.x)) * ${ words }u;
	return ${ unpack( 'address' ) };
}`;
		const tailSize = `fn size() -> vec2u { return vec2u(${ mip6[ 0 ] }u, ${ mip6[ 1 ] }u); }`;

		const phase1 = createKernel( { edges: 'clamp', levels: [ 1, 2, 3, 4, 5, 6 ], reduction: this.reduction, access: this.access, store: phase1Store, bindings } );
		let code = asFunction( phase1, 'spd_phase1' );
		if ( tail > 0 ) {

			// Always the 6-level (32×32) tile, so the 16×16 workgroup matches phase 1;
			// the store hook drops levels past the chain.
			const phase2 = createKernel( { edges: 'clamp', levels: [ 1, 2, 3, 4, 5, 6 ], reduction: this.reduction, load: tailLoad, store: tailStore, size: tailSize } );
			code += '\n' + asFunction( phase2, 'spd_phase2', 't_' );

		}

		// The ticket travels through the tile's first slot: a separate workgroup
		// variable would exceed the default 16 KiB. Phase 1 has finished with the
		// tile (its last level is read by invocation 0 only); phase 2 overwrites it.
		code += `
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	spd_phase1(local, group);
${ tail > 0 ? `	// Not a guaranteed handoff in WGSL: relaxed atomics, workgroup-scoped barrier.
	storageBarrier();
	if (all(local.xy == vec2u(0u))) { ds_tile[0].x = bitcast<f32>(atomicAdd(&spd_state[0], 1u)); }
	let ticket = bitcast<u32>(workgroupUniformLoad(&ds_tile[0]).x);
	if (ticket != ${ groupsX * groupsY - 1 }u) { return; }
	if (all(local.xy == vec2u(0u))) { atomicStore(&spd_state[0], 0u); }
	spd_phase2(local, vec3u(0u));
` : '' }}
`;

		const module = device.createShaderModule( { label: 'atomic spd probe', code } );
		const pipeline = device.createComputePipeline( { label: 'atomic spd probe', layout: 'auto', compute: { module, entryPoint: 'main' } } );
		const state = device.createBuffer( { label: 'atomic spd state', size: 4 + groupsX * groupsY * words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST } );
		const buffer = packed.length ? device.createBuffer( { label: 'atomic spd packed tail', size: offset, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } ) : null;
		const view = level => texture.createView( { dimension: '2d', baseMipLevel: level, mipLevelCount: 1 } );
		const group0 = [ { binding: 1, resource: view( 0 ) } ];
		if ( this.access === 'sample' ) group0.push( { binding: 0, resource: this.sampler } );
		const group1 = Array.from( { length: Math.min( levels, 6 + direct ) }, ( _, i ) => ( { binding: i, resource: view( i + 1 ) } ) );
		group1.push( { binding: 12, resource: { buffer: state } } );
		if ( buffer ) group1.push( { binding: 13, resource: { buffer } } );
		plan = {
			code, pipeline, state, buffer, packed, groupsX, groupsY, direct, tail,
			bindGroups: [ device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: group0 } ), device.createBindGroup( { layout: pipeline.getBindGroupLayout( 1 ), entries: group1 } ) ]
		};
		this.plans.set( texture, plan );
		return plan;

	}

	/** Dispatches recorded per chain: always 1, plus a copy per packed level when `materialize`. */
	passes( texture ) {

		return 1 + this.prepare( texture ).packed.length;

	}

	encode( encoder, texture, { materialize = true } = {} ) {

		const plan = this.prepare( texture );
		const pass = encoder.beginComputePass( { label: 'atomic spd probe' } );
		pass.setPipeline( plan.pipeline );
		pass.setBindGroup( 0, plan.bindGroups[ 0 ] );
		pass.setBindGroup( 1, plan.bindGroups[ 1 ] );
		pass.dispatchWorkgroups( plan.groupsX, plan.groupsY, 1 );
		pass.end();
		if ( materialize ) for ( const { level, width, height, offset, bytesPerRow } of plan.packed ) {

			encoder.copyBufferToTexture( { buffer: plan.buffer, offset, bytesPerRow, rowsPerImage: height }, { texture, mipLevel: level }, [ width, height ] );

		}

	}

	/** The counter word after a chain: 0 when the last workgroup ran and reset it. */
	async counter( texture ) {

		const plan = this.prepare( texture );
		const read = this.device.createBuffer( { size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST } );
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer( plan.state, 0, read, 0, 4 );
		this.device.queue.submit( [ encoder.finish() ] );
		await read.mapAsync( GPUMapMode.READ );
		const value = new Uint32Array( read.getMappedRange() )[ 0 ];
		read.destroy();
		return value;

	}

	release( texture ) {

		const plan = this.plans.get( texture );
		if ( ! plan ) return;
		plan.state.destroy();
		plan.buffer?.destroy();
		this.plans.delete( texture );

	}

}
