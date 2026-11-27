import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Reducer, ScalarReduction, createScalarKernel, scalarBlock, scalarCombineParams, scalarParams, scalarParamsBytes, scalarWorkgroups } from '../../dist/webgpu/index.js';

// Async compilation and the self-dispatch helpers of ScalarReduction and Reducer, on a
// recording host device: no adapter or shader compiler is involved.
globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUBufferUsage = { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

function fixture( { defer = false, failures = 0 } = {} ) {

	const calls = { async: [], sync: [], buffers: [], dispatches: [] };
	const waiting = [];
	const pipeline = descriptor => ( { descriptor } );
	const device = {
		features: new Set(),
		limits: { maxStorageTexturesPerShaderStage: 8, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupStorageSize: 32768, maxStorageBufferBindingSize: 1e9, maxBufferSize: 1e9, maxComputeWorkgroupsPerDimension: 65535, minStorageBufferOffsetAlignment: 256 },
		createShaderModule: descriptor => ( { ...descriptor } ),
		createBindGroupLayout: descriptor => ( { ...descriptor } ),
		createPipelineLayout: descriptor => ( { ...descriptor } ),
		createBindGroup: descriptor => ( { ...descriptor } ),
		createComputePipeline: descriptor => { calls.sync.push( descriptor ); return pipeline( descriptor ); },
		createComputePipelineAsync: descriptor => {

			calls.async.push( descriptor );
			if ( failures > 0 ) { failures --; return Promise.reject( new Error( 'compile failed' ) ); }
			if ( defer ) return new Promise( resolve => waiting.push( () => resolve( pipeline( descriptor ) ) ) );
			return Promise.resolve( pipeline( descriptor ) );

		},
		createBuffer: descriptor => {

			const data = new ArrayBuffer( descriptor.size );
			const buffer = { ...descriptor, getMappedRange: () => data, unmap() {}, destroy() {}, words: () => Array.from( new Uint32Array( data ) ) };
			calls.buffers.push( buffer );
			return buffer;

		},
		queue: { writeBuffer() {}, onSubmittedWorkDone: () => Promise.resolve() }
	};
	const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups( ...groups ) { calls.dispatches.push( groups ); }, end() {} };
	const encoder = { beginComputePass: () => pass, copyBufferToTexture() {} };
	const texture = ( overrides = {} ) => ( { width: 1920, height: 1080, depthOrArrayLayers: 1, mipLevelCount: 1, format: 'rgba16float', dimension: '2d', sampleCount: 1, usage: 31, createView: descriptor => ( { descriptor } ), destroy() {}, ...overrides } );
	const buffer = size => ( { size, usage: GPUBufferUsage.STORAGE, destroy() {} } );
	return { calls, device, pass, encoder, texture, buffer, resolve: () => { for ( const resolve of waiting.splice( 0 ) ) resolve(); } };

}

const bindings = { wgsl: '@group(1) @binding(0) var<storage, read_write> out: array<f32>;', entries: [ { binding: 0, visibility: 4, buffer: { type: 'storage' } } ] };
const load = 'fn load( texel: vec2i, layer: u32 ) -> vec4f { return vec4f( 1.0 ); }';
const finish = 'fn finish( value: vec4f, weight: f32 ) { out[ 0 ] = value.x; }';
const exposure = { reduction: [ 'average', 'sum', 'sum', 'sum' ], bindings, load, finish };
const params = f => f.calls.buffers.filter( buffer => buffer.label === 'downsampler scalar params' ).map( buffer => buffer.words() );

test( 'scalarParams and scalarWorkgroups are what a ScalarReduction writes and dispatches', () => {

	for ( const block of [ 1, 2, 4, 8 ] ) for ( const [ width, height ] of [ [ 1, 1 ], [ 63, 64 ], [ 65, 1 ], [ 960, 540 ], [ 1129, 635 ], [ 1920, 1080 ], [ 3840, 2160 ], [ 8191, 3 ] ] ) for ( const layers of [ [ 0, 1 ], [ 2, 3 ] ] ) {

		const f = fixture(), reduction = new ScalarReduction( f.device, { ...exposure, block } );
		const run = { width, height, layers, bindGroup: {} };
		const prepared = reduction.prepare( run );
		reduction.encodePass( f.pass, run );
		const words = Array.from( scalarParams( { width, height, firstLayer: layers[ 0 ], block } ) );
		// The DsParams words, spelled out: size, firstLayer, groupsX, partialCount, count, padding.
		const tile = 16 * block, gx = Math.ceil( width / tile ), gy = Math.ceil( height / tile );
		assert.deepEqual( words, [ width, height, layers[ 0 ], gx, gx * gy, width * height, 0, 0 ] );
		assert.deepEqual( params( f ), [ words ] );
		assert.equal( scalarParamsBytes, 32 );
		assert.deepEqual( prepared.workgroups, [ ...scalarWorkgroups( width, height, block ), layers[ 1 ] ] );
		assert.deepEqual( prepared.workgroups, reduction.workgroups( width, height, layers[ 1 ] ) );
		assert.deepEqual( f.calls.dispatches, [ prepared.workgroups, [ 1, 1, layers[ 1 ] ] ] );

	}

	assert.equal( scalarBlock, 4 );
	assert.deepEqual( Array.from( scalarParams( { width: 960, height: 540 } ) ), [ 960, 540, 0, 15, 135, 518400, 0, 0 ] );
	assert.throws( () => scalarParams( { width: 0, height: 4 } ), /positive integers/ );
	assert.throws( () => scalarParams( { width: 4, height: 4, block: 3 } ), /block/ );

} );

test( 'scalarCombineParams is what a combine-only run writes', () => {

	for ( const [ count, texels, layers ] of [ [ 510, undefined, [ 0, 1 ] ], [ 3, 99, [ 1, 2 ] ], [ 8161, 0xffffffff, [ 0, 4 ] ] ] ) {

		const f = fixture(), reduction = new ScalarReduction( f.device, { reduction: [ 'weighted', 'weighted', 'max', 'weight' ], bindings, finish } );
		const run = { partials: { buffer: f.buffer( 1 << 20 ), count }, texels, layers, bindGroup: {} };
		const prepared = reduction.prepare( run );
		const words = Array.from( scalarCombineParams( { count, texels, firstLayer: layers[ 0 ] } ) );
		assert.deepEqual( words, [ 0, 0, layers[ 0 ], 0, count, texels ?? 0, 0, 0 ] );
		assert.deepEqual( params( f ), [ words ] );
		assert.equal( prepared.partials, null );
		assert.deepEqual( prepared.combineWorkgroups, [ 1, 1, layers[ 1 ] ] );

	}

	assert.throws( () => scalarCombineParams( { count: 0 } ), /count/ );

} );

test( 'the generated DsParams struct is the six words the helpers fill', () => {

	assert.ok( createScalarKernel( exposure ).startsWith( 'struct DsParams {\n\tsize: vec2u,\n\tfirstLayer: u32,\n\tgroupsX: u32,\n\tpartialCount: u32,\n\tcount: u32,\n}' ) );

} );

for ( const [ name, options, run ] of [
	[ 'load hook', exposure, { width: 960, height: 540, bindGroup: {} } ],
	[ '2d source', { reduction: 'minmax' }, f => ( { source: f.texture(), output: { buffer: f.buffer( 256 ) } } ) ],
	[ 'depth source (splat)', { reduction: 'minmax' }, f => ( { source: f.texture( { format: 'depth32float' } ), output: { buffer: f.buffer( 256 ) } } ) ],
	[ 'array source', { reduction: 'max' }, f => ( { source: f.texture( { depthOrArrayLayers: 6 } ), layers: [ 1, 2 ], output: { buffer: f.buffer( 256 ) } } ) ],
	[ 'combine-only', { reduction: [ 'weighted', 'weighted', 'max', 'weight' ], bindings, finish }, f => ( { partials: { buffer: f.buffer( 1 << 16 ), count: 7 }, bindGroup: {} } ) ],
	[ '8-channel combine-only', { reduction: 'sum', channels: 8 }, f => ( { partials: { buffer: f.buffer( 1 << 16 ), count: 7 }, texels: 99, output: { buffer: f.buffer( 256 ) } } ) ]
] ) test( `ScalarReduction.warmup compiles asynchronously; encode then compiles nothing: ${ name }`, async () => {

	const f = fixture(), reduction = new ScalarReduction( f.device, options );
	const r = typeof run === 'function' ? run( f ) : run;
	await reduction.warmup( r );
	assert.equal( f.calls.sync.length, 0 );
	const compiled = f.calls.async.length;
	assert.equal( compiled, r.partials === undefined ? 2 : 1 );
	assert.equal( f.calls.buffers.length, 0, 'warmup allocates nothing' );
	reduction.encode( f.encoder, r );
	reduction.encodePass( f.pass, r );
	const prepared = reduction.prepare( r );
	for ( const pipeline of [ prepared.partials, prepared.combine ] ) if ( pipeline !== null ) assert.ok( f.calls.async.includes( pipeline.descriptor ) );
	assert.equal( f.calls.sync.length, 0, 'encode, encodePass and prepare use the warmed pipelines' );
	await reduction.warmup( r );
	assert.equal( f.calls.async.length, compiled );

} );

test( 'ScalarReduction pipelines are shared per device across equal reductions, and compiles in flight are shared', async () => {

	const f = fixture( { defer: true } );
	const a = new ScalarReduction( f.device, exposure ), b = new ScalarReduction( f.device, exposure );
	const pending = [ a.warmup(), b.warmup(), a.kernelAsync() ];
	f.resolve();
	await Promise.all( pending );
	assert.equal( f.calls.async.length, 2, 'one compile of each entry point' );
	assert.equal( a.kernel().partials, b.kernel().partials );
	assert.equal( a.kernel().combine, b.kernel().combine );
	// The hooks' group-1 layout is the very one the shared pipelines use, in both reductions.
	assert.equal( a.bindGroupLayouts[ 1 ], b.bindGroupLayouts[ 1 ] );
	assert.equal( a.kernel().partials.descriptor.layout.bindGroupLayouts[ 1 ], b.bindGroupLayouts[ 1 ] );
	// A different device, or different options, compile their own.
	const g = fixture();
	await new ScalarReduction( g.device, exposure ).warmup();
	assert.equal( g.calls.async.length, 2 );
	const other = new ScalarReduction( f.device, { ...exposure, block: 2 } ).warmup();
	f.resolve();
	await other;
	assert.equal( f.calls.async.length, 4 );
	assert.equal( f.calls.sync.length + g.calls.sync.length, 0 );

} );

test( 'ScalarReduction: a rejected compile is not cached, and a synchronous compile is kept', async () => {

	const f = fixture( { failures: 1 } ), reduction = new ScalarReduction( f.device, exposure );
	await assert.rejects( reduction.warmup(), /compile failed/ );
	await reduction.warmup();
	assert.equal( f.calls.async.length, 4 );
	assert.equal( f.calls.sync.length, 0 );

	const g = fixture(), first = new ScalarReduction( g.device, exposure );
	const kernel = first.kernel();
	assert.equal( g.calls.sync.length, 2 );
	assert.equal( ( await new ScalarReduction( g.device, exposure ).kernelAsync() ).partials, kernel.partials );
	assert.equal( g.calls.async.length, 0 );

	// Synchronous use while a compile is in flight compiles synchronously and wins.
	const h = fixture( { defer: true } ), raced = new ScalarReduction( h.device, exposure );
	const pending = raced.warmup();
	const sync = raced.kernel();
	h.resolve();
	await pending;
	assert.equal( ( await raced.kernelAsync() ).partials, sync.partials );
	assert.equal( new ScalarReduction( h.device, exposure ).kernel().partials, sync.partials );

	await assert.rejects( new ScalarReduction( f.device, { reduction: 'sum', channels: 8 } ).kernelAsync(), /only combines partials/ );

} );

test( 'Reducer.create compiles asynchronously; equal Reducers then compile nothing', async () => {

	const f = fixture( { defer: true } );
	const options = { levels: [ 2, 3 ], edges: 'clamp', bindings, load, store: 'fn store( level: u32, texel: vec2i, layer: u32, value: vec4f ) { }' };
	const pending = [ Reducer.create( f.device, options ), Reducer.create( f.device, options ) ];
	f.resolve();
	const [ a, b ] = await Promise.all( pending );
	assert.equal( f.calls.async.length, 1 );
	assert.equal( f.calls.sync.length, 0 );
	assert.equal( a.pipeline, b.pipeline );
	const c = new Reducer( f.device, options );
	assert.equal( c.pipeline, a.pipeline );
	assert.equal( c.bindGroupLayouts[ 1 ], a.bindGroupLayouts[ 1 ] );
	c.encodePass( f.pass, { width: 1917, height: 1077, bindGroup: {} } );
	assert.equal( f.calls.sync.length, 0 );
	assert.equal( c.wgsl, a.wgsl );

	const g = fixture( { failures: 1 } );
	await assert.rejects( Reducer.create( g.device, options ), /compile failed/ );
	await Reducer.create( g.device, options );
	assert.equal( g.calls.async.length, 2 );
	await assert.rejects( Reducer.create( g.device, { ...options, edges: 'exact' } ), /'zero-weight' or 'clamp'/ );

	// A pipeline compiled synchronously first is reused.
	const h = fixture(), sync = new Reducer( h.device, { levels: [ 1 ], format: 'rgba16float' } );
	assert.equal( ( await Reducer.create( h.device, { levels: [ 1 ], format: 'rgba16float' } ) ).pipeline, sync.pipeline );
	assert.deepEqual( [ h.calls.sync.length, h.calls.async.length ], [ 1, 0 ] );

} );
