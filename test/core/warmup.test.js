import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Downsampler } from '../../dist/webgpu/downsampler.js';

// Host scheduling/cache tests: no adapter or shader compiler is involved.
globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUBufferUsage = { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

function fixture( { defer = false, fail = false } = {} ) {

	const calls = { compute: [], render: [], sync: [], resources: [], layouts: [], used: [], usedGroups: [] };
	const waiting = [];
	const pipeline = descriptor => ( { descriptor, getBindGroupLayout: () => ( {} ) } );
	const asyncPipeline = ( kind, descriptor ) => {

		calls[ kind ].push( descriptor );
		if ( fail ) { fail = false; return Promise.reject( new Error( 'compile failed' ) ); }
		if ( defer ) return new Promise( resolve => waiting.push( () => resolve( pipeline( descriptor ) ) ) );
		return Promise.resolve( pipeline( descriptor ) );

	};
	const texture = ( overrides = {} ) => ( {
		width: 133, height: 75, depthOrArrayLayers: 1, mipLevelCount: 8, format: 'rgba8unorm', dimension: '2d', sampleCount: 1, usage: 31,
		createView: descriptor => { calls.resources.push( 'view' ); return { descriptor }; }, destroy() {}, ...overrides
	} );
	const device = {
		features: new Set( [ 'core-features-and-limits' ] ),
		limits: { maxStorageTexturesPerShaderStage: 6, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupStorageSize: 32768, maxStorageBufferBindingSize: 1e9, maxBufferSize: 1e9 },
		createSampler: () => ( {} ),
		createShaderModule: descriptor => descriptor,
		createBindGroupLayout: descriptor => { calls.layouts.push( descriptor ); return descriptor; },
		createPipelineLayout: descriptor => descriptor,
		createComputePipelineAsync: descriptor => asyncPipeline( 'compute', descriptor ),
		createRenderPipelineAsync: descriptor => asyncPipeline( 'render', descriptor ),
		createComputePipeline: descriptor => { calls.sync.push( descriptor ); return pipeline( descriptor ); },
		createRenderPipeline: descriptor => { calls.sync.push( descriptor ); return pipeline( descriptor ); },
		createBindGroup: descriptor => { calls.resources.push( 'bindGroup' ); return { ...descriptor, invalid: descriptor.layout?.invalid === true }; },
		createBuffer: descriptor => { calls.resources.push( 'buffer' ); return { ...descriptor, destroy() {} }; },
		createTexture: descriptor => { calls.resources.push( 'texture' ); return texture( { ...descriptor, mipLevelCount: 1 } ); },
		queue: { writeBuffer() {}, onSubmittedWorkDone: () => Promise.resolve() }
	};
	const pass = { setPipeline( pipeline ) { calls.used.push( pipeline ); }, setBindGroup( _, group ) { calls.usedGroups.push( group ); }, dispatchWorkgroups() {}, draw() {}, setScissorRect() {}, end() {} };
	const encoder = { beginComputePass: () => pass, beginRenderPass: () => pass, copyBufferToTexture() {} };
	return { calls, device, texture, encoder, resolve: () => { for ( const resolve of waiting.splice( 0 ) ) resolve(); } };

}

for ( const [ name, config, textureOptions, options ] of [
	[ 'storage', {}, {}, {} ],
	[ 'dynamic rect', {}, {}, { rect: { x: 3, y: 5, width: 11, height: 9 } } ],
	[ 'dynamic dropped SPD edge', {}, {}, { oddSizes: 'spd', rect: { x: 132, y: 74, width: 1, height: 1 } } ],
	[ 'layer range', {}, { depthOrArrayLayers: 3 }, { layers: [ 1, 1 ], oddSizes: 'box' } ],
	[ 'buffer', { output: 'buffer' }, {}, {} ],
	[ 'buffer rect', { output: 'buffer' }, {}, { rect: { x: 5, y: 3, width: 7, height: 9 } } ],
	[ 'compat', { compatibility: true }, {}, {} ],
	[ 'compat rect', { compatibility: true }, {}, { rect: { x: 5, y: 3, width: 7, height: 9 } } ],
	[ 'sRGB render', { renderPathMinTexels: 0 }, { format: 'rgba8unorm-srgb' }, { oddSizes: 'box' } ],
	[ 'render rect', { output: 'render' }, {}, { rect: { x: 5, y: 3, width: 7, height: 9 } } ],
	[ 'halo box', { halo: true }, {}, { oddSizes: 'box' } ]
] ) test( `warmup covers encode pipelines without allocating execution resources: ${ name }`, async () => {

	const f = fixture(), ds = new Downsampler( f.device, config ), texture = f.texture( textureOptions );
	await ds.warmup( texture, options );
	assert.deepEqual( f.calls.resources, [] );
	const count = f.calls.compute.length + f.calls.render.length;
	assert.ok( count > 0 );
	ds.encode( f.encoder, texture, options );
	assert.equal( f.calls.sync.length, 0, 'encode must use the exact warmed keys/layouts' );
	await ds.warmup( texture, options );
	assert.equal( f.calls.compute.length + f.calls.render.length, count );
	ds.release( texture );
	await ds.warmup( texture, options );
	assert.equal( f.calls.compute.length + f.calls.render.length, count, 'release retains pipelines' );

} );

test( 'warmup shares in-flight requests and snapshots mutable options', async () => {

	const f = fixture( { defer: true } ), ds = new Downsampler( f.device ), texture = f.texture();
	const options = { rect: { x: 1, y: 2, width: 3, height: 4 }, oddSizes: 'box' };
	const first = ds.warmup( texture, options );
	const count = f.calls.compute.length;
	const second = ds.warmup( texture, options );
	assert.equal( f.calls.compute.length, count );
	options.oddSizes = 'spd';
	options.rect.width = 20;
	f.resolve();
	await Promise.all( [ first, second ] );
	ds.encode( f.encoder, texture, { rect: { x: 1, y: 2, width: 3, height: 4 }, oddSizes: 'box' } );
	assert.equal( f.calls.sync.length, 0 );

} );

for ( const output of [ 'storage', 'render' ] ) test( `warmup retries rejected ${ output } pipelines`, async () => {

	const f = fixture( { fail: true } ), ds = new Downsampler( f.device, { output } ), texture = f.texture();
	await assert.rejects( ds.warmup( texture ), /compile failed/ );
	const calls = output === 'render' ? f.calls.render : f.calls.compute;
	const before = calls.length;
	await ds.warmup( texture );
	assert.equal( calls.length, before + 1, 'only the rejected variant is retried' );
	ds.encode( f.encoder, texture );
	assert.equal( f.calls.sync.length, 0 );

} );

test( 'synchronous encode while warmup is pending keeps its pipeline and public prepare binding contract', async () => {

	const f = fixture( { defer: true } ), ds = new Downsampler( f.device ), texture = f.texture();
	const options = { rect: { x: 1, y: 2, width: 3, height: 4 } };
	const pending = ds.warmup( texture, options );
	ds.encode( f.encoder, texture, options );
	assert.ok( f.calls.sync.length > 0 );
	const sync = f.calls.sync.length;
	f.resolve();
	await pending;
	ds.encode( f.encoder, texture, options );
	assert.equal( f.calls.sync.length, sync );
	const batches = ds.prepare( texture, options );
	assert.ok( batches.length > 0 );
	assert.ok( batches.every( batch => batch.dynamicOffsets === undefined ) );
	assert.equal( ds.prepare( texture, options ), batches );
	const staticRegionLayouts = f.calls.layouts.filter( layout => layout.entries.some( entry => entry.buffer?.type === 'uniform' && ! entry.buffer.hasDynamicOffset ) );
	assert.ok( staticRegionLayouts.length > 0, 'prepare retains static layouts distinct from encode' );

} );

test( 'warmup has no pipelines for empty updates and rejects invalid options', async () => {

	const f = fixture(), ds = new Downsampler( f.device ), texture = f.texture();
	await ds.warmup( texture, { rect: { x: 0, y: 0, width: 0, height: 0 } } );
	await assert.rejects( ds.warmup( texture, { mipLevelCount: 20 } ), /mip range/ );
	await assert.rejects( ds.warmup( texture, { layers: [ 4, 1 ] } ), /layer/ );
	await assert.rejects( ds.warmup( texture, { oddSizes: 'spd', reduction: 'max' } ), /average/ );
	assert.equal( f.calls.compute.length, 0 );
	assert.deepEqual( f.calls.resources, [] );

} );

test( 'warmup covers a resampled separate source and colorSpace sRGB alias', async () => {

	for ( const kind of [ 'source', 'srgb' ] ) {

		const f = fixture(), ds = new Downsampler( f.device ), texture = f.texture();
		const options = kind === 'source' ? { source: { texture: f.texture( { width: 79, height: 51 } ), resample: 'area' }, reduction: 'max' } : { colorSpace: 'srgb' };
		await ds.warmup( texture, options );
		assert.deepEqual( f.calls.resources, [] );
		ds.encode( f.encoder, texture, options );
		assert.equal( f.calls.sync.length, 0 );

	}

} );

test( 'warmup observes earlier rejections when a later descriptor throws synchronously', async () => {

	const f = fixture( { fail: true } ), ds = new Downsampler( f.device ), texture = f.texture();
	const create = f.device.createShaderModule;
	let modules = 0;
	f.device.createShaderModule = descriptor => {

		if ( ++ modules === 2 ) throw new Error( 'descriptor failed' );
		return create( descriptor );

	};
	await assert.rejects( ds.warmup( texture ), /compile failed|descriptor failed/ );
	// node:test fails on any unhandled rejection after this test.
	await new Promise( resolve => setImmediate( resolve ) );
	f.device.createShaderModule = create;
	await ds.warmup( texture );
	ds.encode( f.encoder, texture );
	assert.equal( f.calls.sync.length, 0 );

} );


// WebGPU's synchronous creation reports validation errors through error scopes;
// it returns an object even when that pipeline is invalid. Throw-only mocks miss this.
function deferredInvalidSync( f, output ) {

	const name = output === 'render' ? 'createRenderPipeline' : 'createComputePipeline';
	const create = f.device[ name ];
	let validation;
	f.device.pushErrorScope = () => { validation = undefined; };
	f.device.popErrorScope = async () => { await Promise.resolve(); return validation ?? null; };
	f.device[ name ] = descriptor => {

		const pipeline = create( descriptor );
		pipeline.invalid = true;
		pipeline.getBindGroupLayout = () => ( { invalid: true } );
		queueMicrotask( () => { validation = { message: 'deferred validation failure' }; } );
		return pipeline;

	};

}

for ( const output of [ 'storage', 'render' ] ) {

	test( `warmup rejects invalid sync-cached ${ output } objects, retries, and refreshes cached bindings`, async () => {

		const f = fixture( { fail: true } ), ds = new Downsampler( f.device, { output } ), texture = f.texture();
		deferredInvalidSync( f, output );
		f.device.pushErrorScope( 'validation' );
		ds.encode( f.encoder, texture );
		assert.match( ( await f.device.popErrorScope() ).message, /deferred validation/ );
		assert.ok( f.calls.used.every( pipeline => pipeline.invalid ) );
		await assert.rejects( ds.warmup( texture ), /compile failed/ );
		await ds.warmup( texture );
		f.calls.used.length = 0; f.calls.usedGroups.length = 0;
		ds.encode( f.encoder, texture );
		assert.ok( f.calls.used.length > 0 );
		assert.ok( f.calls.used.every( pipeline => ! pipeline.invalid ) );
		assert.ok( f.calls.usedGroups.every( group => ! group.invalid ), 'render must rebuild invalid-layout bind groups' );
		const count = f.calls.compute.length + f.calls.render.length;
		await ds.warmup( texture );
		assert.equal( f.calls.compute.length + f.calls.render.length, count, 'async-validated variants need no second compile' );

	} );

	test( `warmup replaces a deferred-invalid ${ output } sync fallback with its successful async result`, async () => {

		const f = fixture( { defer: true } ), ds = new Downsampler( f.device, { output } ), texture = f.texture();
		deferredInvalidSync( f, output );
		const options = { rect: { x: 1, y: 2, width: 3, height: 4 } };
		const pending = ds.warmup( texture, options );
		f.device.pushErrorScope( 'validation' );
		ds.encode( f.encoder, texture, options );
		assert.match( ( await f.device.popErrorScope() ).message, /deferred validation/ );
		f.resolve();
		await pending;
		f.calls.used.length = 0; f.calls.usedGroups.length = 0;
		ds.encode( f.encoder, texture, options );
		assert.ok( f.calls.used.length > 0 );
		assert.ok( f.calls.used.every( pipeline => ! pipeline.invalid ) );
		assert.ok( f.calls.usedGroups.every( group => ! group.invalid ) );

	} );

	test( `warmup keeps rejecting a permanently invalid ${ output } descriptor already cached by sync creation`, async () => {

		const f = fixture(), ds = new Downsampler( f.device, { output } ), texture = f.texture();
		deferredInvalidSync( f, output );
		const name = output === 'render' ? 'createRenderPipelineAsync' : 'createComputePipelineAsync';
		let attempts = 0;
		f.device[ name ] = () => { attempts ++; return Promise.reject( new Error( 'GPUPipelineError: validation' ) ); };
		ds.encode( f.encoder, texture );
		await assert.rejects( ds.warmup( texture ), /GPUPipelineError/ );
		const before = attempts;
		await assert.rejects( ds.warmup( texture ), /GPUPipelineError/ );
		assert.ok( attempts > before );

	} );

}

test( 'warmup refreshes another texture holding an older shared sync pipeline and retains old prepared batches', async () => {

	const f = fixture(), ds = new Downsampler( f.device ), first = f.texture(), second = f.texture();
	deferredInvalidSync( f, 'storage' );
	const old = ds.prepare( second );
	await ds.warmup( first );
	await ds.warmup( second );
	const fresh = ds.prepare( second );
	assert.notEqual( fresh, old );
	assert.ok( fresh.every( batch => ! batch.pipeline.invalid ) );
	assert.ok( old.every( batch => batch.pipeline.invalid ), 'already returned batches are not mutated' );

} );


for ( const alias of [ false, true ] ) test( `warmup follows auto threshold as updates cross it (${ alias ? 'sRGB alias dynamic storage' : 'sRGB buffer' })`, async () => {

	const f = fixture(), ds = new Downsampler( f.device, { renderPathMinTexels: 64 } );
	const texture = f.texture( { format: alias ? 'rgba8unorm' : 'rgba8unorm-srgb', depthOrArrayLayers: 3 } );
	const base = { oddSizes: 'box', ...( alias ? { colorSpace: 'srgb' } : {} ) };
	await ds.warmup( texture, base );
	assert.ok( f.calls.render.length > 0, 'the whole chain renders above the threshold' );
	assert.equal( f.calls.compute.length, 0 );
	ds.encode( f.encoder, texture, base );
	assert.equal( f.calls.sync.length, 0 );

	const small = { ...base, rect: { x: 123, y: 66, width: 7, height: 9 }, layers: [ 1, 1 ] };
	await ds.warmup( texture, small );
	assert.ok( f.calls.compute.length > 0, '63 texels use compute despite a renderable full texture' );
	const computeCount = f.calls.compute.length, renderCount = f.calls.render.length;
	f.calls.used.length = 0;
	ds.encode( f.encoder, texture, small );
	assert.ok( f.calls.used.every( pipeline => pipeline.descriptor.compute !== undefined ) );
	assert.equal( f.calls.sync.length, 0, 'dynamic full-size template must retain the small rect storage path' );
	if ( alias ) assert.ok( f.calls.layouts.some( layout => layout.entries.some( entry => entry.buffer?.hasDynamicOffset ) ), 'sRGB alias uses the private dynamic-region layout' );

	// Clipped edge footprint: the supplied rect remains in bounds, while workgroups
	// and deeper dependency regions clip against the texture. Reuse warmed keys.
	const clipped = { ...base, rect: { x: 128, y: 68, width: 5, height: 7 }, layers: [ 2, 1 ] };
	await ds.warmup( texture, clipped );
	ds.encode( f.encoder, texture, clipped );
	assert.equal( f.calls.compute.length, computeCount );
	assert.equal( f.calls.render.length, renderCount );
	assert.equal( f.calls.sync.length, 0 );

	const boundary = { ...base, rect: { x: 125, y: 67, width: 8, height: 8 }, layers: [ 0, 1 ] };
	await ds.warmup( texture, boundary );
	f.calls.used.length = 0;
	ds.encode( f.encoder, texture, boundary );
	assert.ok( f.calls.used.every( pipeline => pipeline.descriptor.fragment !== undefined ), 'exactly 64 texels render' );
	assert.equal( f.calls.compute.length, computeCount );
	assert.equal( f.calls.render.length, renderCount, 'full render warm-up covers boundary rect' );
	await ds.warmup( texture, small );
	ds.encode( f.encoder, texture, small );
	assert.equal( f.calls.compute.length, computeCount );
	assert.equal( f.calls.render.length, renderCount );
	assert.equal( f.calls.sync.length, 0 );

} );


for ( const threshold of [ 0, 64, Infinity ] ) test( `auto warms a render-only sRGB alias at threshold ${ threshold } without changing option-blind queries`, async () => {

	const f = fixture(), ds = new Downsampler( f.device, { renderPathMinTexels: threshold } );
	const texture = f.texture( { usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
	assert.equal( ds.outputPath( texture ), null );
	assert.match( ds.unsupportedReason( texture ), /STORAGE_BINDING/ );
	await assert.rejects( ds.warmup( texture ), /STORAGE_BINDING/ );
	const options = { colorSpace: 'srgb', oddSizes: 'box' };
	assert.equal( ds.scratchSize( texture, options ), 0 );
	await ds.warmup( texture, options );
	ds.encode( f.encoder, texture, options );
	const count = f.calls.render.length;
	const partial = { ...options, rect: { x: 128, y: 68, width: 5, height: 7 } };
	assert.equal( ds.scratchSize( texture, partial ), 0 );
	await ds.warmup( texture, partial );
	ds.encode( f.encoder, texture, partial );
	assert.equal( f.calls.render.length, count );
	assert.equal( f.calls.compute.length, 0 );
	assert.equal( f.calls.sync.length, 0 );
	await assert.rejects( new Downsampler( f.device, { output: 'storage' } ).warmup( texture, options ), /STORAGE_BINDING/ );
	await assert.rejects( ds.warmup( texture, { ...options, reduction: 'max' } ), /buffer output path/ );
	await assert.rejects( ds.warmup( texture, { colorSpace: 'srgb', oddSizes: 'spd' } ), /spd/ );
	for ( const [ overrides, message ] of [
		[ { dimension: '3d' }, /dimension/ ],
		[ { sampleCount: 4 }, /multisampled/ ],
		[ { usage: GPUTextureUsage.RENDER_ATTACHMENT }, /TEXTURE_BINDING/ ],
		[ { textureBindingViewDimension: 'cube' }, /textureBindingViewDimension/ ]
	] ) await assert.rejects( ds.warmup( f.texture( { ...texture, ...overrides } ), options ), message );

} );
