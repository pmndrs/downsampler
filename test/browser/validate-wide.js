// GPU suite for 8-channel accumulators (SPEC §14.1 Wide accumulators): Reducer
// and ScalarReduction with `DsWide { lo: vec4f, hi: vec4f }` values, against
// the same oracles as the 4-channel cases (reducePyramid, scalarReference,
// scalarCombineReference) run with 8 channels.
import { levelSize, pyramidTolerance, reducePyramid, scalarCombineReference, scalarReference } from '/dist/index.js';
import { Reducer, ScalarReduction } from '/dist/webgpu/index.js';

const visibility = GPUShaderStage.COMPUTE;
const poisonValue = - 12345.5;

export const wideShapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 64, 64 ], [ 65, 129 ], [ 1, 128 ], [ 128, 8 ],
	[ 192, 108 ], [ 255, 257 ], [ 17, 11, 3 ], [ 1920, 1080 ]
];

// The tool's group 1: the source as 2 vec4f per texel, an atlas of 2 vec4f per
// output texel (levels 1…6 at their own offsets), and the layout uniform.
export const wideBindings = {
	wgsl: `struct Tool { width: u32, height: u32, offset: array<vec4u, 2>, stride: array<vec4u, 2>, rows: array<vec4u, 2> }
@group(1) @binding(0) var<storage, read> toolInput: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> toolAtlas: array<vec4f>;
@group(1) @binding(2) var<uniform> tool: Tool;`,
	entries: () => [
		{ binding: 0, visibility, buffer: { type: 'read-only-storage' } },
		{ binding: 1, visibility, buffer: { type: 'storage' } },
		{ binding: 2, visibility, buffer: { type: 'uniform' } }
	]
};
export const wideLoad = `fn load(texel: vec2i, layer: u32) -> DsWide {
	let i = (layer * tool.height + u32(texel.y)) * tool.width + u32(texel.x);
	return DsWide(toolInput[2u * i], toolInput[2u * i + 1u]);
}`;
const wideStore = `fn store(level: u32, texel: vec2i, layer: u32, value: DsWide) {
	let l = level - 1u;
	let row = layer * tool.rows[l / 4u][l % 4u] + u32(texel.y);
	let i = tool.offset[l / 4u][l % 4u] + row * tool.stride[l / 4u][l % 4u] + u32(texel.x);
	toolAtlas[2u * i] = value.lo;
	toolAtlas[2u * i + 1u] = value.hi;
}`;

// Channel 5 weighs channels 4 and 6.
const wideOps = [ 'sum', 'average', 'min', 'max', 'weighted', 'weight', 'weighted', 'sum' ];
export const wideVariants = [
	{ name: 'wide channel ops', reduction: wideOps },
	{ name: 'wide minmax', reduction: 'minmax', channels: 8 },
	// Per half: max of lo, min of hi. Idempotent.
	{ name: 'wide custom fold', reduction: { wgsl: 'fn reduce(a: DsWide, b: DsWide) -> DsWide { return DsWide(max(a.lo, b.lo), min(a.hi, b.hi)); }' }, channels: 8, reference: { reference: ( a, b, c ) => c < 4 ? Math.max( a, b ) : Math.min( a, b ) } },
	// A sum that counts its combines on channels 1, 3 and 4: skipped or doubled texels change the count.
	{ name: 'wide custom sum', reduction: { wgsl: 'fn reduce(a: DsWide, b: DsWide) -> DsWide { return DsWide(a.lo + b.lo + vec4f(0.0, 1.0, 0.0, 1.0), a.hi + b.hi + vec4f(1.0, 0.0, 0.0, 0.0)); }', kind: 'sum' }, channels: 8, reference: { reference: ( a, b, c ) => a + b + [ 0, 1, 0, 1, 1, 0, 0, 0 ][ c ], kind: 'sum' } }
];

/** 8 floats per texel (exact in f32): values in [−4, 12), and for channels 4–6 non-negative values with a weight (5) that is 0 for a third of the texels. */
export function wideData( width, height, layers, seed ) {

	let state = seed;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) >>> 8;
	return Float32Array.from( { length: width * height * layers * 8 }, ( _, i ) => {

		const c = i % 8;
		if ( c === 5 ) return next() % 3 === 0 ? 0 : next() / 2 ** 24;
		if ( c === 4 || c === 6 ) return next() / 2 ** 22;
		return next() / 2 ** 20 - 4;

	} );

}

function storageBuffer( device, size ) {

	return device.createBuffer( { size: Math.max( 16, size ), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC } );

}

async function readBuffer( device, buffer, offset = 0, size = buffer.size - offset ) {

	const staging = device.createBuffer( { size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer( buffer, offset, staging, 0, size );
	device.queue.submit( [ encoder.finish() ] );
	await staging.mapAsync( GPUMapMode.READ );
	const values = new Float32Array( staging.getMappedRange().slice( 0 ) );
	staging.unmap();
	staging.destroy();
	return values;

}

/** The atlas for levels 1…6, as the store hook addresses it (8 floats per texel). */
function atlasLayout( width, height, layers, edges ) {

	const levels = [ 1, 2, 3, 4, 5, 6 ].map( level => ( { level, width: levelSize( width, level, edges ), height: levelSize( height, level, edges ) } ) );
	let offset = 0;
	for ( const level of levels ) { level.offset = offset; offset += level.width * level.height * layers; }
	const uniform = new Uint32Array( 28 );
	uniform.set( [ width, height ] );
	levels.forEach( ( level, i ) => { uniform[ 4 + i ] = level.offset; uniform[ 12 + i ] = level.width; uniform[ 20 + i ] = level.height; } );
	return { levels, texels: offset, uniform };

}

export function wideTool( device, layout, width, height, layers, edges ) {

	const atlasLayoutOf = atlasLayout( width, height, layers, edges );
	const input = storageBuffer( device, width * height * layers * 32 );
	const atlas = storageBuffer( device, atlasLayoutOf.texels * 32 );
	const uniform = device.createBuffer( { size: atlasLayoutOf.uniform.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( uniform, 0, atlasLayoutOf.uniform );
	const bindGroup = device.createBindGroup( { layout, entries: [ input, atlas, uniform ].map( ( buffer, binding ) => ( { binding, resource: { buffer } } ) ) } );
	return { layout: atlasLayoutOf, input, atlas, uniform, bindGroup, destroy: () => [ input, atlas, uniform ].forEach( buffer => buffer.destroy() ) };

}

function checkLevel( actual, expected, label ) {

	let maxError = 0;
	for ( let i = 0; i < expected.values.length; i ++ ) {

		const limit = pyramidTolerance( expected.magnitude[ i ] );
		const error = Math.abs( actual[ i ] - expected.values[ i ] );
		if ( ! Number.isFinite( actual[ i ] ) || ! ( error <= limit ) ) throw new Error( `${ label }: value ${ i } (texel ${ i >> 3 } channel ${ i & 7 }), actual ${ actual[ i ] }, expected ${ expected.values[ i ] }, error ${ error } > ${ limit }` );
		maxError = Math.max( maxError, error );

	}

	return maxError;

}

// Each kernel width reduces 1…count; complementary selections across variants, so every level is checked.
const wideLevels = [ [ 1 ], [ 1, 3, 4 ], [ 2, 4 ], [ 1, 2, 3, 5 ], [ 3, 5 ] ];

// Case groups (test/ci/shard.js): edges × variant × level selection, over every shape.
async function validateWideReducer( device, { quick, select } ) {

	const results = [];
	const shapes = quick ? wideShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : wideShapes;
	const groupId = ( edges, variant, levels ) => `wide/reducer/${ edges }/${ variant.name.replace( /[^a-z0-9]+/gi, '-' ).replace( /^-|-$/g, '' ) }/L${ levels.join( '' ) }`;
	const wanted = ( edges, variant ) => wideLevels.filter( levels => select.has( groupId( edges, variant, levels ) ) );

	for ( const [ width, height, layers = 1 ] of shapes ) for ( const edges of [ 'zero-weight', 'clamp' ] ) {

		if ( ! wideVariants.some( variant => wanted( edges, variant ).length > 0 ) ) continue;
		const data = wideData( width, height, layers, width * 13 + height );
		let tool = null;

		for ( const variant of wideVariants ) {

			const selected = wanted( edges, variant );
			if ( selected.length === 0 ) continue;
			const expected = reducePyramid( data, width, height, layers, { reduction: variant.reference ?? variant.reduction, levels: [ 1, 2, 3, 4, 5 ], edges, channels: 8 } );

			for ( const levels of selected ) results.push( ...await select.run( groupId( edges, variant, levels ), async () => {

				const reducer = new Reducer( device, { reduction: variant.reduction, channels: variant.channels, levels, edges, dimension: layers > 1 ? '2d-array' : '2d', load: wideLoad, store: wideStore, bindings: { wgsl: wideBindings.wgsl, entries: wideBindings.entries() } } );
				if ( reducer.channels !== 8 ) throw new Error( 'wide reducer: expected 8 channels' );
				if ( tool === null ) {

					tool = wideTool( device, reducer.bindGroupLayouts[ 1 ], width, height, layers, edges );
					device.queue.writeBuffer( tool.input, 0, data );

				}

				device.queue.writeBuffer( tool.atlas, 0, new Float32Array( tool.layout.texels * 8 ).fill( poisonValue ) );
				const encoder = device.createCommandEncoder();
				reducer.encode( encoder, { width, height, layerCount: layers, bindGroup: tool.bindGroup } );
				device.queue.submit( [ encoder.finish() ] );
				const atlas = await readBuffer( device, tool.atlas );
				const label = `wide reducer ${ width }x${ height }x${ layers } ${ edges } ${ variant.name } [${ levels }]`;
				let maxError = 0;

				for ( const { level, width: w, height: h, offset } of tool.layout.levels.slice( 0, 5 ) ) {

					const values = atlas.subarray( offset * 8, ( offset + w * h * layers ) * 8 );
					if ( levels.includes( level ) ) maxError = Math.max( maxError, checkLevel( values, expected[ level - 1 ], `${ label } level ${ level }` ) );
					else if ( values.some( value => value !== poisonValue ) ) throw new Error( `${ label }: unemitted level ${ level } was stored` );

				}

				return [ { reducer: variant.name, width, height, layers, edges, levels, path: 'wide', maxError } ];

			} ) );

		}

		tool?.destroy();

	}

	return results;

}

/** Results per layer from an output buffer: `{ value: DsWide, weight: f32 }`, 48 bytes each. */
async function readWideResults( device, buffer, offset, layers ) {

	const values = await readBuffer( device, buffer, offset, layers * 48 );
	return Array.from( { length: layers }, ( _, i ) => ( { value: Array.from( values.subarray( i * 12, i * 12 + 8 ) ), weight: values[ i * 12 + 8 ] } ) );

}

function checkBound( actual, expected, label ) {

	actual.forEach( ( result, layer ) => {

		const want = expected[ layer ];
		for ( let c = 0; c < 8; c ++ ) {

			const error = Math.abs( result.value[ c ] - want.exact[ c ] );
			if ( ! Number.isFinite( result.value[ c ] ) || error > want.bound[ c ] ) throw new Error( `${ label } layer ${ layer } channel ${ c }: actual ${ result.value[ c ] }, exact ${ want.exact[ c ] }, error ${ error } > bound ${ want.bound[ c ] }` );

		}

		if ( ! ( Math.abs( result.weight - want.exactWeight ) <= want.weightBound ) ) throw new Error( `${ label } layer ${ layer } weight: actual ${ result.weight }, exact ${ want.exactWeight }, bound ${ want.weightBound }` );

	} );

}

const scalarWideOps = [ 'sum', 'average', 'min', 'max', 'weighted', 'weight', 'weighted', 'max' ];

async function validateWideScalar( device, { quick } ) {

	const results = [];
	const output = storageBuffer( device, 256 + 4 * 48 );
	const bindings = { wgsl: wideBindings.wgsl, entries: wideBindings.entries() };
	const scalar = new ScalarReduction( device, { reduction: scalarWideOps, load: wideLoad, bindings } );
	if ( scalar.resultBytes !== 48 || scalar.partialBytes !== 32 ) throw new Error( 'wide scalar: unexpected result or partial size' );

	for ( const [ width, height, layers = 1 ] of quick ? wideShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : wideShapes ) {

		const data = wideData( width, height, layers, width + height * 7 );
		const tool = wideTool( device, scalar.kernel().partials.getBindGroupLayout( 1 ), width, height, layers, 'zero-weight' );
		device.queue.writeBuffer( tool.input, 0, data );
		device.queue.writeBuffer( output, 0, new Float32Array( output.size / 4 ).fill( NaN ) );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		scalar.encodePass( pass, { width, height, layers: [ 0, layers ], bindGroup: tool.bindGroup, output: { buffer: output, offset: 256 } } );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );
		checkBound( await readWideResults( device, output, 256, layers ), scalarReference( data, width, height, layers, scalarWideOps ), `wide scalar ${ width }x${ height }x${ layers }` );
		tool.destroy();
		results.push( { scalar: 'wide', width, height, layers } );

	}

	// Combine-only with 8 channels and no load hook (such a reduction only combines), from tool-written partials.
	const combiner = new ScalarReduction( device, { reduction: scalarWideOps } );
	if ( combiner.unsupportedReason( { width: 4, height: 4, output: { buffer: output } } ) === null ) throw new Error( 'wide scalar: an 8-channel reduction without a load hook must reject image runs' );
	let state = 5;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
	for ( const count of [ 1, 255, 257, 2040 ] ) for ( const layers of [ 1, 2 ] ) {

		const values = Float32Array.from( { length: count * layers * 8 }, ( _, i ) => i % 8 === 5 ? next() * 16 : next() * 512 - 256 );
		const partials = storageBuffer( device, values.byteLength );
		device.queue.writeBuffer( partials, 0, values );
		device.queue.writeBuffer( output, 0, new Float32Array( output.size / 4 ).fill( NaN ) );
		const encoder = device.createCommandEncoder();
		combiner.encode( encoder, { partials: { buffer: partials, count }, texels: count * 64, layers: [ 0, layers ], output: { buffer: output, offset: 256 } } );
		device.queue.submit( [ encoder.finish() ] );
		checkBound( await readWideResults( device, output, 256, layers ), scalarCombineReference( values, count, layers, scalarWideOps, { texels: count * 64 } ), `wide scalar combine-only ${ count } x${ layers }` );
		partials.destroy();
		results.push( { scalar: 'wide combine-only', count, layers } );

	}

	output.destroy();
	return results;

}

function validateWideOptions( device ) {

	const expectThrow = ( run, message, label ) => {

		try { run(); } catch ( error ) {

			if ( ! String( error.message ).includes( message ) ) throw new Error( `${ label }: expected "${ message }", got "${ error.message }"` );
			return;

		}

		throw new Error( `${ label }: expected an error "${ message }"` );

	};

	const bindings = { wgsl: wideBindings.wgsl, entries: wideBindings.entries() };
	expectThrow( () => new Reducer( device, { reduction: wideOps, levels: [ 1 ], format: 'rgba32float' } ), 'load hook and a store hook', 'wide reducer without hooks' );
	expectThrow( () => new Reducer( device, { reduction: wideOps, channels: 4, levels: [ 1 ], load: wideLoad, store: wideStore, bindings } ), 'channels is 4', 'wide ops with 4 channels' );
	expectThrow( () => new Reducer( device, { reduction: 'sum', channels: 6, levels: [ 1 ], load: wideLoad, store: wideStore, bindings } ), 'channels must be 4', 'six channels' );
	// Six levels of DsWide take 32 KiB of workgroup memory: above the default limit.
	if ( device.limits.maxComputeWorkgroupStorageSize < 32768 ) expectThrow( () => new Reducer( device, { reduction: wideOps, levels: [ 6 ], load: wideLoad, store: wideStore, bindings } ), 'maxComputeWorkgroupStorageSize', 'wide six levels' );
	expectThrow( () => new ScalarReduction( device, { reduction: 'sum', channels: 5 } ), 'channels must be 4 or 8', 'scalar five channels' );
	expectThrow( () => new ScalarReduction( device, { reduction: scalarWideOps } ).kernel(), 'combineKernel', 'wide scalar kernel() without a load hook' );
	expectThrow( () => new ScalarReduction( device, { reduce: 'max' } ), 'not `reduce`', 'scalar old reduce key' );
	return [ { option: 'wide validation' } ];

}

/** `select`: the case-group gate (test/ci/shard.js); default all. */
export async function validateWide( device, { quick = false, onProgress = () => {}, select = { has: () => true, run: async ( id, fn ) => fn() } } = {} ) {

	const results = [ ...await validateWideReducer( device, { quick, select } ) ];
	onProgress( 'validated wide reducer' );
	results.push( ...await select.run( 'wide/scalar', () => validateWideScalar( device, { quick } ) ) );
	results.push( ...await select.run( 'wide/options', async () => validateWideOptions( device ) ) );
	onProgress( 'validated wide scalar' );
	return results;

}
