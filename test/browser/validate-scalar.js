// GPU suite for scalar reductions (SPEC §4 Scalar reductions). Every result is
// checked against the CPU oracle: exactly where the values are integers below
// 2^24 (texel counts, bit sums, coordinates, min/max), and within the oracle's
// stated error bound for float sums.
import { fromHalf, levelSize, scalarCombineReference, scalarReference, toHalf } from '/dist/index.js';
import { Reducer, ScalarReduction, scalarResultBytes } from '/dist/webgpu/index.js';
import { createDepthSource, makeData, toFloats, upload } from './gpu.js';

// [ width, height, layers?, viewDimension? ]
export const scalarShapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 15, 17 ], [ 16, 16 ], [ 17, 9 ], [ 31, 33 ], [ 63, 65 ], [ 64, 64 ], [ 65, 129 ],
	[ 128, 1 ], [ 1, 128 ], [ 255, 257 ], [ 17, 11, 3 ], [ 64, 32, 1, '2d-array' ], [ 300, 200, 6 ],
	[ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 4111, 300 ], [ 3840, 2160 ]
];

const big = ( [ w, h ] ) => w * h > 65536;

// Procedural load hooks: integer values, so every result must match exactly.
// The seed lives in the tool's group 1, so reusing a plan with a new seed
// catches stale partials.
const proceduralBindings = {
	wgsl: '@group(1) @binding(0) var<uniform> seed: vec4u;',
	entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } ]
};
const proceduralHash = `fn hash(texel: vec2i, layer: u32) -> u32 {
	let p = vec2u(texel);
	return (p.x * 73856093u) ^ (p.y * 19349663u) ^ (layer * 83492791u) ^ (seed.x * 2654435761u);
}`;
const procedural = [
	{
		name: 'count, bits, max index',
		reduce: [ 'sum', 'sum', 'sum', 'max' ],
		load: `${ proceduralHash }
fn load(texel: vec2i, layer: u32) -> vec4f {
	let h = hash(texel, layer);
	return vec4f(1.0, f32((h >> 7u) & 1u), f32((h >> 13u) & 1u), f32(u32(texel.y) * seed.y + u32(texel.x)));
}`,
		texel: ( h, x, y, width ) => [ 1, ( h >>> 7 ) & 1, ( h >>> 13 ) & 1, y * width + x ]
	},
	{
		name: 'min index, max x, max y, bits',
		reduce: [ 'min', 'max', 'max', 'sum' ],
		load: `${ proceduralHash }
fn load(texel: vec2i, layer: u32) -> vec4f {
	let h = hash(texel, layer);
	return vec4f(f32(u32(texel.y) * seed.y + u32(texel.x)) - f32(layer), f32(texel.x), f32(texel.y), f32((h >> 21u) & 1u));
}`,
		texel: ( h, x, y, width, layer ) => [ y * width + x - layer, x, y, ( h >>> 21 ) & 1 ]
	}
];

function proceduralReference( { texel, reduce }, width, height, layers, seed ) {

	const results = [];
	for ( let layer = 0; layer < layers; layer ++ ) {

		const total = reduce.map( op => op === 'min' ? Infinity : op === 'max' ? - Infinity : 0 );
		for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

			const h = ( Math.imul( x, 73856093 ) ^ Math.imul( y, 19349663 ) ^ Math.imul( layer, 83492791 ) ^ Math.imul( seed, 2654435761 | 0 ) ) >>> 0;
			const v = texel( h, x, y, width, layer );
			for ( let c = 0; c < 4; c ++ ) total[ c ] = reduce[ c ] === 'min' ? Math.min( total[ c ], v[ c ] ) : reduce[ c ] === 'max' ? Math.max( total[ c ], v[ c ] ) : total[ c ] + v[ c ];

		}

		results.push( { value: total, weight: width * height } );

	}

	return results;

}

function makeBuffer( device, size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST ) {

	return device.createBuffer( { size, usage } );

}

/** NaN-poison a buffer so a skipped write fails. */
function poisonBuffer( device, buffer ) {

	device.queue.writeBuffer( buffer, 0, new Float32Array( buffer.size / 4 ).fill( NaN ) );

}

async function readBuffer( device, buffer, offset = 0, size = buffer.size - offset ) {

	const readback = device.createBuffer( { size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer( buffer, offset, readback, 0, size );
	device.queue.submit( [ encoder.finish() ] );
	await readback.mapAsync( GPUMapMode.READ );
	const values = new Float32Array( readback.getMappedRange().slice( 0 ) );
	readback.unmap();
	readback.destroy();
	return values;

}

/** Results per layer from an output buffer: `{ value: vec4f, weight: f32 }`, 32 bytes each. */
async function readResults( device, buffer, offset, layers ) {

	const values = await readBuffer( device, buffer, offset, layers * scalarResultBytes );
	return Array.from( { length: layers }, ( _, i ) => ( { value: Array.from( values.subarray( i * 8, i * 8 + 4 ) ), weight: values[ i * 8 + 4 ] } ) );

}

function checkExact( actual, expected, label ) {

	actual.forEach( ( result, layer ) => {

		const want = expected[ layer ];
		for ( let c = 0; c < 4; c ++ ) if ( ! Object.is( result.value[ c ], want.value[ c ] ) && ! ( result.value[ c ] === 0 && want.value[ c ] === 0 ) ) throw new Error( `${ label } layer ${ layer } channel ${ c }: actual ${ result.value[ c ] }, expected ${ want.value[ c ] }` );
		if ( result.weight !== want.weight ) throw new Error( `${ label } layer ${ layer } weight: actual ${ result.weight }, expected ${ want.weight }` );

	} );

}

// How often the GPU matched the oracle's float32 pairing bit for bit (reported, not required).
const pairing = { same: 0, total: 0 };

/** Within the oracle's bound of the exact (float64) result, for every channel and the weight. */
function checkBound( actual, expected, label ) {

	let worst = 0;
	actual.forEach( ( result, layer ) => {

		const want = expected[ layer ];
		for ( let c = 0; c < 4; c ++ ) {

			const error = Math.abs( result.value[ c ] - want.exact[ c ] );
			if ( ! Number.isFinite( result.value[ c ] ) || error > want.bound[ c ] ) throw new Error( `${ label } layer ${ layer } channel ${ c }: actual ${ result.value[ c ] }, exact ${ want.exact[ c ] }, error ${ error } > bound ${ want.bound[ c ] }` );
			if ( want.bound[ c ] > 0 ) worst = Math.max( worst, error / want.bound[ c ] );
			pairing.total ++;
			if ( Object.is( Math.fround( result.value[ c ] ), Math.fround( want.value[ c ] ) ) ) pairing.same ++;

		}

		if ( ! ( Math.abs( result.weight - want.exactWeight ) <= want.weightBound ) ) throw new Error( `${ label } layer ${ layer } weight: actual ${ result.weight }, exact ${ want.exactWeight }, bound ${ want.weightBound }` );

	} );
	return worst;

}

/** HDR fixtures: channels 0–2 in [−4, 12] (rgba16float: quantized), channel 3 a weight in [0, 1). */
function makeHdr( width, height, layers, format, seed ) {

	let state = seed;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
	const count = width * height * layers * 4;
	if ( format === 'rgba32float' ) return Float32Array.from( { length: count }, ( _, i ) => i % 4 === 3 ? next() : next() * 16 - 4 );
	return Uint16Array.from( { length: count }, ( _, i ) => toHalf( i % 4 === 3 ? next() : next() * 16 - 4 ) );

}

function createSource( device, width, height, layers, format, dimension ) {

	return device.createTexture( {
		label: `scalar source ${ width }x${ height }x${ layers } ${ format }`, size: [ width, height, layers ], format,
		textureBindingViewDimension: dimension ?? ( layers > 1 ? '2d-array' : '2d' ),
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
	} );

}

function run( device, reduction, options, mode = 'encode' ) {

	const encoder = device.createCommandEncoder();
	if ( mode === 'encode' ) reduction.encode( encoder, options );
	else {

		const pass = encoder.beginComputePass();
		reduction.encodePass( pass, options );
		pass.end();

	}

	device.queue.submit( [ encoder.finish() ] );

}

export async function validateScalar( device, { quick = false, onProgress = () => {} } = {} ) {

	const results = [];
	// Compatibility mode defaults to 4096 texels per side (4111×300 is beyond it).
	const fits = ( [ w, h ] ) => Math.max( w, h ) <= device.limits.maxTextureDimension2D;
	const shapes = ( quick ? scalarShapes.filter( shape => ! big( shape ) ) : scalarShapes ).filter( fits );
	const output = makeBuffer( device, 256 + 8 * scalarResultBytes );
	const seedBuffer = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );

	// Procedural hooks: exact counting at every size and layer count.
	for ( const variant of procedural ) {

		const reduction = new ScalarReduction( device, { reduce: variant.reduce, load: variant.load, bindings: proceduralBindings } );
		const bindGroup = device.createBindGroup( { layout: reduction.kernel().partials.getBindGroupLayout( 1 ), entries: [ { binding: 0, resource: { buffer: seedBuffer } } ] } );

		for ( const [ width, height, layers = 1 ] of shapes ) for ( const [ seed, mode ] of [ [ 1, 'encode' ], [ 2, 'encodePass' ] ] ) {

			device.queue.writeBuffer( seedBuffer, 0, Uint32Array.of( seed, width, 0, 0 ) );
			poisonBuffer( device, output );
			run( device, reduction, { size: [ width, height ], layers: [ 0, layers ], bindGroup, output: { buffer: output, offset: 256 } }, mode );
			checkExact( await readResults( device, output, 256, layers ), proceduralReference( variant, width, height, layers, seed ), `scalar procedural ${ variant.name } ${ width }x${ height }x${ layers } seed ${ seed } ${ mode }` );
			results.push( { scalar: `procedural ${ variant.name }`, width, height, layers, seed, mode } );

		}

	}

	onProgress( 'validated scalar procedural' );

	// Textures: float sums within the bound, min/max exact.
	const textureVariants = [
		{ name: 'average', reduce: 'average' },
		{ name: 'sum/average/min/max', reduce: [ 'sum', 'average', 'min', 'max' ] },
		{ name: 'weighted', reduce: [ 'weighted', 'weighted', 'weighted', 'weight' ] },
		{ name: 'weighted, sum', reduce: [ 'weighted', 'sum', 'max', 'weight' ] }
	].map( variant => ( { ...variant, reduction: new ScalarReduction( device, { reduce: variant.reduce } ) } ) );
	let worst = 0;

	for ( const [ shapeIndex, [ width, height, layers = 1, dimension ] ] of shapes.entries() ) {

		// float32 everywhere but 4K, where rgba16float halves the upload.
		for ( const format of width * height > 4e6 ? [ 'rgba16float' ] : [ 'rgba32float', 'rgba16float' ] ) {

			const texture = createSource( device, width, height, layers, format, dimension );

			try {

				const data = makeHdr( width, height, layers, format, shapeIndex + 1 );
				upload( device, texture, data );
				const floats = format === 'rgba16float' ? Float32Array.from( data, fromHalf ) : data;

				for ( const variant of textureVariants ) {

					poisonBuffer( device, output );
					run( device, variant.reduction, { texture, output: { buffer: output, offset: 256 } } );
					const expected = scalarReference( floats, width, height, layers, variant.reduce );
					worst = Math.max( worst, checkBound( await readResults( device, output, 256, layers ), expected, `scalar texture ${ variant.name } ${ width }x${ height }x${ layers } ${ format }` ) );
					results.push( { scalar: `texture ${ variant.name }`, width, height, layers, format } );

				}

				// A layer range, with results at offset 0.
				if ( layers > 1 ) {

					const reduction = textureVariants[ 1 ].reduction;
					poisonBuffer( device, output );
					run( device, reduction, { texture, layers: [ 1, layers - 1 ], output: { buffer: output } }, 'encodePass' );
					const layerFloats = floats.subarray( width * height * 4 );
					checkBound( await readResults( device, output, 0, layers - 1 ), scalarReference( layerFloats, width, height, layers - 1, textureVariants[ 1 ].reduce ), `scalar layer range ${ width }x${ height }x${ layers } ${ format }` );
					results.push( { scalar: 'layer range', width, height, layers, format } );

				}

			} finally {

				texture.destroy();

			}

		}

		onProgress( `validated scalar ${ width }×${ height }×${ layers }` );

	}

	// Normalized sources: unorm → float conversion needn't be correctly rounded,
	// so each input may be off by 2^-21 relative (as SPEC §7 allows for folds).
	for ( const [ width, height, layers = 1 ] of shapes.filter( shape => shape[ 0 ] * shape[ 1 ] <= 2.1e6 ) ) {

		const texture = createSource( device, width, height, layers, 'rgba8unorm' );
		const data = makeData( width, height, layers, 'rgba8unorm', 'random', 5 );
		upload( device, texture, data );
		const reduce = [ 'sum', 'average', 'min', 'max' ];
		const reduction = textureVariants[ 1 ].reduction;
		poisonBuffer( device, output );
		run( device, reduction, { texture, output: { buffer: output, offset: 256 } } );
		checkBound( await readResults( device, output, 256, layers ), scalarReference( toFloats( data, 'rgba8unorm' ), width, height, layers, reduce, { inputError: { relative: 2 ** - 21 } } ), `scalar rgba8unorm ${ width }x${ height }x${ layers }` );
		results.push( { scalar: 'rgba8unorm', width, height, layers } );
		texture.destroy();

	}

	// Depth: min/max of depth32float and depth16unorm (splatted), exact.
	const depthRange = new ScalarReduction( device, { reduce: [ 'min', 'max', 'min', 'max' ] } );
	const depthShapes = shapes.filter( ( [ , , layers = 1 ] ) => layers === 1 ).filter( shape => shape[ 0 ] * shape[ 1 ] <= 2.1e6 );
	for ( const format of [ 'depth32float', 'depth16unorm' ] ) for ( const [ width, height ] of depthShapes ) {

		const values = makeData( width, height, 1, 'r32float', 'random', width + height, { unit: true } );
		// Plant the extremes in the last texel, so dropped edges fail.
		values[ values.length - 1 ] = 0.999;
		if ( values.length > 1 ) values[ values.length - 2 ] = 0.0001;
		const depth = await createDepthSource( device, { width, height, format }, values );
		poisonBuffer( device, output );
		run( device, depthRange, { texture: depth.texture, output: { buffer: output, offset: 256 } } );
		const splatted = new Float32Array( depth.floats.length );
		for ( let i = 0; i < splatted.length; i += 4 ) splatted.fill( depth.floats[ i ], i, i + 4 );
		const slack = format === 'depth16unorm' ? 2 ** - 21 : 0;
		const [ actual ] = await readResults( device, output, 256, 1 );
		const [ expected ] = scalarReference( splatted, width, height, 1, depthRange.ops, { inputError: { relative: slack } } );
		for ( let c = 0; c < 4; c ++ ) if ( ! ( Math.abs( actual.value[ c ] - expected.exact[ c ] ) <= expected.bound[ c ] ) ) throw new Error( `scalar depth ${ format } ${ width }x${ height } channel ${ c }: actual ${ actual.value[ c ] }, expected ${ expected.exact[ c ] }` );
		results.push( { scalar: `depth min/max ${ format }`, width, height } );
		depth.texture.destroy();

	}

	onProgress( 'validated scalar depth' );
	results.push( ...await validateCombineOnly( device, { quick } ) );
	onProgress( 'validated scalar combine-only' );
	results.push( ...await validateExposure( device, { quick } ) );
	results.push( ...validateScalarOptions( device ) );
	output.destroy();
	seedBuffer.destroy();
	return { results, worst, pairing: `${ pairing.same }/${ pairing.total }` };

}

// Combine-only runs (`ScalarRun.partials`): the combine pass over partials a
// tool wrote. First tool-written buffers (integers: exact; floats: within the
// oracle's bound), at counts around the 256-invocation fold and with layers at
// an offset; then the shape it exists for, a Reducer whose store hook writes
// its deepest level's accumulations, combined in the same pass (2 dispatches,
// one read of the source), checked against the whole image's exact result.
const combineVariants = [
	{ name: 'sum/average/min/max', reduce: [ 'sum', 'average', 'min', 'max' ], texels: 1000 },
	{ name: 'weighted', reduce: [ 'weighted', 'weighted', 'max', 'weight' ] },
	{ name: 'weighted, average', reduce: [ 'weighted', 'average', 'min', 'weight' ], texels: 77 }
];

const combineBindings = {
	wgsl: `struct CombineTool { width: u32, height: u32, groupsX: u32, count: u32 }
@group(1) @binding(0) var<storage, read> toolInput: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> toolPartials: array<vec4f>;
@group(1) @binding(2) var<uniform> combineTool: CombineTool;`,
	entries: [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
	]
};
// (w·r, g, b, w): the Reducer sums the weighted term itself, takes the min of b.
const combineLoad = `fn load(texel: vec2i, layer: u32) -> vec4f {
	let v = toolInput[(layer * combineTool.height + u32(texel.y)) * combineTool.width + u32(texel.x)];
	return vec4f(v.x * v.w, v.y, v.z, v.w);
}`;
const combineStore = deepest => `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
	if (level == ${ deepest }u) { toolPartials[layer * combineTool.count + u32(texel.y) * combineTool.groupsX + u32(texel.x)] = value; }
}`;

async function validateCombineOnly( device, { quick } ) {

	const results = [];
	const output = makeBuffer( device, 256 + 8 * scalarResultBytes );
	let state = 11;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;

	for ( const variant of combineVariants ) {

		const reduction = new ScalarReduction( device, { reduce: variant.reduce } );

		for ( const count of quick ? [ 1, 255, 256, 257, 1000 ] : [ 1, 2, 255, 256, 257, 511, 513, 1000, 2040, 8161 ] ) for ( const layers of [ 1, 3 ] ) for ( const integer of [ true, false ] ) {

			// Partials as a tool writes them, at offset 256 (layer after layer), poison around them.
			const values = Float32Array.from( { length: count * layers * 4 }, ( _, i ) => {

				const c = i % 4;
				if ( integer ) return Math.floor( next() * ( c === 3 ? 64 : 4096 ) ) - ( c === 3 ? 0 : 2048 );
				return c === 3 ? next() * 16 : next() * 512 - 256;

			} );
			const partials = makeBuffer( device, 256 + values.byteLength + 64 );
			poisonBuffer( device, partials );
			device.queue.writeBuffer( partials, 256, values );
			poisonBuffer( device, output );
			const texels = variant.texels;
			run( device, reduction, { partials: { buffer: partials, offset: 256, count }, texels, layers: [ 0, layers ], output: { buffer: output, offset: 256 } }, integer ? 'encode' : 'encodePass' );
			const actual = await readResults( device, output, 256, layers );
			const expected = scalarCombineReference( values, count, layers, variant.reduce, { texels } );
			const label = `scalar combine-only ${ variant.name } ${ count } partials x${ layers } ${ integer ? 'integer' : 'float' }`;
			// Integer partials below 2^24 sum exactly; only the final division rounds.
			if ( integer ) actual.forEach( ( result, layer ) => {

				for ( let c = 0; c < 4; c ++ ) if ( Math.abs( result.value[ c ] - expected[ layer ].value[ c ] ) > 2 ** - 22 * Math.abs( expected[ layer ].value[ c ] ) ) throw new Error( `${ label } layer ${ layer } channel ${ c }: actual ${ result.value[ c ] }, expected ${ expected[ layer ].value[ c ] }` );
				if ( result.weight !== expected[ layer ].weight ) throw new Error( `${ label } layer ${ layer } weight: actual ${ result.weight }, expected ${ expected[ layer ].weight }` );

			} );
			else checkBound( actual, expected, label );
			partials.destroy();
			results.push( { scalar: `combine-only ${ variant.name }`, count, layers, integer } );

		}

	}

	// Reducer → combine in one pass: the Reducer's level `deepest` holds one
	// accumulation per 2^deepest tile; the scalar combines them.
	const scalar = new ScalarReduction( device, { reduce: [ 'weighted', 'average', 'min', 'weight' ] } );
	const shapes = quick ? [ [ 1, 1 ], [ 17, 9 ], [ 65, 129 ], [ 17, 11, 3 ] ] : [ [ 1, 1 ], [ 17, 9 ], [ 63, 65 ], [ 64, 64 ], [ 65, 129 ], [ 255, 257 ], [ 17, 11, 3 ], [ 1920, 1080 ] ];
	for ( const deepest of [ 4, 6 ] ) {

		const reducer = new Reducer( device, { reduction: [ 'sum', 'sum', 'min', 'sum' ], levels: [ deepest ], edges: 'zero-weight', dimension: '2d-array', load: combineLoad, store: combineStore( deepest ), bindings: combineBindings } );

		for ( const [ width, height, layers = 1 ] of shapes ) {

			const groupsX = levelSize( width, deepest, 'zero-weight' ), count = groupsX * levelSize( height, deepest, 'zero-weight' );
			const data = Float32Array.from( { length: width * height * layers * 4 }, ( _, i ) => i % 4 === 3 ? ( next() < 0.3 ? 0 : next() ) : next() * 16 - 4 );
			const input = makeBuffer( device, Math.max( 16, data.byteLength ) );
			device.queue.writeBuffer( input, 0, data );
			const partials = makeBuffer( device, count * layers * 16 );
			poisonBuffer( device, partials );
			const uniform = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			device.queue.writeBuffer( uniform, 0, Uint32Array.of( width, height, groupsX, count ) );
			const bindGroup = device.createBindGroup( { layout: reducer.bindGroupLayouts[ 1 ], entries: [ input, partials, uniform ].map( ( buffer, binding ) => ( { binding, resource: { buffer } } ) ) } );
			poisonBuffer( device, output );
			const encoder = device.createCommandEncoder();
			const pass = encoder.beginComputePass();
			reducer.encodePass( pass, { width, height, layers, bindGroup } );
			scalar.encodePass( pass, { partials: { buffer: partials, count }, texels: width * height, layers: [ 0, layers ], output: { buffer: output, offset: 256 } } );
			pass.end();
			device.queue.submit( [ encoder.finish() ] );
			const actual = await readResults( device, output, 256, layers );

			// Exact over the image; bound: the Reducer's two roundings per level, then the combine.
			const expected = [];
			for ( let layer = 0; layer < layers; layer ++ ) {

				const texel = ( i, c ) => data[ ( layer * width * height + i ) * 4 + c ];
				const tiles = new Float64Array( count * 4 ).fill( 0 );
				const magnitude = [ 0, 0, 0, 0 ];
				for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

					const i = y * width + x, tile = ( y >> deepest ) * groupsX + ( x >> deepest );
					const terms = [ Math.fround( texel( i, 0 ) * texel( i, 3 ) ), texel( i, 1 ), texel( i, 2 ), texel( i, 3 ) ];
					for ( let c = 0; c < 4; c ++ ) {

						if ( c === 2 ) tiles[ tile * 4 + c ] = x % ( 1 << deepest ) === 0 && y % ( 1 << deepest ) === 0 ? terms[ c ] : Math.min( tiles[ tile * 4 + c ], terms[ c ] );
						else tiles[ tile * 4 + c ] += terms[ c ];
						magnitude[ c ] += Math.abs( terms[ c ] );

					}

				}

				expected.push( scalarCombineReference( tiles, count, 1, scalar.ops, { texels: width * height, partialDepth: 2 * deepest + 1, magnitude } )[ 0 ] );

			}

			checkBound( actual, expected, `scalar Reducer level ${ deepest } → combine ${ width }x${ height }x${ layers }` );
			for ( const buffer of [ input, partials, uniform ] ) buffer.destroy();
			results.push( { scalar: `Reducer level ${ deepest } → combine-only`, width, height, layers } );

		}

	}

	// Rejected runs, and what prepare() hands a tool that dispatches it.
	const partials = makeBuffer( device, 1024 );
	const average = combineVariants[ 0 ];
	const reduction = new ScalarReduction( device, { reduce: average.reduce } );
	const weighted = new ScalarReduction( device, { reduce: combineVariants[ 1 ].reduce } );
	const texture = createSource( device, 8, 8, 1, 'rgba16float' );
	for ( const [ target, options, message ] of [
		[ reduction, { partials: { buffer: partials, count: 4 }, output: { buffer: output } }, 'texels' ],
		[ reduction, { partials: { buffer: partials, offset: 16, count: 4 }, texels: 9, output: { buffer: output } }, 'offset' ],
		[ reduction, { partials: { buffer: partials, count: 65 }, texels: 9, output: { buffer: output } }, 'too small' ],
		[ reduction, { partials: { buffer: partials, count: 16 }, texels: 9, layers: [ 0, 5 ], output: { buffer: output } }, 'too small' ],
		[ reduction, { partials: { buffer: partials, count: 0 }, texels: 9, output: { buffer: output } }, 'count' ],
		[ reduction, { partials: { buffer: partials, count: 4 }, texels: 9, texture, output: { buffer: output } }, 'no texture' ],
		[ reduction, { texture, texels: 9, output: { buffer: output } }, 'combine-only' ],
		[ weighted, { partials: { buffer: partials, count: 4 }, texels: 0, output: { buffer: output } }, 'positive' ],
		[ weighted, { partials: { buffer: output, count: 4 } }, 'output buffer' ]
	] ) {

		const reason = target.unsupportedReason( options );
		if ( reason === null || ! reason.includes( message ) ) throw new Error( `expected combine-only rejection "${ message }", got ${ reason }` );

	}

	const prepared = weighted.prepare( { partials: { buffer: partials, count: 64 }, layers: [ 2, 1 ], output: { buffer: output } } );
	if ( prepared.partials !== null || prepared.combineWorkgroups.join() !== '1,1,1' || prepared.combine !== weighted.combineKernel().combine ) throw new Error( 'scalar combine-only prepare(): unexpected pipelines or workgroups' );
	if ( weighted.combineKernel().layout === weighted.kernel().layout ) throw new Error( 'scalar combine-only: expected its own group-0 layout (no source texture)' );
	results.push( { option: 'scalar combine-only validation and prepare' } );
	for ( const resource of [ partials, texture, output ] ) resource.destroy();
	return results;

}

// FSR-style auto-exposure (SPEC §14 luma pyramid): the load hook takes the log
// luma of an rgba16float HDR colour texture from the tool's group 1; finish
// smooths it over time into the tool's state buffer and a 1×1 rgba32float
// storage texture, reading the previous value from the state buffer.
export const exposureBindings = {
	wgsl: `struct Exposure { deltaTime: f32, reset: u32 }
@group(1) @binding(0) var colour: texture_2d<f32>;
@group(1) @binding(1) var<uniform> exposureParams: Exposure;
@group(1) @binding(2) var<storage, read_write> exposureState: array<vec4f, 1>;
@group(1) @binding(3) var frameInfo: texture_storage_2d<rgba32float, write>;`,
	entries: [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		{ binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } }
	]
};

export const exposureLoad = `fn load(texel: vec2i, layer: u32) -> vec4f {
	let luma = dot(textureLoad(colour, texel, 0).rgb, vec3f(0.2126, 0.7152, 0.0722));
	return vec4f(log(max(luma, 6.1e-5)));
}`;

// State and frameInfo: (exposure, smoothed log luma, this frame's mean log luma, texel count).
export const exposureFinish = `fn finish(value: vec4f, weight: f32) {
	let previous = exposureState[0].y;
	let rate = 1.0 - exp(-exposureParams.deltaTime);
	let smoothed = select(previous + (value.x - previous) * rate, value.x, exposureParams.reset != 0u);
	let state = vec4f(1.0 / (9.6 * exp(smoothed)), smoothed, value.x, weight);
	exposureState[0] = state;
	textureStore(frameInfo, vec2i(0), state);
}`;

async function validateExposure( device, { quick } ) {

	const results = [];
	const reduction = new ScalarReduction( device, { reduce: 'average', load: exposureLoad, finish: exposureFinish, bindings: exposureBindings } );
	const layout = reduction.kernel().partials.getBindGroupLayout( 1 );
	const params = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
	const state = makeBuffer( device, 16 );
	const frameInfo = device.createTexture( { size: [ 1, 1 ], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC } );
	const readback = device.createBuffer( { size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const shapes = quick ? [ [ 1, 1 ], [ 17, 9 ], [ 255, 257 ] ] : [ [ 1, 1 ], [ 17, 9 ], [ 255, 257 ], [ 1477, 831 ], [ 1707, 960 ], [ 1920, 1080 ], [ 3840, 2160 ] ];

	for ( const [ width, height ] of shapes ) {

		const colour = createSource( device, width, height, 1, 'rgba16float' );
		let smoothed = 0;

		for ( const [ frame, deltaTime ] of [ [ 0, 0.016 ], [ 1, 0.016 ], [ 2, 0.1 ] ] ) {

			// Frames differ, and the right half is brighter in frame 1: an
			// FSR-style top-left 2^n square would miss it.
			let state0 = frame * 977 + width;
			const next = () => ( state0 = ( Math.imul( state0, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
			const data = new Uint16Array( width * height * 4 );
			for ( let i = 0; i < width * height; i ++ ) {

				const x = i % width, scale = frame === 1 && x >= width / 2 ? 64 : 4;
				for ( let c = 0; c < 4; c ++ ) data[ i * 4 + c ] = toHalf( c === 3 ? 1 : next() ** 3 * scale );

			}

			if ( width * height > 1 ) data[ 0 ] = data[ 1 ] = data[ 2 ] = toHalf( 0 ); // below the 6.1e-5 floor
			upload( device, colour, data );
			device.queue.writeBuffer( params, 0, new Uint32Array( [ new Uint32Array( Float32Array.of( deltaTime ).buffer )[ 0 ], frame === 0 ? 1 : 0, 0, 0 ] ) );
			const bindGroup = device.createBindGroup( { layout, entries: [
				{ binding: 0, resource: colour.createView() }, { binding: 1, resource: { buffer: params } },
				{ binding: 2, resource: { buffer: state } }, { binding: 3, resource: frameInfo.createView() }
			] } );
			const encoder = device.createCommandEncoder();
			const pass = encoder.beginComputePass();
			reduction.encodePass( pass, { size: [ width, height ], bindGroup } );
			pass.end();
			encoder.copyTextureToBuffer( { texture: frameInfo }, { buffer: readback, bytesPerRow: 256 }, [ 1, 1 ] );
			device.queue.submit( [ encoder.finish() ] );
			await readback.mapAsync( GPUMapMode.READ );
			const actual = Array.from( new Float32Array( readback.getMappedRange(), 0, 4 ) );
			readback.unmap();
			const stored = await readBuffer( device, state );

			// Oracle: log luma per texel in f32, the GPU's pairing, then the recurrence.
			const logLuma = new Float32Array( width * height * 4 );
			for ( let i = 0; i < width * height; i ++ ) {

				const luma = Math.fround( 0.2126 * fromHalf( data[ i * 4 ] ) + 0.7152 * fromHalf( data[ i * 4 + 1 ] ) + 0.0722 * fromHalf( data[ i * 4 + 2 ] ) );
				logLuma.fill( Math.log( Math.max( luma, Math.fround( 6.1e-5 ) ) ), i * 4, i * 4 + 4 );

			}

			// WGSL log: 2^-21 absolute in [0.5, 2], 3 ulp elsewhere; luma's own rounding adds ~2 ulp.
			const [ mean ] = scalarReference( logLuma, width, height, 1, 'average', { inputError: { absolute: 2 ** - 20, relative: 2 ** - 21 } } );
			smoothed = frame === 0 ? mean.exact[ 0 ] : smoothed + ( mean.exact[ 0 ] - smoothed ) * ( 1 - Math.exp( - Math.fround( deltaTime ) ) );
			const expected = [ 1 / ( 9.6 * Math.exp( actual[ 1 ] ) ), smoothed, mean.exact[ 0 ], width * height ];
			const label = `scalar exposure ${ width }x${ height } frame ${ frame }`;
			if ( ! ( Math.abs( actual[ 2 ] - expected[ 2 ] ) <= mean.bound[ 0 ] ) ) throw new Error( `${ label } mean log luma: actual ${ actual[ 2 ] }, expected ${ expected[ 2 ] }, bound ${ mean.bound[ 0 ] }` );
			if ( ! ( Math.abs( actual[ 1 ] - expected[ 1 ] ) <= mean.bound[ 0 ] + 1e-6 * ( 1 + Math.abs( expected[ 1 ] ) ) ) ) throw new Error( `${ label } smoothed: actual ${ actual[ 1 ] }, expected ${ expected[ 1 ] }` );
			if ( ! ( Math.abs( actual[ 0 ] - expected[ 0 ] ) <= 1e-5 * expected[ 0 ] ) ) throw new Error( `${ label } exposure: actual ${ actual[ 0 ] }, expected ${ expected[ 0 ] }` );
			if ( actual[ 3 ] !== expected[ 3 ] ) throw new Error( `${ label } weight: actual ${ actual[ 3 ] }, expected ${ expected[ 3 ] }` );
			if ( stored.some( ( value, i ) => ! Object.is( value, actual[ i ] ) ) ) throw new Error( `${ label }: state buffer and frameInfo differ` );
			smoothed = actual[ 1 ]; // continue from the GPU's state, so rounding does not accumulate
			results.push( { scalar: 'exposure', width, height, frame } );

		}

		colour.destroy();

	}

	for ( const resource of [ params, state, frameInfo, readback ] ) resource.destroy();
	return results;

}

function validateScalarOptions( device ) {

	const results = [];
	const texture = createSource( device, 8, 8, 1, 'rgba16float' );
	const output = makeBuffer( device, 512 );
	const average = new ScalarReduction( device );
	const hooked = new ScalarReduction( device, { load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return vec4f(1); }', bindings: proceduralBindings } );
	const rejected = [
		[ average, {}, 'source texture' ],
		[ average, { texture }, 'output buffer' ],
		[ average, { texture, output: { buffer: output, offset: 16 } }, 'multiple of 256' ],
		[ average, { texture, output: { buffer: output, offset: 512 } }, 'too small' ],
		[ average, { texture, layers: [ 0, 2 ], output: { buffer: output } }, 'layers' ],
		[ average, { texture, mipLevel: 1, output: { buffer: output } }, 'mipLevel' ],
		[ hooked, { texture, output: { buffer: output } }, 'not a source texture' ],
		[ hooked, { output: { buffer: output } }, 'size' ],
		[ hooked, { size: [ 4, 4 ], output: { buffer: output } }, 'bindGroup' ],
		[ hooked, { size: [ 0, 4 ], output: { buffer: output } }, 'positive' ]
	];

	for ( const [ reduction, options, message ] of rejected ) {

		const reason = reduction.unsupportedReason( options );
		if ( reason === null || ! reason.includes( message ) ) throw new Error( `expected scalar rejection "${ message }", got ${ reason }` );
		let threw = false;
		try { reduction.encode( device.createCommandEncoder(), options ); } catch { threw = true; }
		if ( ! threw ) throw new Error( `scalar encode() accepted a run that should fail with "${ message }"` );

	}

	for ( const [ options, message ] of [
		[ { reduce: [ 'weighted', 'sum', 'sum', 'sum' ] }, 'weight' ],
		[ { load: 'fn loader() {}' }, 'load hook' ],
		[ { finish: 'fn done() {}' }, 'finish hook' ],
		[ { block: 3 }, 'block' ]
	] ) {

		let error = null;
		try { new ScalarReduction( device, options ); } catch ( caught ) { error = caught; }
		if ( ! error?.message.includes( message ) ) throw new Error( `expected ScalarReduction to reject ${ JSON.stringify( options ) } with "${ message }", got ${ error?.message }` );

	}

	// Tools that dispatch themselves: the exposed pipelines, bind group and counts.
	const prepared = average.prepare( { texture, output: { buffer: output } } );
	if ( prepared.workgroups.join() !== '1,1,1' || prepared.combineWorkgroups.join() !== '1,1,1' ) throw new Error( 'scalar prepare(): unexpected workgroup counts' );
	if ( ! /fn partials/.test( average.kernel().wgsl ) || ! /fn combine/.test( average.kernel().wgsl ) ) throw new Error( 'scalar kernel(): missing entry points' );
	texture.destroy();
	output.destroy();
	results.push( { option: 'scalar validation and prepare' } );
	return results;

}
