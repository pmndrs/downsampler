// GPU suite for payload folds (SPEC §14.1 Payload folds): 'argmin'/'argmax'
// keys carrying 'payload' channels, in Reducer (texture and hook paths, 4 and 8
// channels) and ScalarReduction (image and combine-only runs). Keys and the
// first payload take a few integer values, so ties on both are common and the
// lexicographic tie rule decides most texels. Everything selected must match
// the oracle exactly.
import { levelSize, pyramidTolerance, reducePyramid, scalarCombineReference, scalarReference } from '/dist/index.js';
import { Reducer, ScalarReduction } from '/dist/webgpu/index.js';
import { readMips, upload } from './gpu.js';
import { wideBindings, wideLoad, wideTool } from './validate-wide.js';

const poisonValue = - 12345.5;

export const payloadShapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 64, 64 ], [ 65, 129 ], [ 1, 128 ], [ 128, 8 ],
	[ 192, 108 ], [ 255, 257 ], [ 17, 11, 3 ], [ 1920, 1080 ]
];

export const payloadVariants = [
	{ name: 'argmin + 3 payload', reduction: [ 'argmin', 'payload', 'payload', 'payload' ] },
	{ name: 'argmax + 3 payload', reduction: [ 'argmax', 'payload', 'payload', 'payload' ] },
	{ name: 'argmin + payload, argmax + payload', reduction: [ 'argmin', 'payload', 'argmax', 'payload' ] },
	{ name: 'argmax + payload, average, min', reduction: [ 'argmax', 'payload', 'average', 'min' ] }
];

/** RGBA f32: channels 0 and 2 in {0, 1, 2, 3} (keys), 1 in {0, 1, 2} (first payload), 3 in [−4, 12). */
function payloadData( width, height, layers, seed ) {

	let state = seed;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) >>> 8;
	return Float32Array.from( { length: width * height * layers * 4 }, ( _, i ) => {

		const c = i % 4;
		if ( c === 0 || c === 2 ) return next() % 4;
		if ( c === 1 ) return next() % 3;
		return next() / 2 ** 20 - 4;

	} );

}

function checkExact( actual, expected, label ) {

	for ( let i = 0; i < expected.values.length; i ++ ) {

		const limit = pyramidTolerance( expected.magnitude[ i ] );
		if ( ! ( Math.abs( actual[ i ] - expected.values[ i ] ) <= limit ) ) throw new Error( `${ label }: value ${ i } (texel ${ Math.floor( i / 4 ) } channel ${ i % 4 }), actual ${ actual[ i ] }, expected ${ expected.values[ i ] }` );

	}

}

const textureLevels = [ [ 1 ], [ 1, 2, 4 ], [ 2, 5 ], [ 1, 3, 4, 6 ] ];
const slug = name => name.replace( /[^a-z0-9]+/gi, '-' ).replace( /^-|-$/g, '' );

// Case groups (test/ci/shard.js): edges × variant × level selection, over every shape.
async function validatePayloadReducer( device, { quick, select } ) {

	const results = [];
	const groupId = ( edges, variant, levels ) => `payload/reducer/${ edges }/${ slug( variant.name ) }/L${ levels.join( '' ) }`;
	const shapes = quick ? payloadShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : payloadShapes;
	const limit = device.limits.maxStorageTexturesPerShaderStage;

	const levelSets = textureLevels.filter( set => set.length <= limit );
	const wanted = ( edges, variant ) => levelSets.filter( levels => select.has( groupId( edges, variant, levels ) ) );
	const anyWanted = edges => payloadVariants.some( variant => wanted( edges, variant ).length > 0 );

	for ( const [ width, height, layers = 1 ] of shapes ) {

		if ( ! [ 'zero-weight', 'clamp' ].some( anyWanted ) ) continue;
		const dimension = layers > 1 ? '2d-array' : '2d';
		const source = device.createTexture( { size: [ width, height, layers ], format: 'rgba32float', textureBindingViewDimension: dimension, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		const data = payloadData( width, height, layers, width * 31 + height );
		upload( device, source, data );
		const owned = [ source ];

		for ( const edges of [ 'zero-weight', 'clamp' ] ) {

			if ( ! anyWanted( edges ) ) continue;
			const outputs = new Map( [ 1, 2, 3, 4, 5, 6 ].map( level => {

				const texture = device.createTexture( { size: [ levelSize( width, level, edges ), levelSize( height, level, edges ), layers ], format: 'rgba32float', textureBindingViewDimension: dimension, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
				owned.push( texture );
				return [ level, texture ];

			} ) );

			for ( const variant of payloadVariants ) {

				const selected = wanted( edges, variant );
				if ( selected.length === 0 ) continue;
				const expected = reducePyramid( data, width, height, layers, { reduction: variant.reduction, levels: [ 1, 2, 3, 4, 5, 6 ], edges } );
				for ( const levels of selected ) results.push( ...await select.run( groupId( edges, variant, levels ), async () => {

					const reducer = new Reducer( device, { reduction: variant.reduction, levels, edges, format: 'rgba32float', dimension } );
					for ( const level of levels ) {

						const texture = outputs.get( level );
						device.queue.writeTexture( { texture }, new Float32Array( texture.width * texture.height * layers * 4 ).fill( poisonValue ), { bytesPerRow: texture.width * 16, rowsPerImage: texture.height }, [ texture.width, texture.height, layers ] );

					}

					const encoder = device.createCommandEncoder();
					reducer.encode( encoder, { source, outputs: levels.map( level => outputs.get( level ) ) } );
					device.queue.submit( [ encoder.finish() ] );
					for ( const level of levels ) checkExact( ( await readMips( device, outputs.get( level ), 0, 1 ) )[ 0 ], expected[ level - 1 ], `payload reducer ${ width }x${ height }x${ layers } ${ edges } ${ variant.name } [${ levels }] level ${ level }` );
					return [ { reducer: `payload ${ variant.name }`, width, height, layers, edges, levels } ];

				} ) );

			}

		}

		for ( const texture of owned ) texture.destroy();

	}

	return results;

}

// ssgiDev's G-buffer downsample (bench/src/demos/18-temporal/ssrt18-downsample.wgsl):
// box-min depth over the block, the normal of the min-depth texel, mean light.
// As a Reducer: 8 channels (depth key, texel index, normal xyz as payload; light
// rgb averaged), 'zero-weight' (in-bounds texels only), blocks of 2 and 4 (levels
// 1, 2). The index as the first payload breaks depth ties by position, which is
// ssgiDev's rule (a strict < over the block in row-major order). The store hook
// writes (depth, index, normal, light) to the wide atlas.
const gbufferOps = [ 'argmin', 'payload', 'payload', 'payload', 'payload', 'average', 'average', 'average' ];
const gbufferStore = `fn store(level: u32, texel: vec2i, layer: u32, value: DsWide) {
	let l = level - 1u;
	let row = layer * tool.rows[l / 4u][l % 4u] + u32(texel.y);
	let i = tool.offset[l / 4u][l % 4u] + row * tool.stride[l / 4u][l % 4u] + u32(texel.x);
	toolAtlas[2u * i] = value.lo;
	toolAtlas[2u * i + 1u] = value.hi;
}`;

/** ssgiDev's loop, as written: minD starts at 1.0 and bestC at the block's first texel. */
function ssgiDevDownsample( depth, normals, light, width, height, block ) {

	const w = Math.ceil( width / block ), h = Math.ceil( height / block );
	const out = [];
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		let minD = 1.0, best = [ x * block, y * block ], sum = [ 0, 0, 0 ], count = 0;
		for ( let dy = 0; dy < block; dy ++ ) for ( let dx = 0; dx < block; dx ++ ) {

			const cx = x * block + dx, cy = y * block + dy;
			if ( cx < width && cy < height ) {

				const d = depth[ cy * width + cx ];
				if ( d < minD ) { minD = d; best = [ cx, cy ]; }
				for ( let c = 0; c < 3; c ++ ) sum[ c ] += light[ ( cy * width + cx ) * 3 + c ];
				count ++;

			}

		}

		const b = best[ 1 ] * width + best[ 0 ];
		out.push( { depth: minD, normal: [ 0, 1, 2 ].map( c => normals[ b * 3 + c ] ), light: sum.map( v => v / Math.max( 1, count ) ) } );

	}

	return out;

}

async function validateGBuffer( device, { quick } ) {

	const results = [];
	const reducer = new Reducer( device, { reduction: gbufferOps, levels: [ 1, 2 ], edges: 'zero-weight', load: wideLoad, store: gbufferStore, bindings: { wgsl: wideBindings.wgsl, entries: wideBindings.entries() } } );
	for ( const [ width, height ] of quick ? [ [ 17, 9 ], [ 64, 64 ], [ 255, 257 ] ] : [ [ 1, 1 ], [ 17, 9 ], [ 64, 64 ], [ 255, 257 ], [ 1920, 1080 ] ] ) {

		let state = width * 3 + height;
		const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
		// Depth in eighths (many ties), a quarter of it sky (1.0), as raw clip depth.
		const depth = Float32Array.from( { length: width * height }, () => next() < 0.25 ? 1 : Math.floor( next() * 8 ) / 8 );
		const normals = Float32Array.from( { length: width * height * 3 }, () => next() * 2 - 1 );
		const light = Float32Array.from( { length: width * height * 3 }, () => next() * 4 );
		const data = new Float32Array( width * height * 8 );
		for ( let i = 0; i < width * height; i ++ ) data.set( [ depth[ i ], i, normals[ i * 3 ], normals[ i * 3 + 1 ], normals[ i * 3 + 2 ], light[ i * 3 ], light[ i * 3 + 1 ], light[ i * 3 + 2 ] ], i * 8 );
		const tool = wideTool( device, reducer.bindGroupLayouts[ 1 ], width, height, 1, 'zero-weight' );
		device.queue.writeBuffer( tool.input, 0, data );
		device.queue.writeBuffer( tool.atlas, 0, new Float32Array( tool.layout.texels * 8 ).fill( poisonValue ) );
		const encoder = device.createCommandEncoder();
		reducer.encode( encoder, { width, height, bindGroup: tool.bindGroup } );
		device.queue.submit( [ encoder.finish() ] );
		const staging = device.createBuffer( { size: tool.atlas.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		const copy = device.createCommandEncoder();
		copy.copyBufferToBuffer( tool.atlas, 0, staging, 0, tool.atlas.size );
		device.queue.submit( [ copy.finish() ] );
		await staging.mapAsync( GPUMapMode.READ );
		const atlas = new Float32Array( staging.getMappedRange().slice( 0 ) );
		staging.unmap();
		staging.destroy();
		const expected = reducePyramid( data, width, height, 1, { reduction: gbufferOps, levels: [ 1, 2 ], edges: 'zero-weight' } );

		for ( const [ level, block ] of [ [ 1, 2 ], [ 2, 4 ] ] ) {

			const { offset, width: w, height: h } = tool.layout.levels[ level - 1 ];
			const values = atlas.subarray( offset * 8, ( offset + w * h ) * 8 );
			const label = `payload G-buffer ${ width }x${ height } block ${ block }`;
			const reference = ssgiDevDownsample( depth, normals, light, width, height, block );
			reference.forEach( ( texel, i ) => {

				const actual = values.subarray( i * 8, i * 8 + 8 );
				// Depth and normal are selections: exact, against both oracles.
				if ( actual[ 0 ] !== texel.depth || texel.normal.some( ( n, c ) => actual[ 2 + c ] !== n ) ) throw new Error( `${ label } texel ${ i }: depth/normal ${ Array.from( actual.subarray( 0, 5 ) ) }, ssgiDev ${ texel.depth } ${ texel.normal }` );
				for ( let c = 0; c < 8; c ++ ) if ( ! ( Math.abs( actual[ c ] - expected[ level - 1 ].values[ i * 8 + c ] ) <= pyramidTolerance( expected[ level - 1 ].magnitude[ i * 8 + c ] ) ) ) throw new Error( `${ label } texel ${ i } channel ${ c }: actual ${ actual[ c ] }, oracle ${ expected[ level - 1 ].values[ i * 8 + c ] }` );
				for ( let c = 0; c < 3; c ++ ) if ( ! ( Math.abs( actual[ 5 + c ] - texel.light[ c ] ) <= 1e-6 * ( 1 + Math.abs( texel.light[ c ] ) ) ) ) throw new Error( `${ label } texel ${ i } light ${ c }: actual ${ actual[ 5 + c ] }, ssgiDev ${ texel.light[ c ] }` );

			} );
			results.push( { reducer: 'payload G-buffer (ssgiDev)', width, height, block } );

		}

		tool.destroy();

	}

	return results;

}

async function readResults( device, buffer, offset, layers ) {

	const staging = device.createBuffer( { size: layers * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer( buffer, offset, staging, 0, layers * 32 );
	device.queue.submit( [ encoder.finish() ] );
	await staging.mapAsync( GPUMapMode.READ );
	const values = new Float32Array( staging.getMappedRange().slice( 0 ) );
	staging.unmap();
	staging.destroy();
	return Array.from( { length: layers }, ( _, i ) => ( { value: Array.from( values.subarray( i * 8, i * 8 + 4 ) ), weight: values[ i * 8 + 4 ] } ) );

}

function checkScalar( actual, expected, label ) {

	actual.forEach( ( result, layer ) => {

		for ( let c = 0; c < 4; c ++ ) if ( ! ( Math.abs( result.value[ c ] - expected[ layer ].exact[ c ] ) <= expected[ layer ].bound[ c ] ) ) throw new Error( `${ label } layer ${ layer } channel ${ c }: actual ${ result.value[ c ] }, exact ${ expected[ layer ].exact[ c ] }` );

	} );

}

const scalarPayloadVariants = [
	[ 'argmin', 'payload', 'payload', 'payload' ],
	[ 'argmax', 'payload', 'payload', 'sum' ],
	[ 'argmin', 'payload', 'argmax', 'payload' ]
];

async function validatePayloadScalar( device, { quick } ) {

	const results = [];
	const output = device.createBuffer( { size: 256 + 3 * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST } );
	for ( const reduce of scalarPayloadVariants ) {

		const scalar = new ScalarReduction( device, { reduction: reduce } );
		for ( const [ width, height, layers = 1 ] of quick ? payloadShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : payloadShapes ) {

			const source = device.createTexture( { size: [ width, height, layers ], format: 'rgba32float', textureBindingViewDimension: layers > 1 ? '2d-array' : '2d', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
			const data = payloadData( width, height, layers, width + 17 * height );
			upload( device, source, data );
			device.queue.writeBuffer( output, 0, new Float32Array( output.size / 4 ).fill( NaN ) );
			const encoder = device.createCommandEncoder();
			scalar.encode( encoder, { source, output: { buffer: output, offset: 256 } } );
			device.queue.submit( [ encoder.finish() ] );
			checkScalar( await readResults( device, output, 256, layers ), scalarReference( data, width, height, layers, reduce ), `payload scalar ${ reduce } ${ width }x${ height }x${ layers }` );
			source.destroy();
			results.push( { scalar: `payload ${ reduce }`, width, height, layers } );

		}

		// Combine-only: tool-written partials with ties, at counts around the 256-invocation fold.
		for ( const count of [ 1, 255, 257, 2040 ] ) {

			const values = payloadData( count, 1, 2, count );
			const partials = device.createBuffer( { size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
			device.queue.writeBuffer( partials, 0, values );
			device.queue.writeBuffer( output, 0, new Float32Array( output.size / 4 ).fill( NaN ) );
			const encoder = device.createCommandEncoder();
			scalar.encode( encoder, { partials: { buffer: partials, count }, texels: count, layers: [ 0, 2 ], output: { buffer: output, offset: 256 } } );
			device.queue.submit( [ encoder.finish() ] );
			checkScalar( await readResults( device, output, 256, 2 ), scalarCombineReference( values, count, 2, reduce, { texels: count } ), `payload scalar combine-only ${ reduce } ${ count }` );
			partials.destroy();
			results.push( { scalar: `payload combine-only ${ reduce }`, count } );

		}

	}

	output.destroy();
	for ( const [ ops, message ] of [ [ [ 'payload', 'argmin', 'sum', 'sum' ], 'must follow' ], [ [ 'argmin', 'sum', 'payload', 'sum' ], 'must follow' ] ] ) {

		for ( const make of [ () => new ScalarReduction( device, { reduction: ops } ), () => new Reducer( device, { reduction: ops, levels: [ 1 ], format: 'rgba32float' } ) ] ) {

			let error = null;
			try { make(); } catch ( caught ) { error = caught; }
			if ( ! error?.message.includes( message ) ) throw new Error( `expected payload rejection of ${ ops } ("${ message }"), got ${ error?.message }` );

		}

	}

	results.push( { option: 'payload validation' } );
	return results;

}

/** `select`: the case-group gate (test/ci/shard.js); default all. */
export async function validatePayload( device, { quick = false, onProgress = () => {}, select = { has: () => true, run: async ( id, fn ) => fn() } } = {} ) {

	const results = [ ...await validatePayloadReducer( device, { quick, select } ) ];
	onProgress( 'validated payload reducer' );
	results.push( ...await select.run( 'payload/gbuffer', () => validateGBuffer( device, { quick } ) ) );
	results.push( ...await select.run( 'payload/scalar', () => validatePayloadScalar( device, { quick } ) ) );
	onProgress( 'validated payload scalar' );
	return results;

}
