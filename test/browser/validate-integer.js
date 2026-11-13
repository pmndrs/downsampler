// Integer formats (SPEC §6.1 Integer formats; M2.3.7): r32uint, rgba8sint, rgb10a2uint, …
// Every level of every chain must equal `intReference` (core) from the GPU's own previous
// level, bit for bit: integer means are exact rationals rounded half to even, and folds
// select existing values, so nothing is tolerated. Plus the WGSL mean against core's
// `roundedMean` across the u32 range, separate sources (clamped to a narrower destination,
// resampled), partial updates, the buffer and compatibility paths and `encodePass`
// (bit-identical to `encode`), and what is rejected ('spd', Reducer, mixed signedness).
import { bufferWritable, formats, intReference, isIntFormat, roundedMean, sampledChannels, storageWritable } from '/dist/index.js';
import { Downsampler, Reducer, ScalarReduction, intMeanWgsl } from '/dist/webgpu/index.js';
import { createTexture, makeData, poison, readMips, toFloats, upload } from './gpu.js';
import { check, compatibility, expectThrow, levelOptions, shapes, splat, subsetShapes } from './validate.js';

const mipSize = ( size, level ) => Math.max( 1, size >> level );
const slug = name => name.replace( /[^a-z0-9]+/gi, '-' ).replace( /^-|-$/g, '' );

/** Formats that run every shape: the widest unsigned and signed (overflow, sign handling) and an 8-bit one. The rest run the subset. */
export const intFamilyFormats = [ 'rgba8uint', 'rgba32uint', 'rgba32sint' ];

// Custom folds, as integer semilattices. Unsigned: bitwise or (occupancy pyramids). Signed:
// non-negative values first, then the smaller one, a total order that breaks if a kernel hands
// `reduce` offset (biased) values instead of the format's signed integers.
const customFolds = {
	uint: { wgsl: 'fn reduce(a: vec4u, b: vec4u) -> vec4u { return a | b; }', reference: ( a, b ) => ( a | b ) >>> 0 },
	sint: {
		wgsl: `fn reduce(a: vec4i, b: vec4i) -> vec4i {
	let pa = a >= vec4i(0);
	let pb = b >= vec4i(0);
	return select(b, a, (pa & !pb) | ((pa == pb) & (a <= b)));
}`,
		reference: ( a, b ) => ( a >= 0 ) !== ( b >= 0 ) ? ( a >= 0 ? a : b ) : Math.min( a, b )
	}
};

const foldPatterns = [ 'random', 'small', 'edge' ];
const averagePatterns = [ 'constant', 'random', 'small', 'edge', 'tiles' ];
export const intVariants = [
	{ name: 'average', reduction: 'average', patterns: averagePatterns },
	{ name: 'min', reduction: 'min', patterns: foldPatterns },
	{ name: 'max', reduction: 'max', patterns: foldPatterns },
	{ name: 'minmax', reduction: 'minmax', patterns: foldPatterns },
	{ name: 'custom', custom: true, patterns: foldPatterns },
	{ name: 'tail average', reduction: 'average', tail: true, patterns: averagePatterns },
	{ name: 'tail max', reduction: 'max', tail: true, patterns: foldPatterns },
	{ name: 'tail custom', custom: true, tail: true, patterns: foldPatterns },
	{ name: 'box average', reduction: 'average', oddSizes: 'box', patterns: averagePatterns },
	{ name: 'box max', reduction: 'max', oddSizes: 'box', patterns: foldPatterns },
	{ name: 'tail box average', reduction: 'average', oddSizes: 'box', tail: true, patterns: averagePatterns }
];

const signedOf = format => formats[ format ].quantization.startsWith( 's' );
const reductionOf = ( variant, format ) => variant.custom ? { wgsl: customFolds[ signedOf( format ) ? 'sint' : 'uint' ].wgsl } : variant.reduction;
const referenceOf = ( variant, format ) => variant.custom ? customFolds[ signedOf( format ) ? 'sint' : 'uint' ].reference : variant.reduction;

/** An exact comparison of the stored channels (check() with zero tolerance, so --keep-going collects it). */
function compare( actual, expected, format, label ) {

	const { quantization, channels } = formats[ format ];
	return check( actual, { ...expected, filterTolerance: new Float32Array( expected.values.length ) }, quantization, label, 'average', channels, 0, true );

}

/** Every level from level 1 on against `intReference` from the GPU's previous level (level 0 is `input`). */
function checkChain( mips, input, width, height, layers, format, reduction, weights, label, cache ) {

	let previous = input, w = width, h = height;
	mips.forEach( ( actual, i ) => {

		const level = i + 1;
		let expected = level === 1 ? cache?.get( label.key ) : undefined;
		expected ??= intReference( previous, w, h, layers, formats[ format ].quantization, reduction, { weights } );
		if ( level === 1 ) cache?.set( label.key, expected );
		compare( actual, expected, format, `${ label.text } mip ${ level }` );
		previous = actual; w = expected.width; h = expected.height;

	} );

}

/** `part`: 'integer-sources' or 'integer-api' runs that piece alone (for debugging). */
export async function validateInteger( device, { quick = false, onProgress = () => {}, select, part } ) {

	const results = [];
	if ( part === 'integer-sources' ) return select.run( 'integer/sources', () => validateIntSources( device, { quick } ) );
	if ( part === 'integer-api' ) return select.run( 'integer/api', () => validateIntApi( device ) );
	results.push( ...await select.run( 'integer/mean', () => validateMean( device ) ) );
	results.push( ...await validateIntMatrix( device, { quick, onProgress, select } ) );
	results.push( ...await select.run( 'integer/sources', () => validateIntSources( device, { quick } ) ) );
	results.push( ...await select.run( 'integer/api', () => validateIntApi( device ) ) );
	return results;

}

/** Runs a compute shader over `count` cases: `body` reads `input` (u32 words) and writes `output`. */
async function runWgsl( device, helpers, body, input, outputWords, count ) {

	const code = `${ helpers }
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	let i = id.y * 65536u + id.x;
	if (i >= ${ count }u) { return; }
${ body }
}`;
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module: device.createShaderModule( { label: 'integer mean test', code } ), entryPoint: 'main' } } );
	const inBuffer = device.createBuffer( { size: Math.max( 16, input.byteLength ), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( inBuffer, 0, input );
	const outBuffer = device.createBuffer( { size: outputWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
	const readback = device.createBuffer( { size: outputWords * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline( pipeline );
	pass.setBindGroup( 0, device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: { buffer: inBuffer } }, { binding: 1, resource: { buffer: outBuffer } } ] } ) );
	const groups = Math.ceil( count / 64 );
	pass.dispatchWorkgroups( Math.min( groups, 1024 ), Math.ceil( groups / 1024 ) );
	pass.end();
	encoder.copyBufferToBuffer( outBuffer, 0, readback, 0, outputWords * 4 );
	device.queue.submit( [ encoder.finish() ] );
	await readback.mapAsync( GPUMapMode.READ );
	const result = new Uint32Array( readback.getMappedRange().slice( 0 ) );
	readback.unmap();
	for ( const buffer of [ inBuffer, outBuffer, readback ] ) buffer.destroy();
	return result;

}

/**
 * The WGSL means against core's `roundedMean`: `ds_mean` over up to 3×3 taps with per-axis
 * integer weights up to 2^15 (any texture up to 32768 texels per axis), and `ds_mean4`, on
 * values across the u32 range (a third of them at or next to 0, 2^16, 2^31 and 2^32 − 1).
 */
async function validateMean( device ) {

	let state = 99;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 );
	const special = [ 0, 1, 2, 3, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffc, 0xfffffffe, 0xffffffff ];
	const value = () => next() % 3 === 0 ? special[ next() % special.length ] : next();
	const cases = 30000, words = 46;
	const input = new Uint32Array( cases * words );
	const expected = new Uint32Array( cases * 4 );
	for ( let i = 0; i < cases; i ++ ) {

		const base = i * words;
		const kx = 1 + next() % 3, ky = 1 + next() % 3;
		// Weights per axis: random, or the bilinear (2n − r, r) and area extremes, summing to d ≤ 2^15.
		const axis = k => {

			const top = 1 << ( 1 + next() % 15 );
			const w = Array.from( { length: k }, () => next() % 5 === 0 ? 0 : next() % ( Math.floor( top / k ) + 1 ) );
			if ( w.every( x => x === 0 ) ) w[ 0 ] = 1;
			return w;

		};
		const wx = axis( kx ), wy = axis( ky );
		const dx = wx.reduce( ( a, b ) => a + b ), dy = wy.reduce( ( a, b ) => a + b );
		input.set( [ kx, ky, dx, dy ], base );
		input.set( wx, base + 4 );
		input.set( wy, base + 7 );
		for ( let t = 0; t < 36; t ++ ) input[ base + 10 + t ] = value();
		for ( let c = 0; c < 4; c ++ ) {

			const values = [], weights = [];
			for ( let y = 0; y < ky; y ++ ) for ( let x = 0; x < kx; x ++ ) { values.push( input[ base + 10 + ( y * 3 + x ) * 4 + c ] ); weights.push( wx[ x ] * wy[ y ] ); }
			expected[ i * 4 + c ] = roundedMean( values, weights, dx * dy );

		}

	}

	const actual = await runWgsl( device, intMeanWgsl(), `	let base = i * ${ words }u;
	let kx = input[base];
	let ky = input[base + 1u];
	let dx = input[base + 2u];
	let dy = input[base + 3u];
	var m = DsMean();
	for (var y = 0u; y < ky; y++) {
		var hi = vec4u(0u);
		var lo = vec4u(0u);
		for (var x = 0u; x < kx; x++) {
			let o = base + 10u + (y * 3u + x) * 4u;
			let v = vec4u(input[o], input[o + 1u], input[o + 2u], input[o + 3u]);
			let w = input[base + 4u + x];
			hi += w * ds_hi(v);
			lo += w * ds_lo(v);
		}
		m = ds_mean_row(m, hi, lo, dx, input[base + 7u + y]);
	}
	let r = ds_mean(m, dx, dy);
	output[4u * i] = r.x; output[4u * i + 1u] = r.y; output[4u * i + 2u] = r.z; output[4u * i + 3u] = r.w;`, input, cases * 4, cases );
	for ( let i = 0; i < cases * 4; i ++ ) if ( actual[ i ] !== expected[ i ] ) {

		const base = Math.floor( i / 4 ) * words;
		throw new Error( `ds_mean case ${ Math.floor( i / 4 ) } channel ${ i % 4 } (taps ${ input[ base ] }×${ input[ base + 1 ] }, d ${ input[ base + 2 ] }×${ input[ base + 3 ] }): ${ actual[ i ] }, expected ${ expected[ i ] }` );

	}

	// ds_mean4: the 2×2 mean.
	const quads = 50000;
	const quadInput = Uint32Array.from( { length: quads * 4 }, value );
	const quadExpected = Array.from( { length: quads }, ( _, i ) => roundedMean( [ ...quadInput.subarray( i * 4, i * 4 + 4 ) ], [ 1, 1, 1, 1 ], 4 ) );
	const quadActual = await runWgsl( device, intMeanWgsl(), '	output[i] = ds_mean4(vec4u(input[4u * i]), vec4u(input[4u * i + 1u]), vec4u(input[4u * i + 2u]), vec4u(input[4u * i + 3u])).x;', quadInput, quads, quads );
	quadExpected.forEach( ( e, i ) => { if ( quadActual[ i ] !== e ) throw new Error( `ds_mean4(${ [ ...quadInput.subarray( i * 4, i * 4 + 4 ) ] }) = ${ quadActual[ i ] }, expected ${ e }` ); } );
	return [ { mean: 'ds_mean', cases: cases * 4 }, { mean: 'ds_mean4', cases: quads } ];

}

/** Shape × width × format × variant, on the storage path and (subset shapes) the buffer path. */
async function validateIntMatrix( device, { quick, onProgress, select } ) {

	const results = [];
	const cases = quick ? shapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : shapes;
	const subset = subsetShapes.filter( shape => cases.some( c => c.join() === shape.join() ) );
	const set = ( levels, output ) => ( {
		levels, output,
		exact: new Downsampler( device, { levelsPerDispatch: levels, output, tail: false } ),
		tail: new Downsampler( device, { levelsPerDispatch: levels, output, tail: true } )
	} );
	const storageSets = levelOptions( device ).map( levels => set( levels, 'storage' ) );
	const bufferWidths = [ 1, 4, 5, 6 ].filter( levels => ( levels <= 4 || device.limits.maxComputeInvocationsPerWorkgroup >= 256 ) && ! ( quick && levels === 5 ) );
	const bufferSets = bufferWidths.map( levels => set( levels, 'buffer' ) );

	for ( const format of Object.keys( formats ).filter( isIntFormat ) ) {

		const { quantization, feature, coreStorage } = formats[ format ];
		const storagePath = storageWritable( format, device.features ) && ! ( coreStorage && compatibility( device ) );
		// Formats without storage on this device take the buffer path where their texels are whole words.
		const fallback = ! storagePath && bufferWritable( format );
		if ( ! storagePath && ! fallback ) continue;
		const family = intFamilyFormats.includes( format );
		const storageShapes = storagePath ? ( family ? cases : subset ) : [];
		const bufferShapes = fallback || ( family && ! quick && bufferWritable( format ) ) ? subset : [];
		const formatCases = storageShapes === cases ? cases : subset;

		for ( const [ shapeIndex, shape ] of formatCases.entries() ) {

			const [ width, height, layers = 1, dimension ] = shape;
			const onShape = list => list.some( s => s.join() === shape.join() );
			const sets = [ ...( onShape( storageShapes ) ? storageSets : [] ), ...( onShape( bufferShapes ) ? bufferSets : [] ) ];
			const selected = sets.flatMap( s => intVariants.map( variant => [ s, variant, `integer/${ format }/${ s.output }/L${ s.levels }/${ slug( variant.name ) }` ] ) ).filter( ( [ , , id ] ) => select.has( id ) );
			if ( selected.length === 0 ) continue;
			const texture = createTexture( device, { width, height, layers, format, dimension } );
			const inputs = new Map(), firstLevels = new Map();

			try {

				for ( const [ s, variant, id ] of selected ) results.push( ...await select.run( id, async () => {

					const downsampler = variant.tail ? s.tail : s.exact;
					const options = { reduction: reductionOf( variant, format ), oddSizes: variant.oddSizes };
					const plan = downsampler.prepare( texture, options );
					if ( variant.tail && ( ! plan.some( batch => batch.kernel === 'tail' ) || ( quick && plan.some( batch => batch.tile > 16 ) ) ) ) return [];
					const weights = variant.oddSizes === 'box' ? 'area' : 'bilinear';
					for ( const pattern of quick ? variant.patterns.filter( p => p !== 'tiles' && p !== 'constant' ) : variant.patterns ) {

						if ( ! inputs.has( pattern ) ) {

							const data = makeData( width, height, layers, format, pattern, pattern === 'tiles' ? shapeIndex + 3 : 1 );
							inputs.set( pattern, toFloats( data, format ) );
							inputs.set( `${ pattern }:data`, data );

						}

						upload( device, texture, inputs.get( `${ pattern }:data` ) );
						poison( device, texture );
						downsampler.generate( texture, options );
						const mips = await readMips( device, texture, 1 );
						const text = `${ width }x${ height }x${ layers } ${ format } ${ s.output } ${ variant.name } ${ pattern } L${ s.levels }`;
						checkChain( mips, inputs.get( pattern ), width, height, layers, format, referenceOf( variant, format ), weights, { text, key: `${ weights }:${ variant.custom ? 'custom' : variant.reduction }:${ pattern }` }, firstLevels );

					}

					return [ { width, height, layers, format, output: s.output, levelsPerDispatch: s.levels, reduction: variant.name, dispatches: plan.length } ];

				} ) );

				onProgress( `validated integer ${ width }×${ height }×${ layers } ${ format }` );

			} finally {

				texture.destroy();

			}

		}

	}

	return results;

}

// [ source, destination, relation, reductions ]: the same format halved, a wider source clamped
// into a narrower destination (a copy), a signed one halved into fewer bits, and resampled sources.
export const intSourceCases = [
	{ source: 'r32uint', destination: 'r32uint', relation: 'half', reductions: [ 'average', 'max' ] },
	{ source: 'rgba16uint', destination: 'rgba8uint', relation: 'same', reductions: [ 'average', 'max' ] },
	{ source: 'rgba32sint', destination: 'rgba16sint', relation: 'half', reductions: [ 'average', 'min' ] },
	{ source: 'rgba32uint', destination: 'rgb10a2uint', relation: 'half', reductions: [ 'average', 'max' ] },
	{ source: 'r32sint', destination: 'r32sint', relation: [ 512, 256 ], reductions: [ 'average', 'minmax' ] },
	{ source: 'rgba8uint', destination: 'rgba16uint', relation: [ 256, 128 ], reductions: [ 'average', 'max' ] }
];

/** Separate integer sources: copy, half and area-resampled, every level against the oracle. */
async function validateIntSources( device, { quick } ) {

	const results = [];
	const sizes = quick ? [ [ 33, 17 ], [ 64, 32, 3 ] ] : [ [ 33, 17 ], [ 64, 32, 3 ], [ 1129, 635 ], [ 1, 9 ] ];
	const downsamplers = [ new Downsampler( device, { tail: false } ), new Downsampler( device ) ];
	for ( const { source: sourceFormat, destination: format, relation, reductions } of intSourceCases ) {

		if ( ! storageWritable( format, device.features ) && ! bufferWritable( format ) ) continue;
		if ( formats[ format ].feature && ! device.features.has( formats[ format ].feature ) && ! bufferWritable( format ) ) continue;
		for ( const [ width, height, layers = 1 ] of sizes ) {

			const resample = Array.isArray( relation );
			if ( resample && layers > 1 ) continue;
			const [ dw, dh ] = resample ? relation : relation === 'same' ? [ width, height ] : [ Math.max( 1, width >> 1 ), Math.max( 1, height >> 1 ) ];
			const destination = createTexture( device, { width: dw, height: dh, layers, format } );
			const sourceTexture = createTexture( device, { width, height, layers, format: sourceFormat, storage: false } );
			try {

				for ( const pattern of [ 'random', 'small', 'edge' ] ) {

					const data = makeData( width, height, layers, sourceFormat, pattern, 5 );
					upload( device, sourceTexture, data );
					const floats = toFloats( data, sourceFormat );
					const input = sampledChannels( sourceFormat ) === 1 ? splat( floats ) : floats;
					for ( const reduction of reductions ) for ( const [ d, downsampler ] of downsamplers.entries() ) for ( const oddSizes of [ 'exact', 'box' ] ) {

						poison( device, destination, 0 );
						const options = { reduction, oddSizes, source: { texture: sourceTexture, ...( resample ? { resample: 'area' } : {} ) } };
						downsampler.generate( destination, options );
						const mips = await readMips( device, destination, 0 );
						const label = `source ${ sourceFormat } ${ width }x${ height }x${ layers } -> ${ format } ${ dw }x${ dh } ${ reduction } ${ oddSizes } ${ pattern } ${ d ? 'tail' : 'exact' }`;
						const first = intReference( input, width, height, layers, formats[ format ].quantization, reduction, { weights: resample || oddSizes === 'box' ? 'area' : 'bilinear', width: dw, height: dh, source: formats[ sourceFormat ].quantization } );
						compare( mips[ 0 ], first, format, `${ label } mip 0` );
						checkChain( mips.slice( 1 ), mips[ 0 ], dw, dh, layers, format, reduction, oddSizes === 'box' ? 'area' : 'bilinear', { text: label } );

					}

				}

				results.push( { source: sourceFormat, destination: format, width, height, layers, dw, dh } );

			} finally {

				destination.destroy();
				sourceTexture.destroy();

			}

		}

	}

	return results;

}

async function generateBits( device, downsampler, texture, data, options, { pass = false } = {} ) {

	upload( device, texture, data );
	poison( device, texture );
	if ( pass ) {

		const encoder = device.createCommandEncoder();
		const computePass = encoder.beginComputePass();
		downsampler.encodePass( computePass, texture, options );
		computePass.end();
		device.queue.submit( [ encoder.finish() ] );

	} else downsampler.generate( texture, options );
	return ( await readMips( device, texture, 1 ) ).map( level => [ ...level ] );

}

function sameBits( a, b, label ) {

	a.forEach( ( level, i ) => level.forEach( ( value, j ) => {

		if ( value !== b[ i ][ j ] ) throw new Error( `${ label }: mip ${ i + 1 } value ${ j }: ${ value } vs ${ b[ i ][ j ] }` );

	} ) );

}

/** Paths that must give the same bits, partial updates, and rejections. */
async function validateIntApi( device ) {

	const results = [];
	const direct = new Downsampler( device );
	// encodePass, the buffer path and the compatibility path forced: bit-identical to encode.
	for ( const [ format, reduction ] of [ [ 'rgba32sint', 'average' ], [ 'r32uint', 'max' ], [ 'rgba8sint', 'average' ], [ 'rgba16uint', { wgsl: customFolds.uint.wgsl } ] ] ) {

		const texture = createTexture( device, { width: 133, height: 75, format } );
		const data = makeData( 133, 75, 1, format, 'random', 9 );
		const options = { reduction };
		const reference = await generateBits( device, direct, texture, data, options );
		sameBits( await generateBits( device, direct, texture, data, options, { pass: true } ), reference, `${ format } encodePass` );
		if ( bufferWritable( format ) ) sameBits( await generateBits( device, new Downsampler( device, { output: 'buffer' } ), texture, data, options ), reference, `${ format } buffer path` );
		sameBits( await generateBits( device, new Downsampler( device, { compatibility: true } ), texture, data, options ), reference, `${ format } compatibility path` );
		sameBits( await generateBits( device, new Downsampler( device, { tail: false } ), texture, data, options ), reference, `${ format } tail off` );
		// 'box' is 'exact' at powers of two (an even step is the 2×2 mean either way).
		const even = createTexture( device, { width: 128, height: 64, format } );
		const evenData = makeData( 128, 64, 1, format, 'random', 4 );
		sameBits( await generateBits( device, direct, even, evenData, { reduction, oddSizes: 'box' } ), await generateBits( device, direct, even, evenData, options ), `${ format } box at a power of two` );
		even.destroy();

		// A partial update: change a rect of level 0, update it, compare with a full recompute.
		const changed = makeData( 133, 75, 1, format, 'small', 11 );
		const rect = { x: 40, y: 9, width: 23, height: 31 };
		const words = formats[ format ].channels;
		const patched = data.slice();
		for ( let y = rect.y; y < rect.y + rect.height; y ++ ) for ( let x = rect.x; x < rect.x + rect.width; x ++ ) for ( let c = 0; c < words; c ++ ) patched[ ( y * 133 + x ) * words + c ] = changed[ ( y * 133 + x ) * words + c ];
		upload( device, texture, data );
		direct.generate( texture, options );
		upload( device, texture, patched );
		direct.generate( texture, { ...options, rect } );
		const updated = ( await readMips( device, texture, 1 ) ).map( level => [ ...level ] );
		sameBits( updated, await generateBits( device, direct, texture, patched, options ), `${ format } rect update` );
		texture.destroy();
		results.push( { api: format, reduction: typeof reduction === 'string' ? reduction : 'custom' } );

	}

	// Rejections: 'spd', mixed signedness, float sources, integer sources of float destinations, Reducer.
	const texture = createTexture( device, { width: 16, height: 16, format: 'r32uint' } );
	expectThrow( () => direct.prepare( texture, { oddSizes: 'spd' } ), 'oddSizes \'spd\' isn\'t available for integer formats', 'spd on r32uint' );
	const sint = createTexture( device, { width: 16, height: 16, format: 'r32sint', storage: false } );
	const float = createTexture( device, { width: 16, height: 16, format: 'r32float', storage: false } );
	const floatDestination = createTexture( device, { width: 16, height: 16, format: 'r32float' } );
	if ( ! direct.sourceUnsupportedReason( texture, { texture: sint } )?.includes( 'must be a uint integer format too, not r32sint' ) ) throw new Error( `r32sint source of r32uint: ${ direct.sourceUnsupportedReason( texture, { texture: sint } ) }` );
	if ( ! direct.sourceUnsupportedReason( texture, { texture: float } )?.includes( 'not r32float' ) ) throw new Error( 'r32float source of r32uint accepted' );
	if ( ! direct.sourceUnsupportedReason( floatDestination, { texture: sint } )?.includes( 'not float-sampleable' ) ) throw new Error( 'r32sint source of r32float accepted' );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], format: 'r32uint' } ), 'doesn\'t write integer formats', 'Reducer r32uint output' );
	const reducer = new Reducer( device, { levels: [ 1 ], format: 'r32float' } );
	expectThrow( () => reducer.prepare( { source: { texture: sint, mipLevel: 0 }, outputs: [ { texture: floatDestination, mipLevel: 1 } ] } ), 'reads float-sampleable sources, not r32sint', 'Reducer r32sint source' );
	if ( ! new ScalarReduction( device ).unsupportedReason( { source: sint } )?.includes( 'not float-sampleable' ) ) throw new Error( 'ScalarReduction accepted r32sint' );
	// unsupportedReason names tier1 for the 1- and 2-byte formats where the device lacks it.
	if ( ! device.features.has( 'texture-formats-tier1' ) ) {

		const r8 = device.createTexture( { size: [ 4, 4 ], format: 'r8uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		if ( ! direct.unsupportedReason( r8 )?.includes( 'texture-formats-tier1' ) ) throw new Error( `r8uint without tier1: ${ direct.unsupportedReason( r8 ) }` );
		r8.destroy();

	}

	for ( const t of [ texture, sint, float, floatDestination ] ) t.destroy();
	results.push( { api: 'rejections' } );
	return results;

}
