// GPU correctness suite. Every texel of every generated level is checked against
// the CPU reference, computed from the GPU's own previous level so permitted
// rounding differences do not accumulate along the chain.
import { copyReference, formats, fromHalf, isIntegerDomain, levelSize, pyramidTolerance, quantize, reducePyramid, reduceReference, sampledChannels, toHalf, tolerance } from '/dist/index.js';
import { Downsampler, Reducer } from '/dist/webgpu/index.js';
import { createDepthSource, createTexture, makeData, poison, readMip, readMips, toFloats, upload } from './gpu.js';
import { validateScalar } from './validate-scalar.js';
import { validatePayload } from './validate-payload.js';
import { validateWide } from './validate-wide.js';

// [ width, height, layers?, viewDimension? ]: 1×1, 1D tails, odd sizes, partial
// workgroups, rectangles, layers, both 2D view variants, and multi-batch sizes.
export const shapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 7, 4 ], [ 15, 17 ], [ 16, 16 ], [ 17, 9 ], [ 31, 33 ], [ 64, 64 ], [ 65, 129 ],
	[ 128, 8 ], [ 1, 128 ], [ 128, 1 ], [ 192, 108 ], [ 255, 257 ], [ 256, 256 ], [ 512, 32 ],
	[ 64, 32, 3 ], [ 17, 11, 3 ], [ 32, 16, 1, '2d-array' ], [ 17, 9, 1, '2d-array' ],
	[ 2048, 2048 ], [ 96, 4096 ], [ 1920, 1080 ]
];

// One 4-channel format per quantization family runs every shape. The others in
// a family differ only in channel count or storage format string, so they run a
// representative subset: unit and odd sizes, 1D tails, partial tiles, layers,
// both view dimensions and a multi-batch shape (512×32 is multi-batch at every width).
export const familyFormats = [ 'rgba8unorm', 'rgba8snorm', 'rgba16unorm', 'rgba16snorm', 'rgba16float', 'rgba32float' ];
export const subsetShapes = [
	[ 1, 1 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 65, 129 ], [ 1, 128 ], [ 128, 8 ], [ 192, 108 ], [ 512, 32 ],
	[ 17, 11, 3 ], [ 17, 9, 1, '2d-array' ]
];

const patterns = [ 'constant', 'random', 'edge', 'tiles' ];

// Custom fold used to test the hook: per channel, the smallest value >= 0.5, else
// the smallest value. A min under a total order, so associative, commutative and
// idempotent, and sensitive to the value domain (it breaks if a kernel passes
// 0–255 units instead of normalized values).
const preferHigh = {
	wgsl: `fn reduce(a: vec4f, b: vec4f) -> vec4f {
	let ha = a >= vec4f(0.5);
	let hb = b >= vec4f(0.5);
	return select(b, a, (ha & !hb) | ((ha == hb) & (a <= b)));
}`,
	reference: ( a, b ) => ( a >= 0.5 ) !== ( b >= 0.5 ) ? ( a >= 0.5 ? a : b ) : Math.min( a, b )
};

// Each reduction runs the full shape × width × format matrix. 'constant' says
// nothing about a fold, so folds skip it.
export const variants = [
	{ name: 'average', reduction: 'average', patterns },
	// Unfilterable formats always load, so this is only distinct for filterable ones.
	{ name: 'average (load)', reduction: 'average', access: 'load', patterns, filterableOnly: true },
	{ name: 'min', reduction: 'min', patterns: patterns.slice( 1 ) },
	{ name: 'max', reduction: 'max', patterns: patterns.slice( 1 ) },
	{ name: 'minmax', reduction: 'minmax', patterns: patterns.slice( 1 ) },
	{ name: 'custom', reduction: { wgsl: preferHigh.wgsl }, reference: preferHigh.reference, patterns: patterns.slice( 1 ) }
];

/**
 * Compares the stored channels of every texel; the rest are not written.
 * `slack` is a relative allowance for levels read directly from a unorm source:
 * unorm → float conversion needn't be correctly rounded (SwiftShader's
 * depth16unorm differs from k / 65535 by an ulp), so even exact folds can see it.
 */
function check( actual, expected, quantization, label, reduction = 'average', channels = 4, slack = 0, exact = false ) {

	let maxError = 0;

	for ( let i = 0; i < expected.values.length; i ++ ) {

		if ( i % 4 >= channels ) continue;
		const error = Math.abs( actual[ i ] - expected.values[ i ] );
		const limit = exact ? 0 : tolerance( expected.values[ i ], quantization, expected.filterTolerance[ i ], reduction ) + Math.abs( expected.values[ i ] ) * slack;
		if ( ! Number.isFinite( actual[ i ] ) || error > limit ) throw new Error( `${ label }: value ${ i }, actual ${ actual[ i ] }, expected ${ expected.values[ i ] }, error ${ error } > ${ limit }` );
		maxError = Math.max( maxError, error );

	}

	return maxError;

}

/** Kernel widths to cover: single-level, 4 (default limit), 5 (16×16 tiles) and 6 (SPD tiles). */
export function levelOptions( device ) {

	const limit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	return [ 1, 4, 5, 6 ].filter( levels => levels <= limit );

}

/** `only`: run one part of the suite ('mips': all but scalars, 'reducer': Reducer cases only, 'scalar'); default all. */
export async function validate( device, { onProgress = () => {}, quick = false, only } = {} ) {

	const results = [];
	const started = performance.now();
	const cases = only === 'scalar' || only === 'reducer' ? [] : quick ? shapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : shapes;
	let scalarPrecision = null;
	let runs = 0;
	const skipped = [];
	device.pushErrorScope( 'validation' );

	try {

		const downsamplers = levelOptions( device ).map( levels => ( {
			levels,
			sample: new Downsampler( device, { levelsPerDispatch: levels } ),
			load: new Downsampler( device, { levelsPerDispatch: levels, averageAccess: 'load' } )
		} ) );

		for ( const format of Object.keys( formats ) ) {

			const { quantization, channels, filterable, feature } = formats[ format ];
			if ( feature !== undefined && ! device.features.has( feature ) ) {

				skipped.push( `${ format } (needs ${ feature })` );
				continue;

			}

			const formatVariants = variants.filter( variant => filterable || ! variant.filterableOnly );

			const formatCases = familyFormats.includes( format ) ? cases : subsetShapes.filter( shape => cases.some( c => c.join() === shape.join() ) );

			for ( const [ shapeIndex, [ width, height, layers = 1, dimension ] ] of formatCases.entries() ) {

				const texture = createTexture( device, { width, height, layers, format, dimension } );
				// Inputs and first-level references depend only on shape, format,
				// pattern and reduction, so they are shared across kernel widths.
				const inputs = new Map(), firstLevels = new Map();

				try {

					for ( const { levels, ...byAccess } of downsamplers ) for ( const variant of formatVariants ) {

						const downsampler = byAccess[ variant.access ?? 'sample' ];
						const options = { reduction: variant.reduction };
						const reference = variant.reference ?? variant.reduction;
						const result = { width, height, layers, format, levelsPerDispatch: levels, reduction: variant.name, dispatches: 0, maxError: 0 };

						// Reusing one texture with changing input exercises the cached plans.
						for ( const pattern of variant.patterns ) {

							if ( ! inputs.has( pattern ) ) {

								const data = makeData( width, height, layers, format, pattern, pattern === 'tiles' ? shapeIndex + 3 : 1 );
								inputs.set( pattern, { data, floats: toFloats( data, format ) } );

							}

							const input = inputs.get( pattern );
							upload( device, texture, input.data );
							poison( device, texture );
							downsampler.generate( texture, options );
							const mips = await readMips( device, texture, 1 );
							// Levels reduced in workgroup memory use exact integer math in
							// normalized formats: they must match the oracle bit for bit,
							// which is what tests tie rounding. Only a batch's first output
							// (sampler or float weights) gets the tolerance.
							const firstOutputs = new Set( downsampler.prepare( texture, options ).map( batch => batch.firstMipLevel ) );
							let previous = input.floats, w = width, h = height;

							for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

								const key = `${ variant.name }:${ pattern }`;
								let expected = level === 1 ? firstLevels.get( key ) : undefined;
								if ( expected === undefined ) expected = reduceReference( previous, w, h, layers, quantization, reference );
								if ( level === 1 ) firstLevels.set( key, expected );
								const actual = mips[ level - 1 ];
								result.maxError = Math.max( result.maxError, check( actual, expected, quantization, `${ width }x${ height }x${ layers } ${ format } ${ variant.name } ${ pattern } L${ levels } mip ${ level }`, reference, channels, 0, isIntegerDomain( quantization ) && ! firstOutputs.has( level ) ) );
								previous = actual; w = expected.width; h = expected.height;

							}

							runs ++;

						}

						result.dispatches = downsampler.prepare( texture, options ).length;
						results.push( result );

					}

					onProgress( `validated ${ width }×${ height }×${ layers } ${ format }` );

				} finally {

					texture.destroy();

				}

			}

		}

		if ( only !== 'scalar' ) {

			if ( only !== 'reducer' ) {

				results.push( ...await validateSource( device, { quick, onProgress } ) );
				results.push( ...await validateOptions( device ) );

			}

			const reducer = await validateReducer( device, { quick, onProgress } );
			results.push( ...reducer );
			runs += reducer.reduce( ( sum, result ) => sum + ( result.runs ?? 0 ), 0 );
			results.push( ...await validateTools( device, { quick, onProgress } ) );
			results.push( ...await validateReducerOptions( device ) );
			results.push( ...await validateWide( device, { quick, onProgress } ) );
			results.push( ...await validatePayload( device, { quick, onProgress } ) );

		}

		if ( only !== 'mips' && only !== 'reducer' ) {

			const scalar = await validateScalar( device, { quick, onProgress } );
			results.push( ...scalar.results );
			scalarPrecision = { worst: scalar.worst, pairing: scalar.pairing };

		}

		await device.queue.onSubmittedWorkDone();

	} finally {

		const error = await device.popErrorScope();
		if ( error ) throw new Error( `GPU validation error: ${ error.message }` );

	}

	// scalarPrecision: the largest float-sum error seen, as a fraction of the oracle's
	// bound, and how many results matched the oracle's float32 pairing bit for bit.
	return { passed: true, cases: results.length, checks: runs, skipped, scalarPrecision, ms: Math.round( performance.now() - started ), results };

}

// Separate source → destination. 'same' copies the source into level 0, 'half'
// reduces it once; either way every destination level is written and checked.
export const sourceCases = [
	{ source: 'depth32float', destination: 'r32float', relation: 'same', reductions: [ 'min', 'max' ] },
	{ source: 'depth32float', destination: 'rg32float', relation: 'same', reductions: [ 'minmax' ] },
	{ source: 'depth32float', destination: 'rg32float', relation: 'half', reductions: [ 'minmax', 'average' ] },
	{ source: 'depth16unorm', destination: 'r32float', relation: 'half', reductions: [ 'max', 'average' ] },
	{ source: 'r32float', destination: 'rg32float', relation: 'same', reductions: [ 'minmax' ] },
	{ source: 'rgba8unorm', destination: 'rgba16float', relation: 'half', reductions: [ 'average', 'max' ] },
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'same', reductions: [ 'average' ] },
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'half', reductions: [ 'average', 'min' ] }
];

// After the source batch, every batch is the in-place code the main matrix covers
// at 2048², so 512² (multi-batch at every width) stands in for it here.
export const sourceShapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 17, 9 ], [ 64, 64 ], [ 65, 129 ], [ 1, 128 ], [ 255, 257 ], [ 192, 108 ],
	[ 64, 32, 3 ], [ 17, 11, 3 ], [ 512, 512 ], [ 1920, 1080 ]
];

async function makeSource( device, format, width, height, layers, pattern, seed ) {

	if ( format.startsWith( 'depth' ) ) {

		const values = makeData( width, height, layers, 'r32float', pattern, seed, { unit: true } );
		return createDepthSource( device, { width, height, layers, format }, values );

	}

	const texture = createTexture( device, { width, height, layers, format, storage: false } );
	const data = makeData( width, height, layers, format, pattern, seed );
	upload( device, texture, data );
	return { texture, floats: toFloats( data, format ) };

}

/** Single-channel sources are splatted by the kernel; the oracle sees the same. */
function splat( floats ) {

	const output = new Float32Array( floats.length );
	for ( let i = 0; i < floats.length; i += 4 ) output.fill( floats[ i ], i, i + 4 );
	return output;

}

async function validateSource( device, { quick, onProgress } ) {

	const results = [];
	const shapes = quick ? sourceShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : sourceShapes;
	const downsamplers = levelOptions( device ).map( levels => new Downsampler( device, { levelsPerDispatch: levels } ) );

	for ( const { source: sourceFormat, destination: format, relation, reductions } of sourceCases ) {

		const { quantization, channels } = formats[ format ];

		for ( const [ shapeIndex, [ width, height, layers = 1 ] ] of shapes.entries() ) {

			const dw = relation === 'same' ? width : Math.max( 1, width >> 1 ), dh = relation === 'same' ? height : Math.max( 1, height >> 1 );
			const destination = createTexture( device, { width: dw, height: dh, layers, format } );

			try {

				for ( const pattern of [ 'random', 'edge', 'tiles' ] ) {

					const source = await makeSource( device, sourceFormat, width, height, layers, pattern, pattern === 'tiles' ? shapeIndex + 3 : 1 );
					const input = sampledChannels( sourceFormat ) === 1 ? splat( source.floats ) : source.floats;

					try {

						for ( const downsampler of downsamplers ) for ( const reduction of reductions ) {

							poison( device, destination, 0 );
							downsampler.generate( destination, { source: { texture: source.texture }, reduction } );
							const mips = await readMips( device, destination, 0 );
							const label = `${ sourceFormat } ${ width }x${ height }x${ layers } -> ${ format } ${ relation } ${ reduction } ${ pattern } L${ downsampler.levelsPerDispatch }`;
							let previous = input, w = width, h = height;

							for ( let level = 0; level < destination.mipLevelCount; level ++ ) {

								const quantizedInput = level > 0 || formats[ sourceFormat ]?.quantization === quantization;
								const expected = level === 0 && dw === width && dh === height ? copyReference( previous, w, h, layers, quantization ) : reduceReference( previous, w, h, layers, quantization, reduction, { quantizedInput } );
								const slack = level === 0 && /unorm/.test( sourceFormat ) ? 2 ** - 21 : 0;
								check( mips[ level ], expected, quantization, `${ label } mip ${ level }`, reduction, channels, slack );
								previous = mips[ level ]; w = expected.width; h = expected.height;

							}

						}

					} finally {

						source.texture.destroy();

					}

				}

				results.push( { source: sourceFormat, destination: format, relation, width, height, layers, reductions } );
				onProgress( `validated source ${ sourceFormat } ${ width }×${ height }×${ layers } -> ${ format } (${ relation })` );

			} finally {

				destination.destroy();

			}

		}

	}

	return results;

}

async function validateOptions( device ) {

	const results = [];
	const downsampler = new Downsampler( device );

	// Sub-range: only levels after baseMipLevel are written, from that level.
	const texture = createTexture( device, { width: 256, height: 256, format: 'rgba8unorm' } );
	const input = makeData( 256, 256, 1, 'rgba8unorm', 'random' );
	upload( device, texture, input );
	downsampler.generate( texture );
	const kept = await readMip( device, texture, 2 );
	poison( device, texture, 3 );
	const untouched = await readMip( device, texture, 6 );
	downsampler.generate( texture, { baseMipLevel: 2, mipLevelCount: 4 } );
	if ( ( await readMip( device, texture, 2 ) ).some( ( value, i ) => value !== kept[ i ] ) ) throw new Error( 'sub-range modified its source level' );
	let previous = kept;
	for ( let level = 3; level <= 5; level ++ ) {

		const expected = reduceReference( previous, 256 >> ( level - 1 ), 256 >> ( level - 1 ), 1, 'unorm8' );
		previous = await readMip( device, texture, level );
		check( previous, expected, 'unorm8', `sub-range mip ${ level }` );

	}

	if ( ( await readMip( device, texture, 6 ) ).some( ( value, i ) => value !== untouched[ i ] ) ) throw new Error( 'sub-range wrote past mipLevelCount' );
	texture.destroy();
	results.push( { option: 'baseMipLevel/mipLevelCount' } );

	// Unsupported textures are reported, not silently processed.
	const unsupported = [
		device.createTexture( { size: [ 16, 16 ], format: 'rgba8unorm', mipLevelCount: 5, usage: GPUTextureUsage.TEXTURE_BINDING } ),
		device.createTexture( { size: [ 16, 16 ], format: 'rgba8unorm-srgb', mipLevelCount: 5, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } )
	];
	for ( const texture of unsupported ) {

		if ( downsampler.supports( texture ) ) throw new Error( `expected ${ texture.format } without storage to be unsupported` );
		let threw = false;
		try { downsampler.generate( texture ); } catch { threw = true; }
		if ( ! threw ) throw new Error( 'generate() accepted an unsupported texture' );
		texture.destroy();

	}

	// A format whose storage needs a feature the device lacks is reported by name.
	const plain = await ( await navigator.gpu.requestAdapter() ).requestDevice();
	const r8 = plain.createTexture( { size: [ 4, 4 ], format: 'r8unorm', mipLevelCount: 3, usage: GPUTextureUsage.TEXTURE_BINDING } );
	const featureReason = new Downsampler( plain ).unsupportedReason( r8 );
	plain.destroy();
	if ( ! featureReason?.includes( 'texture-formats-tier1' ) ) throw new Error( `expected a missing-feature reason for r8unorm, got ${ featureReason }` );
	results.push( { option: 'unsupported textures' } );

	// Sources that can't feed a destination are rejected with a reason.
	const destination = createTexture( device, { width: 32, height: 32, format: 'r32float' } );
	const layered = createTexture( device, { width: 32, height: 32, layers: 2, format: 'r32float' } );
	const renderOnly = device.createTexture( { size: [ 32, 32 ], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT } );
	const rejected = [
		[ { texture: destination }, 'source is the destination' ],
		[ { texture: createTexture( device, { width: 30, height: 32, format: 'r32float' } ) }, 'must be the source level' ],
		[ { texture: layered }, 'layers' ],
		[ { texture: renderOnly }, 'TEXTURE_BINDING' ],
		[ { texture: layered, mipLevel: 9 }, 'mipLevel' ]
	];
	for ( const [ source, message ] of rejected ) {

		const reason = downsampler.sourceUnsupportedReason( destination, source );
		if ( reason === null || ! reason.includes( message ) ) throw new Error( `expected source rejection "${ message }", got ${ reason }` );
		let threw = false;
		try { downsampler.generate( destination, { source } ); } catch { threw = true; }
		if ( ! threw ) throw new Error( `generate() accepted a source that should fail with "${ message }"` );

	}

	// A non-zero source level, and a new source texture with the same options
	// (a depth buffer recreated on resize) replaces the cached plan.
	const mipped = createTexture( device, { width: 64, height: 64, format: 'r32float' } );
	const mippedData = makeData( 64, 64, 1, 'r32float', 'random', 5 );
	upload( device, mipped, mippedData );
	downsampler.generate( mipped );
	const level1 = await readMip( device, mipped, 1 );
	downsampler.generate( destination, { source: { texture: mipped, mipLevel: 1 }, reduction: 'max' } );
	check( await readMip( device, destination, 0 ), copyReference( level1, 32, 32, 1, 'float32' ), 'float32', 'source mipLevel 1', 'max', 1 );

	for ( const seed of [ 7, 8 ] ) {

		const depth = await createDepthSource( device, { width: 64, height: 64 }, makeData( 64, 64, 1, 'r32float', 'random', seed, { unit: true } ) );
		downsampler.generate( destination, { source: { texture: depth.texture }, reduction: 'max' } );
		check( await readMip( device, destination, 0 ), reduceReference( splat( depth.floats ), 64, 64, 1, 'float32', 'max' ), 'float32', `replaced source ${ seed }`, 'max', 1 );
		depth.texture.destroy();

	}

	for ( const texture of [ destination, layered, renderOnly, mipped, ...rejected.map( ( [ source ] ) => source.texture ) ] ) texture.destroy();
	results.push( { option: 'source validation, mipLevel and replacement' } );

	// encodePass: every batch in one caller-owned pass (odd sizes: several batches)
	// writes what encode() writes, in place and from a depth source, with layers.
	for ( const [ width, height, layers, format, options, label ] of [
		[ 333, 200, 3, 'rgba16float', {}, 'average, 3 layers' ],
		[ 1917, 1079, 1, 'r32float', { reduction: 'min' }, 'depth min' ],
	] ) {

		const a = createTexture( device, { width, height, layers, format } ), b = createTexture( device, { width, height, layers, format } );
		const depth = format === 'r32float' ? await createDepthSource( device, { width, height }, makeData( width, height, 1, 'r32float', 'random', 9, { unit: true } ) ) : null;
		const data = makeData( width, height, layers, format, 'random', 11 );
		if ( depth === null ) { upload( device, a, data ); upload( device, b, data ); }
		const run = { ...options, ...( depth ? { source: { texture: depth.texture } } : {} ) };
		downsampler.generate( a, run );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		downsampler.encodePass( pass, b, run );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );
		if ( downsampler.prepare( b, run ).length < 2 ) throw new Error( `encodePass ${ label }: expected several batches` );
		const ma = await readMips( device, a, depth ? 0 : 1 ), mb = await readMips( device, b, depth ? 0 : 1 );
		ma.forEach( ( level, l ) => { if ( level.some( ( value, i ) => ! Object.is( value, mb[ l ][ i ] ) ) ) throw new Error( `encodePass ${ label }: level ${ l } differs from encode()` ); } );
		for ( const t of [ a, b, depth?.texture ] ) t?.destroy();

	}

	results.push( { option: 'encodePass matches encode' } );
	return results;

}

// ---------------------------------------------------------------------------
// Reducer: tile reductions (SPEC §14.1). Every emitted level is checked against
// `reducePyramid`, which computes it straight from the source in f64.

export const reducerShapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 7, 4 ], [ 17, 9 ], [ 31, 33 ], [ 64, 64 ], [ 65, 129 ], [ 1, 128 ], [ 128, 1 ], [ 128, 8 ],
	[ 192, 108 ], [ 255, 257 ], [ 17, 11, 3 ], [ 64, 32, 3 ], [ 32, 16, 1, '2d-array' ], [ 1920, 1080 ]
];

// Each kernel width reduces levels 1…count; the two paths emit complementary
// subsets so every level of every width is checked, and every path leaves some
// level unemitted (its atlas region must stay poisoned, or it gets no binding).
const textureLevels = { 1: [ 1 ], 4: [ 1, 2, 4 ], 5: [ 2, 5 ], 6: [ 1, 3, 4, 6 ] };
const hookLevels = { 1: [ 1 ], 4: [ 1, 3, 4 ], 5: [ 1, 3, 4, 5 ], 6: [ 1, 2, 3, 5, 6 ] };

// A custom sum: a + b, plus 1 per combine on y and w, so y and w end at Σv + n − 1
// for the n texels counted. Skipped or duplicated texels change n.
const countingSum = 'fn reduce(a: vec4f, b: vec4f) -> vec4f { return a + b + vec4f(0.0, 1.0, 0.0, 1.0); }';

export const reducerVariants = [
	{ name: 'sum/average/min/max', reduction: [ 'sum', 'average', 'min', 'max' ], patterns: [ 'random', 'edge' ] },
	{ name: 'weighted', reduction: [ 'weighted', 'weighted', 'weighted', 'weight' ], patterns: [ 'weights' ] },
	{ name: 'custom fold', reduction: { wgsl: preferHigh.wgsl }, reference: { reference: preferHigh.reference }, patterns: [ 'random' ] },
	{ name: 'custom sum', reduction: { wgsl: countingSum, kind: 'sum' }, reference: { reference: ( a, b, c ) => a + b + ( c % 2 ), kind: 'sum' }, patterns: [ 'random' ] }
];

/**
 * RGBA f32 fixtures (exactly representable): 'random' in [−4, 12), 'edge' non-zero
 * only on the last row and column, 'weights' rgb in [0, 4) with a weight in
 * alpha that is 0 for about a third of the texels (sky).
 */
function reducerData( width, height, layers, pattern, seed = 1 ) {

	const data = new Float32Array( width * height * layers * 4 );
	let state = seed;
	const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) >>> 8;
	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let c = 0; c < 4; c ++ ) {

		let value = next() / 2 ** 20 - 4;
		if ( pattern === 'edge' ) value = x === width - 1 || y === height - 1 ? ( c + layer + 1 ) / 8 : 0;
		if ( pattern === 'weights' ) value = c < 3 ? next() / 2 ** 22 : ( next() % 3 === 0 ? 0 : next() / 2 ** 24 );
		data[ ( ( layer * height + y ) * width + x ) * 4 + c ] = value;

	}

	return data;

}

function checkPyramid( actual, expected, label, { slack = 0, quantization = null } = {} ) {

	let maxError = 0;
	for ( let i = 0; i < expected.values.length; i ++ ) {

		const reference = quantization ? quantize( expected.values[ i ], quantization ) : expected.values[ i ];
		const limit = pyramidTolerance( expected.magnitude[ i ] ) + slack + ( quantization ? tolerance( reference, quantization ) : 0 );
		const error = Math.abs( actual[ i ] - reference );
		if ( ! Number.isFinite( actual[ i ] ) || ! ( error <= limit ) ) throw new Error( `${ label }: value ${ i } (texel ${ i >> 2 } channel ${ i & 3 }), actual ${ actual[ i ] }, expected ${ reference }, error ${ error } > ${ limit }` );
		maxError = Math.max( maxError, error );

	}

	return maxError;

}

async function readBuffer( device, buffer, size = buffer.size ) {

	const staging = device.createBuffer( { size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer( buffer, 0, staging, 0, size );
	device.queue.submit( [ encoder.finish() ] );
	await staging.mapAsync( GPUMapMode.READ );
	const values = new Float32Array( staging.getMappedRange().slice( 0 ) );
	staging.unmap();
	staging.destroy();
	return values;

}

function storageBuffer( device, data, usage = 0 ) {

	const buffer = device.createBuffer( { size: Math.max( 16, data.byteLength ), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | usage } );
	device.queue.writeBuffer( buffer, 0, data );
	return buffer;

}

const poisonValue = - 12345.5;

/**
 * An atlas buffer for levels 1…6 (each level's texels at its own offset), with a
 * tool uniform describing it, as a tool's store hook would see it.
 */
function atlasLayout( width, height, layers, edges ) {

	const levels = [ 1, 2, 3, 4, 5, 6 ].map( level => ( { level, width: levelSize( width, level, edges ), height: levelSize( height, level, edges ) } ) );
	let offset = 0;
	for ( const level of levels ) { level.offset = offset; offset += level.width * level.height * layers; }
	const uniform = new Uint32Array( 28 );
	uniform.set( [ width, height ] );
	levels.forEach( ( level, i ) => { uniform[ 4 + i ] = level.offset; uniform[ 12 + i ] = level.width; uniform[ 20 + i ] = level.height; } );
	return { levels, texels: offset, uniform };

}

// The matrix's tool bindings: the source in a storage buffer, an atlas for the
// output, and a uniform with the source size and the atlas layout.
const matrixBindings = {
	wgsl: `struct Tool { width: u32, height: u32, offset: array<vec4u, 2>, stride: array<vec4u, 2>, rows: array<vec4u, 2> }
@group(1) @binding(0) var<storage, read> toolInput: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> toolAtlas: array<vec4f>;
@group(1) @binding(2) var<uniform> tool: Tool;`,
	entries: () => [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
	]
};
const matrixLoad = `fn load(texel: vec2i, layer: u32) -> vec4f {
	return toolInput[(layer * tool.height + u32(texel.y)) * tool.width + u32(texel.x)];
}`;
const matrixStore = `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
	let i = level - 1u;
	let row = layer * tool.rows[i / 4u][i % 4u] + u32(texel.y);
	toolAtlas[tool.offset[i / 4u][i % 4u] + row * tool.stride[i / 4u][i % 4u] + u32(texel.x)] = value;
}`;
const matrixSize = 'fn size() -> vec2u { return vec2u(tool.width, tool.height); }';

/** The matrix's tool resources for one shape: input buffer, atlas and uniform. */
function matrixTool( device, reducer, width, height, layers, edges ) {

	const layout = atlasLayout( width, height, layers, edges );
	const input = storageBuffer( device, new Float32Array( width * height * layers * 4 ) );
	const atlas = storageBuffer( device, new Float32Array( layout.texels * 4 ) );
	const uniform = device.createBuffer( { size: layout.uniform.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( uniform, 0, layout.uniform );
	const bindGroup = device.createBindGroup( { layout: reducer.bindGroupLayouts[ 1 ], entries: [ input, atlas, uniform ].map( ( buffer, binding ) => ( { binding, resource: { buffer } } ) ) } );
	return { layout, input, atlas, uniform, bindGroup, destroy: () => [ input, atlas, uniform ].forEach( buffer => buffer.destroy() ) };

}

/** Levels of an atlas readback, keyed by level. */
function atlasLevels( values, layout, layers ) {

	return new Map( layout.levels.map( ( { level, width, height, offset } ) => [ level, values.subarray( offset * 4, ( offset + width * height * layers ) * 4 ) ] ) );

}

async function validateReducer( device, { quick, onProgress } ) {

	const results = [];
	const shapes = quick ? reducerShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : reducerShapes;
	const widths = [ 1, 4, 5, 6 ];
	const reducers = new Map();
	const reducerFor = ( path, edges, variant, count, dimension ) => {

		const key = `${ path }:${ edges }:${ variant.name }:${ count }:${ dimension }`;
		if ( ! reducers.has( key ) ) reducers.set( key, new Reducer( device, path === 'texture' ? {
			reduction: variant.reduction, levels: textureLevels[ count ], edges, format: 'rgba32float', dimension
		} : {
			reduction: variant.reduction, levels: hookLevels[ count ], edges, dimension, load: matrixLoad, store: matrixStore,
			bindings: { wgsl: matrixBindings.wgsl, entries: matrixBindings.entries() }
		} ) );
		return reducers.get( key );

	};

	for ( const [ width, height, layers = 1, viewDimension ] of shapes ) {

		const dimension = viewDimension ?? ( layers > 1 ? '2d-array' : '2d' );
		const source = createTexture( device, { width, height, layers, format: 'rgba32float', dimension, storage: false } );
		const destroy = [ source ];

		try {

			for ( const edges of [ 'zero-weight', 'clamp' ] ) {

				// Output textures at exactly each level's size (⌈n/2⌉ under zero-weight).
				const outputs = new Map( [ 1, 2, 3, 4, 5, 6 ].map( level => {

					const texture = device.createTexture( { size: [ levelSize( width, level, edges ), levelSize( height, level, edges ), layers ], format: 'rgba32float', textureBindingViewDimension: dimension, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
					destroy.push( texture );
					return [ level, texture ];

				} ) );
				const tools = new Map();

				for ( const variant of reducerVariants ) for ( const pattern of variant.patterns ) {

					const data = reducerData( width, height, layers, pattern, width * 7 + height );
					upload( device, source, data );
					const expected = reducePyramid( data, width, height, layers, { reduction: variant.reference ?? variant.reduction, levels: [ 1, 2, 3, 4, 5, 6 ], edges } );

					for ( const count of widths ) for ( const path of [ 'texture', 'hook' ] ) {

						const reducer = reducerFor( path, edges, variant, count, dimension );
						const label = `reducer ${ width }x${ height }x${ layers } ${ edges } ${ variant.name } ${ pattern } L${ count } ${ path }`;
						const result = { reducer: variant.name, width, height, layers, edges, levels: reducer.levels, path, pattern, runs: 1, maxError: 0 };
						const encoder = device.createCommandEncoder();

						if ( path === 'texture' ) {

							for ( const level of reducer.levels ) {

								const texture = outputs.get( level );
								device.queue.writeTexture( { texture }, new Float32Array( texture.width * texture.height * layers * 4 ).fill( poisonValue ), { bytesPerRow: texture.width * 16, rowsPerImage: texture.height }, [ texture.width, texture.height, layers ] );

							}

							reducer.encode( encoder, { source, outputs: reducer.levels.map( level => outputs.get( level ) ) } );
							device.queue.submit( [ encoder.finish() ] );
							for ( const level of reducer.levels ) result.maxError = Math.max( result.maxError, checkPyramid( ( await readMips( device, outputs.get( level ), 0, 1 ) )[ 0 ], expected[ level - 1 ], `${ label } level ${ level }` ) );

						} else {

							// Every hook Reducer has an equivalent group-1 layout, so one tool bind group serves them all.
							if ( ! tools.has( edges ) ) tools.set( edges, matrixTool( device, reducer, width, height, layers, edges ) );
							const tool = tools.get( edges );
							device.queue.writeBuffer( tool.input, 0, data );
							device.queue.writeBuffer( tool.atlas, 0, new Float32Array( tool.layout.texels * 4 ).fill( poisonValue ) );
							reducer.encode( encoder, { width, height, layers, bindGroup: tool.bindGroup } );
							device.queue.submit( [ encoder.finish() ] );
							const levels = atlasLevels( await readBuffer( device, tool.atlas ), tool.layout, layers );

							for ( let level = 1; level <= 6; level ++ ) {

								if ( reducer.levels.includes( level ) ) result.maxError = Math.max( result.maxError, checkPyramid( levels.get( level ), expected[ level - 1 ], `${ label } level ${ level }` ) );
								else if ( levels.get( level ).some( value => value !== poisonValue ) ) throw new Error( `${ label }: unemitted level ${ level } was stored` );

							}

						}

						results.push( result );

					}

				}

				for ( const tool of tools.values() ) tool.destroy();

			}

			onProgress( `validated reducer ${ width }×${ height }×${ layers }` );

		} finally {

			for ( const texture of destroy ) texture.destroy();

		}

	}

	return results;

}

// ---------------------------------------------------------------------------
// Tool shapes: the passes M2T exists for, as their tools would write them.

/**
 * FSR 3.1 shading-change pyramid shape: a custom load from two tool textures
 * (current luma × exposure against previous luma), a 2-channel average, levels
 * 1–3 under 'clamp' (SPD's edges), stored to the mips of a half-resolution
 * texture as FSR's SpdMips. z counts loads (4^l under clamp) and w is max |diff|.
 */
const shadingChange = {
	bindings: {
		wgsl: `struct Fsr { exposure: f32 }
@group(1) @binding(0) var fsrCurrent: texture_2d<f32>;
@group(1) @binding(1) var fsrPrevious: texture_2d<f32>;
@group(1) @binding(2) var<uniform> fsr: Fsr;`,
		entries: () => [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
			{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
		]
	},
	load: `fn load(texel: vec2i, layer: u32) -> vec4f {
	let current = textureLoad(fsrCurrent, texel, 0).r * fsr.exposure;
	let previous = textureLoad(fsrPrevious, texel, 0).r;
	let diff = current - previous;
	return vec4f(diff, sign(diff), 1.0, abs(diff));
}`
};

/**
 * Denoiser shape (ReBLUR history fix / FFX block statistics): radiance weighted
 * by a sky mask from depth (sky = weight 0), mean over non-sky texels, levels
 * 1–3 under 'zero-weight' (⌈n/2⌉ sizes), stored by a hook into an atlas buffer.
 */
const denoiser = {
	bindings: {
		wgsl: `struct Atlas { offset: vec4u, stride: vec4u }
@group(1) @binding(0) var denoiserRadiance: texture_2d<f32>;
@group(1) @binding(1) var denoiserDepth: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> denoiserAtlas: array<vec4f>;
@group(1) @binding(3) var<uniform> atlas: Atlas;`,
		entries: () => [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
			{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
			{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
		]
	},
	load: `fn load(texel: vec2i, layer: u32) -> vec4f {
	let depth = textureLoad(denoiserDepth, texel, 0).r;
	let radiance = textureLoad(denoiserRadiance, texel, 0).rgb;
	return vec4f(radiance, select(1.0, 0.0, depth >= 1.0));
}`,
	store: `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
	denoiserAtlas[atlas.offset[level] + u32(texel.y) * atlas.stride[level] + u32(texel.x)] = value;
}`
};

function halfTexture( device, width, height, format, values ) {

	const texture = device.createTexture( { size: [ width, height ], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
	const channels = format.startsWith( 'rgba' ) ? 4 : 1;
	device.queue.writeTexture( { texture }, Uint16Array.from( values, toHalf ), { bytesPerRow: width * channels * 2, rowsPerImage: height }, [ width, height ] );
	return texture;

}

async function validateTools( device, { quick, onProgress } ) {

	const results = [];

	// Shading change: odd sizes drop the last row/column (SPD), partial tiles repeat the edge.
	const shading = new Reducer( device, {
		label: 'fsr shading change', reduction: [ 'average', 'average', 'sum', 'max' ], levels: [ 1, 2, 3 ], edges: 'clamp', format: 'rgba16float',
		load: shadingChange.load, bindings: { wgsl: shadingChange.bindings.wgsl, entries: shadingChange.bindings.entries() }
	} );

	for ( const [ width, height ] of quick ? [ [ 192, 108 ], [ 67, 33 ], [ 5, 3 ] ] : [ [ 1920, 1080 ], [ 1917, 1077 ], [ 67, 33 ], [ 5, 3 ] ] ) {

		let state = width + height;
		const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
		const current = Array.from( { length: width * height }, () => fromHalf( toHalf( next() * 2 ) ) );
		const previous = Array.from( { length: width * height }, () => fromHalf( toHalf( next() * 3 ) ) );
		const exposure = 1.5;
		const base = new Float64Array( width * height * 4 );
		current.forEach( ( c, i ) => {

			const diff = Math.fround( Math.fround( c * exposure ) - previous[ i ] );
			base.set( [ diff, Math.sign( diff ), 1, Math.abs( diff ) ], i * 4 );

		} );

		const cur = halfTexture( device, width, height, 'r16float', current ), prev = halfTexture( device, width, height, 'r16float', previous );
		const uniform = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( uniform, 0, Float32Array.of( exposure, 0, 0, 0 ) );
		// FSR's SpdMips: half resolution, one mip per emitted level; 'clamp' sizes are mip sizes.
		const mips = device.createTexture( { size: [ width >> 1 || 1, height >> 1 || 1 ], mipLevelCount: Math.min( 3, Math.floor( Math.log2( Math.max( width >> 1 || 1, height >> 1 || 1 ) ) ) + 1 ), format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
		const bindGroup = device.createBindGroup( { layout: shading.bindGroupLayouts[ 1 ], entries: [ { binding: 0, resource: cur.createView() }, { binding: 1, resource: prev.createView() }, { binding: 2, resource: { buffer: uniform } } ] } );

		if ( mips.mipLevelCount === 3 ) {

			poison( device, mips, 0 );
			const encoder = device.createCommandEncoder();
			shading.encode( encoder, { width, height, bindGroup, outputs: [ 0, 1, 2 ].map( mipLevel => ( { texture: mips, mipLevel } ) ) } );
			device.queue.submit( [ encoder.finish() ] );
			const actual = await readMips( device, mips, 0, 3 );
			const expected = reducePyramid( base, width, height, 1, { reduction: [ 'average', 'average', 'sum', 'max' ], levels: [ 1, 2, 3 ], edges: 'clamp' } );
			expected.forEach( ( level, i ) => checkPyramid( actual[ i ], level, `fsr shading change ${ width }x${ height } level ${ level.level }`, { quantization: 'float16' } ) );
			results.push( { tool: 'fsr shading change', width, height } );

		} else {

			// Too small for three mips of a half-resolution texture: reject the output.
			let threw = false;
			try { shading.encode( device.createCommandEncoder(), { width, height, bindGroup, outputs: [ 0, 0, 0 ].map( () => ( { texture: mips, mipLevel: 2 } ) ) } ); } catch { threw = true; }
			if ( ! threw ) throw new Error( 'fsr shading change: accepted an output mip outside its texture' );

		}

		for ( const resource of [ cur, prev, mips, uniform ] ) resource.destroy();

	}

	onProgress( 'validated tool shape: fsr shading change' );

	// Denoiser weighted mean: sky (depth 1) weighs 0, so sky-only blocks emit 0.
	const weighted = new Reducer( device, {
		label: 'denoiser weighted mean', reduction: [ 'weighted', 'weighted', 'weighted', 'weight' ], levels: [ 1, 2, 3 ], edges: 'zero-weight',
		load: denoiser.load, store: denoiser.store, bindings: { wgsl: denoiser.bindings.wgsl, entries: denoiser.bindings.entries() }
	} );

	for ( const [ width, height ] of quick ? [ [ 255, 129 ], [ 9, 5 ] ] : [ [ 1920, 1080 ], [ 1917, 1079 ], [ 255, 129 ], [ 9, 5 ] ] ) {

		let state = width * 3 + height;
		const next = () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32;
		// The top third is sky, and a fifth of the rest.
		const sky = Array.from( { length: width * height }, ( _, i ) => Math.floor( i / width ) < height / 3 || next() < 0.2 );
		const radiance = Array.from( { length: width * height * 4 }, () => fromHalf( toHalf( next() * 8 ) ) );
		const depth = Float32Array.from( sky, ( isSky ) => isSky ? 1 : Math.fround( next() * 0.99 ) );
		const base = new Float64Array( width * height * 4 );
		sky.forEach( ( isSky, i ) => base.set( [ radiance[ i * 4 ], radiance[ i * 4 + 1 ], radiance[ i * 4 + 2 ], isSky ? 0 : 1 ], i * 4 ) );

		const radianceTexture = halfTexture( device, width, height, 'rgba16float', radiance );
		const depthTexture = device.createTexture( { size: [ width, height ], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		device.queue.writeTexture( { texture: depthTexture }, depth, { bytesPerRow: width * 4, rowsPerImage: height }, [ width, height ] );
		const layout = atlasLayout( width, height, 1, 'zero-weight' );
		const atlasBuffer = storageBuffer( device, new Float32Array( layout.texels * 4 ).fill( poisonValue ) );
		const uniform = device.createBuffer( { size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( uniform, 0, Uint32Array.of( 0, ...layout.levels.slice( 0, 3 ).map( level => level.offset ), 0, ...layout.levels.slice( 0, 3 ).map( level => level.width ) ) );
		const bindGroup = device.createBindGroup( { layout: weighted.bindGroupLayouts[ 1 ], entries: [ radianceTexture.createView(), depthTexture.createView(), { buffer: atlasBuffer }, { buffer: uniform } ].map( ( resource, binding ) => ( { binding, resource } ) ) } );

		const encoder = device.createCommandEncoder();
		weighted.encode( encoder, { width, height, bindGroup } );
		device.queue.submit( [ encoder.finish() ] );
		const levels = atlasLevels( await readBuffer( device, atlasBuffer ), layout, 1 );
		const expected = reducePyramid( base, width, height, 1, { reduction: [ 'weighted', 'weighted', 'weighted', 'weight' ], levels: [ 1, 2, 3 ], edges: 'zero-weight' } );
		let skyBlocks = 0;

		for ( const level of expected ) {

			const actual = levels.get( level.level );
			checkPyramid( actual, level, `denoiser ${ width }x${ height } level ${ level.level }` );
			for ( let i = 0; i < actual.length; i += 4 ) if ( level.values[ i + 3 ] === 0 ) {

				skyBlocks ++;
				if ( actual[ i ] !== 0 || actual[ i + 1 ] !== 0 || actual[ i + 2 ] !== 0 || actual[ i + 3 ] !== 0 ) throw new Error( `denoiser ${ width }x${ height } level ${ level.level }: a sky-only block emitted ${ Array.from( actual.subarray( i, i + 4 ) ) }` );

			}

		}

		if ( skyBlocks === 0 ) throw new Error( `denoiser ${ width }x${ height }: the fixture has no sky-only block` );
		for ( let level = 4; level <= 6; level ++ ) if ( levels.get( level ).some( value => value !== poisonValue ) ) throw new Error( `denoiser: level ${ level } was written` );
		for ( const resource of [ radianceTexture, depthTexture, atlasBuffer, uniform ] ) resource.destroy();
		results.push( { tool: 'denoiser weighted mean', width, height, skyBlocks } );

	}

	onProgress( 'validated tool shape: denoiser weighted mean' );
	results.push( ...await validateCrossLevel( device, { quick } ) );
	onProgress( 'validated tool shape: cross-level store state' );
	return results;

}

/**
 * Store hooks may keep workgroup state from one level to a deeper one (SPEC
 * §14.1): levels are emitted shallow to deep with a workgroup barrier between
 * them, and the workgroup that emits a texel has emitted all of its in-level
 * descendants, inside an aligned block of at most 16 × 16 texels per level
 * below level 1. The shallow store keeps its values in tool workgroup memory
 * (slot texel % 16); the deep store sums its descendants from there, skipping
 * those past the shallow level (not emitted; under 'zero-weight' they count 0).
 * Both levels and the sum must match the oracle. The upscaler's shading change
 * takes max(4×4, 8×8) responses this way (test/tools/generator.js).
 */
function crossLevelHooks( shallow, deep ) {

	return {
		wgsl: `struct Cross { shallowWidth: u32, deepWidth: u32, deepOffset: u32, sumOffset: u32, shallowSize: vec2u }
@group(1) @binding(0) var crossSource: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> crossOut: array<vec4f>;
@group(1) @binding(2) var<uniform> cross: Cross;`,
		load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return textureLoad(crossSource, texel, 0); }',
		store: `var<workgroup> crossShallow: array<vec4f, 256>;
fn crossSlot(texel: vec2i) -> u32 { return u32(texel.y & 15) * 16u + u32(texel.x & 15); }
fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {
	if (level == ${ shallow }u) {
		crossShallow[crossSlot(texel)] = value;
		crossOut[u32(texel.y) * cross.shallowWidth + u32(texel.x)] = value;
		return;
	}
	var sum = vec4f(0.0);
	for (var y = 0; y < ${ 1 << ( deep - shallow ) }; y++) {
		for (var x = 0; x < ${ 1 << ( deep - shallow ) }; x++) {
			let child = texel * ${ 1 << ( deep - shallow ) } + vec2i(x, y);
			if (all(vec2u(child) < cross.shallowSize)) { sum += crossShallow[crossSlot(child)]; }
		}
	}
	let i = u32(texel.y) * cross.deepWidth + u32(texel.x);
	crossOut[cross.deepOffset + i] = value;
	crossOut[cross.sumOffset + i] = sum;
}`
	};

}

async function validateCrossLevel( device, { quick } ) {

	const results = [];
	const entries = [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
	];

	// Tiles of 8 ([ 2, 3 ], [ 1, 4 ]) and 16 ([ 2, 5 ]) level-1 texels; a 6-level tile leaves no workgroup memory for vec4f state.
	for ( const [ shallow, deep ] of [ [ 2, 3 ], [ 1, 4 ], [ 2, 5 ] ] ) for ( const edges of [ 'clamp', 'zero-weight' ] ) {

		const hooks = crossLevelHooks( shallow, deep );
		const reducer = new Reducer( device, { label: `cross-level ${ shallow }→${ deep } ${ edges }`, reduction: 'sum', levels: [ shallow, deep ], edges, load: hooks.load, store: hooks.store, bindings: { wgsl: hooks.wgsl, entries } } );

		for ( const [ width, height ] of quick ? [ [ 67, 33 ], [ 255, 257 ] ] : [ [ 67, 33 ], [ 192, 108 ], [ 255, 257 ], [ 1917, 1079 ] ] ) {

			const data = reducerData( width, height, 1, 'random', width + 3 * height );
			const source = createTexture( device, { width, height, format: 'rgba32float', storage: false } );
			upload( device, source, data );
			const sizes = [ shallow, deep ].map( level => [ levelSize( width, level, edges ), levelSize( height, level, edges ) ] );
			const shallowTexels = sizes[ 0 ][ 0 ] * sizes[ 0 ][ 1 ], deepTexels = sizes[ 1 ][ 0 ] * sizes[ 1 ][ 1 ];
			const out = storageBuffer( device, new Float32Array( ( shallowTexels + 2 * deepTexels ) * 4 ).fill( poisonValue ) );
			const uniform = device.createBuffer( { size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			device.queue.writeBuffer( uniform, 0, Uint32Array.of( sizes[ 0 ][ 0 ], sizes[ 1 ][ 0 ], shallowTexels, shallowTexels + deepTexels, ...sizes[ 0 ], 0, 0 ) );
			const bindGroup = device.createBindGroup( { layout: reducer.bindGroupLayouts[ 1 ], entries: [ source.createView( { baseMipLevel: 0, mipLevelCount: 1 } ), { buffer: out }, { buffer: uniform } ].map( ( resource, binding ) => ( { binding, resource } ) ) } );

			const encoder = device.createCommandEncoder();
			reducer.encode( encoder, { width, height, bindGroup } );
			device.queue.submit( [ encoder.finish() ] );
			const actual = await readBuffer( device, out );
			const [ expectedShallow, expectedDeep ] = reducePyramid( data, width, height, 1, { reduction: 'sum', levels: [ shallow, deep ], edges } );
			const label = `cross-level ${ shallow }→${ deep } ${ edges } ${ width }x${ height }`;
			checkPyramid( actual.subarray( 0, shallowTexels * 4 ), expectedShallow, `${ label } level ${ shallow }` );
			checkPyramid( actual.subarray( shallowTexels * 4, ( shallowTexels + deepTexels ) * 4 ), expectedDeep, `${ label } level ${ deep }` );
			checkPyramid( actual.subarray( ( shallowTexels + deepTexels ) * 4 ), expectedDeep, `${ label } level ${ deep } from level-${ shallow } workgroup state` );
			for ( const resource of [ source, out, uniform ] ) resource.destroy();
			results.push( { tool: 'cross-level store state', levels: [ shallow, deep ], edges, width, height } );

		}

	}

	return results;

}

async function validateReducerOptions( device ) {

	const results = [];
	const expectThrow = ( run, message, label ) => {

		try { run(); } catch ( error ) {

			if ( ! String( error.message ).includes( message ) ) throw new Error( `${ label }: expected "${ message }", got "${ error.message }"` );
			return;

		}

		throw new Error( `${ label }: expected an error "${ message }"` );

	};

	// The same reduction recorded with encode(), encodePass() and dispatched by
	// the tool itself from { wgsl, layout, bindGroup(), workgroups() }; and with a
	// size hook instead of the Reducer's size uniform. All must match.
	const [ width, height, layers ] = [ 333, 201, 2 ];
	const data = reducerData( width, height, layers, 'random', 5 );
	const options = { reduction: [ 'sum', 'average', 'min', 'max' ], levels: [ 1, 2, 4, 6 ], edges: 'zero-weight', dimension: '2d-array', load: matrixLoad, store: matrixStore };
	const reducer = new Reducer( device, { ...options, bindings: { wgsl: matrixBindings.wgsl, entries: matrixBindings.entries() } } );
	const sized = new Reducer( device, { ...options, size: matrixSize, bindings: { wgsl: matrixBindings.wgsl, entries: matrixBindings.entries() } } );
	const pipeline = device.createComputePipeline( { layout: reducer.layout, compute: { module: device.createShaderModule( { code: reducer.wgsl } ), entryPoint: reducer.entryPoint } } );
	const expected = reducePyramid( data, width, height, layers, { reduction: options.reduction, levels: options.levels, edges: options.edges } );
	const readbacks = [];

	for ( const [ mode, target ] of [ [ 'encode', reducer ], [ 'encodePass', reducer ], [ 'self-dispatch', reducer ], [ 'size hook', sized ] ] ) {

		const tool = matrixTool( device, target, width, height, layers, options.edges );
		device.queue.writeBuffer( tool.input, 0, data );
		device.queue.writeBuffer( tool.atlas, 0, new Float32Array( tool.layout.texels * 4 ).fill( poisonValue ) );
		const encoder = device.createCommandEncoder();
		const dispatch = { width, height, layers, bindGroup: tool.bindGroup };
		if ( mode === 'encode' || mode === 'size hook' ) target.encode( encoder, dispatch );
		else {

			const pass = encoder.beginComputePass();
			if ( mode === 'encodePass' ) target.encodePass( pass, dispatch );
			else {

				pass.setPipeline( pipeline );
				pass.setBindGroup( 0, target.bindGroup( dispatch ) );
				pass.setBindGroup( 1, tool.bindGroup );
				pass.dispatchWorkgroups( ...target.workgroups( width, height, layers ) );

			}

			pass.end();

		}

		device.queue.submit( [ encoder.finish() ] );
		const values = await readBuffer( device, tool.atlas );
		const levels = atlasLevels( values, tool.layout, layers );
		for ( const level of expected ) checkPyramid( levels.get( level.level ), level, `reducer ${ mode } level ${ level.level }` );
		readbacks.push( [ mode, values ] );
		tool.destroy();

	}

	for ( const [ mode, values ] of readbacks.slice( 1 ) ) if ( values.some( ( value, i ) => ! Object.is( value, readbacks[ 0 ][ 1 ][ i ] ) ) ) throw new Error( `reducer ${ mode } differs from encode()` );
	if ( reducer.bindGroup( { width, height, layers } ) !== reducer.bindGroup( { width, height, layers } ) ) throw new Error( 'reducer: bind groups are not cached' );
	results.push( { option: 'reducer encode / encodePass / self-dispatch / size hook' } );

	// One Reducer at two sizes in one submit: each dispatch keeps its own size.
	{

		const sizes = [ [ 37, 21 ], [ 130, 66 ] ];
		const tools = sizes.map( ( [ w, h ] ) => {

			const tool = matrixTool( device, reducer, w, h, 1, options.edges );
			const input = reducerData( w, h, 1, 'random', w );
			device.queue.writeBuffer( tool.input, 0, input );
			device.queue.writeBuffer( tool.atlas, 0, new Float32Array( tool.layout.texels * 4 ).fill( poisonValue ) );
			return { tool, input, w, h };

		} );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		for ( const { tool, w, h } of tools ) reducer.encodePass( pass, { width: w, height: h, layers: 1, bindGroup: tool.bindGroup } );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );

		for ( const { tool, input, w, h } of tools ) {

			const levels = atlasLevels( await readBuffer( device, tool.atlas ), tool.layout, 1 );
			for ( const level of reducePyramid( input, w, h, 1, { reduction: options.reduction, levels: options.levels, edges: options.edges } ) ) checkPyramid( levels.get( level.level ), level, `reducer two sizes ${ w }x${ h } level ${ level.level }` );
			tool.destroy();

		}

		results.push( { option: 'reducer: two sizes in one pass' } );

	}

	// Texture outputs must cover their level: a mip of the source size is one texel
	// short at odd sizes under zero-weight, and fits exactly under clamp.
	const source = createTexture( device, { width: 17, height: 9, format: 'rgba32float', storage: false } );
	const mipped = createTexture( device, { width: 17, height: 9, format: 'rgba32float' } );
	const textureReducer = edges => new Reducer( device, { reduction: 'average', levels: [ 1, 2 ], edges, format: 'rgba32float' } );
	const mipOutputs = [ 1, 2 ].map( mipLevel => ( { texture: mipped, mipLevel } ) );
	expectThrow( () => textureReducer( 'zero-weight' ).encode( device.createCommandEncoder(), { source, outputs: mipOutputs } ), 'smaller than the level', 'zero-weight mip outputs' );
	textureReducer( 'clamp' ).encode( device.createCommandEncoder(), { source, outputs: mipOutputs } );
	expectThrow( () => textureReducer( 'clamp' ).encode( device.createCommandEncoder(), { source, outputs: mipOutputs.slice( 1 ) } ), 'one output per emitted level', 'output count' );
	expectThrow( () => textureReducer( 'clamp' ).encode( device.createCommandEncoder(), { outputs: mipOutputs } ), 'pass `source`', 'missing source' );
	expectThrow( () => new Reducer( device, { reduction: 'average', levels: [ 1, 2 ], format: 'rgba16float' } ).encode( device.createCommandEncoder(), { source, outputs: mipOutputs } ), 'this Reducer writes rgba16float', 'output format' );
	expectThrow( () => reducer.encode( device.createCommandEncoder(), { width, height, layers } ), 'group-1 bindGroup', 'missing tool bind group' );
	expectThrow( () => reducer.encode( device.createCommandEncoder(), { layers } ), 'width and height', 'missing size' );
	expectThrow( () => new Reducer( device, { levels: [ 3, 2 ], format: 'rgba32float' } ), 'strictly increasing', 'unsorted levels' );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], format: 'rgba32float', edges: 'exact' } ), 'Downsampler', 'exact edges' );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], format: 'rgba32float', reduction: [ 'weighted', 'sum', 'sum', 'sum' ] } ), 'need a \'weight\'', 'weighted without weight' );
	expectThrow( () => new Reducer( device, { levels: [ 1 ], load: 'fn sample() {}', store: matrixStore } ), 'load hook', 'bad load hook' );
	const limit = device.limits.maxStorageTexturesPerShaderStage;
	if ( limit < 6 ) expectThrow( () => new Reducer( device, { levels: [ 1, 2, 3, 4, 5, 6 ].slice( 0, limit + 1 ), format: 'rgba32float' } ), 'maxStorageTexturesPerShaderStage', 'storage texture limit' );
	// Sums and channel ops have no 'exact' meaning: Downsampler rejects them.
	expectThrow( () => new Downsampler( device ).generate( mipped, { reduction: 'sum' } ), 'Reducer', 'Downsampler sum' );
	expectThrow( () => new Downsampler( device ).generate( mipped, { reduction: [ 'min', 'max', 'min', 'max' ] } ), 'Reducer', 'Downsampler channel ops' );
	source.destroy();
	mipped.destroy();
	results.push( { option: 'reducer validation' } );
	return results;

}
