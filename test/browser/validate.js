// GPU correctness suite. Every texel of every generated level is checked against
// the CPU reference, computed from the GPU's own previous level so permitted
// rounding differences do not accumulate along the chain.
import { copyReference, formats, isIntegerDomain, reduceReference, sampledChannels, tolerance } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createDepthSource, createTexture, makeData, poison, readMip, readMips, toFloats, upload } from './gpu.js';
import { validateScalar } from './validate-scalar.js';

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

/** `only`: run one part of the suite ('mips', 'scalar'); default all. */
export async function validate( device, { onProgress = () => {}, quick = false, only } = {} ) {

	const results = [];
	const started = performance.now();
	const cases = only === 'scalar' ? [] : quick ? shapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : shapes;
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

			results.push( ...await validateSource( device, { quick, onProgress } ) );
			results.push( ...await validateOptions( device ) );

		}

		if ( only !== 'mips' ) {

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
