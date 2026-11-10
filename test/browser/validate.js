// GPU correctness suite. Every texel of every generated level is checked against
// the CPU reference, computed from the GPU's own previous level so permitted
// rounding differences do not accumulate along the chain.
import { formats, reduceReference, tolerance } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, poison, readMip, toFloats, upload } from './gpu.js';

// [ width, height, layers?, viewDimension? ]: 1×1, 1D tails, odd sizes, partial
// workgroups, rectangles, layers, both 2D view variants, and multi-batch sizes.
export const shapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 7, 4 ], [ 15, 17 ], [ 16, 16 ], [ 17, 9 ], [ 31, 33 ], [ 64, 64 ], [ 65, 129 ],
	[ 128, 8 ], [ 1, 128 ], [ 128, 1 ], [ 192, 108 ], [ 255, 257 ], [ 256, 256 ], [ 512, 32 ],
	[ 64, 32, 3 ], [ 17, 11, 3 ], [ 32, 16, 1, '2d-array' ], [ 17, 9, 1, '2d-array' ],
	[ 2048, 2048 ], [ 96, 4096 ], [ 1920, 1080 ]
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
	{ name: 'average (load)', reduction: 'average', access: 'load', patterns },
	{ name: 'min', reduction: 'min', patterns: patterns.slice( 1 ) },
	{ name: 'max', reduction: 'max', patterns: patterns.slice( 1 ) },
	{ name: 'minmax', reduction: 'minmax', patterns: patterns.slice( 1 ) },
	{ name: 'custom', reduction: { wgsl: preferHigh.wgsl }, reference: preferHigh.reference, patterns: patterns.slice( 1 ) }
];

function check( actual, expected, quantization, label, reduction = 'average' ) {

	let maxError = 0;

	for ( let i = 0; i < expected.values.length; i ++ ) {

		const error = Math.abs( actual[ i ] - expected.values[ i ] );
		const limit = tolerance( expected.values[ i ], quantization, expected.filterTolerance[ i ], reduction );
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

export async function validate( device, { onProgress = () => {}, quick = false } = {} ) {

	const results = [];
	const started = performance.now();
	const cases = quick ? shapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : shapes;
	let runs = 0;
	device.pushErrorScope( 'validation' );

	try {

		for ( const levels of levelOptions( device ) ) {

			const downsamplers = {
				sample: new Downsampler( device, { levelsPerDispatch: levels } ),
				load: new Downsampler( device, { levelsPerDispatch: levels, averageAccess: 'load' } )
			};

			for ( const format of Object.keys( formats ) ) {

				const { quantization } = formats[ format ];

				for ( const [ width, height, layers = 1, dimension ] of cases ) {

					const texture = createTexture( device, { width, height, layers, format, dimension } );

					try {

						for ( const variant of variants ) {

							const downsampler = downsamplers[ variant.access ?? 'sample' ];
							const options = { reduction: variant.reduction };
							const reference = variant.reference ?? variant.reduction;
							const result = { width, height, layers, format, levelsPerDispatch: levels, reduction: variant.name, dispatches: 0, maxError: 0 };

							// Reusing one texture with changing input exercises the cached plans.
							for ( const pattern of variant.patterns ) {

								const input = makeData( width, height, layers, format, pattern, pattern === 'tiles' ? results.length + 3 : 1 );
								upload( device, texture, input );
								poison( device, texture );
								downsampler.generate( texture, options );
								let previous = toFloats( input, format ), w = width, h = height;

								for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

									const expected = reduceReference( previous, w, h, layers, quantization, reference );
									const actual = await readMip( device, texture, level );
									result.maxError = Math.max( result.maxError, check( actual, expected, quantization, `${ width }x${ height }x${ layers } ${ format } ${ variant.name } ${ pattern } L${ levels } mip ${ level }`, reference ) );
									previous = actual; w = expected.width; h = expected.height;

								}

								runs ++;

							}

							result.dispatches = downsampler.prepare( texture, options ).length;
							results.push( result );

						}

						onProgress( `validated ${ width }×${ height }×${ layers } ${ format }, ${ levels } levels/dispatch` );

					} finally {

						texture.destroy();

					}

				}

			}

		}

		results.push( ...await validateOptions( device ) );
		await device.queue.onSubmittedWorkDone();

	} finally {

		const error = await device.popErrorScope();
		if ( error ) throw new Error( `GPU validation error: ${ error.message }` );

	}

	return { passed: true, cases: results.length, checks: runs, ms: Math.round( performance.now() - started ), results };

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

	results.push( { option: 'unsupported textures' } );
	return results;

}
