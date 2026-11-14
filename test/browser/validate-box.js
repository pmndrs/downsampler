// M2.5 (SPEC §2.1, §4 Separate source): oddSizes 'box' at render sizes, and a separate
// source resampled into a destination of another size (`resample: 'area'`, e.g. a
// power-of-two Hi-Z from 1920×1080). Every level is checked against the core oracles
// (`areaReference`, `boxReference`) from the GPU's own previous level, and folds of a
// resampled source must stay conservative against every source texel (`maxViolations`).
import { areaReference, boxReference, copyReference, formats, isIntegerDomain, maxViolations, reduceReference, sampledChannels, spdReference } from '/dist/index.js';
import { Downsampler, maxResampleRatio } from '/dist/webgpu/index.js';
import { createDepthSource, createTexture, makeData, poison, readMips, toFloats, upload } from './gpu.js';
import { check, compatibility, expectThrow, levelOptions, makeSource, preferHigh, splat } from './validate.js';

const mipSize = ( size, level ) => Math.max( 1, size >> level );

// Render sizes, greggman's 31×31, 1×N strips and layers, for the 'box' chain beyond the main matrix's shapes.
export const boxShapes = [ [ 960, 540 ], [ 1129, 635 ], [ 1477, 831 ], [ 1920, 1080 ], [ 31, 31 ], [ 1, 300 ], [ 300, 1 ], [ 33, 17, 3 ], [ 31, 31, 6, '2d-array' ] ];
export const boxShapeFormats = [ 'rgba8unorm', 'rgba16float', 'r32float' ];

// [ source width, height, layers, destination width, height ]: non-2:1 ratios down and up,
// power-of-two destinations, strips, layers, plus a same-size and a half step.
export const resampleShapes = [
	[ 1920, 1080, 1, 1024, 512 ], [ 1920, 1080, 1, 2048, 1024 ], [ 1129, 635, 1, 512, 256 ], [ 1129, 635, 1, 1024, 512 ],
	[ 960, 540, 1, 512, 256 ], [ 1477, 831, 1, 1024, 512 ], [ 31, 31, 1, 16, 16 ], [ 31, 31, 1, 32, 32 ], [ 1, 300, 1, 1, 256 ],
	[ 300, 1, 1, 512, 1 ], [ 65, 33, 3, 32, 32 ], [ 17, 9, 1, 16, 8 ], [ 3, 7, 1, 8, 8 ], [ 128, 64, 1, 16, 8 ], [ 7, 5, 1, 7, 5 ], [ 17, 9, 1, 8, 4 ]
];

export const resampleCases = [
	{ source: 'depth32float', destination: 'r32float', reductions: [ 'min', 'max' ] },
	{ source: 'depth32float', destination: 'rg32float', reductions: [ 'minmax', 'average' ] },
	{ source: 'depth16unorm', destination: 'r32float', reductions: [ 'max', 'average' ] },
	{ source: 'r32float', destination: 'r32float', reductions: [ 'average', 'min' ] },
	{ source: 'rgba16float', destination: 'rgba8unorm', reductions: [ 'average', 'max' ] },
	{ source: 'rgba8unorm', destination: 'rgba16float', reductions: [ 'average' ] },
	{ source: 'rgba16float', destination: 'rg11b10ufloat', reductions: [ 'average' ] }
];

/** Folds must bound every source texel under each texel's normalized area (Hi-Z): 0 violations per channel. */
function conservative( base, width, height, level, w, h, layers, reduction, label ) {

	const kinds = reduction === 'max' ? [ [ 0, 1 ] ] : reduction === 'min' ? [ [ 0, - 1 ] ] : reduction === 'minmax' ? [ [ 0, - 1 ], [ 1, 1 ] ] : [];
	for ( const [ channel, sign ] of kinds ) {

		const pick = values => { const out = new Float32Array( values.length ); for ( let i = 0; i < values.length; i += 4 ) out[ i ] = sign * values[ i + channel ]; return out; };
		const violations = maxViolations( pick( base ), width, height, pick( level ), w, h, layers );
		if ( violations > 0 ) throw new Error( `${ label }: ${ violations } texels don't bound the source texels they cover (channel ${ channel })` );

	}

}

export async function validateBox( device, { quick = false, onProgress = () => {}, select } ) {

	const results = [];
	results.push( ...await validateBoxShapes( device, { quick, onProgress, select } ) );
	results.push( ...await validateResample( device, { quick, onProgress, select } ) );
	results.push( ...await select.run( 'box/options', () => validateBoxOptions( device, { quick } ) ) );
	return results;

}

/** 'box' chains at render sizes, 31×31, strips and layers: average (sample, load), the tail, and max (= 'exact'). */
async function validateBoxShapes( device, { quick, onProgress, select } ) {

	const results = [];
	const shapes = quick ? boxShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : boxShapes;
	const widths = quick ? levelOptions( device ).slice( - 1 ) : levelOptions( device );
	for ( const format of boxShapeFormats ) results.push( ...await select.run( `box/shapes/${ format }`, async () => {

		const { quantization, channels, filterable } = formats[ format ];
		const runs = [];
		for ( const levels of widths ) {

			const sample = new Downsampler( device, { levelsPerDispatch: levels, tail: false } ), load = new Downsampler( device, { levelsPerDispatch: levels, averageAccess: 'load', tail: false } );
			const tail = new Downsampler( device, { levelsPerDispatch: levels } );
			runs.push( [ sample, 'average', 'sample', levels ], [ sample, 'max', 'max', levels ] );
			if ( filterable ) runs.push( [ load, 'average', 'load', levels ] );
			// Tails of 32×32 tiles compile for minutes on SwiftShader.
			if ( ! quick ) runs.push( [ tail, 'average', 'tail', levels ] );

		}

		const done = [];
		for ( const [ width, height, layers = 1, dimension ] of shapes ) {

			const texture = createTexture( device, { width, height, layers, format, dimension } );
			try {

				// Render sizes: the widest kernel only (its oracles dominate the suite's time); 'box'
				// kernels are the same at every width, and the main matrix runs every width at 1920×1080.
				const big = width * height > 65536;
				for ( const pattern of quick ? [ 'random' ] : big ? [ 'random', 'edge' ] : [ 'random', 'edge', 'tiles' ] ) {

					const data = makeData( width, height, layers, format, pattern, 3 );
					const input = toFloats( data, format );
					for ( const [ downsampler, reduction, kind, levels ] of runs ) {

						if ( big && levels !== widths[ widths.length - 1 ] ) continue;
						const options = { oddSizes: 'box', reduction };
						upload( device, texture, data );
						poison( device, texture );
						downsampler.generate( texture, options );
						const mips = await readMips( device, texture, 1 );
						const batches = downsampler.prepare( texture, options );
						const label = `box ${ width }x${ height }x${ layers } ${ format } ${ kind } ${ pattern } L${ levels }`;
						// Exact (normalized formats) wherever a level is an even step reduced in workgroup memory.
						const first = new Set( batches.map( batch => batch.firstMipLevel ) );
						const tailOdd = level => batches.some( batch => batch.kernel === 'tail' && level > batch.firstMipLevel && level < batch.firstMipLevel + batch.count ) && ( mipSize( width, level - 1 ) % 2 === 1 && mipSize( width, level - 1 ) > 1 || mipSize( height, level - 1 ) % 2 === 1 && mipSize( height, level - 1 ) > 1 );
						let previous = input, w = width, h = height;
						for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

							const expected = boxReference( previous, w, h, layers, quantization, reduction, { weights: kind === 'sample' && first.has( level ) ? 'sampler' : 'shader' } );
							const exact = reduction !== 'average' || ( isIntegerDomain( quantization ) && ! first.has( level ) && ! tailOdd( level ) );
							check( mips[ level - 1 ], expected, quantization, `${ label } mip ${ level }`, reduction, channels, 0, exact );
							previous = mips[ level - 1 ]; w = expected.width; h = expected.height;

						}

						if ( reduction === 'average' && pattern === 'random' && width === 31 && height === 31 && layers === 1 ) {

							// greggman's 31×31: the 1×1 is the mean of every texel (to the per-level rounding).
							const one = mips[ mips.length - 1 ];
							for ( let c = 0; c < channels; c ++ ) {

								let sum = 0;
								for ( let i = c; i < input.length; i += 4 ) sum += input[ i ];
								const mean = sum / ( width * height ), limit = quantization === 'unorm8' ? 2.5 / 255 : quantization === 'float16' ? 0.004 : 1e-5;
								if ( Math.abs( one[ c ] - mean ) > limit ) throw new Error( `${ label }: 1×1 channel ${ c } is ${ one[ c ] }, the mean is ${ mean }` );

							}

						}

					}

				}

				done.push( { box: format, width, height, layers } );
				onProgress( `validated box ${ width }×${ height }×${ layers } ${ format }` );

			} finally {

				texture.destroy();

			}

		}

		return done;

	} ) );

	return results;

}

/** Separate sources resampled into another size: every mode, the first level from the source by area, the rest by the mode's rule. */
async function validateResample( device, { quick, onProgress, select } ) {

	const results = [];
	const shapes = quick ? resampleShapes.filter( ( [ w, h, , dw, dh ] ) => w * h <= 65536 && dw * dh <= 65536 ) : resampleShapes;
	const widths = quick ? levelOptions( device ).slice( - 1 ) : levelOptions( device );
	const modes = widths.flatMap( levels => {

		const plain = new Downsampler( device, { levelsPerDispatch: levels, tail: false } );
		const tail = new Downsampler( device, { levelsPerDispatch: levels } );
		const list = [ [ plain, 'exact', levels ], [ plain, 'box', levels ], [ plain, 'spd', levels ] ];
		return quick ? list : [ ...list, [ tail, 'exact', levels, 'tail' ], [ tail, 'box', levels, 'tail' ] ];

	} );

	for ( const { source: sourceFormat, destination: format, reductions } of resampleCases ) {

		if ( formats[ format ].coreStorage && compatibility( device ) ) continue;
		const { quantization, channels } = formats[ format ];
		results.push( ...await select.run( `box/resample/${ sourceFormat }/${ format }`, async () => {

			const done = [];
			for ( const [ width, height, layers, dw, dh ] of shapes ) {

				const destination = createTexture( device, { width: dw, height: dh, layers, format } );
				try {

					for ( const pattern of quick ? [ 'random' ] : [ 'random', 'edge' ] ) {

						const source = await makeSource( device, sourceFormat, width, height, layers, pattern, 5 );
						const input = sampledChannels( sourceFormat ) === 1 ? splat( source.floats ) : source.floats;
						const slack = /unorm/.test( sourceFormat ) && sourceFormat.startsWith( 'depth' ) ? 2 ** - 21 : 0;
						const firstLevels = new Map();
						try {

							for ( const [ downsampler, mode, levels, tail ] of modes ) for ( const reduction of reductions ) {

								// Render-size sources: the widest kernel only (the first batch is the same at every width).
								if ( width * height > 65536 && levels !== widths[ widths.length - 1 ] ) continue;
								if ( mode === 'spd' && reduction !== 'average' ) continue;
								const options = { source: { texture: source.texture, resample: 'area' }, reduction, oddSizes: mode };
								if ( tail && ! downsampler.prepare( destination, options ).some( batch => batch.kernel === 'tail' ) ) continue;
								poison( device, destination, 0 );
								downsampler.generate( destination, options );
								const mips = await readMips( device, destination, 0 );
								const label = `resample ${ sourceFormat } ${ width }x${ height }x${ layers } -> ${ format } ${ dw }x${ dh } ${ reduction } ${ pattern } L${ levels } ${ tail ?? '' }${ mode }`;

								// Level 0: by area from the source, whatever the mode (the same for every mode and width).
								if ( ! firstLevels.has( reduction ) ) firstLevels.set( reduction, areaReference( input, width, height, layers, quantization, reduction, dw, dh, { quantizedInput: formats[ sourceFormat ]?.quantization === quantization } ) );
								check( mips[ 0 ], firstLevels.get( reduction ), quantization, `${ label } mip 0`, reduction, channels, slack );

								if ( mode === 'spd' ) {

									for ( const batch of downsampler.prepare( destination, options ).filter( batch => batch.sourceMipLevel !== null && ! batch.copy ) ) {

										spdReference( mips[ batch.sourceMipLevel ], mipSize( dw, batch.sourceMipLevel ), mipSize( dh, batch.sourceMipLevel ), layers, quantization, reduction, batch.count ).forEach( ( level, i ) => {

											check( mips[ batch.firstMipLevel + i ], level, quantization, `${ label } mip ${ batch.firstMipLevel + i }`, reduction, channels );

										} );

									}

									continue;

								}

								let previous = mips[ 0 ], w = dw, h = dh;
								for ( let level = 1; level < destination.mipLevelCount; level ++ ) {

									const expected = ( mode === 'box' ? boxReference : reduceReference )( previous, w, h, layers, quantization, reduction );
									check( mips[ level ], expected, quantization, `${ label } mip ${ level }`, reduction, channels );
									previous = mips[ level ]; w = expected.width; h = expected.height;

								}

								// Hi-Z: every level bounds every source texel its normalized area covers (float32
								// sources and destinations: a conversion may round a value below its source).
								if ( levels === widths[ widths.length - 1 ] && quantization === 'float32' && /32float$/.test( sourceFormat ) ) mips.forEach( ( level, l ) => conservative( input, width, height, level, mipSize( dw, l ), mipSize( dh, l ), layers, reduction, `${ label } mip ${ l }` ) );

							}

						} finally {

							source.texture.destroy();

						}

					}

					done.push( { resample: `${ sourceFormat } -> ${ format }`, width, height, layers, dw, dh } );
					onProgress( `validated resample ${ sourceFormat } ${ width }×${ height } -> ${ format } ${ dw }×${ dh }` );

				} finally {

					destination.destroy();

				}

			}

			return done;

		} ) );

	}

	return results;

}

async function validateBoxOptions( device, { quick } ) {

	const results = [];
	const same = ( a, b, label ) => a.forEach( ( level, l ) => { if ( level.some( ( value, i ) => ! Object.is( value, b[ l ][ i ] ) ) ) throw new Error( `${ label }: level ${ l } differs` ); } );
	const plain = new Downsampler( device, { tail: false } ), tail = new Downsampler( device );
	const levelsPerDispatch = plain.levelsPerDispatch;
	const run = async ( owner, texture, data, options, record = 'encode', from = 1 ) => {

		if ( data ) upload( device, texture, data );
		poison( device, texture, from );
		const encoder = device.createCommandEncoder();
		if ( record === 'pass' ) {

			const pass = encoder.beginComputePass();
			owner.encodePass( pass, texture, options );
			pass.end();

		} else owner.encode( encoder, texture, options );
		device.queue.submit( [ encoder.finish() ] );
		return readMips( device, texture, from );

	};

	// 'box' plans the batches of 'exact' (with and without the tail); at even sizes it is 'exact'
	// bit for bit, and its folds are the 'exact' folds at any size.
	const policy = createTexture( device, { width: 133, height: 75, format: 'rgba16float' } );
	const policyData = makeData( 133, 75, 1, 'rgba16float', 'random', 21 );
	for ( const owner of [ plain, tail ] ) {

		const shape = options => owner.prepare( policy, options ).map( batch => `${ batch.kernel }:${ batch.firstMipLevel }:${ batch.count }` ).join();
		if ( shape( { oddSizes: 'box' } ) !== shape( {} ) ) throw new Error( `box: batches differ from 'exact' (tail ${ owner.tail })` );

	}

	for ( const reduction of [ 'max', 'minmax', { wgsl: preferHigh.wgsl } ] ) same( await run( plain, policy, policyData, { reduction } ), await run( plain, policy, policyData, { reduction, oddSizes: 'box' } ), `box ${ reduction.wgsl ? 'custom' : reduction } = exact` );
	const even = createTexture( device, { width: 256, height: 128, format: 'rgba8unorm' } );
	const evenData = makeData( 256, 128, 1, 'rgba8unorm', 'random', 22 );
	same( await run( plain, even, evenData, {} ), await run( plain, even, evenData, { oddSizes: 'box' } ), 'box at a power of two = exact' );
	even.destroy();
	results.push( { option: 'box: exact batches, exact folds, exact at powers of two' } );

	// encodePass, the buffer output path and the compatibility path record the same bits as
	// encode for 'box' and its tail (rgba16float: every path rounds with ds_round_f16).
	const buffer = ( policy.usage & GPUTextureUsage.COPY_DST ) !== 0 ? { plain: new Downsampler( device, { output: 'buffer', levelsPerDispatch, tail: false } ), tail: new Downsampler( device, { output: 'buffer', levelsPerDispatch } ) } : null;
	const compat = compatibility( device ) ? null : { plain: new Downsampler( device, { compatibility: true, levelsPerDispatch, tail: false } ), tail: new Downsampler( device, { compatibility: true, levelsPerDispatch } ) };
	for ( const [ owner, kind ] of [ [ plain, 'plain' ], [ tail, 'tail' ] ] ) {

		const options = { oddSizes: 'box' };
		const reference = await run( owner, policy, policyData, options );
		same( reference, await run( owner, policy, policyData, options, 'pass' ), `box ${ kind } encodePass` );
		if ( buffer ) same( reference, await run( buffer[ kind ], policy, policyData, options ), `box ${ kind } buffer output` );
		if ( compat ) same( reference, await run( compat[ kind ], policy, policyData, options, 'pass' ), `box ${ kind } compatibility path` );

	}

	if ( ! tail.prepare( policy, { oddSizes: 'box' } ).some( batch => batch.kernel === 'tail' ) ) throw new Error( 'box: expected a tail batch for 133×75' );
	policy.destroy();
	results.push( { option: 'box: encodePass, buffer output and compatibility path match encode' } );

	// Resample: API. Without 'area' another size is rejected and the reason names the option;
	// unknown values and ratios over maxResampleRatio are rejected.
	const hiZ = createTexture( device, { width: 64, height: 32, format: 'r32float' } );
	const depth = await createDepthSource( device, { width: 120, height: 67 }, makeData( 120, 67, 1, 'r32float', 'random', 23, { unit: true } ) );
	const reasons = [
		[ { texture: depth.texture }, 'resample \'area\'' ],
		[ depth.texture, 'resample \'area\'' ],
		[ { texture: depth.texture, resample: 'bilinear' }, 'unknown' ],
		[ { texture: depth.texture, resample: 'area' }, null ],
		[ { texture: depth.texture, resample: 'area', mipLevel: 0 }, null ]
	];
	for ( const [ source, message ] of reasons ) {

		const reason = plain.sourceUnsupportedReason( hiZ, source );
		if ( message === null ? reason !== null : ! reason?.includes( message ) ) throw new Error( `resample: expected ${ message === null ? 'acceptance' : `"${ message }"` }, got ${ reason }` );
		if ( message !== null ) expectThrow( () => plain.prepare( hiZ, { source, reduction: 'max' } ), message, `resample ${ message }` );

	}

	const tiny = createTexture( device, { width: 8, height: 8, format: 'r32float' } );
	const wide = await createDepthSource( device, { width: 8 * maxResampleRatio + 1, height: 8 }, makeData( 8 * maxResampleRatio + 1, 8, 1, 'r32float', 'random', 24, { unit: true } ) );
	expectThrow( () => plain.prepare( tiny, { source: { texture: wide.texture, resample: 'area' }, reduction: 'max' } ), `more than ${ maxResampleRatio }×`, 'resample ratio' );
	expectThrow( () => plain.prepare( hiZ, { source: { texture: depth.texture, resample: 'area' }, reduction: 'max', oddSizes: 'spd' } ), '\'zero-weight\'', 'resample spd max' );
	// The cap is an option (validation only: the kernel loops over each texel's span at run time).
	if ( plain.maxResampleRatio !== maxResampleRatio ) throw new Error( `resample: expected maxResampleRatio ${ maxResampleRatio } by default, got ${ plain.maxResampleRatio }` );
	for ( const ratio of [ 0.5, NaN, Infinity, '8' ] ) expectThrow( () => new Downsampler( device, { maxResampleRatio: ratio } ), 'maxResampleRatio', `maxResampleRatio ${ ratio }` );
	const raised = new Downsampler( device, { maxResampleRatio: 17 } );
	if ( raised.sourceUnsupportedReason( tiny, { texture: wide.texture, resample: 'area' } ) !== null ) throw new Error( 'resample: maxResampleRatio 17 must accept a 65 → 8 step' );
	const narrow = createTexture( device, { width: 4, height: 4, format: 'r32float' } );
	for ( const reduction of [ 'max', 'average' ] ) {

		const levels = await run( raised, narrow, null, { source: { texture: wide.texture, resample: 'area' }, reduction }, 'encode', 0 );
		check( levels[ 0 ], areaReference( splat( wide.floats ), wide.texture.width, 8, 1, 'float32', reduction, 4, 4 ), 'float32', `resample ${ reduction } at ratio 16.25 (maxResampleRatio 17)`, reduction, 1 );

	}

	tiny.destroy(); wide.texture.destroy(); narrow.destroy();
	results.push( { option: 'resample: rejections, maxResampleRatio' } );

	// Resample: a power-of-two Hi-Z from a render-size depth buffer takes 2 dispatches at 6 levels
	// per dispatch, and encodePass / compatibility path / buffer path match encode. A new depth
	// texture of another size with the same options (a resize) replaces the cached plan.
	const big = quick ? [ 240, 135, 128, 64 ] : [ 1920, 1080, 1024, 512 ];
	const pyramid = createTexture( device, { width: big[ 2 ], height: big[ 3 ], format: 'r32float' } );
	const frame = await createDepthSource( device, { width: big[ 0 ], height: big[ 1 ] }, makeData( big[ 0 ], big[ 1 ], 1, 'r32float', 'random', 25, { unit: true } ) );
	const hiZOptions = { source: { texture: frame.texture, resample: 'area' }, reduction: 'min' };
	const batches = plain.prepare( pyramid, hiZOptions );
	// Reduction dispatches; compatibility mode adds its copy dispatches (`copy`) between batches.
	const reductions = batches.filter( batch => ! batch.copy ).length;
	if ( levelsPerDispatch === 6 && reductions !== Math.ceil( pyramid.mipLevelCount / 6 ) ) throw new Error( `resample: expected ${ Math.ceil( pyramid.mipLevelCount / 6 ) } reduction dispatches, got ${ reductions }` );
	const reference = await run( plain, pyramid, null, hiZOptions, 'encode', 0 );
	check( reference[ 0 ], areaReference( splat( frame.floats ), big[ 0 ], big[ 1 ], 1, 'float32', 'min', big[ 2 ], big[ 3 ] ), 'float32', 'resample hi-z mip 0', 'min', 1 );
	same( reference, await run( plain, pyramid, null, hiZOptions, 'pass', 0 ), 'resample encodePass' );
	same( reference, await run( tail, pyramid, null, hiZOptions, 'encode', 0 ), 'resample tail' );
	if ( compat ) same( reference, await run( compat.plain, pyramid, null, hiZOptions, 'pass', 0 ), 'resample compatibility path' );
	if ( buffer ) same( reference, await run( buffer.plain, pyramid, null, hiZOptions, 'encode', 0 ), 'resample buffer output' );
	const resized = await createDepthSource( device, { width: big[ 0 ] * 2 / 3 | 0, height: big[ 1 ] * 2 / 3 | 0 }, makeData( big[ 0 ] * 2 / 3 | 0, big[ 1 ] * 2 / 3 | 0, 1, 'r32float', 'random', 26, { unit: true } ) );
	const after = await run( plain, pyramid, null, { ...hiZOptions, source: { texture: resized.texture, resample: 'area' } }, 'encode', 0 );
	check( after[ 0 ], areaReference( splat( resized.floats ), resized.texture.width, resized.texture.height, 1, 'float32', 'min', big[ 2 ], big[ 3 ] ), 'float32', 'resample after a resize mip 0', 'min', 1 );
	// A same-size source with resample 'area' is a copy.
	const copy = createTexture( device, { width: 120, height: 67, format: 'r32float' } );
	const copied = await run( plain, copy, null, { source: { texture: depth.texture, resample: 'area' }, reduction: 'max' }, 'encode', 0 );
	check( copied[ 0 ], copyReference( splat( depth.floats ), 120, 67, 1, 'float32' ), 'float32', 'resample same size', 'max', 1 );
	for ( const t of [ hiZ, depth.texture, pyramid, frame.texture, resized.texture, copy ] ) t.destroy();
	results.push( { option: `resample: ${ batches.length } dispatches for a ${ big[ 2 ] }×${ big[ 3 ] } Hi-Z; encodePass, tail, compatibility path, buffer output and resizes` } );
	return results;

}
