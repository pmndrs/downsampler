// GPU correctness suite. Every texel of every generated level is checked against
// the CPU reference, computed from the GPU's own previous level so permitted
// rounding differences do not accumulate along the chain.
import { areaReference, boxReference, bufferWritable, copyReference, formats, maxViolations, fromHalf, isIntegerDomain, isIntQuantization, isPackedQuantization, levelSize, pyramidTolerance, quantize, reducePyramid, reduceReference, sampledChannels, spdReference, storageWritable, toHalf, tolerance } from '/dist/index.js';
import { Downsampler, Reducer, requiredLimits } from '/dist/webgpu/index.js';
import * as three from '/dist/three/index.js';
import { createDepthSource, createTexture, makeData, poison, readMip, readMips, toFloats, trackBuffers, upload } from './gpu.js';
import { validateBox } from './validate-box.js';
import { validateMipmap } from './validate-mipmap.js';
import { validateInteger } from './validate-integer.js';
import { validatePacked } from './validate-packed.js';
import { validateScalar } from './validate-scalar.js';
import { validatePayload } from './validate-payload.js';
import { validateRegion } from './validate-region.js';
import { validateSrgb } from './validate-srgb.js';
import { validateWide } from './validate-wide.js';
import { assignShards, digest, Selection } from '../ci/shard.js';

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
export const familyFormats = [ 'rgba8unorm', 'rgba8snorm', 'rgba16unorm', 'rgba16snorm', 'rgba16float', 'rgba32float', 'rgb10a2unorm' ];
export const subsetShapes = [
	[ 1, 1 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 65, 129 ], [ 1, 128 ], [ 128, 8 ], [ 192, 108 ], [ 512, 32 ],
	[ 17, 11, 3 ], [ 17, 9, 1, '2d-array' ]
];

// Formats whose buffer output path (SPEC §6.1) runs the subset shapes besides the packed
// float formats: each family's 4-channel format, plus distinct word layouts (bgra byte
// order, two 16-bit floats in one word, one and two float32 words).
export const bufferFormats = [ ...familyFormats, 'bgra8unorm', 'rg16float', 'r32float', 'rg32float' ];

const patterns = [ 'constant', 'random', 'edge', 'tiles' ];

// Custom fold used to test the hook: per channel, the smallest value >= 0.5, else
// the smallest value. A min under a total order, so associative, commutative and
// idempotent, and sensitive to the value domain (it breaks if a kernel passes
// 0–255 units instead of normalized values).
export const preferHigh = {
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
	{ name: 'custom', reduction: { wgsl: preferHigh.wgsl }, reference: preferHigh.reference, patterns: patterns.slice( 1 ) },
	// oddSizes: 'spd' (SPD semantics, 'average' only: 'clamp' tile batches, ⌊n/2⌋
	// sizes), checked per batch against `spdReference` from the GPU's own batch source level.
	{ name: 'spd average', reduction: 'average', oddSizes: 'spd', patterns },
	{ name: 'spd average (load)', reduction: 'average', oddSizes: 'spd', access: 'load', patterns, filterableOnly: true },
	// The exact tail kernel ('exact' results, fewer batches). Only shapes whose
	// plan has a tail batch run these; elsewhere the plan is the 'exact' one.
	{ name: 'tail average', reduction: 'average', tail: true, patterns },
	{ name: 'tail average (load)', reduction: 'average', tail: true, access: 'load', patterns, filterableOnly: true },
	{ name: 'tail max', reduction: 'max', tail: true, patterns: patterns.slice( 1 ) },
	{ name: 'tail minmax', reduction: 'minmax', tail: true, patterns: patterns.slice( 1 ) },
	{ name: 'tail custom', reduction: { wgsl: preferHigh.wgsl }, reference: preferHigh.reference, tail: true, patterns: patterns.slice( 1 ) },
	// oddSizes: 'box' (area-weighted odd sizes, M2.5), each level against `boxReference` from the
	// GPU's previous level; on the box formats only (every quantization path, two packed, r32float).
	// Batches reading an odd level load; even ones sample, as 'exact' does. 'box max' must equal
	// the 'exact' folds (the same footprint).
	{ name: 'box average', reduction: 'average', oddSizes: 'box', patterns, box: true },
	{ name: 'box average (load)', reduction: 'average', oddSizes: 'box', access: 'load', patterns, filterableOnly: true, box: true },
	{ name: 'box max', reduction: 'max', oddSizes: 'box', patterns: patterns.slice( 1 ), box: true },
	{ name: 'tail box average', reduction: 'average', oddSizes: 'box', tail: true, patterns, box: true }
];

/** Formats the 'box' variants run on: each quantization family's 4-channel format, the packed floats and r32float. */
export const boxFormats = [ ...familyFormats, 'rgb9e5ufloat', 'rg11b10ufloat', 'r32float', 'rgba8unorm-srgb' ];

const mipSize = ( size, level ) => Math.max( 1, size >> level );

export function expectThrow( run, message, label ) {

	try { run(); } catch ( error ) {

		if ( ! String( error.message ).includes( message ) ) throw new Error( `${ label }: expected "${ message }", got "${ error.message }"` );
		return;

	}

	throw new Error( `${ label }: expected an error "${ message }"` );

}

/**
 * Levels of a plan that need the declared tolerance instead of a bit-exact
 * match (normalized and packed formats): each batch's first output, and in a
 * tail batch every level reduced from an odd, non-unit size (shader bilinear
 * weights), or every tail level for packed formats (the tail's f32 pairing
 * differs from the oracle's float32 pairing). Compatibility copies are skipped.
 */
function toleratedLevels( batches, width, height, packed ) {

	const levels = new Set();
	for ( const batch of batches ) {

		if ( batch.copy ) continue;
		levels.add( batch.firstMipLevel );
		if ( batch.kernel !== 'tail' ) continue;
		for ( let level = batch.firstMipLevel + 1; level < batch.firstMipLevel + batch.count; level ++ ) {

			const w = mipSize( width, level - 1 ), h = mipSize( height, level - 1 );
			if ( packed || ( w > 1 && w % 2 === 1 ) || ( h > 1 && h % 2 === 1 ) ) levels.add( level );

		}

	}

	return levels;

}

/**
 * Compares the stored channels of every texel; the rest are not written.
 * `slack` is a relative allowance for levels read directly from a unorm source:
 * unorm → float conversion needn't be correctly rounded (SwiftShader's
 * depth16unorm differs from k / 65535 by an ulp), so even exact folds can see it.
 */
export function check( actual, expected, quantization, label, reduction = 'average', channels = 4, slack = 0, exact = false ) {

	let maxError = 0, firstFailure = null, failing = 0;
	const packed = isPackedQuantization( quantization );

	for ( let i = 0; i < expected.values.length; i ++ ) {

		if ( i % 4 >= channels ) continue;
		const error = Math.abs( actual[ i ] - expected.values[ i ] );
		// Packed formats: the step depends on the texel (rgb9e5's shared exponent) or the channel (rg11b10's 10-bit b).
		// The largest channel may be off by its own filter allowance, and so cross an exponent: take its upper bound.
		const t = i - i % 4;
		const upper = c => expected.values[ t + c ] + expected.filterTolerance[ t + c ];
		// The channel also sets rgb10a2unorm's step (2-bit alpha) and sRGB's (linear alpha).
		const texel = packed || quantization === 'srgb8' ? { channel: i % 4, texelMax: Math.max( upper( 0 ), upper( 1 ), upper( 2 ) ) } : { channel: i % 4, texelMax: 0 };
		const limit = exact ? 0 : tolerance( expected.values[ i ], quantization, expected.filterTolerance[ i ], reduction, texel ) + Math.abs( expected.values[ i ] ) * slack;
		if ( ! Number.isFinite( actual[ i ] ) || error > limit ) {

			const message = `${ label }: value ${ i }, actual ${ actual[ i ] }, expected ${ expected.values[ i ] }, error ${ error } > ${ limit }`;
			if ( ! failures ) throw new Error( message );
			firstFailure ??= message;
			failing ++;

		}
		maxError = Math.max( maxError, error );

	}

	if ( firstFailure ) failures.push( { label, first: firstFailure, failing, values: expected.values.length, maxError } );

	return maxError;

}

/**
 * --keep-going: failing checks are collected here instead of thrown (one entry per check: its
 * first failing value and how many failed), so one run maps every failure on a new device.
 * Null (the default): the first failing value throws.
 */
let failures = null;

/** Kernel widths to cover: single-level, 4 (default limit), 5 (16×16 tiles) and 6 (SPD tiles). */
export function levelOptions( device ) {

	const limit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	return [ 1, 4, 5, 6 ].filter( levels => levels <= limit );

}

/** A compatibility-mode device: no 'core-features-and-limits' (SPEC §4 Compatibility mode). */
export const compatibility = device => ! device.features.has( 'core-features-and-limits' );

/**
 * `only`: run one part of the suite; default all. The parts 'downsampler', 'reducer', 'scalar' and 'fixtures'
 * (the tool fixtures' oracle checks) partition it; 'mips' is 'downsampler' plus 'reducer'. 'packed' runs the packed float
 * formats and the buffer output path ('packed-source' and 'packed-api' one piece of it, for debugging).
 * 'odd' runs only the 'spd' and tail variants of the main matrix (part of 'downsampler', for debugging),
 * 'box' the main matrix's 'box' variants and validate-box.js (M2.5; also part of 'downsampler').
 * 'region' runs only the partial updates (`rect`, `layers`; validate-region.js; part of 'downsampler').
 * 'mipmap' runs only the `downsampler/mipmap` entry's chains against a Downsampler's (validate-mipmap.js; part of 'downsampler').
 * 'srgb' runs the sRGB formats of the main matrix, sRGB separate sources and validate-srgb.js (M2.3.5; part of 'downsampler').
 * 'integer' runs only the integer formats (validate-integer.js; part of 'downsampler'), and 'format:<name>'
 * the main matrix of one format (for debugging and mutation checks).
 *
 * `shard` ({ index, total }, 1-based) runs one shard of the case groups (after `only`), split by
 * `durations` (group id → ms; test/ci/README.md). `list` runs nothing and returns the group ids
 * (`ids`). The result carries `groups` (id → { ms, cases } for the groups that ran) and the
 * listing's size and digest, so the shards of one suite can be checked against each other.
 */
export async function validate( device, { onProgress = () => {}, quick = false, only, shard, durations = {}, list = false, keepGoing = false } = {} ) {

	failures = keepGoing ? [] : null;
	let select = new Selection();
	let shards = null;

	if ( shard || list ) {

		// A first pass registers every group id and runs nothing (no textures, pipelines or oracles).
		const listing = new Selection( { list: true } );
		await suite( device, { quick, only, select: listing } );
		if ( list ) return { listed: true, ids: listing.ids, digest: digest( listing.ids ) };
		shards = assignShards( listing.ids, shard.total, durations );
		select = new Selection( { assigned: shards[ shard.index - 1 ] } );
		select.listing = listing.ids;

	}

	// A group that throws outside check() (a direct assertion, a lost device) is recorded too.
	if ( keepGoing ) select.onError = ( id, error ) => failures.push( { label: id, first: `${ id } threw: ${ error.message }`, thrown: true } );
	const report = await suite( device, { onProgress, quick, only, select } );
	const ids = select.listing ?? select.ids;
	const failed = failures ?? [];
	failures = null;
	return { ...report, passed: failed.length === 0, failures: failed, shard: shard ? { ...shard, groups: shards[ shard.index - 1 ].length } : null, listing: { groups: ids.length, digest: digest( ids ), ids }, groups: select.groups };

}

// --quick checks the tool fixtures at an even and an odd render size (the full suite: all five).
const quickFixtureSizes = [ [ 960, 540 ], [ 1129, 635 ] ];

/** Variant names as id segments: 'spd average (load)' → 'spd-average-load'. */
const slug = name => name.replace( /[^a-z0-9]+/gi, '-' ).replace( /^-|-$/g, '' );

async function suite( device, { onProgress = () => {}, quick = false, only, select } ) {

	const results = [];
	const started = performance.now();
	const cases = only === 'scalar' || only === 'reducer' || only === 'fixtures' || only === 'region' || only === 'mipmap' || only?.startsWith( 'integer' ) ? [] : quick ? shapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : shapes;
	let scalarPrecision = null;
	let runs = 0;
	const skipped = [];
	device.pushErrorScope( 'validation' );

	try {

		// Storage path: kernel widths up to the device's storage texture limit. Buffer
		// path: every width the workgroup size allows (5 and 6 levels take 256
		// invocations), since it binds one buffer however many levels it writes.
		const downsamplerSet = ( levels, output ) => ( {
			levels, output,
			sample: new Downsampler( device, { levelsPerDispatch: levels, output, tail: false } ),
			load: new Downsampler( device, { levelsPerDispatch: levels, averageAccess: 'load', output, tail: false } ),
			tailsample: new Downsampler( device, { levelsPerDispatch: levels, output, tail: true } ),
			tailload: new Downsampler( device, { levelsPerDispatch: levels, averageAccess: 'load', output, tail: true } )
		} );
		const storageSets = levelOptions( device ).map( levels => downsamplerSet( levels, 'storage' ) );
		const bufferSets = [ 1, 4, 5, 6 ].filter( levels => levels <= 4 || device.limits.maxComputeInvocationsPerWorkgroup >= 256 ).map( levels => downsamplerSet( levels, 'buffer' ) );
		const subset = subsetShapes.filter( shape => cases.some( c => c.join() === shape.join() ) );

		for ( const format of Object.keys( formats ) ) {

			const { quantization, channels, filterable, feature, coreStorage } = formats[ format ];
			const packed = isPackedQuantization( quantization ), srgb = quantization === 'srgb8';
			if ( only === 'srgb' && ! srgb ) continue;
			// Compatibility mode has no bgra8unorm-srgb textures at all.
			if ( format === 'bgra8unorm-srgb' && compatibility( device ) ) { skipped.push( 'bgra8unorm-srgb (no such texture in compatibility mode)' ); continue; }
			if ( only?.startsWith( 'packed' ) && ( ! packed || only !== 'packed' ) ) continue;
			const int = isIntQuantization( quantization );
			if ( only?.startsWith( 'integer' ) && ( ! int || only !== 'integer' ) ) continue;
			// 'format:<name>': one format's main matrix alone (for debugging and mutation checks).
			if ( only?.startsWith( 'format:' ) && format !== only.slice( 7 ) ) continue;
			const compatNoStorage = coreStorage === true && compatibility( device );
			const storagePath = storageWritable( format, device.features ) && ! compatNoStorage;
			// Without storage, packed and sRGB formats, rgb10a2unorm (tier1) and rg32float in compatibility mode
			// fall back to the buffer path ('auto' takes it for them); the rest are skipped.
			const fallback = packed || srgb || compatNoStorage || ( ( format === 'rgb10a2unorm' || int ) && bufferWritable( format ) );
			if ( compatNoStorage ) {

				skipped.push( `${ format } storage path (no storage in compatibility mode)` );
				// The Downsampler names the reason (a texture without storage usage can still be created).
				const probe = device.createTexture( { size: [ 1, 1 ], format, usage: GPUTextureUsage.TEXTURE_BINDING } );
				if ( ! new Downsampler( device ).unsupportedReason( probe )?.includes( 'compatibility mode' ) ) throw new Error( `compat: Downsampler does not reject ${ format }, which has no storage in compatibility mode` );
				probe.destroy();

			} else if ( ! storagePath ) {

				if ( formats[ format ].storage !== false ) skipped.push( `${ format }${ fallback ? ' storage path' : '' } (needs ${ feature })` );
				if ( ! fallback ) continue;

			}

			// Integer formats have their own matrix and exact oracle (validate-integer.js; --only=integer), which
			// takes the buffer path where they have no storage. Missing paths are reported as skipped above.
			if ( int ) continue;

			// --quick (CI, SwiftShader) runs the 'spd' variants on the family formats and
			// the tail on rgba8unorm and rgba16float (integer and float16 paths), with two
			// patterns. SwiftShader takes 30 s or more to compile a 32×32-tile pipeline.
			const odd = variant => variant.oddSizes !== undefined || variant.tail === true;
			const quickOdd = variant => quick && odd( variant );
			const quickFormats = variant => variant.tail ? [ 'rgba8unorm', 'rgba16float' ] : [ ...familyFormats, 'rgba8unorm-srgb' ];
			const formatVariants = variants.filter( variant => ( filterable || ! variant.filterableOnly ) && ( only !== 'odd' || odd( variant ) ) && ! ( quickOdd( variant ) && ! quickFormats( variant ).includes( format ) ) && ( ! variant.box || boxFormats.includes( format ) ) && ( only !== 'box' || variant.box ) );
			const family = familyFormats.includes( format ) || packed || format === 'rgba8unorm-srgb';
			const storageShapes = ! storagePath ? [] : family ? cases : subset;
			// The buffer path runs every shape when it is a packed format's only path (rgb9e5ufloat;
			// rg11b10ufloat without tier1), else the subset, for each family format plus a few
			// others with a distinct word layout (bgra byte order, two 16-bit channels, 1–2 float words).
			// --quick (CI, SwiftShader, where the packed encoders are slow): the buffer path runs
			// the subset shapes at widths 1, 4 and 6, for the fallback formats only.
			const bufferShapes = ( packed || srgb ) && ! storagePath && ! quick ? cases : fallback || ( bufferFormats.includes( format ) && ! quick ) ? subset : [];
			const formatBufferSets = quick ? bufferSets.filter( set => set.levels !== 5 ) : bufferSets;
			const formatCases = storageShapes === cases || bufferShapes === cases ? cases : subset;
			for ( const [ shapeIndex, shape ] of formatCases.entries() ) {

				const [ width, height, layers = 1, dimension ] = shape;
				const onShape = shapes => shapes.some( s => s.join() === shape.join() );
				const sets = [ ...( onShape( storageShapes ) ? storageSets : [] ), ...( onShape( bufferShapes ) ? formatBufferSets : [] ) ];
				// Case groups: format × output path × kernel width × variant, over every shape
				// (a group's pipelines are its own, so shards don't compile each other's).
				const selected = sets.flatMap( set => formatVariants.map( variant => [ set, variant, `downsampler/${ format }/${ set.output }/L${ set.levels }/${ slug( variant.name ) }` ] ) ).filter( ( [ , , id ] ) => select.has( id ) );
				if ( selected.length === 0 ) continue;
				const texture = createTexture( device, { width, height, layers, format, dimension } );
				// Inputs and first-level references depend only on shape, format,
				// pattern and reduction, so they are shared across kernel widths.
				const inputs = new Map(), firstLevels = new Map();

				try {

					for ( const [ { levels, output, ...byAccess }, variant, id ] of selected ) results.push( ...await select.run( id, async () => {

						const downsampler = byAccess[ `${ variant.tail ? 'tail' : '' }${ variant.access ?? 'sample' }` ];
						const options = { reduction: variant.reduction, oddSizes: variant.oddSizes };
						// Tail variants run where the plan has a tail batch. --quick skips 32×32-tile
						// tails: SwiftShader takes minutes to compile each of those pipelines.
						const tails = variant.tail ? downsampler.prepare( texture, options ).filter( batch => batch.kernel === 'tail' ) : [];
						if ( variant.tail && ( tails.length === 0 || ( quick && tails.some( batch => batch.tile > 16 ) ) ) ) return [];
						const reference = variant.reference ?? variant.reduction;
						const result = { width, height, layers, format, output, levelsPerDispatch: levels, reduction: variant.name, dispatches: 0, maxError: 0 };

						// Reusing one texture with changing input exercises the cached plans.
						for ( const pattern of quickOdd( variant ) ? variant.patterns.filter( pattern => pattern === 'random' || pattern === 'edge' ) : variant.patterns ) {

							if ( ! inputs.has( pattern ) ) {

								const data = makeData( width, height, layers, format, pattern, pattern === 'tiles' ? shapeIndex + 3 : 1 );
								inputs.set( pattern, { data, floats: toFloats( data, format ) } );

							}

							const input = inputs.get( pattern );
							upload( device, texture, input.data );
							poison( device, texture );
							downsampler.generate( texture, options );
							const mips = await readMips( device, texture, 1 );
							const label = `${ width }x${ height }x${ layers } ${ format } ${ output } ${ variant.name } ${ pattern } L${ levels }`;
							// The hardware sampler weights a batch's first output (averages of filterable formats,
							// unless the variant loads); its tolerance models the samplers' 8-bit weights
							// (samplerWeightSteps). Loads and in-tile levels use shader weights: the strict model.
							const sampled = ( variant.access ?? 'sample' ) === 'sample' && filterable;
							const sampledLevels = new Set( sampled ? downsampler.prepare( texture, options ).filter( batch => ! batch.copy ).map( batch => batch.firstMipLevel ) : [] );

							if ( variant.oddSizes === 'spd' ) {

								// Each batch from the level it reads (the input, or the GPU's own).
								for ( const batch of downsampler.prepare( texture, options ).filter( batch => ! batch.copy ) ) {

									const source = batch.sourceMipLevel === 0 ? input.floats : mips[ batch.sourceMipLevel - 1 ];
									const expected = spdReference( source, mipSize( width, batch.sourceMipLevel ), mipSize( height, batch.sourceMipLevel ), layers, quantization, reference, batch.count, { weights: sampled ? 'sampler' : 'shader' } );
									expected.forEach( ( level, i ) => {

										const mip = batch.firstMipLevel + i;
										result.maxError = Math.max( result.maxError, check( mips[ mip - 1 ], level, quantization, `${ label } mip ${ mip }`, reference, channels ) );

									} );

								}

								runs ++;
								continue;

							}

							// Levels reduced in workgroup memory use exact integer math in
							// normalized formats: they must match the oracle bit for bit,
							// which is what tests tie rounding. Only a batch's first output
							// (sampler or float weights) gets the tolerance. Packed float
							// formats are exact there too, against an oracle that sums in
							// float32 in the kernel's order (their sums aren't always exact).
							// A tail's odd steps get the tolerance too (toleratedLevels).
							const firstOutputs = toleratedLevels( downsampler.prepare( texture, options ), width, height, packed || srgb );
							let previous = input.floats, w = width, h = height;

							for ( let level = 1; level < texture.mipLevelCount; level ++ ) {

								const key = `${ variant.name }:${ pattern }`;
								const inTile = ! firstOutputs.has( level );
								let expected = level === 1 ? firstLevels.get( key ) : undefined;
								if ( expected === undefined ) expected = ( variant.oddSizes === 'box' ? boxReference : reduceReference )( previous, w, h, layers, quantization, reference, { ...( ( packed || srgb ) && inTile ? { pairing: 'float32' } : {} ), weights: sampledLevels.has( level ) ? 'sampler' : 'shader' } );
								if ( level === 1 ) firstLevels.set( key, expected );
								const actual = mips[ level - 1 ];
								result.maxError = Math.max( result.maxError, check( actual, expected, quantization, `${ label } mip ${ level }`, reference, channels, 0, ( isIntegerDomain( quantization ) || packed || srgb ) && inTile ) );
								previous = actual; w = expected.width; h = expected.height;

							}

							runs ++;

						}

						result.dispatches = downsampler.prepare( texture, options ).length;
						return [ result ];

					} ) );

					onProgress( `validated ${ width }×${ height }×${ layers } ${ format }` );

				} finally {

					texture.destroy();

				}

			}

		}

		if ( only === 'odd' ) {

			// The main matrix's 'spd' and tail variants only (above).

		} else if ( only === 'box' ) {

			results.push( ...await validateBox( device, { quick, onProgress, select } ) );

		} else if ( only === 'region' ) {

			results.push( ...await validateRegion( device, { quick, onProgress, select } ) );

		} else if ( only === 'mipmap' ) {

			results.push( ...await select.run( 'downsampler/mipmap', () => validateMipmap( device, { quick } ) ) );

		} else if ( only === 'srgb' ) {

			// The main matrix's sRGB formats (above), sRGB separate sources and validate-srgb.js.
			results.push( ...await validateSource( device, { quick, onProgress, select, srgbOnly: true } ) );
			results.push( ...await validateSrgb( device, { quick, onProgress, select } ) );

		} else if ( only?.startsWith( 'format:' ) ) {

			// The main matrix of one format (above).

		} else if ( only?.startsWith( 'integer' ) ) {

			results.push( ...await validateInteger( device, { quick, onProgress, select, part: only } ) );

		} else if ( only?.startsWith( 'packed' ) ) {

			// 'packed-source' and 'packed-api' run one part (for debugging).
			if ( only !== 'packed-api' ) results.push( ...await validateSource( device, { quick, onProgress, select, packedOnly: true } ) );
			if ( only !== 'packed-source' ) results.push( ...await validatePacked( device, { quick, onProgress, select } ) );

		} else if ( only !== 'scalar' && only !== 'fixtures' ) {

			if ( only !== 'reducer' ) {

				results.push( ...await validateSource( device, { quick, onProgress, select } ) );
				results.push( ...await select.run( 'downsampler/options', () => validateOptions( device ) ) );
				results.push( ...await select.run( 'downsampler/mipmap', () => validateMipmap( device, { quick } ) ) );
				results.push( ...await validateRegion( device, { quick, onProgress, select } ) );
				results.push( ...await validatePacked( device, { quick, onProgress, part: 'downsampler', select } ) );
				results.push( ...await validateBox( device, { quick, onProgress, select } ) );
				results.push( ...await validateSrgb( device, { quick, onProgress, select } ) );
				results.push( ...await validateInteger( device, { quick, onProgress, select } ) );

			}

			if ( only !== 'downsampler' ) {

				const reducer = await validateReducer( device, { quick, onProgress, select } );
				results.push( ...reducer );
				runs += reducer.reduce( ( sum, result ) => sum + ( result.runs ?? 0 ), 0 );
				results.push( ...await validateTools( device, { quick, onProgress, select } ) );
				results.push( ...await select.run( 'reducer/options', () => validateReducerOptions( device ) ) );
				results.push( ...await validateWide( device, { quick, onProgress, select } ) );
				results.push( ...await validatePayload( device, { quick, onProgress, select } ) );
				results.push( ...await validatePacked( device, { quick, onProgress, part: 'reducer', select } ) );

			}

		}

		if ( only === undefined || only === 'scalar' ) {

			const scalar = await validateScalar( device, { quick, onProgress, select } );
			results.push( ...scalar.results );
			if ( scalar.pairing !== null ) scalarPrecision = { worst: scalar.worst, pairing: scalar.pairing };

		}

		// The tool fixtures' oracle checks (test/tools/README.md; `node test/tools/run.js --check`), one
		// group per size. They make their own core device, so a compatibility-mode run leaves them out.
		if ( ( only === undefined || only === 'fixtures' ) && ! compatibility( device ) ) {

			const { defaultSizes } = await import( '../tools/scene.js' );
			for ( const [ width, height ] of quick ? quickFixtureSizes : defaultSizes ) results.push( ...await select.run( `fixtures/${ width }x${ height }`, async () => {

				const { runTools } = await import( '../tools/harness.js' );
				const report = await runTools( { sizes: [ [ width, height ] ], bench: false, onProgress } );
				const checked = Object.entries( report.results[ 0 ].fixtures ).filter( ( [ , fixture ] ) => fixture.check );
				const failed = checked.filter( ( [ , fixture ] ) => ! fixture.check.ok ).map( ( [ name ] ) => name );
				if ( failed.length ) throw new Error( `tool fixtures ${ width }x${ height }: oracle checks failed: ${ failed.join( ', ' ) }` );
				onProgress( `validated tool fixtures ${ width }×${ height }` );
				return checked.map( ( [ name ] ) => ( { fixture: name, width, height } ) );

			} ) );

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
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'half', reductions: [ 'average', 'min' ] },
	// HDR into the GI formats, on the buffer path (rg11b10ufloat: storage when tier1 is available).
	{ source: 'rgba16float', destination: 'rgb9e5ufloat', relation: 'half', reductions: [ 'average', 'max' ] },
	{ source: 'rgba32float', destination: 'rgb9e5ufloat', relation: 'same', reductions: [ 'average' ] },
	{ source: 'rgba16float', destination: 'rg11b10ufloat', relation: 'half', reductions: [ 'average', 'min' ] },
	// sRGB (M2.3.5): decoded (and snapped to the exact table on loads), reduced in linear,
	// encoded as stored; HDR clamps into it, and an sRGB source converts to linear float.
	{ source: 'rgba8unorm-srgb', destination: 'rgba8unorm-srgb', relation: 'same', reductions: [ 'average', 'max' ] },
	{ source: 'bgra8unorm-srgb', destination: 'bgra8unorm-srgb', relation: 'half', reductions: [ 'average', 'min' ] },
	{ source: 'rgba16float', destination: 'rgba8unorm-srgb', relation: 'half', reductions: [ 'average', 'max' ] },
	{ source: 'rgba8unorm-srgb', destination: 'rgba16float', relation: 'half', reductions: [ 'average' ] }
];

// After the source batch, every batch is the in-place code the main matrix covers
// at 2048², so 512² (multi-batch at every width) stands in for it here.
export const sourceShapes = [
	[ 1, 1 ], [ 2, 2 ], [ 3, 7 ], [ 17, 9 ], [ 64, 64 ], [ 65, 129 ], [ 1, 128 ], [ 255, 257 ], [ 192, 108 ],
	[ 64, 32, 3 ], [ 17, 11, 3 ], [ 512, 512 ], [ 1920, 1080 ]
];

export async function makeSource( device, format, width, height, layers, pattern, seed ) {

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
export function splat( floats ) {

	// The input's array type: integer sources come as Float64Arrays (a Float32Array can't hold every u32).
	const output = new floats.constructor( floats.length );
	for ( let i = 0; i < floats.length; i += 4 ) output.fill( floats[ i ], i, i + 4 );
	return output;

}

async function validateSource( device, { quick, onProgress, select, packedOnly = false, srgbOnly = false } ) {

	const results = [];
	const shapes = quick ? sourceShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : sourceShapes;
	// 'exact', 'spd' ('average' only; a same-size source: an exact copy batch, then
	// clamp batches) and the exact tail (only where the plan has a tail batch).
	const downsamplers = levelOptions( device ).flatMap( levels => {

		const exact = new Downsampler( device, { levelsPerDispatch: levels, tail: false } );
		// 'box': area-weighted odd sizes, folds included (they must match 'exact').
		const modes = [ { downsampler: exact, mode: 'exact' }, { downsampler: exact, mode: 'spd' }, { downsampler: exact, mode: 'box' } ];
		// Tail pipelines compile slowly on SwiftShader; --quick leaves them to the main matrix.
		const tail = new Downsampler( device, { levelsPerDispatch: levels, tail: true } );
		return quick ? modes : [ ...modes, { downsampler: tail, mode: 'tail' }, { downsampler: tail, mode: 'tail box' } ];

	} );

	for ( const { source: sourceFormat, destination: format, relation, reductions } of sourceCases ) {

		if ( packedOnly && ! isPackedQuantization( formats[ format ].quantization ) ) continue;
		if ( srgbOnly && ! [ format, sourceFormat ].some( f => f.endsWith( '-srgb' ) ) ) continue;
		if ( formats[ format ].coreStorage && compatibility( device ) ) continue;
		if ( [ format, sourceFormat ].includes( 'bgra8unorm-srgb' ) && compatibility( device ) ) continue;

		const { quantization, channels } = formats[ format ];

		// One case group per source case (its pipelines), over every shape.
		results.push( ...await select.run( `source/${ sourceFormat }/${ format }/${ relation }`, async () => {

			const results = [];
			for ( const [ shapeIndex, [ width, height, layers = 1 ] ] of shapes.entries() ) {

				const dw = relation === 'same' ? width : Math.max( 1, width >> 1 ), dh = relation === 'same' ? height : Math.max( 1, height >> 1 );
				const destination = createTexture( device, { width: dw, height: dh, layers, format } );

				try {

					for ( const pattern of [ 'random', 'edge', 'tiles' ] ) {

						const source = await makeSource( device, sourceFormat, width, height, layers, pattern, pattern === 'tiles' ? shapeIndex + 3 : 1 );
						const input = sampledChannels( sourceFormat ) === 1 ? splat( source.floats ) : source.floats;

						try {

							for ( const { downsampler, mode } of downsamplers ) for ( const reduction of reductions ) {

								if ( mode === 'spd' && reduction !== 'average' ) continue;
								const box = mode.endsWith( 'box' );
								const options = { source: { texture: source.texture }, reduction, oddSizes: mode === 'spd' ? 'spd' : box ? 'box' : 'exact' };
								if ( mode.startsWith( 'tail' ) &&! downsampler.prepare( destination, options ).some( batch => batch.kernel === 'tail' ) ) continue;
								poison( device, destination, 0 );
								downsampler.generate( destination, options );
								const mips = await readMips( device, destination, 0 );
								const label = `${ sourceFormat } ${ width }x${ height }x${ layers } -> ${ format } ${ relation } ${ reduction } ${ pattern } L${ downsampler.levelsPerDispatch } ${ mode }`;
								const unormSource = /unorm/.test( sourceFormat ) ? 2 ** - 21 : 0;

								if ( mode === 'spd' ) {

									for ( const batch of downsampler.prepare( destination, options ).filter( batch => ! batch.copy ) ) {

										const external = batch.sourceMipLevel === null;
										if ( batch.kernel === 'exact' ) {

											check( mips[ 0 ], copyReference( input, width, height, layers, quantization ), quantization, `${ label } copy`, reduction, channels, unormSource );
											continue;

										}

										const from = external ? input : mips[ batch.sourceMipLevel ];
										const sw = external ? width : mipSize( dw, batch.sourceMipLevel ), sh = external ? height : mipSize( dh, batch.sourceMipLevel );
										spdReference( from, sw, sh, layers, quantization, reduction, batch.count ).forEach( ( level, i ) => {

											const mip = batch.firstMipLevel + i;
											check( mips[ mip ], level, quantization, `${ label } mip ${ mip }`, reduction, channels, external ? unormSource : 0 );

										} );

									}

									continue;

								}

								let previous = input, w = width, h = height;

								for ( let level = 0; level < destination.mipLevelCount; level ++ ) {

									const quantizedInput = level > 0 || formats[ sourceFormat ]?.quantization === quantization;
									const expected = level === 0 && dw === width && dh === height ? copyReference( previous, w, h, layers, quantization ) : ( box ? boxReference : reduceReference )( previous, w, h, layers, quantization, reduction, { quantizedInput } );
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

			return results;

		} ) );

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

	// exactTile: the workgroup shape of 'exact' batches changes no bits. 16 and 32 need
	// 256 invocations, and are skipped on software adapters (slow 32×32-tile compiles).
	{

		const big = device.limits.maxComputeInvocationsPerWorkgroup >= 256 && device.adapterInfo?.isFallbackAdapter !== true;
		for ( const bad of [ 4, 12, 64, ...( device.limits.maxComputeInvocationsPerWorkgroup >= 256 ? [] : [ 16, 32 ] ) ] ) {

			let threw = false;
			try { new Downsampler( device, { exactTile: bad } ); } catch { threw = true; }
			if ( ! threw ) throw new Error( `exactTile ${ bad } was accepted` );

		}

		// Without the tail kernel (default on), so every level comes from an 'exact' batch.
		const untailed = new Downsampler( device, { tail: false } );
		if ( downsampler.tail !== true ) throw new Error( 'the tail kernel is not the default' );
		const tiles = [ 8, ...( big ? [ 16, 32 ] : [] ) ].map( tile => [ tile, new Downsampler( device, { exactTile: tile, tail: false } ) ] );
		for ( const [ width, height, layers, format, options, label ] of [
			[ 333, 200, 3, 'rgba16float', {}, 'average, 3 layers' ],
			[ 1280, 720, 1, 'rgba8unorm', {}, 'average 1280×720' ],
			[ 1917, 1079, 1, 'r32float', { reduction: 'max' }, 'max' ],
			[ 255, 257, 1, 'rgba8unorm', { reduction: 'minmax' }, 'minmax' ],
		] ) {

			const data = makeData( width, height, layers, format, 'random', 12 );
			const reference = createTexture( device, { width, height, layers, format } );
			upload( device, reference, data );
			untailed.generate( reference, options );
			const expected = await readMips( device, reference, 1 );
			reference.destroy();
			for ( const [ tile, owner ] of tiles ) {

				const texture = createTexture( device, { width, height, layers, format } );
				upload( device, texture, data );
				owner.generate( texture, options );
				const actual = await readMips( device, texture, 1 );
				texture.destroy();
				actual.forEach( ( level, l ) => { if ( level.some( ( value, i ) => ! Object.is( value, expected[ l ][ i ] ) ) ) throw new Error( `exactTile ${ tile } ${ label }: level ${ l + 1 } differs from the default (${ untailed.exactTile })` ); } );

			}

		}

		results.push( { option: `exactTile ${ tiles.map( ( [ tile ] ) => tile ).join( '/' ) } match the default (${ untailed.exactTile }); default tail ${ downsampler.tail }` } );

	}

	// The compatibility-mode path (scratch level per batch output, a copy dispatch
	// per level) writes the same bits as the direct path: in place, from a depth
	// source, with layers, through encode and encodePass. On core devices only
	// (a compatibility-mode device rejects the direct path).
	const compat = new Downsampler( device, { compatibility: true } ), direct = new Downsampler( device, { compatibility: false } );
	for ( const [ width, height, layers, format, options, label ] of compatibility( device ) ? [] : [
		[ 333, 200, 3, 'rgba16float', {}, 'average, 3 layers' ],
		[ 255, 257, 1, 'rgba8unorm', { reduction: 'minmax' }, 'minmax' ],
		[ 17, 9, 1, 'rgba8unorm', { baseMipLevel: 1 }, 'sub-range' ],
		[ 1917, 1079, 1, 'r32float', { reduction: 'min' }, 'depth min' ],
	] ) {

		const a = createTexture( device, { width, height, layers, format } ), b = createTexture( device, { width, height, layers, format } );
		const depth = format === 'r32float' ? await createDepthSource( device, { width, height }, makeData( width, height, 1, 'r32float', 'random', 9, { unit: true } ) ) : null;
		const data = makeData( width, height, layers, format, 'random', 13 );
		if ( depth === null ) { upload( device, a, data ); upload( device, b, data ); }
		poison( device, b, depth ? 0 : ( options.baseMipLevel ?? 0 ) + 1 );
		const run = { ...options, ...( depth ? { source: { texture: depth.texture } } : {} ) };
		direct.generate( a, run );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		compat.encodePass( pass, b, run );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );
		const batches = compat.prepare( b, run );
		const written = b.mipLevelCount - ( depth ? 0 : ( options.baseMipLevel ?? 0 ) + 1 );
		if ( batches.filter( batch => batch.copy ).length !== written || batches.some( batch => ! batch.copy && batch.firstMipLevel === undefined ) ) throw new Error( `compat ${ label }: expected one copy per written level (${ written })` );
		const ma = await readMips( device, a ), mb = await readMips( device, b );
		ma.forEach( ( level, l ) => { if ( level.some( ( value, i ) => ! Object.is( value, mb[ l ][ i ] ) ) ) throw new Error( `compat ${ label }: level ${ l } differs from the direct path` ); } );
		compat.release( b );
		for ( const t of [ a, b, depth?.texture ] ) t?.destroy();

	}

	if ( direct.prepare( createTexture( device, { width: 8, height: 8, format: 'rgba8unorm' } ) ).some( batch => batch.copy ) ) throw new Error( 'compat: the direct path has copy dispatches' );
	if ( ! compatibility( device ) ) results.push( { option: 'compatibility path matches the direct path' } );

	// Odd-size policies (M2T.4). Unknown policies, and folds under 'spd', are
	// rejected. 'spd' and the tail record the same bits through encodePass as
	// through encode; on the buffer output path as on the storage path (float16:
	// both round with pack2x16float); and through the compatibility-mode path
	// (scratch levels and copies) as directly. On a compatibility-mode device
	// (fewer than 256 invocations) tail tiles are 8×8.
	// 133×75: the tail starts at 16×9, a 16×16 tile (32×32 tiles compile for minutes on SwiftShader).
	const policy = createTexture( device, { width: 133, height: 75, format: 'rgba16float' } );
	expectThrow( () => downsampler.prepare( policy, { oddSizes: 'fast' } ), 'not implemented', 'oddSizes fast' );
	for ( const reduction of [ 'min', 'max', 'minmax', { wgsl: preferHigh.wgsl } ] ) expectThrow( () => downsampler.prepare( policy, { oddSizes: 'spd', reduction } ), '\'zero-weight\'', `spd ${ typeof reduction === 'string' ? reduction : 'custom' }` );
	const policyData = makeData( 133, 75, 1, 'rgba16float', 'random', 12 );
	const tail = new Downsampler( device, { tail: true } );
	const runs = [ [ downsampler, { oddSizes: 'spd' }, 'spd' ], [ tail, {}, 'tail' ], [ tail, { reduction: 'max' }, 'tail max' ] ];
	const chain = async ( owner, options, record ) => {

		upload( device, policy, policyData );
		poison( device, policy );
		const encoder = device.createCommandEncoder();
		if ( record === 'pass' ) {

			const pass = encoder.beginComputePass();
			owner.encodePass( pass, policy, options );
			pass.end();

		} else owner.encode( encoder, policy, options );
		device.queue.submit( [ encoder.finish() ] );
		return readMips( device, policy, 1 );

	};
	const same = ( a, b, label ) => a.forEach( ( level, l ) => { if ( level.some( ( value, i ) => ! Object.is( value, b[ l ][ i ] ) ) ) throw new Error( `${ label }: level ${ l + 1 } differs` ); } );
	// The same levels per dispatch on every path: 'spd' results depend on where batches end.
	const levelsPerDispatch = downsampler.levelsPerDispatch;
	const buffer = ( policy.usage & GPUTextureUsage.COPY_DST ) !== 0 ? { spd: new Downsampler( device, { output: 'buffer', levelsPerDispatch } ), tail: new Downsampler( device, { output: 'buffer', levelsPerDispatch, tail: true } ) } : null;
	const compatPaths = compatibility( device ) ? null : { spd: new Downsampler( device, { compatibility: true, levelsPerDispatch } ), tail: new Downsampler( device, { compatibility: true, levelsPerDispatch, tail: true } ) };
	for ( const [ owner, options, label ] of runs ) {

		const reference = await chain( owner, options, 'encode' );
		same( reference, await chain( owner, options, 'pass' ), `encodePass ${ label }` );
		const kind = owner === tail ? 'tail' : 'spd';
		if ( buffer ) same( reference, await chain( buffer[ kind ], options, 'encode' ), `buffer output ${ label }` );
		if ( compatPaths ) same( reference, await chain( compatPaths[ kind ], options, 'pass' ), `compatibility path ${ label }` );

	}

	if ( downsampler.prepare( policy, { oddSizes: 'spd' } ).filter( batch => ! batch.copy ).length !== Math.ceil( ( policy.mipLevelCount - 1 ) / downsampler.levelsPerDispatch ) ) throw new Error( 'spd: expected ⌈levels / levelsPerDispatch⌉ dispatches' );
	if ( device.limits.maxComputeInvocationsPerWorkgroup < 256 && tail.prepare( policy ).some( batch => batch.kernel === 'tail' && batch.tile !== 8 ) ) throw new Error( 'tail: tiles must be 8×8 below 256 invocations per workgroup' );
	if ( ! tail.prepare( policy ).some( batch => batch.kernel === 'tail' ) ) throw new Error( 'tail: expected a tail batch for 133×75' );
	tail.release( policy ); buffer?.spd.release( policy ); buffer?.tail.release( policy ); compatPaths?.spd.release( policy ); compatPaths?.tail.release( policy );
	policy.destroy();
	results.push( { option: 'odd-size policies: rejections, encodePass, buffer output, compatibility path' } );
	results.push( ...await validateThree( device ) );
	return results;

}

// Resolved fields and requiredLimits; the three.js adapter against a structural
// stand-in for WebGPURenderer: shared Downsamplers per option set, the reason
// it declines, and recording into the caller's encoder without submitting.
async function validateThree( device ) {

	const adapter = await navigator.gpu.requestAdapter( device.features.has( 'core-features-and-limits' ) ? {} : { featureLevel: 'compatibility' } );
	if ( JSON.stringify( requiredLimits( adapter ) ) !== JSON.stringify( Downsampler.requiredLimits( adapter ) ) || three.requiredLimits !== requiredLimits ) throw new Error( 'requiredLimits: the standalone function, Downsampler.requiredLimits and downsampler/three differ' );
	if ( new Downsampler( device ).averageAccess !== 'sample' || new Downsampler( device, { averageAccess: 'load' } ).averageAccess !== 'load' ) throw new Error( 'Downsampler.averageAccess must read the resolved option (default \'sample\')' );

	const gpuTextures = new Map();
	const renderer = { backend: { isWebGPUBackend: true, device, get: object => ( { texture: gpuTextures.get( object ) } ) } };
	const shared = three.getDownsampler( renderer );
	if ( three.getDownsampler( renderer ) !== shared || three.getDownsampler( renderer, {} ) !== shared ) throw new Error( 'three: getDownsampler must share one Downsampler per device and option set' );
	const load = three.getDownsampler( renderer, { averageAccess: 'load', f16Rounding: 'device' } );
	if ( load === shared || load !== three.getDownsampler( renderer, { f16Rounding: 'device', averageAccess: 'load' } ) || load.averageAccess !== 'load' || load.f16Rounding !== 'device' ) throw new Error( 'three: getDownsampler options must select their own shared Downsampler, whatever the key order' );

	const [ unallocated, single, noStorage, chain ] = [ {}, {}, {}, {} ];
	gpuTextures.set( single, device.createTexture( { size: [ 4, 4 ], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING } ) );
	gpuTextures.set( noStorage, device.createTexture( { size: [ 4, 4 ], format: 'rgba8unorm', mipLevelCount: 3, usage: GPUTextureUsage.TEXTURE_BINDING } ) );
	gpuTextures.set( chain, createTexture( device, { width: 64, height: 48, format: 'rgba8unorm' } ) );
	for ( const [ object, message ] of [ [ unallocated, 'not allocated' ], [ single, 'single mip level' ], [ noStorage, 'STORAGE_BINDING' ] ] ) {

		const reason = three.unsupportedReason( renderer, object );
		if ( ! reason?.includes( message ) ) throw new Error( `three: expected unsupportedReason "${ message }", got ${ reason }` );
		if ( three.generateMipmaps( renderer, object ) !== false ) throw new Error( `three: generateMipmaps must decline (${ message })` );

	}

	if ( three.unsupportedReason( renderer, chain ) !== null ) throw new Error( `three: unexpected reason ${ three.unsupportedReason( renderer, chain ) }` );
	const texture = gpuTextures.get( chain );
	const data = makeData( 64, 48, 1, 'rgba8unorm', 'random', 3 );
	upload( device, texture, data );
	poison( device, texture );
	const poisoned = await readMip( device, texture, 1 );
	const encoder = device.createCommandEncoder();
	if ( three.generateMipmaps( renderer, chain, { encoder, downsampler: { averageAccess: 'load' } } ) !== true ) throw new Error( 'three: generateMipmaps with an encoder declined' );
	if ( ( await readMip( device, texture, 1 ) ).some( ( value, i ) => value !== poisoned[ i ] ) ) throw new Error( 'three: generateMipmaps with an encoder must not submit' );
	device.queue.submit( [ encoder.finish() ] );
	check( await readMip( device, texture, 1 ), reduceReference( toFloats( data, 'rgba8unorm' ), 64, 48, 1, 'unorm8', 'average', { weights: 'shader' } ), 'unorm8', 'three generateMipmaps into an encoder, mip 1' );
	for ( const gpuTexture of gpuTextures.values() ) gpuTexture.destroy();
	return [ { option: 'three adapter: options, encoder, unsupportedReason; requiredLimits; averageAccess' } ];

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
export function reducerData( width, height, layers, pattern, seed = 1 ) {

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

export function checkPyramid( actual, expected, label, { slack = 0, quantization = null } = {} ) {

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

async function validateReducer( device, { quick, onProgress, select } ) {

	const results = [];
	const shapes = quick ? reducerShapes.filter( ( [ w, h ] ) => w * h <= 65536 ) : reducerShapes;
	const widths = [ 1, 4, 5, 6 ];
	// Case groups: edges × variant × kernel width × path (one pipeline per view dimension), over every shape.
	const groupId = ( edges, variant, count, path ) => `reducer/${ edges }/${ slug( variant.name ) }/L${ count }/${ path }`;
	const combos = widths.flatMap( count => [ 'texture', 'hook' ].map( path => [ count, path ] ) );
	const wanted = ( edges, variant ) => combos.filter( ( [ count, path ] ) => select.has( groupId( edges, variant, count, path ) ) );
	const edgeModes = [ 'zero-weight', 'clamp' ];
	const anyWanted = edges => reducerVariants.filter( variant => wanted( edges, variant ).length > 0 ).length > 0;
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

		if ( edgeModes.filter( anyWanted ).length === 0 ) continue;
		const dimension = viewDimension ?? ( layers > 1 ? '2d-array' : '2d' );
		const source = createTexture( device, { width, height, layers, format: 'rgba32float', dimension, storage: false } );
		const destroy = [ source ];

		try {

			for ( const edges of edgeModes ) {

				if ( ! anyWanted( edges ) ) continue;
				// Output textures at exactly each level's size (⌈n/2⌉ under zero-weight).
				const outputs = new Map( [ 1, 2, 3, 4, 5, 6 ].map( level => {

					const texture = device.createTexture( { size: [ levelSize( width, level, edges ), levelSize( height, level, edges ), layers ], format: 'rgba32float', textureBindingViewDimension: dimension, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST } );
					destroy.push( texture );
					return [ level, texture ];

				} ) );
				const tools = new Map();

				for ( const variant of reducerVariants ) for ( const pattern of variant.patterns ) {

					const selected = wanted( edges, variant );
					if ( selected.length === 0 ) continue;
					const data = reducerData( width, height, layers, pattern, width * 7 + height );
					upload( device, source, data );
					const expected = reducePyramid( data, width, height, layers, { reduction: variant.reference ?? variant.reduction, levels: [ 1, 2, 3, 4, 5, 6 ], edges } );

					for ( const [ count, path ] of selected ) results.push( ...await select.run( groupId( edges, variant, count, path ), async () => {

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

						return [ result ];

					} ) );

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

async function validateTools( device, { quick, onProgress, select } ) {

	const results = [];

	// Shading change: odd sizes drop the last row/column (SPD), partial tiles repeat the edge.
	results.push( ...await select.run( 'tools/fsr-shading-change', async () => {

		const results = [];
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

			if ( mips.mipLevelCount === 3 && compatibility( device ) ) {

				// Compatibility mode: three mips of one texture in one dispatch are rejected up front.
				let error = null;
				try { shading.encode( device.createCommandEncoder(), { width, height, bindGroup, outputs: [ 0, 1, 2 ].map( mipLevel => ( { texture: mips, mipLevel } ) ) } ); } catch ( caught ) { error = caught; }
				if ( ! error?.message.includes( 'compatibility mode' ) ) throw new Error( `compat: the Reducer accepted three mips of one texture (${ error?.message })` );
				results.push( { tool: 'fsr shading change (compat: rejected)', width, height } );

			} else if ( mips.mipLevelCount === 3 ) {

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
		return results;

	} ) );

	// Denoiser weighted mean: sky (depth 1) weighs 0, so sky-only blocks emit 0.
	results.push( ...await select.run( 'tools/denoiser-weighted-mean', async () => {

		const results = [];
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
		return results;

	} ) );

	results.push( ...await validateCrossLevel( device, { quick, select } ) );
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

async function validateCrossLevel( device, { quick, select } ) {

	const results = [];
	const entries = [
		{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
	];

	// Tiles of 8 ([ 2, 3 ], [ 1, 4 ]) and 16 ([ 2, 5 ]) level-1 texels; a 6-level tile leaves no workgroup memory for vec4f state.
	for ( const [ shallow, deep ] of [ [ 2, 3 ], [ 1, 4 ], [ 2, 5 ] ] ) for ( const edges of [ 'clamp', 'zero-weight' ] ) results.push( ...await select.run( `tools/cross-level/${ shallow }-${ deep }/${ edges }`, async () => {

		const results = [];
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

		return results;

	} ) );

	return results;

}

async function validateReducerOptions( device ) {

	const results = [];

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
	if ( compatibility( device ) ) expectThrow( () => textureReducer( 'clamp' ).encode( device.createCommandEncoder(), { source, outputs: mipOutputs } ), 'compatibility mode', 'compat: two mips of one texture' );
	else textureReducer( 'clamp' ).encode( device.createCommandEncoder(), { source, outputs: mipOutputs } );
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
	results.push( ...await validateReducerLifetime( device ) );
	return results;

}

// Buffers a Reducer creates (size uniforms, buffer-output buffers) are destroyed
// by release() and, deferred until the queue is done, when the 16-entry cache
// evicts them; a dispatch recorded before its entry was evicted still submits.
async function validateReducerLifetime( device ) {

	const results = [];
	const tracker = trackBuffers( device );
	try {

		const hooked = new Reducer( device, { reduction: 'sum', levels: [ 1 ], load: 'fn load(texel: vec2i, layer: u32) -> vec4f { return vec4f(1.0); }', store: 'fn store(level: u32, texel: vec2i, layer: u32, value: vec4f) {}', label: 'lifetime hooked' } );
		const first = hooked.prepare( { width: 1, height: 1 } );
		if ( first.pipeline !== hooked.pipeline || first.bindGroup !== hooked.bindGroup( { width: 1, height: 1 } ) || first.toolBindGroup !== undefined || first.workgroups.join() !== '1,1,1' || first.buffer !== undefined ) throw new Error( 'reducer prepare(): unexpected pipeline, bind group, tool group, workgroups or buffer' );
		// Recorded now, submitted after its entry is evicted.
		device.pushErrorScope( 'validation' );
		const early = device.createCommandEncoder();
		hooked.encode( early, { width: 1, height: 1 } );
		for ( let width = 2; width <= 17; width ++ ) hooked.prepare( { width, height: 1 } );
		device.queue.submit( [ early.finish() ] );
		const error = await device.popErrorScope();
		if ( error !== null ) throw new Error( `reducer: a dispatch recorded before its cache entry was evicted failed to submit: ${ error.message }` );
		await tracker.settle();
		const sizes = tracker.labelled( 'lifetime hooked size' );
		if ( sizes.length !== 17 ) throw new Error( `reducer: expected 17 size uniforms, got ${ sizes.length }` );
		if ( ! tracker.destroyed.has( sizes[ 0 ] ) || sizes.slice( 1 ).some( buffer => tracker.destroyed.has( buffer ) ) ) throw new Error( 'reducer: eviction must destroy the evicted entry\'s size uniform, and only it' );
		hooked.release();
		if ( sizes.some( buffer => ! tracker.destroyed.has( buffer ) ) ) throw new Error( 'reducer: release() must destroy every size uniform' );

		// Buffer output: release( texture ) destroys the entries that use it, and only them.
		{

			const reducer = new Reducer( device, { reduction: 'average', levels: [ 1, 2 ], format: 'rgba32float', output: 'buffer', label: 'lifetime buffer' } );
			const textures = [ 8, 8, 4, 2 ].map( size => device.createTexture( { size: [ size, size ], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } ) );
			const [ source, other, half, quarter ] = textures;
			const dispatch = { source, outputs: [ half, quarter ] };
			const prepared = reducer.prepare( dispatch );
			reducer.prepare( { source: other, outputs: [ half, quarter ] } );
			const owned = tracker.labelled( 'lifetime buffer output' );
			if ( owned.length !== 4 || prepared.buffer !== owned[ 0 ] || prepared.copies?.length !== 2 ) throw new Error( `reducer buffer output: expected 2 entries of 2 buffers and 2 copies, got ${ owned.length } buffers` );
			reducer.release( other );
			if ( tracker.destroyed.has( owned[ 0 ] ) || tracker.destroyed.has( owned[ 1 ] ) || ! tracker.destroyed.has( owned[ 2 ] ) || ! tracker.destroyed.has( owned[ 3 ] ) ) throw new Error( 'reducer: release( texture ) must destroy only the entries that use the texture' );
			if ( reducer.prepare( dispatch ).bindGroup !== prepared.bindGroup ) throw new Error( 'reducer: release( other ) dropped an unrelated entry' );
			reducer.release( half );
			if ( ! tracker.destroyed.has( owned[ 0 ] ) || ! tracker.destroyed.has( owned[ 1 ] ) ) throw new Error( 'reducer: release( output ) must destroy that entry\'s buffers' );
			for ( const texture of textures ) texture.destroy();

		}

	} finally {

		tracker.restore();

	}

	results.push( { option: 'reducer release, eviction and prepare' } );
	return results;

}
