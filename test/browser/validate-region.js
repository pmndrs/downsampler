// Partial updates (SPEC §5 `rect` and `layers`, §7): a chain is updated after its source
// changed inside a rect (and a layer range), and must then equal
//  1. a full GPU recompute of the changed source, bit for bit, and the CPU oracle per level
//     (from the GPU's own previous level) at the usual tolerance;
//  2. the previous chain outside each level's derived region (and outside the layer range);
//  3. on a poisoned chain, the poison outside the derived regions (no stray writes).
// Updates are applied in sequence to one texture, each from the state the last one left,
// as an app would; the source texels change with writeTexture of the rect alone.
import { areaReference, boxReference, copyReference, formats, reduceReference, sampledChannels, spdReference } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createDepthSource, createTexture, makeData, poison, readMips, toFloats, trackBuffers, upload } from './gpu.js';
import { check, compatibility } from './validate.js';

const mipSize = ( size, level ) => Math.max( 1, size >> level );

// Per-channel smallest value ≥ 0.5, else the smallest (the main matrix's custom fold).
const preferHigh = {
	wgsl: `fn reduce(a: vec4f, b: vec4f) -> vec4f {
	let ha = a >= vec4f(0.5);
	let hb = b >= vec4f(0.5);
	return select(b, a, (ha & !hb) | ((ha == hb) & (a <= b)));
}`,
	reference: ( a, b ) => ( a >= 0.5 ) !== ( b >= 0.5 ) ? ( a >= 0.5 ? a : b ) : Math.min( a, b )
};

const renderSizes = [ [ 1129, 635 ], [ 1920, 1080 ], [ 1477, 831 ] ];
// Unit, odd and prime sizes, 1D tails, partial tiles; 1×1 has no level to write.
const smallShapes = [ [ 1, 1 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 65, 129 ], [ 1, 128 ], [ 128, 1 ], [ 133, 75 ], [ 255, 257 ] ];
const quickShapes = [ [ 1, 1 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 65, 129 ], [ 1, 128 ], [ 133, 75 ] ];

/** Rects at corners, edges, 1×1, the whole level and seeded random places (`count` of them in all). */
function rectsFor( width, height, seed, count = 9 ) {

	let state = seed;
	const next = n => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) % n;
	const fixed = [
		{ x: 0, y: 0, width: 1, height: 1 }, { x: width - 1, y: height - 1, width: 1, height: 1 },
		{ x: width - 1, y: 0, width: 1, height }, { x: 0, y: height - 1, width, height: 1 },
		{ x: Math.floor( width / 3 ), y: Math.floor( height / 3 ), width: Math.max( 1, Math.floor( width / 5 ) ), height: Math.max( 1, Math.floor( height / 7 ) ) },
		{ x: 0, y: 0, width, height }
	];
	const rects = fixed.slice( 0, count );
	while ( rects.length < count ) {

		const x = next( width ), y = next( height );
		rects.push( { x, y, width: 1 + next( Math.min( width - x, 97 ) ), height: 1 + next( Math.min( height - y, 61 ) ) } );

	}

	return rects;

}

/** Values per texel in a `makeData` array. */
const perTexel = ( data, width, height, layers ) => data.length / ( width * height * layers );

/** `data` with `fresh` inside `rect` of layers `range`. */
function merge( data, fresh, width, height, layers, rect, [ first, count ] ) {

	const out = data.slice(), n = perTexel( data, width, height, layers );
	for ( let layer = first; layer < first + count; layer ++ ) for ( let y = rect.y; y < rect.y + rect.height; y ++ ) {

		const start = ( ( layer * height + y ) * width + rect.x ) * n;
		out.set( fresh.subarray( start, start + rect.width * n ), start );

	}

	return out;

}

/** Writes only `rect` of layers `range` of level 0 (what an app that changed a region uploads). */
function uploadRect( device, texture, data, rect, [ first, count ] ) {

	const { width, height } = texture, layers = texture.depthOrArrayLayers, n = perTexel( data, width, height, layers );
	const part = new data.constructor( rect.width * rect.height * count * n );
	for ( let layer = 0; layer < count; layer ++ ) for ( let y = 0; y < rect.height; y ++ ) {

		const start = ( ( ( first + layer ) * height + rect.y + y ) * width + rect.x ) * n;
		part.set( data.subarray( start, start + rect.width * n ), ( layer * rect.height + y ) * rect.width * n );

	}

	device.queue.writeTexture( { texture, origin: { x: rect.x, y: rect.y, z: first } }, part, { bytesPerRow: rect.width * formats[ texture.format ].bytesPerTexel, rowsPerImage: rect.height }, [ rect.width, rect.height, count ] );

}

/** Level → the rect each batch of a partial plan writes (null: none); compatibility copies write the same. */
function writtenRegions( batches ) {

	const written = new Map();
	for ( const batch of batches ) if ( ! batch.copy ) batch.regions.forEach( ( region, i ) => written.set( batch.firstMipLevel + i, region ) );
	return written;

}

const inside = ( region, x, y ) => region !== null && region !== undefined && x >= region.x && x < region.x + region.width && y >= region.y && y < region.y + region.height;

/**
 * Compares two chains (arrays of RGBA levels from level 0) texel by texel, bit for bit
 * (Object.is), where `where( level, x, y, layer )` holds. Throws on the first difference.
 */
function same( a, b, width, height, layers, where, label, fromLevel = 0 ) {

	for ( let level = fromLevel; level < a.length; level ++ ) {

		const w = mipSize( width, level ), h = mipSize( height, level );
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			if ( ! where( level, x, y, layer ) ) continue;
			const t = ( ( layer * h + y ) * w + x ) * 4;
			for ( let c = 0; c < 4; c ++ ) if ( ! Object.is( a[ level ][ t + c ], b[ level ][ t + c ] ) ) throw new Error( `${ label }: level ${ level } texel (${ x }, ${ y }) layer ${ layer } channel ${ c }: ${ a[ level ][ t + c ] } vs ${ b[ level ][ t + c ] }` );

		}

	}

}

/**
 * The CPU oracle for every level a plan writes, each from the GPU's own source level
 * (`mips`, from level 0; `sourceFloats` for a separate source). The tolerance is the
 * sampler model (looser than shader weights); folds are exact.
 */
function checkOracle( batches, mips, sourceFloats, sourceSize, sourceFormat, width, height, layers, format, reference, label, { oddSizes = 'exact', resample = false } = {} ) {

	const { quantization, channels } = formats[ format ];
	for ( const batch of batches ) {

		if ( batch.copy ) continue;
		const external = batch.sourceMipLevel === null;
		const from = external ? sourceFloats : mips[ batch.sourceMipLevel ];
		const sw = external ? sourceSize[ 0 ] : mipSize( width, batch.sourceMipLevel ), sh = external ? sourceSize[ 1 ] : mipSize( height, batch.sourceMipLevel );
		if ( batch.kernel === 'clamp' ) {

			spdReference( from, sw, sh, layers, quantization, reference, batch.count ).forEach( ( level, i ) => check( mips[ batch.firstMipLevel + i ], level, quantization, `${ label } oracle mip ${ batch.firstMipLevel + i }`, reference, channels ) );
			continue;

		}

		let previous = from, w = sw, h = sh;
		for ( let i = 0; i < batch.count; i ++ ) {

			const level = batch.firstMipLevel + i;
			const first = external && i === 0;
			const quantizedInput = ! first || formats[ sourceFormat ]?.quantization === quantization;
			const step = oddSizes === 'box' ? boxReference : reduceReference;
			// A resampled source's first level is an area step to the destination's size (M2.5).
			const expected = first && w === mipSize( width, level ) && h === mipSize( height, level ) ? copyReference( previous, w, h, layers, quantization ) : first && resample ? areaReference( previous, w, h, layers, quantization, reference, mipSize( width, level ), mipSize( height, level ), { quantizedInput } ) : step( previous, w, h, layers, quantization, reference, { quantizedInput } );
			check( mips[ level ], expected, quantization, `${ label } oracle mip ${ level }`, reference, channels, external && i === 0 ? 2 ** - 21 : 0 );
			previous = mips[ level ]; w = expected.width; h = expected.height;

		}

	}

}

/**
 * The `levelsPerDispatch` option that gives a Downsampler `downsampler`'s levels per dispatch on
 * both paths. Passing the option sets both; left out, the storage path takes the storage limit
 * and the buffer path 6 (`kernelDefaults`). Below 6 storage textures (SwiftShader: 4) the two
 * differ, and passing the storage value would also pin sRGB's buffer path to it: a different
 * plan, so a first chain whose untouched texels a full recompute doesn't reproduce.
 */
export const levelsOption = downsampler => downsampler.levelsPerDispatch === downsampler.bufferLevelsPerDispatch ? { levelsPerDispatch: downsampler.levelsPerDispatch } : {};

/**
 * A Downsampler with the same settings (so the same plans and results) but its own caches: first
 * chains come from it, so a partial update never relies on state its own instance left behind
 * (the compatibility path's scratch levels, which a chain made elsewhere doesn't fill).
 */
const twins = new WeakMap();
function twin( device, downsampler ) {

	if ( ! twins.has( downsampler ) ) {

		const { output, averageAccess, compatibility, tail, f16Rounding, exactTile } = downsampler;
		twins.set( downsampler, new Downsampler( device, { ...levelsOption( downsampler ), output, averageAccess, compatibility, tail, f16Rounding, exactTile } ) );

	}

	return twins.get( downsampler );

}

function record( device, downsampler, texture, options, how ) {

	const encoder = device.createCommandEncoder();
	if ( how === 'pass' ) {

		const pass = encoder.beginComputePass();
		downsampler.encodePass( pass, texture, options );
		pass.end();

	} else downsampler.encode( encoder, texture, options );
	device.queue.submit( [ encoder.finish() ] );

}

/**
 * One in-place sequence: a full chain of random data, then each update (`{ rect?, layers? }`)
 * in turn, checked against a full recompute, the previous chain and a poisoned chain.
 */
async function inPlaceSequence( device, { downsampler, width, height, layers = 1, dimension, format, options, updates, how = 'encode', label, oracle = true } ) {

	const texture = createTexture( device, { width, height, layers, format, dimension } );
	const full = createTexture( device, { width, height, layers, format, dimension } );
	const probe = createTexture( device, { width, height, layers, format, dimension } );
	const reference = options.reduction === undefined || typeof options.reduction === 'string' ? options.reduction ?? 'average' : preferHigh.reference;

	try {

		// The first chain's untouched texels stand for the full recompute's only if both plan alike.
		const shape = owner => owner.prepare( full, options ).map( ( { kernel, firstMipLevel, count, tile, output } ) => `${ kernel }:${ firstMipLevel }:${ count }:${ tile }:${ output }` ).join( ' ' );
		if ( shape( twin( device, downsampler ) ) !== shape( downsampler ) ) throw new Error( `${ label }: the first chain's Downsampler plans ${ shape( twin( device, downsampler ) ) }, the one under test ${ shape( downsampler ) }` );
		let data = makeData( width, height, layers, format, 'random', 1 );
		upload( device, texture, data );
		poison( device, texture );
		twin( device, downsampler ).generate( texture, options );
		let previous = await readMips( device, texture, 0 );
		poison( device, probe );
		const poisoned = await readMips( device, probe, 0 );

		for ( const [ u, update ] of updates.entries() ) {

			const rect = update.rect ?? { x: 0, y: 0, width, height }, range = update.layers ?? [ 0, layers ];
			const run = { ...options, ...( update.rect ? { rect: update.rect } : {} ), ...( update.layers ? { layers: update.layers } : {} ) };
			const tag = `${ label } ${ JSON.stringify( update ) }`;
			data = merge( data, makeData( width, height, layers, format, 'random', 1000 + u ), width, height, layers, rect, range );

			uploadRect( device, texture, data, rect, range );
			record( device, downsampler, texture, run, how );
			const mips = await readMips( device, texture, 0 );

			// 1. A full GPU recompute of the changed source, bit for bit.
			upload( device, full, data );
			poison( device, full );
			downsampler.generate( full, options );
			const expected = await readMips( device, full, 0 );
			same( mips, expected, width, height, layers, () => true, `${ tag } vs full recompute` );

			// 2. Outside the derived regions and layers, the previous chain.
			const batches = downsampler.prepare( texture, run );
			const written = writtenRegions( batches );
			const changed = ( level, x, y, layer ) => layer >= range[ 0 ] && layer < range[ 0 ] + range[ 1 ] && ( level === 0 ? inside( rect, x, y ) : inside( written.get( level ), x, y ) );
			same( mips, previous, width, height, layers, ( ...at ) => ! changed( ...at ), `${ tag } outside the regions vs the previous chain` );
			if ( oracle ) checkOracle( batches, mips, null, null, format, width, height, layers, format, reference, tag, { oddSizes: options.oddSizes } );

			// 3. No stray writes: on a poisoned chain, everything outside the regions stays poisoned.
			upload( device, probe, data );
			poison( device, probe );
			record( device, downsampler, probe, run, how );
			same( await readMips( device, probe, 0 ), poisoned, width, height, layers, ( ...at ) => ! changed( ...at ), `${ tag } stray writes`, 1 );

			previous = mips;

		}

		return updates.length;

	} finally {

		for ( const owner of [ downsampler, twin( device, downsampler ) ] ) for ( const t of [ texture, full, probe ] ) owner.release( t );
		for ( const t of [ texture, full, probe ] ) t.destroy();

	}

}

const slug = name => name.replace( /[^a-z0-9]+/gi, '-' ).replace( /^-|-$/g, '' );

export async function validateRegion( device, { quick, onProgress = () => {}, select } ) {

	const results = [];
	const compat = compatibility( device );
	const storageLimit = Math.min( 6, device.limits.maxStorageTexturesPerShaderStage );
	const software = device.adapterInfo?.isFallbackAdapter === true;
	// Odd-size policy and kernel choices, each with its own Downsampler. 'tail' is the default
	// (tail on); 'exact' pins whole chains to 'exact' batches. 'compat' forces the
	// compatibility-mode path on a core device (a compatibility device takes it anyway).
	const modes = [
		{ name: 'exact', downsampler: new Downsampler( device, { tail: false } ) },
		{ name: 'tail', downsampler: new Downsampler( device ) },
		{ name: 'exact L1', downsampler: new Downsampler( device, { tail: false, levelsPerDispatch: 1 } ) },
		...( storageLimit > 4 ? [ { name: 'tail L4', downsampler: new Downsampler( device, { levelsPerDispatch: 4 } ) } ] : [] ),
		{ name: 'spd', downsampler: new Downsampler( device ), options: { oddSizes: 'spd' } },
		// oddSizes 'box' (M2.5): the same footprint, so the same regions; with and without the tail.
		{ name: 'box', downsampler: new Downsampler( device, { tail: false } ), options: { oddSizes: 'box' } },
		{ name: 'tail box', downsampler: new Downsampler( device ), options: { oddSizes: 'box' } },
		{ name: 'buffer', downsampler: new Downsampler( device, { output: 'buffer' } ) },
		...( compat ? [] : [ { name: 'compat', downsampler: new Downsampler( device, { compatibility: true } ) } ] ),
		{ name: 'pass', downsampler: new Downsampler( device ), how: 'pass' }
	];
	// Format × reduction per mode. Folds are rejected under 'spd'; the buffer path, the
	// compatibility path and encodePass run a representative pair.
	const reductions = [
		{ name: 'average', format: 'rgba8unorm', options: {} },
		{ name: 'average (load)', format: 'rgba8unorm', options: {}, access: 'load' },
		{ name: 'max', format: 'rgba8unorm', options: { reduction: 'max' } },
		{ name: 'minmax', format: 'rgba8unorm', options: { reduction: 'minmax' } },
		{ name: 'custom', format: 'rgba8unorm', options: { reduction: { wgsl: preferHigh.wgsl } } },
		{ name: 'average', format: 'rgba16float', options: {} },
		{ name: 'max', format: 'rgba16float', options: { reduction: 'max' } },
		{ name: 'min', format: 'r32float', options: { reduction: 'min' } },
		// sRGB (M2.3.5, buffer path): under 'exact', the tail, 'spd' and 'box'.
		{ name: 'average', format: 'rgba8unorm-srgb', options: {} },
		{ name: 'max', format: 'rgba8unorm-srgb', options: { reduction: 'max' } }
	];
	const pick = mode => reductions.filter( r => {

		if ( mode === 'spd' ) return r.name.startsWith( 'average' );
		if ( mode.endsWith( 'box' ) ) return r.name.startsWith( 'average' ) || ( r.name === 'max' && r.format === 'rgba8unorm' );
		if ( mode === 'exact' || mode === 'tail' ) return true;
		if ( mode === 'pass' ) return r.format === 'rgba16float';
		return r.format === 'rgba8unorm' ? r.name === 'average' || r.name === 'max' : r.name === 'average' && r.format === 'rgba16float';

	} );
	const loads = new Map();

	for ( const { name: mode, downsampler: owner, options: modeOptions = {}, how } of modes ) for ( const variant of pick( mode ) ) {

		const id = `region/${ slug( mode ) }/${ variant.format }/${ slug( variant.name ) }`;
		results.push( ...await select.run( id, async () => {

			const key = `${ mode }:${ variant.access ?? '' }`;
			let downsampler = owner;
			if ( variant.access === 'load' ) {

				if ( ! loads.has( key ) ) loads.set( key, new Downsampler( device, { tail: owner.tail, ...levelsOption( owner ), averageAccess: 'load', output: owner.output, compatibility: owner.compatibility } ) );
				downsampler = loads.get( key );

			}

			// Render sizes for the main policies' first formats; everything runs the small shapes.
			const primary = ( mode === 'exact' || mode === 'tail' || mode === 'spd' || mode === 'box' ) && ( variant.format !== 'rgba8unorm' || variant.name === 'average' || variant.name === 'max' );
			const shapes = quick ? [ ...quickShapes, ...( primary && ! software ? [ renderSizes[ 0 ] ] : [] ) ] : [ ...smallShapes, ...( primary ? renderSizes : [ renderSizes[ 0 ] ] ) ];
			const options = { ...variant.options, ...modeOptions };
			const out = [];
			for ( const [ width, height ] of shapes ) {

				// SwiftShader compiles 32×32-tile tail pipelines for minutes: --quick skips those plans.
				if ( quick && downsampler.tail && planHasBigTail( device, downsampler, width, height, variant.format, options ) ) continue;
				const updates = rectsFor( width, height, width * 7 + height, renderSizes.some( s => s[ 0 ] === width ) ? 5 : 9 ).map( rect => ( { rect } ) );
				const label = `region ${ mode } ${ variant.format } ${ variant.name } ${ width }x${ height }`;
				const count = await inPlaceSequence( device, { downsampler, width, height, format: variant.format, options, updates, how, label } );
				out.push( { region: mode, format: variant.format, reduction: variant.name, width, height, updates: count } );
				onProgress( `validated ${ label }` );

			}

			return out;

		} ) );

	}

	// SwiftShader's split on a real GPU: a device with WebGPU's default limits (4 storage textures,
	// 256 invocations) runs the storage path at 4 levels per dispatch and sRGB's buffer path at 6,
	// so each Downsampler's twin (above) must plan sRGB at 6 too. Devices at 4 run this above.
	if ( storageLimit > 4 ) results.push( ...await select.run( 'region/default-limits/rgba8unorm-srgb/average', async () => {

		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance', ...( compat ? { featureLevel: 'compatibility' } : {} ) } );
		const plain = await adapter.requestDevice();
		const out = [];
		try {

			if ( plain.limits.maxStorageTexturesPerShaderStage !== 4 ) throw new Error( `region default limits: the device has ${ plain.limits.maxStorageTexturesPerShaderStage } storage textures, not WebGPU's default 4` );
			for ( const [ mode, downsampler, options ] of [ [ 'tail', new Downsampler( plain ), {} ], [ 'exact', new Downsampler( plain, { tail: false } ), {} ], [ 'spd', new Downsampler( plain ), { oddSizes: 'spd' } ] ] ) {

				if ( downsampler.levelsPerDispatch !== 4 || downsampler.bufferLevelsPerDispatch !== 6 ) throw new Error( `region default limits: ${ downsampler.levelsPerDispatch } storage and ${ downsampler.bufferLevelsPerDispatch } buffer levels per dispatch, not 4 and 6` );
				for ( const [ width, height ] of quick ? quickShapes : smallShapes ) {

					const updates = rectsFor( width, height, width * 7 + height, 9 ).map( rect => ( { rect } ) );
					const label = `region default limits ${ mode } rgba8unorm-srgb average ${ width }x${ height }`;
					const count = await inPlaceSequence( plain, { downsampler, width, height, format: 'rgba8unorm-srgb', options, updates, label } );
					out.push( { region: `default limits ${ mode }`, format: 'rgba8unorm-srgb', reduction: 'average', width, height, updates: count } );
					onProgress( `validated ${ label }` );

				}

			}

		} finally {

			plain.destroy();

		}

		return out;

	} ) );

	// Layer ranges, alone and with a rect, on both array view dimensions (and a 1-layer array).
	for ( const { format, options, name } of [ { format: 'rgba8unorm', options: {}, name: 'average' }, { format: 'rgba16float', options: { reduction: 'max' }, name: 'max' } ] ) {

		results.push( ...await select.run( `region/layers/${ format }/${ name }`, async () => {

			const out = [];
			const owners = [ [ 'tail', new Downsampler( device ) ], [ 'exact', new Downsampler( device, { tail: false } ) ], [ 'buffer', new Downsampler( device, { output: 'buffer' } ) ], ...( compat ? [] : [ [ 'compat', new Downsampler( device, { compatibility: true } ) ] ] ) ];
			for ( const [ width, height, layers, dimension ] of [ [ 64, 32, 3 ], [ 17, 11, 3 ], [ 33, 65, 6 ], [ 17, 9, 1, '2d-array' ] ] ) for ( const [ mode, downsampler ] of owners ) {

				const rect = { x: Math.floor( width / 2 ), y: 1, width: Math.max( 1, Math.floor( width / 4 ) ), height: Math.min( height - 1, 5 ) };
				const updates = layers === 1 ? [ { layers: [ 0, 1 ] }, { layers: [ 0, 1 ], rect } ] : [
					{ layers: [ 1, 1 ] }, { layers: [ 0, 2 ] }, { layers: [ layers - 1, 1 ] }, { layers: [ 1, layers - 1 ], rect },
					{ layers: [ 0, layers ], rect: { x: width - 1, y: height - 1, width: 1, height: 1 } }, { rect: { x: 0, y: 0, width: 1, height } }
				];
				const label = `region layers ${ mode } ${ format } ${ name } ${ width }x${ height }x${ layers }`;
				const count = await inPlaceSequence( device, { downsampler, width, height, layers, dimension, format, options, updates, label } );
				out.push( { region: `layers ${ mode }`, format, reduction: name, width, height, layers, updates: count } );
				onProgress( `validated ${ label }` );

			}

			return out;

		} ) );

	}

	results.push( ...await validateRegionSources( device, { quick, onProgress, select } ) );
	results.push( ...await select.run( 'region/options', () => validateRegionOptions( device ) ) );
	return results;

}

/** Whether the plan for a `width` × `height` texture has a tail batch with a tile over 16. */
function planHasBigTail( device, downsampler, width, height, format, options ) {

	const texture = createTexture( device, { width, height, format } );
	try { return downsampler.prepare( texture, options ).some( batch => batch.kernel === 'tail' && batch.tile > 16 ); } finally { downsampler.release( texture ); texture.destroy(); }

}

// Separate sources: the rect marks texels of the source level; level 0 of the destination is
// the source's size (a copy) or half.
const sourceCases = [
	{ source: 'r32float', destination: 'rg32float', relation: 'same', reduction: 'minmax' },
	{ source: 'depth32float', destination: 'r32float', relation: 'half', reduction: 'max' },
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'half', reduction: 'average' },
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'same', reduction: 'average', oddSizes: 'spd' },
	{ source: 'rgba16float', destination: 'rgba16float', relation: 'half', reduction: 'average', oddSizes: 'spd' },
	// Resampled sources (M2.5, resample 'area'): the next power of two below or above the source size.
	{ source: 'depth32float', destination: 'r32float', relation: 'pow2 down', reduction: 'max' },
	{ source: 'depth32float', destination: 'rg32float', relation: 'pow2 up', reduction: 'minmax' },
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'pow2 down', reduction: 'average' },
	{ source: 'r32float', destination: 'r32float', relation: 'pow2 up', reduction: 'average', oddSizes: 'box' },
	{ source: 'rgba16float', destination: 'rgba16float', relation: 'pow2 down', reduction: 'average', oddSizes: 'spd' },
	// 'box' with a half step from an odd source.
	{ source: 'rgba16float', destination: 'rgba8unorm', relation: 'half', reduction: 'average', oddSizes: 'box' }
];

async function makeSource( device, format, width, height, layers, values ) {

	if ( format === 'depth32float' ) return createDepthSource( device, { width, height, layers, format }, values );
	const texture = createTexture( device, { width, height, layers, format, storage: false } );
	upload( device, texture, values );
	return { texture, floats: toFloats( values, format ) };

}

async function validateRegionSources( device, { quick, onProgress, select } ) {

	const results = [];
	const compat = compatibility( device );
	for ( const { source: sourceFormat, destination: format, relation, reduction, oddSizes = 'exact' } of sourceCases ) {

		if ( formats[ format ].coreStorage && compat ) continue;
		const resample = relation.startsWith( 'pow2' );
		const pow2 = size => relation === 'pow2 up' ? 2 ** Math.ceil( Math.log2( size ) ) : 2 ** Math.floor( Math.log2( size ) );
		results.push( ...await select.run( `region/source/${ sourceFormat }/${ format }/${ relation.replace( ' ', '-' ) }${ oddSizes !== 'exact' ? `/${ oddSizes }` : '' }`, async () => {

			const out = [];
			const owners = [ [ 'tail', new Downsampler( device ) ], [ 'exact', new Downsampler( device, { tail: false } ) ], ...( compat ? [] : [ [ 'compat', new Downsampler( device, { compatibility: true } ) ] ] ) ];
			for ( const [ width, height, layers = 1 ] of quick ? [ [ 17, 9 ], [ 65, 129 ], [ 17, 11, 3 ] ] : [ [ 17, 9 ], [ 65, 129 ], [ 255, 257 ], [ 17, 11, 3 ], [ 1129, 635 ] ] ) for ( const [ mode, downsampler ] of owners ) {

				const dw = resample ? pow2( width ) : relation === 'same' ? width : Math.max( 1, width >> 1 ), dh = resample ? pow2( height ) : relation === 'same' ? height : Math.max( 1, height >> 1 );
				const area = resample ? { resample: 'area' } : {};
				const destination = createTexture( device, { width: dw, height: dh, layers, format } );
				const full = createTexture( device, { width: dw, height: dh, layers, format } );
				const probe = createTexture( device, { width: dw, height: dh, layers, format } );
				const depth = sourceFormat.startsWith( 'depth' );
				const valuesOf = seed => makeData( width, height, layers, depth ? 'r32float' : sourceFormat, 'random', seed, { unit: depth } );
				let values = valuesOf( 3 );
				let source = await makeSource( device, sourceFormat, width, height, layers, values );
				const label = `region source ${ mode } ${ sourceFormat } ${ width }x${ height }x${ layers } -> ${ format } ${ relation } ${ reduction } ${ oddSizes }`;

				try {

					const options = { reduction, oddSizes };
					poison( device, destination, 0 );
					twin( device, downsampler ).generate( destination, { ...options, source: { texture: source.texture, ...area } } );
					let previous = await readMips( device, destination, 0 );
					poison( device, probe, 0 );
					const poisoned = await readMips( device, probe, 0 );
					const updates = [ ...rectsFor( width, height, width + height, 5 ).map( rect => ( { rect } ) ), ...( layers > 1 ? [ { layers: [ 1, 2 ], rect: { x: 1, y: 2, width: 3, height: 4 } } ] : [] ) ];

					for ( const [ u, update ] of updates.entries() ) {

						const rect = update.rect, range = update.layers ?? [ 0, layers ];
						values = merge( values, valuesOf( 500 + u ), width, height, layers, rect, range );
						// A depth buffer is rendered again; a colour source gets the rect written.
						if ( depth ) { source.texture.destroy(); source = await makeSource( device, sourceFormat, width, height, layers, values ); } else { uploadRect( device, source.texture, values, rect, range ); source.floats = toFloats( values, sourceFormat ); }
						const run = { ...options, source: { texture: source.texture, ...area }, rect, ...( update.layers ? { layers: update.layers } : {} ) };
						const tag = `${ label } ${ JSON.stringify( update ) }`;
						downsampler.generate( destination, run );
						const mips = await readMips( device, destination, 0 );
						poison( device, full, 0 );
						downsampler.generate( full, { ...options, source: { texture: source.texture, ...area } } );
						same( mips, await readMips( device, full, 0 ), dw, dh, layers, () => true, `${ tag } vs full recompute` );
						const batches = downsampler.prepare( destination, run );
						const written = writtenRegions( batches );
						const changed = ( level, x, y, layer ) => layer >= range[ 0 ] && layer < range[ 0 ] + range[ 1 ] && inside( written.get( level ), x, y );
						same( mips, previous, dw, dh, layers, ( ...at ) => ! changed( ...at ), `${ tag } outside the regions vs the previous chain` );
						const floats = sampledChannels( sourceFormat ) === 1 ? Float32Array.from( source.floats, ( _, i ) => source.floats[ i - i % 4 ] ) : source.floats;
						checkOracle( batches, mips, floats, [ width, height ], sourceFormat, dw, dh, layers, format, reduction, tag, { oddSizes, resample } );
						poison( device, probe, 0 );
						downsampler.generate( probe, run );
						same( await readMips( device, probe, 0 ), poisoned, dw, dh, layers, ( ...at ) => ! changed( ...at ), `${ tag } stray writes` );
						previous = mips;

					}

					out.push( { region: `source ${ mode }`, source: sourceFormat, destination: format, relation, width, height, layers, updates: updates.length } );
					onProgress( `validated ${ label }` );

				} finally {

					for ( const t of [ destination, full, probe ] ) { downsampler.release( t ); twin( device, downsampler ).release( t ); t.destroy(); }
					source.texture.destroy();

				}

			}

			return out;

		} ) );

	}

	return results;

}

function expectThrow( run, message, label ) {

	try { run(); } catch ( error ) {

		if ( ! String( error.message ).includes( message ) ) throw new Error( `${ label }: expected "${ message }", got "${ error.message }"` );
		return;

	}

	throw new Error( `${ label }: expected an error "${ message }"` );

}

async function validateRegionOptions( device ) {

	const results = [];
	const downsampler = new Downsampler( device );
	const texture = createTexture( device, { width: 37, height: 21, layers: 3, format: 'rgba8unorm' } );

	// Rejections name the problem and the level the rect refers to.
	for ( const [ options, message ] of [
		[ { rect: { x: 0, y: 0, width: - 1, height: 1 } }, 'must not be negative' ],
		[ { rect: { x: 38, y: 0, width: 0, height: 1 } }, 'outside the 37×21 level' ],
		[ { rect: { x: 30, y: 0, width: 8, height: 1 } }, 'outside the 37×21 level' ],
		[ { rect: { x: - 1, y: 0, width: 2, height: 1 } }, 'outside' ],
		[ { rect: { x: 0.5, y: 0, width: 2, height: 1 } }, 'integers' ],
		[ { rect: { x: 0, y: 0, width: 19, height: 1 }, baseMipLevel: 1 }, 'outside the 18×10 level it marks dirty (level 1)' ],
		[ { layers: [ 0, 0 ] }, 'at least 1' ],
		[ { layers: [ 2, 2 ] }, "outside the texture's 3 layers" ],
		[ { layers: [ - 1, 1 ] }, 'outside' ],
		[ { layers: [ 1 ] }, '[ first, count ]' ]
	] ) expectThrow( () => downsampler.prepare( texture, options ), message, JSON.stringify( options ) );
	results.push( { option: 'rect and layers rejections' } );

	// A rect that fits the (larger) separate source but not the destination is accepted, and named against the source.
	{

		const source = createTexture( device, { width: 74, height: 42, format: 'rgba16float', storage: false } );
		const destination = createTexture( device, { width: 37, height: 21, format: 'rgba8unorm' } );
		downsampler.prepare( destination, { source: { texture: source }, rect: { x: 60, y: 30, width: 14, height: 12 } } );
		expectThrow( () => downsampler.prepare( destination, { source: { texture: source }, rect: { x: 60, y: 30, width: 15, height: 12 } } ), 'outside the 74×42 level it marks dirty (the source level)', 'source rect' );
		source.destroy(); destination.destroy();

	}

	// encodePass can't record the buffer path's copies, with or without a rect.
	{

		const buffer = new Downsampler( device, { output: 'buffer' } );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		expectThrow( () => buffer.encodePass( pass, texture, { rect: { x: 1, y: 1, width: 2, height: 2 } } ), 'use encode()', 'buffer encodePass' );
		pass.end();
		encoder.finish();

	}

	// A 1×1 texture has no level to write; a partial plan is empty.
	{

		const one = createTexture( device, { width: 1, height: 1, format: 'rgba8unorm' } );
		if ( downsampler.prepare( one, { rect: { x: 0, y: 0, width: 1, height: 1 } } ).length !== 0 ) throw new Error( '1×1: expected no batches' );
		one.destroy();

	}

	// An empty rect (width or height 0) is a no-op, as WebGPU's zero-size copies are: no
	// batches, nothing recorded on either path or in a pass, no buffers, the chain untouched.
	{

		const tracker = trackBuffers( device );
		try {

			const empty = [ { x: 5, y: 3, width: 0, height: 4 }, { x: 37, y: 21, width: 0, height: 0 }, { x: 0, y: 7, width: 9, height: 0 } ];
			const buffer = new Downsampler( device, { output: 'buffer' } );
			const target = createTexture( device, { width: 37, height: 21, layers: 3, format: 'rgba8unorm' } );
			upload( device, target, makeData( 37, 21, 3, 'rgba8unorm', 'random', 8 ) );
			poison( device, target );
			const before = await readMips( device, target, 0 );
			for ( const rect of empty ) for ( const owner of [ downsampler, buffer ] ) {

				for ( const options of [ { rect }, { rect, layers: [ 1, 1 ] }, { rect, oddSizes: 'spd' }, { rect, reduction: 'max' } ] ) if ( owner.prepare( target, options ).length !== 0 ) throw new Error( `empty rect ${ JSON.stringify( options ) }: expected no batches` );
				const encoder = device.createCommandEncoder();
				let recorded = 0;
				for ( const method of [ 'beginComputePass', 'copyBufferToTexture' ] ) {

					const original = encoder[ method ].bind( encoder );
					encoder[ method ] = ( ...args ) => { recorded ++; return original( ...args ); };

				}

				owner.encode( encoder, target, { rect } );
				if ( owner === downsampler ) {

					const pass = encoder.beginComputePass();
					recorded --;
					const setPipeline = pass.setPipeline.bind( pass );
					pass.setPipeline = pipeline => { recorded ++; setPipeline( pipeline ); };
					owner.encodePass( pass, target, { rect } );
					pass.end();

				}

				device.queue.submit( [ encoder.finish() ] );
				if ( recorded !== 0 ) throw new Error( `empty rect ${ JSON.stringify( rect ) }: recorded ${ recorded } passes, copies or dispatches` );

			}

			// Still validated: the rest of the options, and a source.
			expectThrow( () => downsampler.prepare( target, { rect: empty[ 0 ], oddSizes: 'fast' } ), 'not implemented', 'empty rect, bad oddSizes' );
			expectThrow( () => downsampler.prepare( target, { rect: empty[ 0 ], layers: [ 3, 1 ] } ), 'outside', 'empty rect, bad layers' );
			if ( tracker.labelled( 'downsampler' ).length !== 0 ) throw new Error( `empty rect: created ${ tracker.labelled( 'downsampler' ).length } buffers` );
			same( await readMips( device, target, 0 ), before, 37, 21, 3, () => true, 'empty rect leaves the chain untouched' );
			downsampler.release( target ); buffer.release( target );
			target.destroy();

		} finally {

			tracker.restore();

		}

	}

	// Region uniforms of plans evicted from the 16-entry cache (and of a plan whose source
	// texture was replaced) are destroyed once the queue is done; a chain recorded before its
	// plan was evicted still submits; release() destroys the rest.
	{

		const tracker = trackBuffers( device );
		try {

			const owner = new Downsampler( device );
			const target = createTexture( device, { width: 64, height: 32, format: 'rgba16float' } );
			upload( device, target, makeData( 64, 32, 1, 'rgba16float', 'random', 9 ) );
			device.pushErrorScope( 'validation' );
			const early = device.createCommandEncoder();
			owner.encode( early, target, { rect: { x: 0, y: 0, width: 1, height: 1 } } );
			for ( let i = 1; i <= 16; i ++ ) owner.prepare( target, { rect: { x: i, y: 0, width: 1, height: 1 } } );
			device.queue.submit( [ early.finish() ] );
			const error = await device.popErrorScope();
			if ( error !== null ) throw new Error( `region eviction: a chain recorded before its plan was evicted failed to submit: ${ error.message }` );
			await tracker.settle();
			const uniforms = tracker.labelled( 'downsampler region' );
			if ( uniforms.length !== 17 ) throw new Error( `region eviction: expected 17 region uniforms, got ${ uniforms.length }` );
			if ( ! tracker.destroyed.has( uniforms[ 0 ] ) || uniforms.slice( 1 ).some( buffer => tracker.destroyed.has( buffer ) ) ) throw new Error( 'region eviction: must destroy the evicted plan\'s region uniform, and only it' );
			owner.release( target );
			if ( uniforms.some( buffer => ! tracker.destroyed.has( buffer ) ) ) throw new Error( 'region eviction: release() must destroy every region uniform' );

			// A new source texture for the same rect replaces the plan; the old uniform is retired.
			const destination = createTexture( device, { width: 32, height: 16, format: 'rgba8unorm' } );
			const sources = [ 0, 1 ].map( () => createTexture( device, { width: 64, height: 32, format: 'rgba8unorm', storage: false } ) );
			const rect = { x: 3, y: 2, width: 5, height: 4 };
			for ( const texture of sources ) owner.prepare( destination, { source: texture, rect } );
			await tracker.settle();
			const replaced = tracker.labelled( 'downsampler region' ).slice( 17 );
			if ( replaced.length !== 2 || ! tracker.destroyed.has( replaced[ 0 ] ) || tracker.destroyed.has( replaced[ 1 ] ) ) throw new Error( 'region plans: a replaced source must retire the old plan\'s region uniform, and only it' );
			owner.release( destination );
			for ( const texture of [ target, destination, ...sources ] ) texture.destroy();

		} finally {

			tracker.restore();

		}

	}

	// Several partial updates recorded in one encoder before a submit, and more than the 16
	// cached regions: each keeps its own region uniform, so the result equals updates
	// submitted one at a time.
	{

		const [ width, height ] = [ 133, 75 ];
		const a = createTexture( device, { width, height, format: 'rgba16float' } ), b = createTexture( device, { width, height, format: 'rgba16float' } );
		const data = makeData( width, height, 1, 'rgba16float', 'random', 4 );
		for ( const t of [ a, b ] ) { upload( device, t, data ); downsampler.generate( t ); }
		const rects = Array.from( { length: 20 }, ( _, i ) => ( { x: ( i * 37 ) % ( width - 9 ), y: ( i * 23 ) % ( height - 5 ), width: 9, height: 5 } ) );
		// The source changes in every rect up front; each update then recomputes its own regions, and
		// the last update to touch a texel sees all its inputs final, so the chain ends up exact.
		let changed = data;
		for ( const [ i, rect ] of rects.entries() ) changed = merge( changed, makeData( width, height, 1, 'rgba16float', 'random', 50 + i ), width, height, 1, rect, [ 0, 1 ] );
		upload( device, a, changed ); upload( device, b, changed );
		const encoder = device.createCommandEncoder();
		for ( const rect of rects ) downsampler.encode( encoder, a, { rect } );
		device.queue.submit( [ encoder.finish() ] );
		for ( const rect of rects ) downsampler.generate( b, { rect } );
		const [ ma, mb ] = [ await readMips( device, a, 0 ), await readMips( device, b, 0 ) ];
		same( ma, mb, width, height, 1, () => true, 'one encoder vs one submit per update' );
		const full = createTexture( device, { width, height, format: 'rgba16float' } );
		upload( device, full, changed ); downsampler.generate( full );
		same( ma, await readMips( device, full, 0 ), width, height, 1, () => true, '20 partial updates vs a full recompute' );
		for ( const t of [ a, b, full ] ) { downsampler.release( t ); t.destroy(); }

	}

	results.push( { option: 'source rects, encodePass on the buffer path, 1×1, empty rects, batched updates and cache eviction' } );
	downsampler.release( texture );
	texture.destroy();
	return results;

}
