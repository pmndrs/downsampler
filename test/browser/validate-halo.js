// Halo batches (SPEC §13 item 3; Downsampler `halo`): odd steps carried inside a batch, each
// workgroup computing the texels past its tile that the odd steps read. They must not change
// results: wherever both chains apply shader weights (averageAccess 'load', folds, 'box' with
// 'load'), a halo chain is bit-identical to the plain one in exact formats (normalized integers,
// float16, packed, sRGB) and within an ulp in float32, where a level computed by another kernel
// may round differently. With the default 'sample' every level is checked against the core
// oracle from the GPU's previous level, as the main matrix checks tail kernels.
// Partial updates (rect, layers) of a halo chain equal its full recompute.
import { formats, isIntegerDomain, isPackedQuantization, reduceReference } from '/dist/index.js';
import { Downsampler } from '/dist/webgpu/index.js';
import { createTexture, makeData, poison, readMips, toFloats, upload } from './gpu.js';
import { check, compatibility } from './validate.js';

const mipSize = ( size, level ) => Math.max( 1, size >> level );

// Shapes whose plans have halo batches (odd steps from a 16- or 32-texel tile), layers and a
// 2d-array; --quick keeps the small ones.
export const haloShapes = [ [ 255, 257 ], [ 191, 107 ], [ 129, 67 ], [ 383, 215, 3 ], [ 959, 539 ], [ 1129, 635 ], [ 1477, 831 ], [ 1920, 1080 ] ];
const quickShapes = haloShapes.filter( ( [ w, h ] ) => w * h <= 100000 );
// [ format, output ]: unorm8 (exact integer levels), float16 (rounding), float32 (loads only),
// snorm, sRGB and rgb9e5 (buffer path).
export const haloFormats = [ [ 'rgba8unorm', 'storage' ], [ 'rgba16float', 'storage' ], [ 'rgba16float', 'buffer' ], [ 'r32float', 'storage' ], [ 'rgba8snorm', 'storage' ], [ 'rgba8unorm-srgb', 'auto' ], [ 'rgb9e5ufloat', 'auto' ] ];
const custom = { wgsl: 'fn reduce(a: vec4f, b: vec4f) -> vec4f { return max(a, b); }' };
// Modes compared bit for bit with the plain chain.
const identical = [
	{ name: 'average-load', oddSizes: 'exact', reduction: 'average', access: 'load' },
	{ name: 'min', oddSizes: 'exact', reduction: 'min' },
	{ name: 'minmax', oddSizes: 'exact', reduction: 'minmax' },
	{ name: 'custom', oddSizes: 'exact', reduction: custom },
	{ name: 'box-average-load', oddSizes: 'box', reduction: 'average', access: 'load' },
	{ name: 'box-max', oddSizes: 'box', reduction: 'max' }
];

export async function validateHalo( device, { quick = false, onProgress = () => {}, select } ) {

	const results = [];
	if ( compatibility( device ) || device.limits.maxComputeInvocationsPerWorkgroup < 256 ) return results;
	const shapes = quick ? quickShapes : haloShapes;

	for ( const [ format, output ] of haloFormats ) {

		if ( ! formats[ format ] ) continue;
		const filterable = format !== 'r32float';
		results.push( ...await select.run( `halo/${ format }/${ output }`, async () => {

			const out = [];
			const make = options => new Downsampler( device, { output, ...options } );
			for ( const [ width, height, layers = 1 ] of shapes ) {

				const a = createTexture( device, { width, height, layers, format } ), b = createTexture( device, { width, height, layers, format } );
				const data = makeData( width, height, layers, format, 'random', width + height );
				try {

					for ( const mode of identical ) {

						const access = mode.access ?? 'sample';
						if ( mode.reduction !== 'average' && format === 'rgb9e5ufloat' && mode.reduction === 'minmax' ) continue;
						const options = { reduction: mode.reduction, oddSizes: mode.oddSizes };
						const plain = make( { averageAccess: access } ), halo = make( { averageAccess: access, halo: true } );
						const halos = halo.prepare( a, options ).filter( batch => batch.halo !== undefined ).length;
						if ( halos === 0 ) continue;
						for ( const [ texture, downsampler ] of [ [ a, plain ], [ b, halo ] ] ) { upload( device, texture, data ); poison( device, texture ); downsampler.generate( texture, options ); }
						const [ expected, actual ] = [ await readMips( device, a, 1 ), await readMips( device, b, 1 ) ];
						same( actual, expected, `halo ${ format } ${ output } ${ width }x${ height }x${ layers } ${ mode.name }`, formats[ format ].quantization === 'float32' );
						out.push( { format, output, width, height, layers, mode: mode.name, haloBatches: halos, dispatches: [ plain.prepare( a, options ).length, halo.prepare( b, options ).length ] } );

					}

					// The default 'sample' path: every level against the oracle from the GPU's previous level.
					if ( filterable ) {

						const halo = make( { halo: true } );
						const batches = halo.prepare( b );
						if ( batches.some( batch => batch.halo !== undefined ) ) {

							upload( device, b, data ); poison( device, b );
							halo.generate( b );
							const mips = await readMips( device, b, 1 );
							const { quantization, channels } = formats[ format ];
							const srgb = quantization === 'srgb8', packed = isPackedQuantization( quantization ) || srgb;
							const first = new Set( batches.filter( batch => ! batch.copy ).map( batch => batch.firstMipLevel ) );
							// Odd steps inside halo and tail batches use shader weights in float math: the tolerance, as for tails.
							const odd = new Set();
							for ( const batch of batches ) for ( let level = batch.firstMipLevel + 1; level < batch.firstMipLevel + batch.count; level ++ ) {

								const w = mipSize( width, level - 1 ), h = mipSize( height, level - 1 );
								if ( packed || ( w > 1 && w % 2 ) || ( h > 1 && h % 2 ) ) odd.add( level );

							}

							let previous = toFloats( data, format ), w = width, h = height;
							for ( let level = 1; level < b.mipLevelCount; level ++ ) {

								const tolerated = first.has( level ) || odd.has( level );
								const expected = reduceReference( previous, w, h, layers, quantization, 'average', { ...( packed && ! first.has( level ) ? { pairing: 'float32' } : {} ), weights: first.has( level ) ? 'sampler' : 'shader' } );
								check( mips[ level - 1 ], expected, quantization, `halo ${ format } ${ output } ${ width }x${ height }x${ layers } sample mip ${ level }`, 'average', channels, 0, ( isIntegerDomain( quantization ) || packed ) && ! tolerated );
								previous = mips[ level - 1 ]; w = expected.width; h = expected.height;

							}

							out.push( { format, output, width, height, layers, mode: 'average-sample', haloBatches: batches.filter( batch => batch.halo !== undefined ).length } );

						}

					}

					// Partial updates: a rect (and a layer range) of a halo chain equals its full recompute.
					if ( format === 'rgba8unorm' || format === 'rgba16float' ) {

						const halo = make( { halo: true } );
						const rect = { x: Math.floor( width / 3 ), y: Math.floor( height / 4 ), width: Math.max( 1, Math.floor( width / 5 ) ), height: Math.max( 1, Math.floor( height / 6 ) ) };
						const changed = makeData( width, height, layers, format, 'random', width * 7 + height );
						for ( const options of [ { rect }, ...( layers > 1 ? [ { rect, layers: [ 1, 1 ] } ] : [] ) ] ) {

							// a: the old chain, then the changed texels and a partial update; b: a full recompute of the result.
							upload( device, a, data ); halo.generate( a );
							const mixed = mixRect( data, changed, width, height, layers, format, rect, options.layers );
							upload( device, a, mixed );
							const encoder = device.createCommandEncoder();
							halo.encode( encoder, a, options );
							device.queue.submit( [ encoder.finish() ] );
							upload( device, b, mixed ); poison( device, b ); halo.generate( b );
							same( await readMips( device, a, 1 ), await readMips( device, b, 1 ), `halo ${ format } ${ width }x${ height }x${ layers } rect${ options.layers ? ' layers' : '' }` );
							out.push( { format, output, width, height, layers, mode: `rect${ options.layers ? '+layers' : '' }` } );

						}

					}

				} finally {

					a.destroy(); b.destroy();

				}

			}

			onProgress( `validated halo ${ format } ${ output }` );
			return out;

		} ) );

	}

	return results;

}

/** `base` with `changed`'s texels inside `rect` (and the layer range, if any) of level 0. */
function mixRect( base, changed, width, height, layers, format, rect, range = [ 0, layers ] ) {

	const mixed = base.slice();
	const per = base.length / ( width * height * layers );
	for ( let layer = range[ 0 ]; layer < range[ 0 ] + range[ 1 ]; layer ++ ) for ( let y = rect.y; y < rect.y + rect.height; y ++ ) {

		const start = ( ( layer * height + y ) * width + rect.x ) * per;
		mixed.set( changed.subarray( start, start + rect.width * per ), start );

	}

	return mixed;

}

/** Every level bit for bit (NaN equal to NaN), or within 2⁻²⁰ relative for float32 (`ulps`). */
function same( actual, expected, label, ulps = false ) {

	for ( let level = 0; level < expected.length; level ++ ) {

		const a = actual[ level ], e = expected[ level ];
		for ( let i = 0; i < e.length; i ++ ) {

			if ( Object.is( a[ i ], e[ i ] ) || a[ i ] === e[ i ] ) continue;
			if ( ulps && Math.abs( a[ i ] - e[ i ] ) <= 2 ** - 20 * Math.max( 1, Math.abs( e[ i ] ) ) ) continue;
			throw new Error( `${ label }: level ${ level + 1 } value ${ i } is ${ a[ i ] }, expected ${ e[ i ] } (the plain chain's)` );

		}

	}

}
