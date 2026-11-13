// Diff native level dumps (d3d12-mips --dump=<folder>) against this library's CPU oracles.
//
//   spd-*                  spdChainReference( …, levelsPerDispatch 6 ): AMD SPD semantics as
//                          our oddSizes: 'spd' implements them (floor sizes, unrounded within a
//                          batch, the next batch reads stored mip 6)
//   render*, d3d11-*, blit 'exact' level by level (reduceReference): render-pass semantics
//   nvpro-*                'box' level by level (defined here; the library's 'box' is planned for
//                          M2.5): nvpro_pyramid's area weights (n−x, n, x+1)/(2n+1) for size 2n+1,
//                          each level from the stored one below it; max over the same footprint
//
// Each reference starts from the dumped level 0, so the input generator doesn't matter.
// Usage: npm run build && node native/d3d12-mips/check-dump.mjs <dump folder> [--json=out.json]
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { quantize, reduceReference, spdChainReference, tolerance } from '../../dist/index.js';

/** Source texels and weights for output x along an axis of `size` texels (nvpro_pyramid). */
function boxTaps( size, x ) {

	if ( size === 1 ) return [ [ 0, 1 ] ];
	if ( size % 2 === 0 ) return [ [ 2 * x, 0.5 ], [ 2 * x + 1, 0.5 ] ];
	const n = ( size - 1 ) / 2;
	return [ [ 2 * x, ( n - x ) / ( 2 * n + 1 ) ], [ 2 * x + 1, n / ( 2 * n + 1 ) ], [ 2 * x + 2, ( x + 1 ) / ( 2 * n + 1 ) ] ];

}

/** One 'box' level from RGBA `source` (w × h), quantized as stored. */
function boxLevel( source, w, h, quantization, reduction ) {

	const width = Math.max( 1, w >> 1 ), height = Math.max( 1, h >> 1 ), values = new Float32Array( width * height * 4 );
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let c = 0; c < 4; c ++ ) {

		let value = reduction === 'max' ? - Infinity : 0;
		for ( const [ sy, wy ] of boxTaps( h, y ) ) for ( const [ sx, wx ] of boxTaps( w, x ) ) {

			const texel = source[ ( sy * w + sx ) * 4 + c ];
			value = reduction === 'max' ? Math.max( value, texel ) : value + wx * wy * texel;

		}
		values[ ( y * width + x ) * 4 + c ] = quantize( value, quantization );

	}
	return { width, height, values, filterTolerance: new Float32Array( values.length ) };

}

const folder = process.argv[ 2 ];
const jsonOut = process.argv.find( arg => arg.startsWith( '--json=' ) )?.slice( 7 );
if ( ! folder ) throw new Error( 'usage: node check-dump.mjs <dump folder> [--json=out.json]' );

const quantizations = { rgba8unorm: 'unorm8', rgba16float: 'float16', r32float: 'float32' };
// One storage step near 1 (the inputs are in [0, 1)): 8-bit unit, f16 ulp at [0.5, 1), f32 ulp.
const steps = { rgba8unorm: 1 / 255, rgba16float: 2 ** -11, r32float: 2 ** -24 };

const report = [];
for ( const file of readdirSync( folder ).filter( name => name.endsWith( '.json' ) ).sort() ) {

	const meta = JSON.parse( readFileSync( join( folder, file ), 'utf8' ) );
	const raw = readFileSync( join( folder, file.replace( /\.json$/, '.bin' ) ) );
	const all = new Float32Array( raw.buffer, raw.byteOffset, raw.byteLength / 4 );
	const levels = [];
	let offset = 0;
	for ( const [ w, h ] of meta.levels ) { levels.push( all.subarray( offset, offset + w * h * 4 ) ); offset += w * h * 4; }

	const { width, height, format, variant } = meta;
	const quantization = quantizations[ format ];
	const reduction = format === 'r32float' ? 'max' : 'average';
	const isSpd = variant.startsWith( 'spd-' ), isBox = variant.startsWith( 'nvpro-' );
	const channels = format === 'r32float' ? 1 : 4;

	let reference;
	if ( isSpd ) reference = spdChainReference( levels[ 0 ], width, height, 1, quantization, reduction, levels.length - 1, 6 ); // ReferenceLevel objects
	else {

		reference = [];
		let input = Float32Array.from( levels[ 0 ] ), w = width, h = height;
		for ( let level = 1; level < levels.length; level ++ ) {

			const next = isBox ? boxLevel( input, w, h, quantization, reduction ) : reduceReference( input, w, h, 1, quantization, reduction );
			reference.push( next );
			input = Float32Array.from( next.values ); w = next.width; h = next.height;

		}

	}

	const perLevel = [];
	for ( let level = 1; level < levels.length; level ++ ) {

		const actual = levels[ level ], expected = reference[ level - 1 ].values, allowance = reference[ level - 1 ].filterTolerance;
		if ( actual.length !== expected.length ) throw new Error( `${ file } level ${ level }: ${ actual.length / 4 } texels, reference ${ expected.length / 4 }` );
		let max = 0, differing = 0, values = 0, outside = 0;
		for ( let i = 0; i < actual.length; i ++ ) {

			if ( i % 4 >= channels ) continue;
			values ++;
			const error = Math.abs( actual[ i ] - expected[ i ] );
			if ( error > 0 ) differing ++;
			if ( error > tolerance( expected[ i ], quantization, allowance[ i ] ?? 0, reduction ) ) outside ++;
			max = Math.max( max, error );

		}
		perLevel.push( { level, size: meta.levels[ level ], maxSteps: +( max / steps[ format ] ).toFixed( 3 ), differing, outside, values } );

	}

	const maxSteps = Math.max( ...perLevel.map( level => level.maxSteps ) );
	const outside = perLevel.reduce( ( sum, level ) => sum + level.outside, 0 ), failing = perLevel.filter( level => level.outside ).map( level => level.level );
	const differing = perLevel.reduce( ( sum, level ) => sum + level.differing, 0 ), values = perLevel.reduce( ( sum, level ) => sum + level.values, 0 );
	report.push( { variant, format, width, height, oracle: isSpd ? 'spdChainReference (6 per batch)' : isBox ? "'box' level by level" : "'exact' level by level", maxSteps, differing, outside, failingLevels: failing, values, perLevel } );
	console.log( `${ variant.padEnd( 24 ) } ${ format.padEnd( 12 ) } ${ width }x${ height } vs ${ isSpd ? 'spd  ' : isBox ? 'box  ' : 'exact' }: max ${ maxSteps } steps, ${ differing }/${ values } values differ (${ ( 100 * differing / values ).toFixed( 3 ) } %), ${ outside } outside tolerance()${ failing.length ? ` at levels ${ failing.join( ', ' ) }` : '' }` );

}

if ( jsonOut ) writeFileSync( jsonOut, JSON.stringify( { about: 'Native level dumps diffed against the library oracles; maxSteps is in storage steps near 1 (8-bit unit, f16 ulp in [0.5, 1), f32 ulp)', report }, null, '\t' ) + '\n' );
