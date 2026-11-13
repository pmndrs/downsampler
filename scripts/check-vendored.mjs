#!/usr/bin/env node
// Checks a vendored copy of the mipmap generator against the golden WGSL in test/golden/mipmap/,
// byte for byte (SPEC.md §10 Vendoring).
//
//   node scripts/check-vendored.mjs <vendored-file> [--golden=<dir>] [--diffs=<n>]
//
// The file is imported as an ES module from a temporary copy (relative imports resolved
// against its own directory, so an engine file can import its siblings). Its generator is:
// - an exported `createMipmapKernel( options )` (dist/mipmap.js, or a copy of it), or
// - a module-level `function createKernel(...)`: called with an options object if it takes one
//   parameter, else positionally as ( format, dimension, count, access, tail, tile ), the shape
//   three.js's WebGPUTextureMipmapUtils uses (no f16Rounding: 'device' cases aren't covered).
// Golden cases the copy rejects (formats it leaves out) are reported as not covered, which
// doesn't fail the check. Exit status: 0 every covered case matches, 1 some differ, 2 no
// generator or nothing covered.
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = process.argv.slice( 2 );
const flag = ( name, fallback ) => args.find( arg => arg.startsWith( `--${ name }=` ) )?.slice( name.length + 3 ) ?? fallback;
const file = args.find( arg => ! arg.startsWith( '--' ) );
if ( file === undefined ) {

	console.error( 'usage: node scripts/check-vendored.mjs <vendored-file> [--golden=<dir>] [--diffs=<n>]' );
	process.exit( 2 );

}

const goldenDir = resolve( flag( 'golden', fileURLToPath( new URL( '../test/golden/mipmap/', import.meta.url ) ) ) );
const maxDiffs = Number( flag( 'diffs', 3 ) );
const cases = Object.entries( JSON.parse( readFileSync( join( goldenDir, 'cases.json' ), 'utf8' ) ) );
if ( readdirSync( goldenDir ).length !== cases.length + 1 ) console.warn( `warning: ${ goldenDir } holds files that cases.json doesn't list.` );

// Import the copy with its relative imports made absolute.
const path = resolve( file );
let source = readFileSync( path, 'utf8' );
const absolute = specifier => pathToFileURL( resolve( dirname( path ), specifier ) ).href;
source = source.replace( /(\bfrom\s*|\bimport\s*\(?\s*)(['"])(\.\.?\/[^'"]+)\2/g, ( _, head, quote, specifier ) => `${ head }${ quote }${ absolute( specifier ) }${ quote }` );
let name;
if ( /export\s+function\s+createMipmapKernel\b|export\s*\{[^}]*\bcreateMipmapKernel\b/.test( source ) ) name = 'createMipmapKernel';
else if ( /(^|\n)\s*(export\s+)?function\s+createKernel\s*\(/.test( source ) ) {

	name = '__checkVendoredKernel';
	source += '\nexport { createKernel as __checkVendoredKernel };\n';

} else {

	console.error( `${ file }: no exported createMipmapKernel and no module-level function createKernel.` );
	process.exit( 2 );

}

const tmp = mkdtempSync( join( tmpdir(), 'downsampler-check-vendored-' ) );
let generator;
try {

	const copy = join( tmp, 'vendored.mjs' );
	writeFileSync( copy, source );
	generator = ( await import( pathToFileURL( copy ).href ) )[ name ];

} finally {

	rmSync( tmp, { recursive: true, force: true } );

}

const positional = generator.length > 1;
const generate = options => positional ? generator( options.format, options.dimension, options.count, options.access, options.tail, options.tile ) : generator( options );

/** Line diff of golden (-) against vendored (+), with two lines of context. */
function diff( a, b ) {

	const x = a.split( '\n' ), y = b.split( '\n' );
	// Longest common subsequence, by suffix lengths.
	const lcs = Array.from( { length: x.length + 1 }, () => new Uint32Array( y.length + 1 ) );
	for ( let i = x.length - 1; i >= 0; i -- ) for ( let j = y.length - 1; j >= 0; j -- ) lcs[ i ][ j ] = x[ i ] === y[ j ] ? lcs[ i + 1 ][ j + 1 ] + 1 : Math.max( lcs[ i + 1 ][ j ], lcs[ i ][ j + 1 ] );
	const ops = [];
	let i = 0, j = 0;
	while ( i < x.length || j < y.length ) {

		if ( i < x.length && j < y.length && x[ i ] === y[ j ] ) { ops.push( [ ' ', x[ i ++ ], i ] ); j ++; }
		else if ( j < y.length && ( i === x.length || lcs[ i ][ j + 1 ] >= lcs[ i + 1 ][ j ] ) ) ops.push( [ '+', y[ j ++ ], i ] );
		else ops.push( [ '-', x[ i ++ ], i ] );

	}

	const keep = ops.map( ( op, k ) => ops.slice( Math.max( 0, k - 2 ), k + 3 ).some( near => near[ 0 ] !== ' ' ) );
	const lines = [];
	ops.forEach( ( [ kind, line, at ], k ) => {

		if ( ! keep[ k ] ) return;
		if ( k > 0 && ! keep[ k - 1 ] ) lines.push( `@@ golden line ${ at } @@` );
		lines.push( `${ kind } ${ line }` );

	} );
	return lines.join( '\n' );

}

const matched = [], uncovered = [], differ = new Map();
for ( const [ caseName, options ] of cases ) {

	const golden = readFileSync( join( goldenDir, caseName ), 'utf8' );
	if ( positional && options.f16Rounding !== 'nearest' ) {

		uncovered.push( `${ caseName } (a positional generator takes no f16Rounding)` );
		continue;

	}

	let code;
	try {

		code = generate( options );

	} catch ( error ) {

		uncovered.push( `${ caseName } (threw: ${ error.message })` );
		continue;

	}

	if ( code === golden ) matched.push( caseName );
	else {

		const text = diff( golden, String( code ) );
		if ( ! differ.has( text ) ) differ.set( text, [] );
		differ.get( text ).push( caseName );

	}

}

const different = [ ...differ.values() ].flat();
console.log( `check-vendored: ${ file } (${ name === 'createMipmapKernel' ? 'createMipmapKernel' : `createKernel, ${ positional ? 'positional' : 'options' }` })` );
console.log( `  ${ matched.length } of ${ cases.length } golden kernels match` );
if ( different.length > 0 ) console.log( `  ${ different.length } differ (${ differ.size } distinct diffs): ${ different.join( ', ' ) }` );
if ( uncovered.length > 0 ) console.log( `  ${ uncovered.length } not covered:\n    ${ uncovered.join( '\n    ' ) }` );
if ( differ.size > 0 ) {

	// Changed lines across every differing case (digits as #), most frequent first.
	const tally = new Map();
	for ( const [ text, names ] of differ ) {

		const lines = new Set( text.split( '\n' ).filter( line => line[ 0 ] === '-' || line[ 0 ] === '+' ).map( line => line.replace( /\d+/g, '#' ).trimEnd() ) );
		for ( const line of lines ) tally.set( line, ( tally.get( line ) ?? 0 ) + names.length );

	}

	console.log( '\n  changed lines (- golden, + vendored; digits as #), by cases:' );
	for ( const [ line, count ] of [ ...tally ].sort( ( a, b ) => b[ 1 ] - a[ 1 ] ) ) console.log( `  ${ String( count ).padStart( 5 ) }  ${ line }` );

}

[ ...differ ].slice( 0, maxDiffs ).forEach( ( [ text, names ], k ) => {

	console.log( `\n--- diff ${ k + 1 } of ${ differ.size }, ${ names.length } case${ names.length > 1 ? 's' : '' } (- golden, + vendored), e.g. ${ names[ 0 ] } ---\n${ text }` );

} );
process.exit( different.length > 0 ? 1 : matched.length === 0 ? 2 : 0 );
