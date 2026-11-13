// Golden WGSL of `downsampler/mipmap` (SPEC.md §10 Vendoring). Fails when the generator's
// output changes; after an intended change, regenerate with
//   npm run build && node test/core/golden.test.js --update
// and commit test/golden/mipmap/. Vendored copies are checked against these files with
// scripts/check-vendored.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createMipmapKernel, mipmapAccess, mipmapFormats } from '../../dist/mipmap.js';
import { createKernel } from '../../dist/webgpu/kernel.js';

const dir = new URL( '../golden/mipmap/', import.meta.url );

/** The fixed option sets: every format at a 4-level batch and a 6-level tail with its usual access; every batch shape for rgba8unorm; both accesses, both dimensions and 'device' float16 rounding. */
function goldenCases() {

	const cases = [];
	const add = ( format, dimension, [ tail, count, tile ], access, f16Rounding = 'nearest' ) => cases.push( { format, dimension, count, access, tail, tile, f16Rounding } );
	for ( const format of Object.keys( mipmapFormats ) ) for ( const shape of [ [ false, 4, 8 ], [ true, 6, 32 ] ] ) add( format, '2d', shape, mipmapAccess( format ) );
	for ( const shape of [ [ false, 1, 8 ], [ false, 1, 32 ], [ false, 4, 16 ], [ false, 5, 16 ], [ false, 6, 32 ], [ true, 2, 8 ], [ true, 4, 16 ] ] ) add( 'rgba8unorm', '2d', shape, 'sample' );
	for ( const shape of [ [ false, 6, 32 ], [ true, 6, 32 ] ] ) add( 'rgba8unorm', '2d-array', shape, 'load' );
	for ( const shape of [ [ false, 1, 8 ], [ false, 6, 32 ], [ true, 2, 8 ] ] ) add( 'rgba16float', '2d', shape, 'load' );
	add( 'rgba16float', '2d', [ false, 5, 16 ], 'sample' );
	for ( const shape of [ [ false, 4, 16 ], [ true, 4, 16 ] ] ) add( 'rgba32float', '2d-array', shape, 'load' );
	add( 'rgba32float', '2d', [ false, 4, 8 ], 'sample' );
	add( 'rgba16float', '2d', [ false, 4, 8 ], 'sample', 'device' );
	add( 'r16float', '2d', [ true, 6, 32 ], 'sample', 'device' );
	return cases.map( options => [ caseName( options ), options ] ).sort( ( a, b ) => a[ 0 ] < b[ 0 ] ? - 1 : 1 );

}

function caseName( { format, dimension, count, access, tail, tile, f16Rounding } ) {

	return `${ format }.${ dimension }.${ tail ? 'tail' : 'exact' }${ count }.tile${ tile }.${ access }${ f16Rounding === 'device' ? '.f16device' : '' }.wgsl`;

}

const cases = goldenCases();

if ( process.argv.includes( '--update' ) ) {

	rmSync( dir, { recursive: true, force: true } );
	mkdirSync( dir, { recursive: true } );
	for ( const [ name, options ] of cases ) writeFileSync( new URL( name, dir ), createMipmapKernel( options ) );
	writeFileSync( new URL( 'cases.json', dir ), JSON.stringify( Object.fromEntries( cases ), null, '\t' ) + '\n' );
	console.log( `wrote ${ cases.length } golden kernels to ${ fileURLToPath( dir ) }` );

}

test( 'the golden manifest lists exactly the golden cases', () => {

	assert.deepEqual( JSON.parse( readFileSync( new URL( 'cases.json', dir ), 'utf8' ) ), Object.fromEntries( cases ) );
	const files = readdirSync( dir ).filter( name => name !== 'cases.json' ).sort();
	assert.deepEqual( files, cases.map( ( [ name ] ) => name ) );

} );

test( 'createMipmapKernel and createKernel produce the golden WGSL byte for byte (rerun with --update after an intended change)', () => {

	for ( const [ name, options ] of cases ) {

		const golden = readFileSync( new URL( name, dir ), 'utf8' );
		assert.equal( createMipmapKernel( options ), golden, name );
		assert.equal( createKernel( { ...options, f16Pack: false } ), golden, name );

	}

} );

test( 'check-vendored passes dist/mipmap.js and reports a drifted copy', () => {

	const script = fileURLToPath( new URL( '../../scripts/check-vendored.mjs', import.meta.url ) );
	const run = file => spawnSync( process.execPath, [ script, file ], { encoding: 'utf8' } );
	const self = run( fileURLToPath( new URL( '../../dist/mipmap.js', import.meta.url ) ) );
	assert.equal( self.status, 0, self.stdout + self.stderr );
	assert.match( self.stdout, new RegExp( `${ cases.length } of ${ cases.length } golden kernels match` ) );

	// A copy with one changed line fails, naming the cases.
	const tmp = new URL( '../../dist/.check-vendored-mutant.js', import.meta.url );
	const code = readFileSync( new URL( '../../dist/mipmap.js', import.meta.url ), 'utf8' );
	writeFileSync( tmp, code.replace( 'fraction == vec4f(0.5)', 'fraction >= vec4f(0.5)' ) );
	try {

		const mutant = run( fileURLToPath( tmp ) );
		assert.equal( mutant.status, 1, mutant.stdout + mutant.stderr );
		assert.match( mutant.stdout, /rgba8unorm\.2d\.exact4\.tile8\.sample\.wgsl/ );
		assert.match( mutant.stdout, /fraction >= vec4f\(0\.5\)/ );

	} finally {

		rmSync( tmp, { force: true } );

	}

} );
