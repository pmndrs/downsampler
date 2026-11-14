// Mutation checks for the integer formats and rgb10a2unorm (SPEC §6.1, §7): patch the built
// dist/, run the matching GPU suite part, record whether (and where) it fails. Each run stops at
// its first failure, so caught mutations are quick.
// Run after npm run build: node results/probes/mutate-integer.mjs [--quick] [--names=ties,sign]
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL( '../..', import.meta.url ).pathname.replace( /\/$/, '' );
const extra = process.argv.slice( 2 );
const only = extra.find( a => a.startsWith( '--names=' ) )?.slice( 8 ).split( ',' );
const runArgs = extra.filter( a => ! a.startsWith( '--names=' ) );
const integer = 'dist/webgpu/integer.js', kernel = 'dist/webgpu/kernel.js';
// [ name, file, from, to, suite part ]
const mutations = [
	[ '2×2 mean: ties round up', integer, '(rem > vec4u(2u)) | ((rem == vec4u(2u)) & ((f & vec4u(1u)) == vec4u(1u)))', '(rem >= vec4u(2u))' ],
	[ '2×2 mean: truncates', integer, 'return f + select(vec4u(0u), vec4u(1u), (rem > vec4u(2u)) | ((rem == vec4u(2u)) & ((f & vec4u(1u)) == vec4u(1u))));', 'return f;' ],
	[ '2×2 mean: plain u32 sum (overflows at 32 bits, ties up)', integer, 'return f + select(vec4u(0u), vec4u(1u), (rem > vec4u(2u)) | ((rem == vec4u(2u)) & ((f & vec4u(1u)) == vec4u(1u))));', 'return (a + b + c + d + vec4u(2u)) >> vec4u(2u);' ],
	[ 'weighted mean: ties round up', integer, '(twice > vec4u(d)) | ((twice == vec4u(d)) & ((f & vec4u(1u)) == vec4u(1u)))', '(twice >= vec4u(d))' ],
	[ 'weighted mean: low half remainder dropped', integer, '	r += tLo % d;\n', '' ],
	[ 'weighted mean: long division a step short', integer, 'for (var i = 0u; i < 8u; i++) {', 'for (var i = 0u; i < 7u; i++) {' ],
	[ 'bilinear weights swapped', integer, 'let wx = vec2u(d.x - r.x, r.x);', 'let wx = vec2u(r.x, d.x - r.x);' ],
	[ 'area overlap off by one', integer, 'return min(n * (t + 1u), (x + 1u) * s) - max(n * t, x * s); }', 'return min(n * (t + 1u), (x + 1u) * s) - max(n * t, x * s) + 1u; }' ],
	[ 'fold footprint misses 2x + 2', integer, '& (${sourceSize} > vec2u(1));', '& (${sourceSize} > vec2u(1)) & vec2<bool>(false);' ],
	[ 'sign offset not removed on store', integer, "${signed ? 'bitcast<vec4i>(value ^ DS_BIAS)'", "${signed ? 'bitcast<vec4i>(value)'" ],
	[ 'no sign offset on load (signed compared as unsigned)', integer, "${signed ? ' ^ DS_BIAS' : ''};\n}", "${signed ? '' : ''};\n}" ],
	[ 'custom fold sees offset values', integer, "'bitcast<vec4u>(reduce(bitcast<vec4i>(a ^ DS_BIAS), bitcast<vec4i>(b ^ DS_BIAS))) ^ DS_BIAS'", "'bitcast<vec4u>(reduce(bitcast<vec4i>(a), bitcast<vec4i>(b)))'" ],
	[ 'no clamp to a narrower destination', integer, 'fn quantize(v: vec4u) -> vec4u { return clamp(v,', 'fn quantize(v: vec4u) -> vec4u { return v; }\nfn ds_unused(v: vec4u) -> vec4u { return clamp(v,' ],
	[ 'buffer path: 8-bit integers masked to 7 bits', kernel, "const mask = bits === 8 ? '0xffu' : '0xffffu';", "const mask = bits === 8 ? '0x7fu' : '0xffffu';" ],
	[ 'rgb10a2unorm: alpha in 10-bit units', kernel, 'const domains = [0, 1, 2, 3].map(c => channelDomain(quantization, c));', 'const domains = [0, 1, 2, 3].map(c => channelDomain(quantization, 0));', 'format:rgb10a2unorm' ],
	[ 'rgb10a2: buffer word with a at bit 28', kernel, '(${unit(\'w\', 3)} << 30u)', '(${unit(\'w\', 3)} << 28u)', 'format:rgb10a2unorm' ]
];

const results = [];
for ( const [ name, file, from, to, part = 'integer' ] of mutations ) {

	if ( only && ! only.some( n => name.includes( n ) ) ) continue;
	const path = `${ root }/${ file }`;
	const original = readFileSync( path, 'utf8' );
	if ( ! original.includes( from ) ) { results.push( { name, status: 'PATCH NOT FOUND' } ); console.log( name, 'PATCH NOT FOUND' ); continue; }
	writeFileSync( path, original.replace( from, to ) );
	const started = Date.now();
	let status, detail = '';
	try {

		// 'format:rgb10a2unorm' runs its main matrix: the storage path (tier1) and the buffer path's subset shapes.
		execFileSync( 'node', [ 'test/run.js', `--only=${ part }`, ...runArgs ], { cwd: root, stdio: 'pipe', maxBuffer: 1 << 26 } );
		status = 'MISSED';

	} catch ( error ) {

		status = 'caught';
		detail = String( error.stderr || error.stdout ).split( '\n' ).find( line => /Error|error|differs|expected|actual/.test( line ) )?.slice( 0, 220 ) ?? '';

	} finally {

		writeFileSync( path, original );

	}

	results.push( { name, status, seconds: Math.round( ( Date.now() - started ) / 1000 ), detail } );
	console.log( `${ status }\t${ name }\t${ detail }` );

}

console.log( JSON.stringify( results, null, 1 ) );
