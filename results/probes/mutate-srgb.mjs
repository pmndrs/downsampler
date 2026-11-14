// Mutation checks for sRGB destinations (M2.3.5, SPEC §6.1 sRGB): patch the built dist/,
// run the sRGB part of the GPU suite (main-matrix sRGB formats, sRGB sources,
// validate-srgb.js), record whether (and where) it fails. Core-oracle mutations are
// covered by test/core/srgb.test.js (the standard's encode formula is its independent oracle).
// Run after npm run build: node results/probes/mutate-srgb.mjs [--quick] [--names=snap,alpha]
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL( '../..', import.meta.url ).pathname.replace( /\/$/, '' );
const extra = process.argv.slice( 2 );
const only = extra.find( a => a.startsWith( '--names=' ) )?.slice( 8 ).split( ',' );
const runArgs = extra.filter( a => ! a.startsWith( '--names=' ) );
const kernel = 'dist/webgpu/kernel.js';
// Encoded-space helpers for the mutations that average sRGB codes instead of linear light.
const encoded = `fn ds_mut_e(x: vec3f) -> vec3f { return 1.055 * pow(clamp(x, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.4)) - 0.055; }
fn ds_mut_d(x: vec3f) -> vec3f { return pow((x + 0.055) / 1.055, vec3f(2.4)); }
fn ds_mut_avg(a: vec4f, b: vec4f, c: vec4f, d: vec4f) -> vec4f { return vec4f(ds_mut_d((ds_mut_e(a.xyz) + ds_mut_e(b.xyz) + ds_mut_e(c.xyz) + ds_mut_e(d.xyz)) * 0.25), (a.w + b.w + c.w + d.w) * 0.25); }
fn ds_srgb_quantize(v: vec3f) -> vec3f {`;
const mutations = [
	// No decode: the kernel averages the stored codes, as a UNORM alias of the texture would
	// (SPD's sRGB path in WebGPU terms): loads read codes, levels are quantized and stored as codes.
	[ 'no decode (codes averaged as UNORM)', [
		[ 'let v = vec4f(ds_srgb_quantize(t.xyz), t.w) * SCALE;', 'let v = vec4f(vec3f(f32(ds_srgb_code(t.x)), f32(ds_srgb_code(t.y)), f32(ds_srgb_code(t.z))) / 255.0, t.w) * SCALE;' ],
		[ 'return vec3f(ds_srgb_table[ds_srgb_code(v.x)], ds_srgb_table[ds_srgb_code(v.y)], ds_srgb_table[ds_srgb_code(v.z)]);', 'return round(clamp(v, vec3f(0.0), vec3f(1.0)) * 255.0) / 255.0;' ],
		[ 'return [`ds_srgb_code(v.${r}) | (ds_srgb_code(v.${g}) << 8u) | (ds_srgb_code(v.${b}) << 16u) | (${alpha} << 24u)`];', 'return [`u32(round(clamp(v.${r}, 0.0, 1.0) * 255.0)) | (u32(round(clamp(v.${g}, 0.0, 1.0) * 255.0)) << 8u) | (u32(round(clamp(v.${b}, 0.0, 1.0) * 255.0)) << 16u) | (${alpha} << 24u)`];' ]
	] ],
	// Linear-space filtering skipped in the tile: each 2×2 average is taken of encoded values.
	[ 'in-tile averages in encoded space', [
		[ 'fn ds_srgb_quantize(v: vec3f) -> vec3f {', encoded ],
		[ 'const combine = (a, b, c, d) => average ? `quantize((${a} + ${b} + ${c} + ${d}) * 0.25)`', 'const combine = (a, b, c, d) => average ? `quantize(ds_mut_avg(${a}, ${b}, ${c}, ${d}))`' ]
	] ],
	[ 'loads not snapped to the exact decode (hardware decode kept)', [
		[ 'let v = vec4f(ds_srgb_quantize(t.xyz), t.w) * SCALE;', 'let v = t * SCALE;' ]
	] ],
	[ 'encode truncates (floor, no correction)', [
		[ 'var k = u32(clamp(round(e * 255.0), 0.0, 255.0));\n\tif (k < 255u && c >= ds_srgb_table[256u + k]) { k += 1u; } else if (k > 0u && c < ds_srgb_table[255u + k]) { k -= 1u; }', 'var k = u32(clamp(floor(e * 255.0), 0.0, 255.0));' ]
	] ],
	[ 'encode by pow alone (no threshold correction)', [
		[ 'if (k < 255u && c >= ds_srgb_table[256u + k]) { k += 1u; } else if (k > 0u && c < ds_srgb_table[255u + k]) { k -= 1u; }', '' ]
	] ],
	[ 'encode with gamma 2.2 instead of the sRGB curve', [
		[ 'let e = select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);\n\tvar k = u32(clamp(round(e * 255.0), 0.0, 255.0));\n\tif (k < 255u && c >= ds_srgb_table[256u + k]) { k += 1u; } else if (k > 0u && c < ds_srgb_table[255u + k]) { k -= 1u; }', 'let e = pow(c, 1.0 / 2.2);\n\tvar k = u32(clamp(round(e * 255.0), 0.0, 255.0));' ]
	] ],
	[ 'no intermediate quantization (sRGB)', [
		[ 'return vec4f(ds_srgb_quantize(value.xyz), alpha);', 'return vec4f(value.xyz, alpha);' ]
	] ],
	[ 'alpha ties round up, not to even', [
		[ 'let alpha = select(select(lower, lower + 1.0, fraction > 0.5), lower + f32(i32(lower) & 1), fraction == 0.5);', 'let alpha = select(lower, lower + 1.0, fraction >= 0.5);' ]
	] ],
	[ 'alpha encoded as sRGB', [
		[ "const alpha = normalized ? 'u32(floor(clamp(v.w, 0.0, 1.0) * 255.0 + 0.5))' : 'u32(v.w)';", "const alpha = normalized ? 'ds_srgb_code(v.w)' : 'ds_srgb_code(v.w / 255.0)';" ]
	] ],
	[ 'bgra8unorm-srgb in rgba byte order', [
		[ "const [r, g, b] = format === 'bgra8unorm-srgb' ? ['z', 'y', 'x'] : ['x', 'y', 'z'];", "const [r, g, b] = ['x', 'y', 'z'];" ]
	] ]
];

const results = [];
for ( const [ name, patches ] of mutations ) {

	if ( only && ! only.some( n => name.includes( n ) ) ) continue;
	const path = `${ root }/${ kernel }`;
	const original = readFileSync( path, 'utf8' );
	const missing = patches.filter( ( [ from ] ) => ! original.includes( from ) );
	if ( missing.length ) { results.push( { name, status: 'PATCH NOT FOUND' } ); console.log( name, 'PATCH NOT FOUND', missing[ 0 ][ 0 ].slice( 0, 80 ) ); continue; }
	writeFileSync( path, patches.reduce( ( text, [ from, to ] ) => text.replace( from, to ), original ) );
	const started = Date.now();
	let status, detail = '';
	try {

		execFileSync( 'node', [ 'test/run.js', '--only=srgb', ...runArgs ], { cwd: root, stdio: 'pipe', maxBuffer: 1 << 26 } );
		status = 'MISSED';

	} catch ( error ) {

		status = 'caught';
		detail = String( error.stderr || error.stdout ).split( '\n' ).find( line => /Error|error|differs|expected|actual/.test( line ) )?.slice( 0, 240 ) ?? '';

	} finally {

		writeFileSync( path, original );

	}

	results.push( { name, status, seconds: Math.round( ( Date.now() - started ) / 1000 ), detail } );
	console.log( `${ status }\t${ name }\t${ detail }` );

}

console.log( JSON.stringify( results, null, 1 ) );
