// Mutation checks for the packed formats and the buffer output path (SPEC §6.1): patch the
// built dist/, run the packed GPU suite, record whether (and where) it fails.
// Run after npm run build: node results/probes/mutate-packed.mjs [--quick] [--names=bump,layers]
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL( '../..', import.meta.url ).pathname.replace( /\/$/, '' );
const extra = process.argv.slice( 2 );
const only = extra.find( a => a.startsWith( '--names=' ) )?.slice( 8 ).split( ',' );
const runArgs = extra.filter( a => ! a.startsWith( '--names=' ) );
const mutations = [
	[ 'exponent from log2', 'dist/webgpu/kernel.js', 'i32((bitcast<u32>(m) >> 23u) & 0xffu) - 127', 'i32(floor(log2(m)))' ],
	[ 'no 512 bump', 'dist/webgpu/kernel.js', '== 512.0) { e += 1; }', '== 512.0) { }' ],
	[ 'rgb9e5 round half to even (WGSL round)', 'dist/webgpu/kernel.js', 'return select(f, f + vec3f(1.0), v - f >= vec3f(0.5));', 'return round(v);' ],
	[ 'rgb9e5 NaN not cleared', 'dist/webgpu/kernel.js', 'let c = select(clamp(value, vec3f(0.0), vec3f(65408.0)), vec3f(0.0), nan);', 'let c = clamp(value, vec3f(0.0), vec3f(65408.0));' ],
	[ 'rg11b10 ties round up, not to even', 'dist/webgpu/kernel.js', '(remainder == halfway && (code & 1u) == 1u)', '(remainder == halfway)' ],
	[ 'rg11b10 truncates (round toward zero)', 'dist/webgpu/kernel.js', 'if (remainder > halfway || (remainder == halfway && (code & 1u) == 1u)) { code += 1u; }', '' ],
	[ 'rg11b10 overflow to Inf', 'dist/webgpu/kernel.js', 'return min(code, inf - 1u);', 'return code;' ],
	[ 'rg11b10 subnormal decode exponent', 'dist/webgpu/kernel.js', 'f32(mantissa) * ds_pow2(-14 - i32(bits))', 'f32(mantissa) * ds_pow2(-15 - i32(bits))' ],
	[ 'no intermediate quantization (packed)', 'dist/webgpu/kernel.js', 'fn quantize(value: vec4f) -> vec4f { return vec4f(${unpack}(${pack}(value.xyz)), 1.0); }', 'fn quantize(value: vec4f) -> vec4f { return vec4f(value.xyz, 1.0); }' ],
	[ 'unpadded rows (bytesPerRow = width × bytes)', 'dist/core/plan.js', 'const bytesPerRow = Math.ceil(width * bytesPerTexel / 256) * 256;', 'const bytesPerRow = width * bytesPerTexel;' ],
	[ 'kernel row stride ignores padding', 'dist/webgpu/downsampler.js', 'words[12 + i] = level.bytesPerRow / 4;', 'words[12 + i] = level.width * formats[format].bytesPerTexel / 4;' ],
	[ 'layers share one region', 'dist/webgpu/kernel.js', '(layer * ds_output.rows[i / 4u][i % 4u] + texel.y)', '(texel.y)' ],
	[ 'level offsets not 256-aligned (packed back to back)', 'dist/core/plan.js', 'size += bytesPerRow * height * layers;', 'size += width * bytesPerTexel * height * layers;' ],
	[ 'only the first level of a batch copied', 'dist/webgpu/downsampler.js', 'for (const copy of batch.copies ?? []) {', 'for (const copy of (batch.copies ?? []).slice(0, 1)) {' ]
];

const results = [];
for ( const [ name, file, from, to ] of mutations ) {

	if ( only && ! only.some( n => name.includes( n ) ) ) continue;
	const path = `${ root }/${ file }`;
	const original = readFileSync( path, 'utf8' );
	if ( ! original.includes( from ) ) { results.push( { name, status: 'PATCH NOT FOUND' } ); console.log( name, 'PATCH NOT FOUND' ); continue; }
	writeFileSync( path, original.replace( from, to ) );
	const started = Date.now();
	let status, detail = '';
	try {

		execFileSync( 'node', [ 'test/run.js', '--only=packed', ...runArgs ], { cwd: root, stdio: 'pipe', maxBuffer: 1 << 26 } );
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
