import { formats, type SupportedFormat } from '../core/formats.js';
import { tileSize, workgroupSide } from '../core/plan.js';

export type ViewDimension = '2d' | '2d-array';

export interface KernelOptions {
	format: SupportedFormat;
	dimension: ViewDimension;
	/** Output levels written by one dispatch (1–6). */
	count: number;
}

/**
 * WGSL for one batch. Bindings: 0 sampler, 1 source level (sampled), 2… output
 * levels (write-only storage). Workgroup z selects the array layer.
 *
 * Each workgroup writes a tile×tile block of the first output (32×32 for six
 * levels, i.e. a 64×64 source tile as in SPD). Adjacent invocations sample and
 * store adjacent texels — the layout matters: assigning each invocation a 2×2
 * block measured ~30–45 % slower. Later levels are 2:1 reductions of the tile in
 * workgroup memory, with clamped reads for unit dimensions and partial tiles.
 */
export function createKernel( { format, dimension, count }: KernelOptions ): string {

	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const tile = tileSize( count );
	const side = workgroupSide( tile );
	const samples = tile / side;
	let code = `
@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var source: texture_2d${ suffix }<f32>;
var<workgroup> tile: array<vec4f, ${ tile * tile }>;
`;

	for ( let i = 0; i < count; i ++ ) {

		code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;

	}

	// Intermediate levels are quantized exactly as they would be stored. 8-bit
	// values stay in integer 0–255 units so 2:1 averages are exact and ties round
	// to even; with k/255 floats, float error turns ties into round-ups that drift
	// by several steps along long 1D tails.
	code += formats[ format ].quantization === 'unorm8' ? `
const SCALE = 255.0;
fn quantize(value: vec4f) -> vec4f {
	let scaled = clamp(value, vec4f(0), vec4f(255));
	let lower = floor(scaled);
	let fraction = scaled - lower;
	let nearest = select(lower, lower + vec4f(1), fraction > vec4f(0.5));
	return select(nearest, lower + vec4f(vec4u(lower) & vec4u(1)), fraction == vec4f(0.5));
}
` : `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f {
	return vec4f(unpack2x16float(pack2x16float(value.xy)), unpack2x16float(pack2x16float(value.zw)));
}
`;

	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	let size0 = textureDimensions(mip0);
	var value: vec4f;
`;

	for ( let i = 0; i < samples * samples; i ++ ) {

		code += `
	{
		let p = local.xy + vec2u(${ ( i % samples ) * side }u, ${ Math.floor( i / samples ) * side }u);
		let pos0 = group.xy * ${ tile }u + p;
		let uv = (vec2f(min(pos0, size0 - vec2u(1))) + 0.5) / vec2f(size0);
		value = quantize(textureSampleLevel(source, imageSampler, uv${ layer }, 0.0) * SCALE);
		tile[p.y * ${ tile }u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0)${ layer }, value / SCALE); }
	}
`;

	}

	for ( let i = 1; i < count; i ++ ) {

		const extent = tile >> i;
		code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = textureDimensions(mip${ i - 1 }) - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = quantize((tile[lo.y * ${ tile }u + lo.x] + tile[lo.y * ${ tile }u + hi.x] + tile[hi.y * ${ tile }u + lo.x] + tile[hi.y * ${ tile }u + hi.x]) * 0.25);
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		tile[local.y * ${ tile }u + local.x] = value;
		let pos = group.xy * ${ extent }u + local.xy;
		if (all(pos < textureDimensions(mip${ i }))) { textureStore(mip${ i }, vec2i(pos)${ layer }, value / SCALE); }
	}
`;

	}

	return code + '}\n';

}
