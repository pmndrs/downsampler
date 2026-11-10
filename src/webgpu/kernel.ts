import { formats, type SupportedFormat } from '../core/formats.js';
import { tileSize, workgroupSide } from '../core/plan.js';
import { isBuiltinReduction, reductionKey, type Reduction } from '../core/reduction.js';

export type ViewDimension = '2d' | '2d-array';

/**
 * How the first output of a batch reads its source level:
 * - 'sample': one hardware bilinear sample (average only; needs a filterable format)
 * - 'load': `textureLoad` of the footprint, with bilinear weights computed in the
 *   shader for 'average'. Required for fold reductions and unfilterable formats.
 */
export type SourceAccess = 'sample' | 'load';

export interface KernelOptions {
	format: SupportedFormat;
	dimension: ViewDimension;
	/** Output levels written by one dispatch (1–6). */
	count: number;
	/** Default 'average'. */
	reduction?: Reduction;
	/** Default 'sample' for 'average', otherwise 'load' (the only option for folds). */
	access?: SourceAccess;
}

/** Bind group layout entries matching {@link createKernel}. */
export function kernelLayoutEntries( { format, dimension, count, access }: { format: SupportedFormat, dimension: ViewDimension, count: number, access: SourceAccess } ): GPUBindGroupLayoutEntry[] {

	const visibility = GPUShaderStage.COMPUTE;
	const entries: GPUBindGroupLayoutEntry[] = [];
	if ( access === 'sample' ) entries.push( { binding: 0, visibility, sampler: { type: 'filtering' } } );
	entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
	for ( let i = 0; i < count; i ++ ) entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format, viewDimension: dimension } } );
	return entries;

}

const builtinFolds = {
	min: 'min(a, b)',
	max: 'max(a, b)',
	minmax: 'vec4f(min(a.x, b.x), max(a.y, b.y), min(a.z, b.z), max(a.w, b.w))'
};

/**
 * WGSL for one batch. Bindings: 0 sampler ('sample' access only), 1 source level,
 * 2… output levels (write-only storage). Workgroup z selects the array layer.
 *
 * Each workgroup writes a tile×tile block of the first output (32×32 for six
 * levels, i.e. a 64×64 source tile as in SPD). Adjacent invocations read and
 * store adjacent texels — the layout matters: assigning each invocation a 2×2
 * block measured ~30–45 % slower. Later levels are 2:1 reductions of the tile in
 * workgroup memory, with clamped reads for unit dimensions and partial tiles.
 */
export function createKernel( { format, dimension, count, reduction = 'average', access }: KernelOptions ): string {

	reductionKey( reduction );
	const average = reduction === 'average';
	access ??= average ? 'sample' : 'load';
	if ( ! average && access !== 'load' ) throw new Error( 'downsampler: fold reductions read their source with textureLoad (access "load").' );

	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const unorm8 = formats[ format ].quantization === 'unorm8';
	const tile = tileSize( count );
	const side = workgroupSide( tile );
	const samples = tile / side;

	let code = access === 'sample' ? '@group(0) @binding(0) var imageSampler: sampler;\n' : '';
	code += `@group(0) @binding(1) var source: texture_2d${ suffix }<f32>;
var<workgroup> tile: array<vec4f, ${ tile * tile }>;
`;
	for ( let i = 0; i < count; i ++ ) code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;

	// Intermediate levels are quantized exactly as they would be stored. 8-bit
	// values stay in integer 0–255 units so 2:1 averages are exact and ties round
	// to even; with k/255 floats, float error turns ties into round-ups that drift
	// by several steps along long 1D tails.
	code += unorm8 ? `
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

	if ( ! average ) {

		// Folds see normalized values; the tile holds scaled ones.
		if ( isBuiltinReduction( reduction ) ) code += `fn fold(a: vec4f, b: vec4f) -> vec4f { return ${ builtinFolds[ reduction as keyof typeof builtinFolds ] }; }\n`;
		else code += `${ reduction.wgsl }\nfn fold(a: vec4f, b: vec4f) -> vec4f { return ${ unorm8 ? 'reduce(a / SCALE, b / SCALE) * SCALE' : 'reduce(a, b)' }; }\n`;

	}

	if ( access === 'load' ) {

		// unorm8 texels are rounded back to exact integers so later sums are exact.
		code += `
fn load(p: vec2u, layer: u32) -> vec4f {
	let v = textureLoad(source, vec2i(p)${ array ? ', i32(layer)' : '' }, 0) * SCALE;
	return ${ unorm8 ? 'round(v)' : 'v' };
}
`;

	}

	code += `
fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
`;

	if ( access === 'sample' ) {

		code += `	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv${ array ? ', i32(layer)' : '' }, 0.0) * SCALE);
`;

	} else if ( average ) {

		// Bilinear weights at the output texel centre, as the sampler would apply
		// them but without its fixed-point weight rounding.
		code += `	let sourceSize = textureDimensions(source);
	let last = sourceSize - vec2u(1);
	let s = (vec2f(pos) + 0.5) * vec2f(sourceSize) / vec2f(size) - 0.5;
	let base = floor(s);
	let f = s - base;
	let lo = vec2u(clamp(vec2i(base), vec2i(0), vec2i(last)));
	let hi = vec2u(clamp(vec2i(base) + 1, vec2i(0), vec2i(last)));
	let top = mix(load(lo, layer), load(vec2u(hi.x, lo.y), layer), f.x);
	let bottom = mix(load(vec2u(lo.x, hi.y), layer), load(hi, layer), f.x);
	return quantize(mix(top, bottom, f.y));
`;

	} else {

		// The footprint: 2x and 2x + 1, plus 2x + 2 along odd source dimensions.
		code += `	let sourceSize = textureDimensions(source);
	let last = sourceSize - vec2u(1);
	let lo = min(pos * 2u, last);
	let hi = min(lo + vec2u(1), last);
	var v = fold(fold(load(lo, layer), load(vec2u(hi.x, lo.y), layer)), fold(load(vec2u(lo.x, hi.y), layer), load(hi, layer)));
	let odd = ((sourceSize & vec2u(1)) == vec2u(1)) & (sourceSize > vec2u(1));
	let far = min(lo + vec2u(2), last);
	if (odd.x) { v = fold(v, fold(load(vec2u(far.x, lo.y), layer), load(vec2u(far.x, hi.y), layer))); }
	if (odd.y) { v = fold(v, fold(load(vec2u(lo.x, far.y), layer), load(vec2u(hi.x, far.y), layer))); }
	if (odd.x && odd.y) { v = fold(v, load(far, layer)); }
	return quantize(v);
`;

	}

	code += `}

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
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * ${ tile }u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0)${ layer }, value / SCALE); }
	}
`;

	}

	const combine = ( a: string, b: string, c: string, d: string ) => average ? `quantize((${ a } + ${ b } + ${ c } + ${ d }) * 0.25)` : `quantize(fold(fold(${ a }, ${ b }), fold(${ c }, ${ d })))`;

	for ( let i = 1; i < count; i ++ ) {

		const extent = tile >> i;
		const at = ( x: string, y: string ) => `tile[${ y }.y * ${ tile }u + ${ x }.x]`;
		code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = textureDimensions(mip${ i - 1 }) - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = ${ combine( at( 'lo', 'lo' ), at( 'hi', 'lo' ), at( 'lo', 'hi' ), at( 'hi', 'hi' ) ) };
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
