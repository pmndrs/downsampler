import { formats, integerDomains, isIntegerDomain, isSupportedFormat, sampledChannels, type SupportedFormat } from '../core/formats.js';
import { tailTile, tileSize, workgroupSide } from '../core/plan.js';
import { checkLevels } from '../core/pyramid.js';
import { isBuiltinReduction, needsTileEdges, reductionKey, resolveReduction, type CustomReduction, type EdgePolicy, type Reduction } from '../core/reduction.js';

export type ViewDimension = '2d' | '2d-array';

/**
 * How the first output of a batch reads its source level:
 * - 'sample': one hardware bilinear sample (average only; needs a filterable format)
 * - 'load': `textureLoad` of the footprint, with bilinear weights computed in the
 *   shader for 'average'. Required for fold reductions and unfilterable formats.
 * Tile kernels ('zero-weight', 'clamp') load by default. Under 'clamp', an
 * all-'average' reduction without a load hook can 'sample' instead: one
 * bilinear sample at the shared corner of each 2×2 block (as SPD does).
 */
export type SourceAccess = 'sample' | 'load';

export interface KernelOptions {
	/** Storage format of the output levels. Required unless a `store` hook writes them. */
	format?: SupportedFormat;
	/** View dimension of the source and output textures. Default '2d'. */
	dimension?: ViewDimension;
	/** Levels reduced by one dispatch (1–6). Tile kernels default to the deepest of `levels`. */
	count?: number;
	/** Default 'average'. Channel ops and custom `kind: 'sum'` need a tile edge policy. */
	reduction?: Reduction;
	/**
	 * 'exact': default 'sample' for 'average', otherwise 'load' (the only option
	 * for folds). Tile kernels: default 'load'; 'sample' only under 'clamp' for
	 * an all-'average' reduction read from a texture (no load hook).
	 */
	access?: SourceAccess;
	/**
	 * 'exact' only: a tail kernel. One workgroup (dispatch 1×1×layers) writes all
	 * `count` levels; the first output (at most 32×32) reads the source level as
	 * usual, and every further level is reduced from the previous one in
	 * workgroup memory by the same 'exact' rule, odd sizes included.
	 */
	tail?: boolean;
	/**
	 * Format of the texture the first output reads, when it isn't `format` (a
	 * separate source, e.g. depth32float). Single-channel sources are splatted to
	 * all four channels; loads are rounded to exact integer units only when the
	 * source has the destination's normalized encoding ('exact' only).
	 */
	sourceFormat?: string;
	/**
	 * Default 'exact' (mip chains: render-pass parity, a batch per odd size).
	 * 'zero-weight' and 'clamp' select the tile kernel: every level of the dispatch
	 * in one tile, no intermediate rounding, sums unnormalised until emitted.
	 */
	edges?: EdgePolicy;
	/**
	 * Tile kernels: the levels to emit, source-relative (1 = half size), strictly
	 * increasing, at most 6. Default every level 1…count. Unemitted levels are
	 * reduced in workgroup memory but never stored, and get no binding.
	 */
	levels?: readonly number[];
	/** Tile kernels: WGSL defining `fn load(texel: vec2i, layer: u32) -> vec4f`, replacing the source texture. */
	load?: string;
	/** Tile kernels: WGSL defining `fn store(level: u32, texel: vec2i, layer: u32, value: vec4f)`, replacing the output textures. */
	store?: string;
	/** Tile kernels with a `load` hook: WGSL defining `fn size() -> vec2u`, the source size, replacing the size uniform. */
	size?: string;
	/** Tile kernels: WGSL declarations of the tool's `@group(1)` bindings, used by the hooks. */
	bindings?: string;
}

/** Group 0 binding of the size uniform (`struct { size: vec2u }`) of tile kernels with a load hook. */
export const sizeBinding = 8;

const hookPatterns = {
	load: /\bfn\s+load\s*\(/,
	store: /\bfn\s+store\s*\(/,
	size: /\bfn\s+size\s*\(/
};

/** Validates tile kernel options and resolves their defaults. */
function tileOptions( options: KernelOptions ) {

	const { format, dimension = '2d', reduction = 'average', edges, load, store, size, bindings = '', sourceFormat } = options;
	if ( edges !== 'zero-weight' && edges !== 'clamp' ) throw new Error( `downsampler: unknown edge policy ${ JSON.stringify( edges ) }; expected 'exact', 'zero-weight' or 'clamp'.` );
	const levels = options.levels ?? Array.from( { length: options.count ?? 0 }, ( _, i ) => i + 1 );
	checkLevels( levels );
	const count = levels[ levels.length - 1 ];
	if ( options.count !== undefined && options.count !== count ) throw new Error( `downsampler: count ${ options.count } must be the deepest emitted level (${ count }).` );
	const access = options.access ?? 'load';
	if ( access !== 'load' && access !== 'sample' ) throw new Error( `downsampler: unknown access ${ JSON.stringify( access ) }.` );
	if ( access === 'sample' && ( edges !== 'clamp' || load !== undefined || resolveReduction( reduction ).ops?.some( op => op !== 'average' ) !== false ) ) throw new Error( 'downsampler: tile kernels sample (access "sample") only under \'clamp\', for an all-\'average\' reduction read from a texture; otherwise they read with textureLoad.' );
	if ( options.tail !== undefined ) throw new Error( 'downsampler: tail is an \'exact\' kernel option.' );
	for ( const [ name, hook ] of Object.entries( { load, store, size } ) ) {

		if ( hook !== undefined && ( typeof hook !== 'string' || ! hookPatterns[ name as keyof typeof hookPatterns ].test( hook ) ) ) throw new Error( `downsampler: the ${ name } hook must be WGSL defining \`fn ${ name }(…)\`.` );

	}

	if ( size !== undefined && load === undefined ) throw new Error( 'downsampler: a size hook replaces the size uniform of a load hook; without a load hook the size is the source texture\'s.' );
	if ( typeof bindings !== 'string' ) throw new Error( 'downsampler: bindings must be WGSL declarations for @group(1).' );
	if ( store === undefined && ( format === undefined || ! isSupportedFormat( format ) ) ) throw new Error( `downsampler: output format ${ JSON.stringify( format ) } is not supported; pass a supported format or a store hook.` );
	if ( dimension !== '2d' && dimension !== '2d-array' ) throw new Error( `downsampler: view dimension ${ JSON.stringify( dimension ) } is not supported.` );
	return { format, dimension, reduction, edges, access, levels, count, load, store, size, bindings, sourceFormat: sourceFormat ?? format ?? 'rgba32float' };

}

/** Bind group layout entries (group 0) matching {@link createKernel}. */
export function kernelLayoutEntries( options: KernelOptions ): GPUBindGroupLayoutEntry[] {

	const visibility = GPUShaderStage.COMPUTE;
	const entries: GPUBindGroupLayoutEntry[] = [];

	if ( ( options.edges ?? 'exact' ) !== 'exact' ) {

		const { format, dimension, access, levels, load, store, size } = tileOptions( options );
		if ( access === 'sample' ) entries.push( { binding: 0, visibility, sampler: { type: 'filtering' } } );
		if ( load === undefined ) entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
		if ( store === undefined ) levels.forEach( ( _, i ) => entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format: format!, viewDimension: dimension } } ) );
		if ( load !== undefined && size === undefined ) entries.push( { binding: sizeBinding, visibility, buffer: { type: 'uniform', minBindingSize: 8 } } );
		return entries;

	}

	const { format, dimension = '2d', count, reduction = 'average' } = options;
	const access = options.access ?? ( reduction === 'average' ? 'sample' : 'load' );
	if ( access === 'sample' ) entries.push( { binding: 0, visibility, sampler: { type: 'filtering' } } );
	entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );
	for ( let i = 0; i < count!; i ++ ) entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format: format!, viewDimension: dimension } } );
	return entries;

}

const builtinFolds = {
	min: 'min(a, b)',
	max: 'max(a, b)',
	minmax: 'vec4f(min(a.x, b.x), max(a.y, b.y), min(a.z, b.z), max(a.w, b.w))'
};

/**
 * WGSL for one dispatch. `edges` selects the kernel family:
 * - 'exact' (default): a mip chain batch, as `Downsampler` plans them.
 * - 'zero-weight' or 'clamp': a tile reduction, as `Reducer` runs it (SPEC §14.1).
 */
export function createKernel( options: KernelOptions ): string {

	if ( ( options.edges ?? 'exact' ) !== 'exact' ) return createTileKernel( options );
	const { reduction = 'average' } = options;
	if ( needsTileEdges( reduction ) ) throw new Error( 'downsampler: sums, per-channel reductions and custom sums need edges \'zero-weight\' or \'clamp\' (a Reducer); \'exact\' only has meaning for averages and folds.' );
	for ( const name of [ 'load', 'store', 'size', 'bindings', 'levels' ] as const ) {

		if ( options[ name ] !== undefined ) throw new Error( `downsampler: ${ name } needs edges 'zero-weight' or 'clamp' (a Reducer).` );

	}

	if ( options.format === undefined || options.count === undefined ) throw new Error( 'downsampler: exact kernels need a format and a level count.' );
	return createExactKernel( { ...options, format: options.format, dimension: options.dimension ?? '2d', count: options.count } );

}

/**
 * WGSL for one batch. Bindings: 0 sampler ('sample' access only), 1 source level,
 * 2… output levels (write-only storage). Workgroup z selects the array layer.
 *
 * Each workgroup writes a tile×tile block of the first output (32×32 for six
 * levels, i.e. a 64×64 source tile as in SPD). Adjacent invocations read and
 * store adjacent texels — the layout matters: assigning each invocation a 2×2
 * block measured ~30–45 % slower. Later levels are 2:1 reductions of the tile in
 * workgroup memory, with clamped reads for unit dimensions and partial tiles.
 *
 * A tail kernel (`tail`) is one workgroup with a 32×32 tile: the first output
 * as above, then every further level from the previous one in the tile by the
 * 'exact' rule at any parity (shader bilinear weights for 'average', the
 * `2x … 2x+2` footprint for folds), each quantized as stored.
 */
function createExactKernel( { format, dimension, count, reduction = 'average', access, sourceFormat = format, tail = false }: KernelOptions & { format: SupportedFormat, dimension: ViewDimension, count: number } ): string {

	reductionKey( reduction );
	const average = reduction === 'average';
	access ??= average ? 'sample' : 'load';
	if ( ! average && access !== 'load' ) throw new Error( 'downsampler: fold reductions read their source with textureLoad (access "load").' );

	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const quantization = formats[ format ].quantization;
	const integer = isIntegerDomain( quantization );
	const exactLoads = integer && isSupportedFormat( sourceFormat ) && formats[ sourceFormat ].quantization === quantization;
	const splat = sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '';
	const tile = tail ? tailTile : tileSize( count );
	const side = workgroupSide( tile );
	const samples = tile / side;

	let code = access === 'sample' ? '@group(0) @binding(0) var imageSampler: sampler;\n' : '';
	code += `@group(0) @binding(1) var source: texture_2d${ suffix }<f32>;
var<workgroup> tile: array<vec4f, ${ tile * tile }>;
`;
	for ( let i = 0; i < count; i ++ ) code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;

	// Intermediate levels are quantized exactly as they would be stored.
	// Normalized values stay in integer units (value × scale) so 2:1 averages are
	// exact and ties round to even; with k/255 floats, float error turns ties into
	// round-ups that drift by several steps along long 1D tails. The parity test
	// uses i32: converting a negative float to u32 is undefined (snorm).
	if ( integer ) {

		const { scale, min, max } = integerDomains[ quantization ];
		code += `
const SCALE = ${ scale }.0;
fn quantize(value: vec4f) -> vec4f {
	let scaled = clamp(value, vec4f(${ min }.0), vec4f(${ max }.0));
	let lower = floor(scaled);
	let fraction = scaled - lower;
	let nearest = select(lower, lower + vec4f(1), fraction > vec4f(0.5));
	return select(nearest, lower + vec4f(vec4i(lower) & vec4i(1)), fraction == vec4f(0.5));
}
`;

	} else if ( quantization === 'float16' ) {

		code += `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f {
	return vec4f(unpack2x16float(pack2x16float(value.xy)), unpack2x16float(pack2x16float(value.zw)));
}
`;

	} else {

		code += `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return value; }
`;

	}

	if ( ! average ) {

		// Folds see normalized values; the tile holds scaled ones.
		if ( isBuiltinReduction( reduction ) ) code += `fn fold(a: vec4f, b: vec4f) -> vec4f { return ${ builtinFolds[ reduction as keyof typeof builtinFolds ] }; }\n`;
		else code += `${ ( reduction as CustomReduction ).wgsl }\nfn fold(a: vec4f, b: vec4f) -> vec4f { return ${ integer ? 'reduce(a / SCALE, b / SCALE) * SCALE' : 'reduce(a, b)' }; }\n`;

	}

	if ( access === 'load' ) {

		// Normalized texels are rounded back to exact integer units so later sums are exact.
		code += `
fn load(p: vec2u, layer: u32) -> vec4f {
	let v = textureLoad(source, vec2i(p)${ array ? ', i32(layer)' : '' }, 0)${ splat } * SCALE;
	return ${ exactLoads ? 'round(v)' : 'v' };
}
`;

	}

	code += `
fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
`;

	if ( access === 'sample' ) {

		code += `	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv${ array ? ', i32(layer)' : '' }, 0.0)${ splat } * SCALE);
`;

	} else if ( average ) {

		// Bilinear weights at the output texel centre, as the sampler would apply
		// them but without its fixed-point weight rounding. The sample position
		// (x + 0.5) * S / n - 0.5 is ((2x + 1) * S - n) / 2n: integer math keeps
		// the base texel exact (float division isn't correctly rounded on GPUs),
		// and gives weights of exactly 0 or 0.5 for copies and even sizes.
		code += `	let sourceSize = textureDimensions(source);
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(load(lo, layer), load(vec2u(hi.x, lo.y), layer), f.x);
	let bottom = mix(load(vec2u(lo.x, hi.y), layer), load(hi, layer), f.x);
	return quantize(mix(top, bottom, f.y));
`;

	} else {

		// The footprint: 2x and 2x + 1, plus 2x + 2 along odd source dimensions.
		// A same-size source (separate source copied into the first level) reads
		// one texel per axis.
		code += `	let sourceSize = textureDimensions(source);
	let last = sourceSize - vec2u(1);
	let same = sourceSize == size;
	let lo = select(min(pos * 2u, last), pos, same);
	let hi = select(min(lo + vec2u(1), last), lo, same);
	var v = fold(fold(load(lo, layer), load(vec2u(hi.x, lo.y), layer)), fold(load(vec2u(lo.x, hi.y), layer), load(hi, layer)));
	let odd = ((sourceSize & vec2u(1)) == vec2u(1)) & (sourceSize > vec2u(1)) & !same;
	let far = min(lo + vec2u(2), last);
	if (odd.x) { v = fold(v, fold(load(vec2u(far.x, lo.y), layer), load(vec2u(far.x, hi.y), layer))); }
	if (odd.y) { v = fold(v, fold(load(vec2u(lo.x, far.y), layer), load(vec2u(hi.x, far.y), layer))); }
	if (odd.x && odd.y) { v = fold(v, load(far, layer)); }
	return quantize(v);
`;

	}

	code += '}\n';

	if ( tail ) {

		// The 'exact' rule from a level held in the tile (scaled, quantized units),
		// as `first` applies it to a texture level with 'load' access.
		code += `
fn tileAt(p: vec2u) -> vec4f { return tile[p.y * ${ tile }u + p.x]; }
fn tailStep(pos: vec2u, sourceSize: vec2u, size: vec2u) -> vec4f {
	let last = sourceSize - vec2u(1);
`;
		code += average ? `	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(tileAt(lo), tileAt(vec2u(hi.x, lo.y)), f.x);
	let bottom = mix(tileAt(vec2u(lo.x, hi.y)), tileAt(hi), f.x);
	return quantize(mix(top, bottom, f.y));
}
` : `	let lo = min(pos * 2u, last);
	let hi = min(lo + vec2u(1), last);
	var v = fold(fold(tileAt(lo), tileAt(vec2u(hi.x, lo.y))), fold(tileAt(vec2u(lo.x, hi.y)), tileAt(hi)));
	let odd = ((sourceSize & vec2u(1)) == vec2u(1)) & (sourceSize > vec2u(1));
	let far = min(lo + vec2u(2), last);
	if (odd.x) { v = fold(v, fold(tileAt(vec2u(far.x, lo.y)), tileAt(vec2u(far.x, hi.y)))); }
	if (odd.y) { v = fold(v, fold(tileAt(vec2u(lo.x, far.y)), tileAt(vec2u(hi.x, far.y)))); }
	if (odd.x && odd.y) { v = fold(v, tileAt(far)); }
	return quantize(v);
}
`;

	}

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
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * ${ tile }u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0)${ layer }, value / SCALE); }
	}
`;

	}

	if ( tail ) {

		// The first output is at most 32×32, so every later level is at most 16×16:
		// one texel per invocation.
		for ( let i = 1; i < count; i ++ ) {

			code += `
	workgroupBarrier();
	{
		let size = textureDimensions(mip${ i });
		if (all(local.xy < size)) { value = tailStep(local.xy, textureDimensions(mip${ i - 1 }), size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < textureDimensions(mip${ i }))) {
		tile[local.y * ${ tile }u + local.x] = value;
		textureStore(mip${ i }, vec2i(local.xy)${ layer }, value / SCALE);
	}
`;

		}

		return code + '}\n';

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

/**
 * WGSL for a tile reduction ('zero-weight' or 'clamp' edges): every level up to
 * the deepest emitted one, reduced in one workgroup tile, in one dispatch.
 *
 * Group 0 (ours): 1 source texture (no load hook), 2… one storage texture per
 * emitted level (no store hook), 8 the size uniform (load hook without a size
 * hook). Group 1 is the tool's (`bindings`). Workgroup z is the array layer.
 * Identifiers starting with `ds_` are reserved, as are the entry point `main`
 * and the hook names.
 *
 * Workgroup memory holds unnormalised accumulations per channel (sums, weighted
 * sums, min/max or a custom combine); a level is normalised only when it is
 * emitted. Nothing is rounded between levels: each emitted level is the
 * reduction of its source block (`reducePyramid` in core is the oracle).
 */
function createTileKernel( options: KernelOptions ): string {

	const { format, dimension, reduction, edges, access, levels, count, load, store, size, bindings, sourceFormat } = tileOptions( options );
	const resolved = resolveReduction( reduction );
	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const tile = tileSize( count );
	const side = workgroupSide( tile );
	const samples = tile / side;
	const zeroWeight = edges === 'zero-weight';
	const channels = [ 'x', 'y', 'z', 'w' ];

	let code = access === 'sample' ? '@group(0) @binding(0) var ds_sampler: sampler;\n' : '';
	if ( load === undefined ) code += `@group(0) @binding(1) var ds_source: texture_2d${ suffix }<f32>;\n`;
	if ( store === undefined ) levels.forEach( ( level, i ) => code += `@group(0) @binding(${ i + 2 }) var ds_out${ level }: texture_storage_2d${ suffix }<${ format }, write>;\n` );
	if ( load !== undefined && size === undefined ) code += `struct DsParams { size: vec2u }\n@group(0) @binding(${ sizeBinding }) var<uniform> ds_params: DsParams;\n`;
	if ( bindings !== '' ) code += `\n// Tool bindings (group 1)\n${ bindings }\n`;
	if ( load !== undefined ) code += `\n// load hook\n${ load }\n`;
	if ( store !== undefined ) code += `\n// store hook\n${ store }\n`;
	if ( size !== undefined ) code += `\n// size hook\n${ size }\n`;

	code += `
var<workgroup> ds_tile: array<vec4f, ${ tile * tile }>;
`;

	if ( load === undefined ) {

		const splat = sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '';
		code += `
fn load(texel: vec2i, layer: u32) -> vec4f {
	return textureLoad(ds_source, texel${ array ? ', i32(layer)' : '' }, 0)${ splat };
}
fn ds_size() -> vec2u { return textureDimensions(ds_source); }
`;

	} else {

		code += `fn ds_size() -> vec2u { return ${ size === undefined ? 'ds_params.size' : 'size()' }; }\n`;

	}

	// Combine two accumulations, prepare a loaded texel (weighted channels carry
	// w·v), and normalise an accumulation for emission.
	let prepare = ( v: string ) => v;
	let emitted = false;
	if ( resolved.custom !== undefined ) {

		code += `${ ( reduction as { wgsl: string } ).wgsl }\nfn ds_combine(a: vec4f, b: vec4f) -> vec4f { return reduce(a, b); }\n`;

	} else {

		const ops = resolved.ops;
		const weight = channels[ ops.indexOf( 'weight' ) ];
		const terms = ops.map( ( op, c ) => {

			const ch = channels[ c ];
			return op === 'min' || op === 'max' ? `${ op }(a.${ ch }, b.${ ch })` : `a.${ ch } + b.${ ch }`;

		} );
		code += `fn ds_combine(a: vec4f, b: vec4f) -> vec4f { return vec4f(${ terms.join( ', ' ) }); }\n`;

		if ( ops.includes( 'weighted' ) ) {

			code += `fn ds_prepare(v: vec4f) -> vec4f { return vec4f(${ ops.map( ( op, c ) => op === 'weighted' ? `v.${ channels[ c ] } * v.${ weight }` : `v.${ channels[ c ] }` ).join( ', ' ) }); }\n`;
			prepare = v => `ds_prepare(${ v })`;

		}

		if ( ops.includes( 'average' ) || ops.includes( 'weighted' ) ) {

			emitted = true;
			code += `fn ds_emit(v: vec4f, n: f32) -> vec4f {
	return vec4f(${ ops.map( ( op, c ) => {

				const ch = channels[ c ];
				if ( op === 'average' ) return `v.${ ch } / n`;
				if ( op === 'weighted' ) return `select(0.0, v.${ ch } / v.${ weight }, v.${ weight } != 0.0)`;
				return `v.${ ch }`;

			} ).join( ', ' ) });
}
`;

		}

	}

	const averages = resolved.ops?.includes( 'average' ) ?? false;
	// Level sizes: ⌈n / 2^l⌉ ('zero-weight') or ⌊n / 2^l⌋, at least 1 ('clamp').
	const levelSize = ( level: number ) => zeroWeight ? `((ds_size() + vec2u(${ ( 1 << level ) - 1 }u)) >> vec2u(${ level }u))` : `max(ds_size() >> vec2u(${ level }u), vec2u(1u))`;
	// Texels behind an output texel: the in-bounds part of its block, or the whole clamped block.
	const texelCount = ( level: number, pos: string ) => ! averages ? '1.0' : zeroWeight ? `ds_count(${ level }u, ${ pos })` : `${ 4 ** level }.0`;
	if ( averages && zeroWeight ) code += `
fn ds_count(level: u32, texel: vec2u) -> f32 {
	let lo = texel << vec2u(level);
	let n = min(ds_size(), lo + vec2u(1u << level)) - lo;
	return f32(n.x * n.y);
}
`;

	// 'zero-weight': b, c and d count only when inside the level they come from (a
	// always is, for any output inside its own level). 'clamp': all four count.
	code += zeroWeight ? `
fn ds_reduce4(a: vec4f, b: vec4f, c: vec4f, d: vec4f, inside: vec2<bool>) -> vec4f {
	let top = select(a, ds_combine(a, b), inside.x);
	let bottom = select(c, ds_combine(c, d), inside.x);
	return select(top, ds_combine(top, bottom), inside.y);
}
` : `
fn ds_reduce4(a: vec4f, b: vec4f, c: vec4f, d: vec4f) -> vec4f {
	return ds_combine(ds_combine(a, b), ds_combine(c, d));
}
`;

	// Level 1 from a 2×2 block of the source. Coordinates are clamped, so the load
	// hook only sees in-bounds texels; under 'zero-weight' the clamped duplicates
	// are then left out. 'sample' takes one bilinear sample at the block's shared
	// corner (weights ½, ½): clamp-to-edge addressing repeats the edge texel past
	// the image exactly as clamped loads do. The tile holds sums, hence × 4.
	if ( access === 'sample' ) code += `
fn ds_first(p: vec2u, layer: u32) -> vec4f {
	let uv = vec2f(p * 2u + vec2u(1u)) / vec2f(ds_size());
	return textureSampleLevel(ds_source, ds_sampler, uv${ array ? ', i32(layer)' : '' }, 0.0)${ sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '' } * 4.0;
}
`;
	else code += `
fn ds_first(p: vec2u, layer: u32) -> vec4f {
	let size = ds_size();
	let last = vec2i(size) - vec2i(1);
	let lo = min(vec2i(p * 2u), last);
	let hi = min(vec2i(p * 2u + vec2u(1u)), last);
	let a = ${ prepare( 'load(lo, layer)' ) };
	let b = ${ prepare( 'load(vec2i(hi.x, lo.y), layer)' ) };
	let c = ${ prepare( 'load(vec2i(lo.x, hi.y), layer)' ) };
	let d = ${ prepare( 'load(hi, layer)' ) };
	return ds_reduce4(a, b, c, d${ zeroWeight ? ', p * 2u + vec2u(1u) < size' : '' });
}
`;

	const emit = ( level: number, pos: string ) => {

		const value = emitted ? `ds_emit(value, ${ texelCount( level, pos ) })` : 'value';
		return store !== undefined ? `store(${ level }u, vec2i(${ pos }), group.z, ${ value });` : `textureStore(ds_out${ level }, vec2i(${ pos })${ array ? ', i32(group.z)' : '' }, ${ value });`;

	};

	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	var value: vec4f;
`;
	if ( levels.includes( 1 ) ) code += `	let size1 = ${ levelSize( 1 ) };\n`;

	for ( let i = 0; i < samples * samples; i ++ ) {

		code += `
	{
		let p = local.xy + vec2u(${ ( i % samples ) * side }u, ${ Math.floor( i / samples ) * side }u);
		let pos = group.xy * ${ tile }u + p;
		value = ds_first(pos, group.z);
${ count > 1 ? `		ds_tile[p.y * ${ tile }u + p.x] = value;\n` : '' }${ levels.includes( 1 ) ? `		if (all(pos < size1)) { ${ emit( 1, 'pos' ) } }\n` : '' }	}
`;

	}

	for ( let level = 2; level <= count; level ++ ) {

		const extent = tile >> ( level - 1 );
		const at = ( dx: number, dy: number ) => `ds_tile[(lo.y + ${ dy }u) * ${ tile }u + lo.x + ${ dx }u]`;
		code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let lo = local.xy * 2u;
		value = ds_reduce4(${ at( 0, 0 ) }, ${ at( 1, 0 ) }, ${ at( 0, 1 ) }, ${ at( 1, 1 ) }${ zeroWeight ? `, group.xy * ${ extent * 2 }u + lo + vec2u(1u) < ${ levelSize( level - 1 ) }` : '' });
	}
`;
		if ( level < count ) code += `	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) { ds_tile[local.y * ${ tile }u + local.x] = value; }
`;
		if ( levels.includes( level ) ) code += `	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let pos = group.xy * ${ extent }u + local.xy;
		if (all(pos < ${ levelSize( level ) })) { ${ emit( level, 'pos' ) } }
	}
`;

	}

	return code + '}\n';

}
