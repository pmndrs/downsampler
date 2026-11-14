/**
 * WGSL for 'exact' batches of integer formats (r32uint, rgba8sint, rgb10a2uint, …;
 * SPEC.md §6.1 Integer formats). The float kernel's structure (`createExactKernel`)
 * with `vec4u` working values: tiles, tail kernels, area-weighted first outputs
 * (`oddSizes: 'box'`, resampled sources), partial updates and both output paths.
 *
 * Signed formats are offset by 2^31 on load (`v ^ 0x80000000`) and back on store:
 * that maps i32 order onto u32 order, so min and max work unchanged, and adds an even
 * constant, so averages and their tie parity are unchanged too. Every average is the
 * exact weighted mean rounded to nearest, ties to even (`intReference` in core),
 * computed in u32 arithmetic without overflow: each value splits into 16-bit halves,
 * and the division is carried as quotient and remainder (`ds_mean`).
 */
import { formats, intDomains, intRange, isIntQuantization, isSupportedFormat, sampledChannels, sampleKind, type FormatInfo, type IntQuantization, type SupportedFormat } from '../core/formats.js';
import { tailTile, tileSize, workgroupSide } from '../core/plan.js';
import { isBuiltinReduction, reductionKey, type CustomReduction } from '../core/reduction.js';
import { bufferOutputWgsl, regionWgsl, type KernelOptions, type ViewDimension } from './kernel.js';

const builtinFolds = {
	min: 'min(a, b)',
	max: 'max(a, b)',
	minmax: 'vec4u(min(a.x, b.x), max(a.y, b.y), min(a.z, b.z), max(a.w, b.w))'
};

/**
 * WGSL helpers for exact integer means of `vec4u` values below 2^32 (signed values
 * offset by 2^31), rounded to nearest with ties to even:
 * - `ds_mean4(a, b, c, d)`: the 2×2 mean, as ⌊Σ v/4⌋ + ⌊Σ (v mod 4) / 4⌋ plus the remainder.
 * - Weighted means with integer weights per axis summing to dx and dy (dx, dy ≤ 2^15):
 *   for each row of taps, `ds_hi`/`ds_lo` sums Σ w·(v >> 16) and Σ w·(v & 0xffff)
 *   (each below 2^31) go into `ds_mean_row( m, hi, lo, dx, wy )`, which keeps
 *   quotient and remainder by dx; `ds_mean( m, dx, dy )` then divides by dx·dy.
 */
export function intMeanWgsl(): string {

	return `fn ds_mean4(a: vec4u, b: vec4u, c: vec4u, d: vec4u) -> vec4u {
	let q = (a >> vec4u(2u)) + (b >> vec4u(2u)) + (c >> vec4u(2u)) + (d >> vec4u(2u));
	let r = (a & vec4u(3u)) + (b & vec4u(3u)) + (c & vec4u(3u)) + (d & vec4u(3u));
	let f = q + (r >> vec4u(2u));
	let rem = r & vec4u(3u);
	return f + select(vec4u(0u), vec4u(1u), (rem > vec4u(2u)) | ((rem == vec4u(2u)) & ((f & vec4u(1u)) == vec4u(1u))));
}
// Σ w·v per 16-bit half: A = Σ_rows wy·⌊R/dx⌋ and B = Σ_rows wy·(R mod dx), so Σ w·v = dx·A + B.
struct DsMean { aHi: vec4u, bHi: vec4u, aLo: vec4u, bLo: vec4u }
fn ds_hi(v: vec4u) -> vec4u { return v >> vec4u(16u); }
fn ds_lo(v: vec4u) -> vec4u { return v & vec4u(0xffffu); }
fn ds_mean_row(m: DsMean, hi: vec4u, lo: vec4u, dx: u32, wy: u32) -> DsMean {
	return DsMean(m.aHi + wy * (hi / dx), m.bHi + wy * (hi % dx), m.aLo + wy * (lo / dx), m.bLo + wy * (lo % dx));
}
fn ds_mean(m: DsMean, dx: u32, dy: u32) -> vec4u {
	let d = dx * dy;
	// Per half, (dx·A + B) / d = ⌊A / dy⌋ + t / d with t = (A mod dy)·dx + B < 2d.
	let tHi = (m.aHi % dy) * dx + m.bHi;
	let tLo = (m.aLo % dy) * dx + m.bLo;
	let qHi = m.aHi / dy + tHi / d;
	let qLo = m.aLo / dy + tLo / d;
	// 2^16 · (tHi mod d) / d by long division, two bits at a time (remainders stay below d ≤ 2^30).
	var q = vec4u(0u);
	var r = tHi % d;
	for (var i = 0u; i < 8u; i++) {
		r = r << vec4u(2u);
		q = (q << vec4u(2u)) + r / d;
		r = r % d;
	}
	r += tLo % d;
	q += r / d;
	r = r % d;
	let f = (qHi << vec4u(16u)) + qLo + q;
	let twice = 2u * r;
	return f + select(vec4u(0u), vec4u(1u), (twice > vec4u(d)) | ((twice == vec4u(d)) & ((f & vec4u(1u)) == vec4u(1u))));
}
`;

}

const hex = ( value: number ) => `0x${ ( value >>> 0 ).toString( 16 ) }u`;

/**
 * WGSL for one 'exact' batch of an integer format: the integer form of `createExactKernel`
 * (same bindings, tiles, tail, `area`/`box`, `region` and `output` options). Sources are
 * always loaded. Custom folds define `fn reduce(a: vec4u, b: vec4u) -> vec4u` for uint
 * formats and `fn reduce(a: vec4i, b: vec4i) -> vec4i` for sint formats.
 */
export function createIntKernel( { format, dimension, count, reduction = 'average', access = 'load', sourceFormat = format, output = 'storage', tail = false, box = false, area = box, tile: tileOption, region = false }: KernelOptions & { format: SupportedFormat, dimension: ViewDimension, count: number } ): string {

	reductionKey( reduction );
	const quantization = formats[ format ].quantization;
	if ( ! isIntQuantization( quantization ) ) throw new Error( `downsampler: ${ format } is not an integer format.` );
	const average = reduction === 'average';
	if ( access !== 'load' ) throw new Error( `downsampler: integer formats (${ format }) can't be sampled; they load (access "load").` );
	if ( box && ! area ) throw new Error( 'downsampler: a box kernel\'s first output is area-weighted (area true).' );
	const { signed } = intDomains[ quantization ];
	const kind = sampleKind( sourceFormat );
	if ( kind !== ( signed ? 'sint' : 'uint' ) ) throw new Error( `downsampler: ${ format } reads an integer source of the same signedness, not ${ sourceFormat }.` );
	if ( isSupportedFormat( sourceFormat ) && ! isIntQuantization( formats[ sourceFormat ].quantization ) ) throw new Error( `downsampler: ${ format } needs an integer source, not ${ sourceFormat }.` );

	const array = dimension === '2d-array';
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const scalar = signed ? 'i32' : 'u32';
	const splat = sampledChannels( sourceFormat ) === 1 ? '.xxxx' : '';
	const tile = tileOption ?? ( tail ? tailTile : tileSize( count ) );
	if ( tile !== 8 && tile !== 16 && tile !== 32 ) throw new Error( `downsampler: a tile is 8, 16 or 32, got ${ tile }.` );
	if ( ! tail && tile < tileSize( count ) ) throw new Error( `downsampler: ${ count } levels need a tile of at least ${ tileSize( count ) }, got ${ tile }.` );
	const side = workgroupSide( tile );
	const samples = tile / side;
	const buffer = output === 'buffer';
	if ( output !== 'storage' && ! buffer ) throw new Error( `downsampler: unknown output path ${ JSON.stringify( output ) }; expected 'storage' or 'buffer'.` );
	if ( ! buffer && ( formats[ format ] as FormatInfo ).storage === false ) throw new Error( `downsampler: ${ format } can't be a storage texture; use output 'buffer'.` );

	let code = `@group(0) @binding(1) var source: texture_2d${ suffix }<${ scalar }>;
var<workgroup> tile: array<vec4u, ${ tile * tile }>;
`;
	if ( buffer ) code += bufferOutputWgsl( format, false, false );
	else for ( let i = 0; i < count; i ++ ) code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;
	if ( region ) code += regionWgsl();

	// Working values are u32; signed values carry an offset of 2^31. Results are clamped
	// to the destination's range (it only binds for a wider separate source).
	const ranges = [ 0, 1, 2, 3 ].map( c => intRange( quantization as IntQuantization, c ) );
	const biased = ( value: number ) => hex( signed ? value + 2 ** 31 : value );
	code += `
const DS_BIAS = vec4u(${ signed ? '0x80000000u' : '0u' });
fn quantize(v: vec4u) -> vec4u { return clamp(v, vec4u(${ ranges.map( r => biased( r[ 0 ] ) ).join( ', ' ) }), vec4u(${ ranges.map( r => biased( r[ 1 ] ) ).join( ', ' ) })); }
fn load(p: vec2u, layer: u32) -> vec4u {
	return ${ signed ? 'bitcast<vec4u>' : '' }(textureLoad(source, vec2i(p)${ array ? ', i32(layer)' : '' }, 0)${ splat })${ signed ? ' ^ DS_BIAS' : '' };
}
`;

	if ( average ) code += intMeanWgsl();
	else if ( isBuiltinReduction( reduction ) ) code += `fn fold(a: vec4u, b: vec4u) -> vec4u { return ${ builtinFolds[ reduction as keyof typeof builtinFolds ] }; }\n`;
	else code += `${ ( reduction as CustomReduction ).wgsl }\nfn fold(a: vec4u, b: vec4u) -> vec4u { return ${ signed ? 'bitcast<vec4u>(reduce(bitcast<vec4i>(a ^ DS_BIAS), bitcast<vec4i>(b ^ DS_BIAS))) ^ DS_BIAS' : 'reduce(a, b)' }; }\n`;

	// Integer overlaps of area-weighted steps (see `areaOverlap` in core): weights overlap / S.
	if ( area ) code += `fn ds_area_first(x: vec2u, s: vec2u, n: vec2u) -> vec2u { return (x * s) / n; }
fn ds_area_last(x: vec2u, s: vec2u, n: vec2u) -> vec2u { return ((x + vec2u(1u)) * s - vec2u(1u)) / n; }
${ average ? `fn ds_overlap(t: u32, x: u32, s: u32, n: u32) -> u32 { return min(n * (t + 1u), (x + 1u) * s) - max(n * t, x * s); }
` : '' }`;

	// The 'exact' and area rules from any level: `at` reads a texel of it.
	const rule = ( at: ( p: string ) => string, size: string, sourceSize: string, kindOf: 'bilinear' | 'area' | 'fold' | 'areaFold' ) => {

		if ( kindOf === 'area' ) return `	let lo = ds_area_first(pos, ${ sourceSize }, ${ size });
	let hi = ds_area_last(pos, ${ sourceSize }, ${ size });
	var m = DsMean();
	for (var y = lo.y; y <= hi.y; y++) {
		var rowHi = vec4u(0u);
		var rowLo = vec4u(0u);
		for (var x = lo.x; x <= hi.x; x++) {
			let w = ds_overlap(x, pos.x, ${ sourceSize }.x, ${ size }.x);
			let v = ${ at( 'vec2u(x, y)' ) };
			rowHi += w * ds_hi(v);
			rowLo += w * ds_lo(v);
		}
		m = ds_mean_row(m, rowHi, rowLo, ${ sourceSize }.x, ds_overlap(y, pos.y, ${ sourceSize }.y, ${ size }.y));
	}
	return quantize(ds_mean(m, ${ sourceSize }.x, ${ sourceSize }.y));
`;
		if ( kindOf === 'areaFold' ) return `	let lo = ds_area_first(pos, ${ sourceSize }, ${ size });
	let hi = ds_area_last(pos, ${ sourceSize }, ${ size });
	var v = ${ at( 'lo' ) };
	for (var y = lo.y; y <= hi.y; y++) {
		for (var x = lo.x; x <= hi.x; x++) { v = fold(v, ${ at( 'vec2u(x, y)' ) }); }
	}
	return quantize(v);
`;
		if ( kindOf === 'bilinear' ) return `	let last = ${ sourceSize } - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * ${ sourceSize } - ${ size };
	let d = 2u * ${ size };
	let lo = min(numerator / d, last);
	let hi = min(lo + vec2u(1), last);
	let r = numerator % d;
	let a = ${ at( 'lo' ) };
	let b = ${ at( 'vec2u(hi.x, lo.y)' ) };
	let c = ${ at( 'vec2u(lo.x, hi.y)' ) };
	let e = ${ at( 'hi' ) };
	let wx = vec2u(d.x - r.x, r.x);
	var m = ds_mean_row(DsMean(), wx.x * ds_hi(a) + wx.y * ds_hi(b), wx.x * ds_lo(a) + wx.y * ds_lo(b), d.x, d.y - r.y);
	m = ds_mean_row(m, wx.x * ds_hi(c) + wx.y * ds_hi(e), wx.x * ds_lo(c) + wx.y * ds_lo(e), d.x, r.y);
	return quantize(ds_mean(m, d.x, d.y));
`;
		// Folds: 2x and 2x + 1, plus 2x + 2 along odd source dimensions.
		return `	let last = ${ sourceSize } - vec2u(1);
	let lo = min(pos * 2u, last);
	let hi = min(lo + vec2u(1), last);
	var v = fold(fold(${ at( 'lo' ) }, ${ at( 'vec2u(hi.x, lo.y)' ) }), fold(${ at( 'vec2u(lo.x, hi.y)' ) }, ${ at( 'hi' ) }));
	let odd = ((${ sourceSize } & vec2u(1)) == vec2u(1)) & (${ sourceSize } > vec2u(1));
	let far = min(lo + vec2u(2), last);
	if (odd.x) { v = fold(v, fold(${ at( 'vec2u(far.x, lo.y)' ) }, ${ at( 'vec2u(far.x, hi.y)' ) })); }
	if (odd.y) { v = fold(v, fold(${ at( 'vec2u(lo.x, far.y)' ) }, ${ at( 'vec2u(hi.x, far.y)' ) })); }
	if (odd.x && odd.y) { v = fold(v, ${ at( 'far' ) }); }
	return quantize(v);
`;

	};

	// The first output: a same-size source (a separate source's copy) reads its texel.
	const fromSource = ( p: string ) => `load(${ p }, layer)`;
	code += `
fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4u {
	let sourceSize = textureDimensions(source);
${ area ? '' : '	if (all(sourceSize == size)) { return quantize(load(pos, layer)); }\n' }${ rule( fromSource, 'size', 'sourceSize', area ? ( average ? 'area' : 'areaFold' ) : average ? 'bilinear' : 'fold' ) }}
`;

	if ( buffer ) code += `
fn ds_level_size(i: u32) -> vec2u { return max(ds_output.size >> vec2u(i), vec2u(1u)); }
`;
	const dims = ( i: number ) => buffer ? `ds_level_size(${ i }u)` : i === 0 ? 'textureDimensions(mip0)' : `max(size0 >> vec2u(${ i }u), vec2u(1u))`;
	// Stores write the format's integers: signed values lose their offset.
	const put = ( i: number, pos: string ) => buffer ? `ds_put(${ i }u, ${ pos }, group.z, value${ signed ? ' ^ DS_BIAS' : '' });` : `textureStore(mip${ i }, vec2i(${ pos })${ layer }, ${ signed ? 'bitcast<vec4i>(value ^ DS_BIAS)' : 'value' });`;
	const stored = ( i: number, pos: string, size: string ) => region ? `ds_inside(${ i }u, ${ pos })` : `all(${ pos } < ${ size })`;
	const entry = region ? `@builtin(workgroup_id) ds_group: vec3u) {
	let group = ds_group + ds_region.origin.xyz;` : '@builtin(workgroup_id) group: vec3u) {';

	if ( tail ) code += `
fn tileAt(p: vec2u) -> vec4u { return tile[p.y * ${ tile }u + p.x]; }
fn tailStep(pos: vec2u, sourceSize: vec2u, size: vec2u) -> vec4u {
${ rule( p => `tileAt(${ p })`, 'size', 'sourceSize', average ? ( box ? 'area' : 'bilinear' ) : 'fold' ) }}
`;

	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, ${ entry }
	let size0 = ${ dims( 0 ) };
	var value: vec4u;
`;

	for ( let i = 0; i < samples * samples; i ++ ) code += `
	{
		let p = local.xy + vec2u(${ ( i % samples ) * side }u, ${ Math.floor( i / samples ) * side }u);
		let pos0 = group.xy * ${ tile }u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * ${ tile }u + p.x] = value;
		if (${ stored( 0, 'pos0', 'size0' ) }) { ${ put( 0, 'pos0' ) } }
	}
`;

	if ( tail ) {

		for ( let i = 1; i < count; i ++ ) code += `
	workgroupBarrier();
	{
		let size = ${ dims( i ) };
		if (all(local.xy < size)) { value = tailStep(local.xy, ${ dims( i - 1 ) }, size); }
	}
	workgroupBarrier();
	if (all(local.xy < ${ dims( i ) })) {
		tile[local.y * ${ tile }u + local.x] = value;
		${ region ? `if (ds_inside(${ i }u, local.xy)) { ${ put( i, 'local.xy' ) } }` : put( i, 'local.xy' ) }
	}
`;
		return code + '}\n';

	}

	const combine = ( a: string, b: string, c: string, d: string ) => average ? `quantize(ds_mean4(${ a }, ${ b }, ${ c }, ${ d }))` : `quantize(fold(fold(${ a }, ${ b }), fold(${ c }, ${ d })))`;
	for ( let i = 1; i < count; i ++ ) {

		const extent = tile >> i;
		const at = ( x: string, y: string ) => `tile[${ y }.y * ${ tile }u + ${ x }.x]`;
		code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = ${ dims( i - 1 ) } - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = ${ combine( at( 'lo', 'lo' ), at( 'hi', 'lo' ), at( 'lo', 'hi' ), at( 'hi', 'hi' ) ) };
	}
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		tile[local.y * ${ tile }u + local.x] = value;
		let pos = group.xy * ${ extent }u + local.xy;
		if (${ stored( i, 'pos', dims( i ) ) }) { ${ put( i, 'pos' ) } }
	}
`;

	}

	return code + '}\n';

}

/** The `sampleType` integer sources bind with: 'uint' or 'sint'. */
export function intSampleType( format: SupportedFormat ): GPUTextureSampleType {

	return ( intDomains[ formats[ format ].quantization as IntQuantization ].signed ? 'sint' : 'uint' );

}
