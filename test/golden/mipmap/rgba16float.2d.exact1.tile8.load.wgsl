@group(0) @binding(1) var source: texture_2d<f32>;
var<workgroup> tile: array<vec4f, 64>;
fn ds_round_f16(v: vec4f) -> vec4f {
	let bits = bitcast<vec4u>(v);
	let a = bits & vec4u(0x7fffffffu);
	let normal = (a + vec4u(0xfffu) + ((a >> vec4u(13u)) & vec4u(1u))) & vec4u(0xffffe000u);
	let subnormal = bitcast<vec4u>(round(bitcast<vec4f>(a) * 16777216.0) * (1.0 / 16777216.0));
	let r = select(subnormal, normal, a >= vec4u(0x38800000u));
	let q = select(select(r, vec4u(0x7f800000u), a >= vec4u(0x477ff000u)), vec4u(0x7fc00000u), a > vec4u(0x7f800000u));
	return bitcast<vec4f>(q | (bits & vec4u(0x80000000u)));
}
@group(0) @binding(2) var mip0: texture_storage_2d<rgba16float, write>;

const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return ds_round_f16(value); }

fn load(p: vec2u, layer: u32) -> vec4f {
	let v = textureLoad(source, vec2i(p), 0) * SCALE;
	return v;
}

fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
	let sourceSize = textureDimensions(source);
	if (all(sourceSize == size)) { return quantize(load(pos, layer)); }
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(load(lo, layer), load(vec2u(hi.x, lo.y), layer), f.x);
	let bottom = mix(load(vec2u(lo.x, hi.y), layer), load(hi, layer), f.x);
	return quantize(mix(top, bottom, f.y));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	let size0 = textureDimensions(mip0);
	var value: vec4f;

	{
		let p = local.xy + vec2u(0u, 0u);
		let pos0 = group.xy * 8u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * 8u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0), value / SCALE); }
	}
}
