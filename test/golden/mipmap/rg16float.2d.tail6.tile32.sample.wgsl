@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
var<workgroup> tile: array<vec4f, 1024>;
fn ds_round_f16(v: vec4f) -> vec4f {
	let bits = bitcast<vec4u>(v);
	let a = bits & vec4u(0x7fffffffu);
	let normal = (a + vec4u(0xfffu) + ((a >> vec4u(13u)) & vec4u(1u))) & vec4u(0xffffe000u);
	let subnormal = bitcast<vec4u>(round(bitcast<vec4f>(a) * 16777216.0) * (1.0 / 16777216.0));
	let r = select(subnormal, normal, a >= vec4u(0x38800000u));
	let q = select(select(r, vec4u(0x7f800000u), a >= vec4u(0x477ff000u)), vec4u(0x7fc00000u), a > vec4u(0x7f800000u));
	return bitcast<vec4f>(q | (bits & vec4u(0x80000000u)));
}
@group(0) @binding(2) var mip0: texture_storage_2d<rg16float, write>;
@group(0) @binding(3) var mip1: texture_storage_2d<rg16float, write>;
@group(0) @binding(4) var mip2: texture_storage_2d<rg16float, write>;
@group(0) @binding(5) var mip3: texture_storage_2d<rg16float, write>;
@group(0) @binding(6) var mip4: texture_storage_2d<rg16float, write>;
@group(0) @binding(7) var mip5: texture_storage_2d<rg16float, write>;

const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return ds_round_f16(value); }

fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv, 0.0) * SCALE);
}

fn tileAt(p: vec2u) -> vec4f { return tile[p.y * 32u + p.x]; }
fn tailStep(pos: vec2u, sourceSize: vec2u, size: vec2u) -> vec4f {
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(tileAt(lo), tileAt(vec2u(hi.x, lo.y)), f.x);
	let bottom = mix(tileAt(vec2u(lo.x, hi.y)), tileAt(hi), f.x);
	return quantize(mix(top, bottom, f.y));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	let size0 = textureDimensions(mip0);
	var value: vec4f;

	{
		let p = local.xy + vec2u(0u, 0u);
		let pos0 = group.xy * 32u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * 32u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0), value / SCALE); }
	}

	{
		let p = local.xy + vec2u(16u, 0u);
		let pos0 = group.xy * 32u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * 32u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0), value / SCALE); }
	}

	{
		let p = local.xy + vec2u(0u, 16u);
		let pos0 = group.xy * 32u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * 32u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0), value / SCALE); }
	}

	{
		let p = local.xy + vec2u(16u, 16u);
		let pos0 = group.xy * 32u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * 32u + p.x] = value;
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0), value / SCALE); }
	}

	workgroupBarrier();
	{
		let size = max(size0 >> vec2u(1u), vec2u(1u));
		if (all(local.xy < size)) { value = tailStep(local.xy, textureDimensions(mip0), size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < max(size0 >> vec2u(1u), vec2u(1u)))) {
		tile[local.y * 32u + local.x] = value;
		textureStore(mip1, vec2i(local.xy), value / SCALE);
	}

	workgroupBarrier();
	{
		let size = max(size0 >> vec2u(2u), vec2u(1u));
		if (all(local.xy < size)) { value = tailStep(local.xy, max(size0 >> vec2u(1u), vec2u(1u)), size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < max(size0 >> vec2u(2u), vec2u(1u)))) {
		tile[local.y * 32u + local.x] = value;
		textureStore(mip2, vec2i(local.xy), value / SCALE);
	}

	workgroupBarrier();
	{
		let size = max(size0 >> vec2u(3u), vec2u(1u));
		if (all(local.xy < size)) { value = tailStep(local.xy, max(size0 >> vec2u(2u), vec2u(1u)), size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < max(size0 >> vec2u(3u), vec2u(1u)))) {
		tile[local.y * 32u + local.x] = value;
		textureStore(mip3, vec2i(local.xy), value / SCALE);
	}

	workgroupBarrier();
	{
		let size = max(size0 >> vec2u(4u), vec2u(1u));
		if (all(local.xy < size)) { value = tailStep(local.xy, max(size0 >> vec2u(3u), vec2u(1u)), size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < max(size0 >> vec2u(4u), vec2u(1u)))) {
		tile[local.y * 32u + local.x] = value;
		textureStore(mip4, vec2i(local.xy), value / SCALE);
	}

	workgroupBarrier();
	{
		let size = max(size0 >> vec2u(5u), vec2u(1u));
		if (all(local.xy < size)) { value = tailStep(local.xy, max(size0 >> vec2u(4u), vec2u(1u)), size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < max(size0 >> vec2u(5u), vec2u(1u)))) {
		tile[local.y * 32u + local.x] = value;
		textureStore(mip5, vec2i(local.xy), value / SCALE);
	}
}
