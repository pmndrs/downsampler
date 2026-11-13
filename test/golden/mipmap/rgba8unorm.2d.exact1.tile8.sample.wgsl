@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
var<workgroup> tile: array<vec4f, 64>;
@group(0) @binding(2) var mip0: texture_storage_2d<rgba8unorm, write>;

const SCALE = 255.0;
fn quantize(value: vec4f) -> vec4f {
	let scaled = clamp(value, vec4f(0.0), vec4f(255.0));
	let lower = floor(scaled);
	let fraction = scaled - lower;
	let nearest = select(lower, lower + vec4f(1), fraction > vec4f(0.5));
	return select(nearest, lower + vec4f(vec4i(lower) & vec4i(1)), fraction == vec4f(0.5));
}

fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv, 0.0) * SCALE);
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
