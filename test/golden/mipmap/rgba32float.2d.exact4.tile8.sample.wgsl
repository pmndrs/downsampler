@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
var<workgroup> tile: array<vec4f, 64>;
@group(0) @binding(2) var mip0: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var mip1: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var mip2: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var mip3: texture_storage_2d<rgba32float, write>;

const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return value; }

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

	workgroupBarrier();
	if (local.x < 4u && local.y < 4u) {
		let last = textureDimensions(mip0) - group.xy * 8u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = quantize((tile[lo.y * 8u + lo.x] + tile[lo.y * 8u + hi.x] + tile[hi.y * 8u + lo.x] + tile[hi.y * 8u + hi.x]) * 0.25);
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < 4u && local.y < 4u) {
		tile[local.y * 8u + local.x] = value;
		let pos = group.xy * 4u + local.xy;
		if (all(pos < max(size0 >> vec2u(1u), vec2u(1u)))) { textureStore(mip1, vec2i(pos), value / SCALE); }
	}

	workgroupBarrier();
	if (local.x < 2u && local.y < 2u) {
		let last = max(size0 >> vec2u(1u), vec2u(1u)) - group.xy * 4u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = quantize((tile[lo.y * 8u + lo.x] + tile[lo.y * 8u + hi.x] + tile[hi.y * 8u + lo.x] + tile[hi.y * 8u + hi.x]) * 0.25);
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < 2u && local.y < 2u) {
		tile[local.y * 8u + local.x] = value;
		let pos = group.xy * 2u + local.xy;
		if (all(pos < max(size0 >> vec2u(2u), vec2u(1u)))) { textureStore(mip2, vec2i(pos), value / SCALE); }
	}

	workgroupBarrier();
	if (local.x < 1u && local.y < 1u) {
		let last = max(size0 >> vec2u(2u), vec2u(1u)) - group.xy * 2u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = quantize((tile[lo.y * 8u + lo.x] + tile[lo.y * 8u + hi.x] + tile[hi.y * 8u + lo.x] + tile[hi.y * 8u + hi.x]) * 0.25);
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < 1u && local.y < 1u) {
		tile[local.y * 8u + local.x] = value;
		let pos = group.xy * 1u + local.xy;
		if (all(pos < max(size0 >> vec2u(3u), vec2u(1u)))) { textureStore(mip3, vec2i(pos), value / SCALE); }
	}
}
