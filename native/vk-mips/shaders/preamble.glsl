// nvpro_pyramid interface for this harness, modelled on nvpro's srgba8_mipmap_preamble.glsl
// (vk_compute_mipmaps 6c248f5), but storing linear rgba8unorm / rgba16float / r32float directly.
//
//   AVERAGE: area-weighted average (nvpro's odd-size weights), bilinear LOAD_REDUCE4 as nvpro
//            recommends for speed comparisons
//   MAX:     max over the same footprint (Hi-Z), texel loads
//   IMAGE_FORMAT: rgba8 | rgba16f | r32f

layout(set = 0, binding = 0) uniform sampler2D srcTex;
layout(set = 1, binding = 0, IMAGE_FORMAT) uniform writeonly image2D imageMipLevels[16];

ivec2 levelSize(int level) { return imageSize(imageMipLevels[level]); }
#define NVPRO_PYRAMID_LEVEL_SIZE levelSize

#if defined(MAX)

#define NVPRO_PYRAMID_TYPE float
#define NVPRO_PYRAMID_LOAD(coord, level, out_) out_ = texelFetch(srcTex, coord, level).r
#define NVPRO_PYRAMID_REDUCE(a0, v0, a1, v1, a2, v2, out_) out_ = max(max(v0, v1), v2)
#define NVPRO_PYRAMID_REDUCE2(v0, v1, out_) out_ = max(v0, v1)
#define NVPRO_PYRAMID_REDUCE4(v00, v01, v10, v11, out_) out_ = max(max(v00, v01), max(v10, v11))
#define NVPRO_PYRAMID_STORE(coord, level, in_) imageStore(imageMipLevels[level], coord, vec4(in_, 0, 0, 1))

#else

#define NVPRO_PYRAMID_TYPE vec4
#define NVPRO_PYRAMID_LOAD(coord, level, out_) out_ = texelFetch(srcTex, coord, level)
#define NVPRO_PYRAMID_REDUCE(a0, v0, a1, v1, a2, v2, out_) out_ = a0 * v0 + a1 * v1 + a2 * v2
#define NVPRO_PYRAMID_REDUCE2(v0, v1, out_) out_ = 0.5 * (v0 + v1)
#define NVPRO_PYRAMID_REDUCE4(v00, v01, v10, v11, out_) out_ = 0.25 * ((v00 + v01) + (v10 + v11))
#define NVPRO_PYRAMID_STORE(coord, level, in_) imageStore(imageMipLevels[level], coord, in_)

void loadReduce4(in ivec2 srcTexelCoord, in int srcLevel, out vec4 out_)
{
	// The exact centre of the 2x2 square, as nvpro's preamble samples it.
	vec2 normCoord = (vec2(srcTexelCoord) + vec2(1)) / vec2(imageSize(imageMipLevels[srcLevel]));
	out_ = textureLod(srcTex, normCoord, srcLevel);
}
#define NVPRO_PYRAMID_LOAD_REDUCE4 loadReduce4

#endif
