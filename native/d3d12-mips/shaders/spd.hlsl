// AMD FidelityFX SPD, compiled from AMD's own ffx_spd.h (FidelityFX SDK 7539ae8,
// Kits/Cauldron2/framework/shaders/fidelityfx; byte-identical to the FSR 3.1.6 copy
// that test/tools/spd.js ports). The callbacks follow AMD's SPD sample: source level
// through an SRV, mip 6 through a globallycoherent UAV for the last workgroup, the
// other levels through a UAV array, and a global atomic counter for the handoff.
//
// Compile-time options:
//   SPD_LINEAR_SAMPLER            AMD's option: one bilinear sample per 2x2 source quad
//   FFX_SPD_NO_WAVE_OPERATIONS    AMD's option: LDS reduction instead of quad intrinsics
//   SPD_MAX                       max instead of average (Hi-Z)
//   SPD_UNORM                     destinations declared `unorm float4` (rgba8unorm)
//   SPD_CHANNELS                  1 keeps only red in LDS (r32float), else 4
//   SPD_SRGB                      rgba8unorm-srgb, averaged in linear light: the source is read
//                                 through an _SRGB SRV (the hardware decodes; the sampler filters
//                                 in linear), LDS holds linear values, every store encodes in the
//                                 shader to a UNORM UAV, and mip 6 read back is decoded
//   SPD_SRGB_AMD                  rgba8unorm-srgb as AMD's SPD sample and FidelityFX SDK 1.x do it:
//                                 the linear sampler path encodes the sampled (decoded, filtered)
//                                 value with ffxSrgbFromLinear and SPD then averages encoded
//                                 values; the load path reads the codes through a UNORM view
//                                 (no decode at all). Stores write the encoded values as they are.
//
// Which levels a dispatch writes is set on the CPU (see main.cpp): `mips` <= 6 never
// touches the counter, so the same shader also runs as 6 levels per dispatch.

#define FFX_GPU 1
#define FFX_HLSL 1
#include "ffx_core.h"

#ifndef SPD_CHANNELS
#define SPD_CHANNELS 4
#endif

#ifdef SPD_UNORM
#define DST_TYPE unorm float4
#else
#define DST_TYPE float4
#endif

cbuffer Constants : register(b0)
{
	uint mips;
	uint numWorkGroups;
	uint2 workGroupOffset;
	float2 invInputSize;
	float2 padding;
};

globallycoherent RWStructuredBuffer<uint> spdGlobalAtomic : register(u0);
globallycoherent RWTexture2D<DST_TYPE> imgDst6 : register(u1);
RWTexture2D<DST_TYPE> imgDst[12] : register(u2);
Texture2D<float4> imgSrc : register(t0);
SamplerState srcSampler : register(s0);

groupshared uint spdCounter;

#ifdef SPD_SRGB
// IEC 61966-2-1, as the library's oracle (srgb.ts); the UNORM store rounds to the nearest code.
float SrgbEncode(float v)
{
	v = saturate(v);
	return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

float SrgbDecode(float c)
{
	return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float4 SrgbEncode4(float4 v) { return float4(SrgbEncode(v.r), SrgbEncode(v.g), SrgbEncode(v.b), v.a); }
float4 SrgbDecode4(float4 c) { return float4(SrgbDecode(c.r), SrgbDecode(c.g), SrgbDecode(c.b), c.a); }
#endif

void SpdIncreaseAtomicCounter(uint slice)
{
	InterlockedAdd(spdGlobalAtomic[0], 1, spdCounter);
}

uint SpdGetAtomicCounter()
{
	return spdCounter;
}

void SpdResetAtomicCounter(uint slice)
{
	spdGlobalAtomic[0] = 0;
}

groupshared float spdIntermediateR[16][16];
#if SPD_CHANNELS == 4
groupshared float spdIntermediateG[16][16];
groupshared float spdIntermediateB[16][16];
groupshared float spdIntermediateA[16][16];
#endif

float4 SpdLoadSourceImage(int2 p, uint slice)
{
#ifdef SPD_LINEAR_SAMPLER
	// p is the top-left texel of a 2x2 quad: sample its center.
	float2 uv = float2(p) * invInputSize + invInputSize;
	float4 result = imgSrc.SampleLevel(srcSampler, uv, 0);
#ifdef SPD_SRGB_AMD
	// AMD's SpdLoadSourceImage with the linear sampler (FidelityFX SDK 1.1.4 ffx_spd_callbacks_hlsl.h).
	result = float4(ffxSrgbFromLinear(result.x), ffxSrgbFromLinear(result.y), ffxSrgbFromLinear(result.z), result.w);
#endif
	return result;
#else
	return imgSrc[p];
#endif
}

float4 SpdLoad(int2 p, uint slice)
{
#ifdef SPD_SRGB
	return SrgbDecode4(imgDst6[p]);
#else
	return imgDst6[p];
#endif
}

void SpdStore(int2 p, float4 value, uint mip, uint slice)
{
#ifdef SPD_SRGB
	value = SrgbEncode4(value);
#endif
	if (mip == 5)
	{
		imgDst6[p] = value;
		return;
	}
	imgDst[mip][p] = value;
}

float4 SpdLoadIntermediate(uint x, uint y)
{
#if SPD_CHANNELS == 4
	return float4(spdIntermediateR[x][y], spdIntermediateG[x][y], spdIntermediateB[x][y], spdIntermediateA[x][y]);
#else
	return float4(spdIntermediateR[x][y], 0, 0, 0);
#endif
}

void SpdStoreIntermediate(uint x, uint y, float4 value)
{
	spdIntermediateR[x][y] = value.x;
#if SPD_CHANNELS == 4
	spdIntermediateG[x][y] = value.y;
	spdIntermediateB[x][y] = value.z;
	spdIntermediateA[x][y] = value.w;
#endif
}

float4 SpdReduce4(float4 v0, float4 v1, float4 v2, float4 v3)
{
#ifdef SPD_MAX
	return max(max(v0, v1), max(v2, v3));
#else
	return (v0 + v1 + v2 + v3) * 0.25;
#endif
}

#include "spd/ffx_spd.h"

[numthreads(256, 1, 1)]
void main(uint3 workGroupId : SV_GroupID, uint localThreadIndex : SV_GroupIndex)
{
	SpdDownsample(workGroupId.xy, localThreadIndex, mips, numWorkGroups, workGroupId.z, workGroupOffset);
}
