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
	// p is the top-left texel of a 2x2 quad: sample its centre.
	float2 uv = float2(p) * invInputSize + invInputSize;
	return imgSrc.SampleLevel(srcSampler, uv, 0);
#else
	return imgSrc[p];
#endif
}

float4 SpdLoad(int2 p, uint slice)
{
	return imgDst6[p];
}

void SpdStore(int2 p, float4 value, uint mip, uint slice)
{
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
