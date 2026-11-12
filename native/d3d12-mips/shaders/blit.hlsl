// One render pass per level, sampling the previous level with a linear sampler:
// the same full-screen triangle and UVs as test/bench/render-baseline.js.

Texture2D<float4> source : register(t0);
SamplerState imageSampler : register(s0);

struct Varyings
{
	float4 position : SV_Position;
	float2 uv : TEXCOORD0;
};

Varyings vs(uint index : SV_VertexID)
{
	const float2 positions[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
	float2 p = positions[index];
	Varyings output;
	output.position = float4(p, 0, 1);
	output.uv = p * float2(0.5, -0.5) + 0.5;
	return output;
}

float4 ps(Varyings input) : SV_Target
{
	return source.SampleLevel(imageSampler, input.uv, 0);
}
