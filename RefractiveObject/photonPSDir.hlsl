//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
cbuffer cbPerObject : register(b0)
{
	float4		g_vObjectColor			: packoffset(c0);
};

cbuffer cbPerFrame : register(b1)
{
	float3		g_vLightDir				: packoffset(c0);
	float		g_fAmbient : packoffset(c0.w);
};

//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
Texture2D	g_txDiffuse : register(t0);
SamplerState g_samLinear : register(s0);

//--------------------------------------------------------------------------------------
// Input / Output structures
//--------------------------------------------------------------------------------------


struct PS_INPUT
{
	float4 PosH  : SV_POSITION;
	float4 Color : COLOR;
};
struct PS_OUTPUT
{
	float4 Color  : SV_TARGET0;
};
PS_OUTPUT PSMain(PS_INPUT Input)
{
	PS_OUTPUT tmp;

	tmp.Color.x = (Input.Color.x - g_vLightDir.x);
	tmp.Color.y = (Input.Color.y - g_vLightDir.y);
	tmp.Color.z = (Input.Color.z - g_vLightDir.z);
	tmp.Color.w = 1.0f;

	return tmp;
}


