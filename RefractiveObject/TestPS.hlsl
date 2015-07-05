//--------------------------------------------------------------------------------------
// File: BasicHLSL11_PS.hlsl
//
// The pixel shader file for the BasicHLSL11 sample.  
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

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
	float3 vNormal		: NORMAL;
	float2 vTexcoord	: TEXCOORD0;
};

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PSMain(PS_INPUT Input) : SV_TARGET
{
		float direct = dot(float3(0.0f, 0.0f, 1.0f), Input.vNormal);
		return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

