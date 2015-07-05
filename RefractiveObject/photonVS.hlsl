//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
cbuffer cbPerObject : register(b0)
{
	matrix		g_mWorldViewProjection	: packoffset(c0);
	matrix		g_mWorld				: packoffset(c4);
};

struct VS_INPUT
{
	float3 PosL  : POSITION;
	float4 Color : COLOR;
};

struct VS_OUTPUT
{
	float4 PosH  : SV_POSITION;
	float4 Color : COLOR;
};

VS_OUTPUT VSMain(VS_INPUT Input)
{
	VS_OUTPUT vout;

	// Transform to homogeneous clip space.
	vout.PosH = mul(float4(Input.PosL, 1.0f), g_mWorldViewProjection);
	// Just pass vertex color into the pixel shader.
	vout.Color = float4(Input.PosL, 1.0f);

	return vout;
}

