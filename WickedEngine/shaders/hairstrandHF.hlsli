#ifndef WI_HAIRSTRAND_HF
#define WI_HAIRSTRAND_HF

#include "globals.hlsli"
#include "ShaderInterop_HairStrand.h"

struct HairStrandVSOut
{
	float4 pos : SV_POSITION;
	float3 worldPos : WORLD_POS;
	float3 tangent : TANGENT;
	float4 strandData : STRAND_DATA; // x = u (0..1 root-to-tip), y = h (-1..1 across width), z = coverage (0..1), w = strandSeed
	float clip : SV_ClipDistance0;
};

struct HairStrandPush
{
	uint camera_index;
};

#endif // WI_HAIRSTRAND_HF
