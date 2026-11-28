#include "globals.hlsli"
#include "ShaderInterop_HairStrand.h"

// Optional 3x3x3 box-smoothing pass over the 64^3 SDF to remove faceted triangle-edge gradient jumps
// so hair strands glide smoothly around the face, ears, jaw, and shoulders.
Texture3D<float> sdfInput : register(t0);
RWTexture3D<float> sdfOutput : register(u0);

[numthreads(THREADCOUNT_HAIRSTAND_SDF, THREADCOUNT_HAIRSTAND_SDF, THREADCOUNT_HAIRSTAND_SDF)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= HAIRSTAND_SDF_RESOLUTION ||
	    DTid.y >= HAIRSTAND_SDF_RESOLUTION ||
	    DTid.z >= HAIRSTAND_SDF_RESOLUTION)
		return;

	int3 coord = int3(DTid);
	int maxC = int(HAIRSTAND_SDF_RESOLUTION) - 1;

	float center = sdfInput.Load(int4(coord, 0));
	float sum = center * 4.0f;
	float weight = 4.0f;

	const int3 offsets[6] = {
		int3(-1, 0, 0), int3(1, 0, 0),
		int3(0, -1, 0), int3(0, 1, 0),
		int3(0, 0, -1), int3(0, 0, 1)
	};

	[unroll]
	for (int i = 0; i < 6; ++i)
	{
		int3 c = clamp(coord + offsets[i], int3(0, 0, 0), int3(maxC, maxC, maxC));
		sum += sdfInput.Load(int4(c, 0));
		weight += 1.0f;
	}

	sdfOutput[DTid] = sum / weight;
}
