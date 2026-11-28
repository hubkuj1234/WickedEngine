#include "hairstrandHF.hlsli"

void main(HairStrandVSOut input)
{
	float coverage = input.strandData.z;
	if (coverage < 0.99f)
	{
		float noise = dither(input.pos.xy + float2(GetFrame().frame_count * 1.618f, 0.0f));
		clip(coverage - noise);
	}
}
