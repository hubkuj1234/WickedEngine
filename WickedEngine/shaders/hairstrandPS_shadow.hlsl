#include "hairstrandHF.hlsli"

void main(HairStrandVSOut input)
{
	float coverage = input.strandData.z;
	if (coverage < 0.99f)
	{
		float noise = dither(input.pos.xy);
		clip(coverage - noise);
	}
}
