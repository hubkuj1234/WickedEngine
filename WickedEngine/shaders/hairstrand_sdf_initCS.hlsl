#include "globals.hlsli"
#include "ShaderInterop_HairStrand.h"

// Builds a 64^3 local-space 3D Signed Distance Field from head/upper-body collision triangles.
// Pass 0 (this shader): computes exact unsigned distance & closest surface normal sign
// for every voxel in the 64^3 volume by iterating over the bound head/upper-body triangle list.
// Because a MakeHuman head+neck+upper-chest subset is ~1500-3500 triangles and this only runs
// when the character mesh changes (or on morph edits), direct exact distance evaluation per
// voxel is completely artifact-free (no JFA sign-flip holes!).

StructuredBuffer<float4> headTriVertices : register(t0); // 3 float4s per triangle: (v0, v1, v2) in head-local space
RWTexture3D<float> sdfVolume : register(u0);

inline float3 ClosestPointOnTriangle(float3 p, float3 a, float3 b, float3 c)
{
	float3 ab = b - a;
	float3 ac = c - a;
	float3 ap = p - a;
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a;

	float3 bp = p - b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) return b;

	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		float v = d1 / max(d1 - d3, 1e-8f);
		return a + v * ab;
	}

	float3 cp = p - c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) return c;

	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float w = d2 / max(d2 - d6, 1e-8f);
		return a + w * ac;
	}

	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
	{
		float w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-8f);
		return b + w * (c - b);
	}

	float denom = 1.0f / max(va + vb + vc, 1e-8f);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab * v + ac * w;
}

[numthreads(THREADCOUNT_HAIRSTAND_SDF, THREADCOUNT_HAIRSTAND_SDF, THREADCOUNT_HAIRSTAND_SDF)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= HAIRSTAND_SDF_RESOLUTION ||
	    DTid.y >= HAIRSTAND_SDF_RESOLUTION ||
	    DTid.z >= HAIRSTAND_SDF_RESOLUTION)
		return;

	float3 uvw = (float3(DTid) + 0.5f) / float(HAIRSTAND_SDF_RESOLUTION);
	float3 p = lerp(xHairSDFBoundsMin, xHairSDFBoundsMax, uvw);

	float minDistSq = 1e20f;
	float3 bestNormal = float3(0, 1, 0);
	float3 bestClosest = p;

	uint triCount = xHairSDFTriangleCount;
	for (uint t = 0; t < triCount; ++t)
	{
		float3 a = headTriVertices[t * 3 + 0].xyz;
		float3 b = headTriVertices[t * 3 + 1].xyz;
		float3 c = headTriVertices[t * 3 + 2].xyz;

		float3 closest = ClosestPointOnTriangle(p, a, b, c);
		float3 diff = p - closest;
		float dSq = dot(diff, diff);
		if (dSq < minDistSq)
		{
			minDistSq = dSq;
			bestClosest = closest;
			float3 n = cross(b - a, c - a);
			float nLen = length(n);
			bestNormal = nLen > 1e-7f ? (n / nLen) : float3(0, 1, 0);
		}
	}

	float dist = sqrt(minDistSq);
	// Determine inside/outside sign using closest triangle outward normal
	// (WickedEngine uses CW front-facing in LH, so flip if needed; MakeHuman normals point outward)
	float sgn = dot(p - bestClosest, bestNormal) < 0.0f ? -1.0f : 1.0f;
	sdfVolume[DTid] = dist * sgn;
}
