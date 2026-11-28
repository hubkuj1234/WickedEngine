#include "globals.hlsli"
#include "ShaderInterop_HairStrand.h"

// Follow-hair GPU interpolation, smooth tangent/radius computation,
// and procedural AABB generation for hardware raytracing BLAS.

StructuredBuffer<float4> simPositions : register(t0);
StructuredBuffer<HairStrandRestVertex> restVertices : register(t1);
StructuredBuffer<HairStrandFollowDesc> followDescs : register(t2);

RWStructuredBuffer<float4> renderPositions : register(u0);     // xyz = world pos, w = radius
RWStructuredBuffer<float4> renderPrevPositions : register(u1); // xyz = prev world pos, w = radius
RWStructuredBuffer<float4> renderTangents : register(u2);      // xyz = unit tangent, w = u (0..1)
RWStructuredBuffer<HairStrandAABB> segmentAABBs : register(u3);
RWStructuredBuffer<HairStrandSegment> segmentIndices : register(u4);

[numthreads(THREADCOUNT_HAIRSTAND_FOLLOW, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	uint strandIdx = DTid.x;
	if (strandIdx >= xHairTotalRenderStrands)
		return;

	uint numVerts = xHairVerticesPerStrand;
	if (numVerts < 2)
		return;

	bool isGuide = (strandIdx < xHairGuideCount);
	uint guideIdx = isGuide ? strandIdx : followDescs[strandIdx - xHairGuideCount].guide_index;
	guideIdx = min(guideIdx, xHairGuideCount - 1);

	float3 rootOffset = float3(0, 0, 0);
	float tipSep = 0.0f;
	float radiusScale = 1.0f;
	float strandSeed = frac(sin(float(strandIdx) * 12.9898f + 78.233f) * 43758.5453f);

	if (!isGuide)
	{
		HairStrandFollowDesc fd = followDescs[strandIdx - xHairGuideCount];
		rootOffset = mul((float3x3)xHairWorld, fd.root_offset);
		tipSep = fd.tip_separation;
		radiusScale = fd.radius_scale;
		strandSeed = fd.random_seed;
	}

	uint guideBase = guideIdx * numVerts;
	uint renderBase = strandIdx * numVerts;
	uint segBase = strandIdx * (numVerts - 1);

	HairStrandRestVertex rootRest = restVertices[guideBase];
	float rRoot = max(rootRest.root_radius * radiusScale, 0.00015f);
	float rTip = max(rootRest.tip_radius * radiusScale, 0.00005f);

	bool resetSim = (xHairFlags & HAIRSTRAND_FLAG_RESET_SIM) != 0;

	// Build deterministic perpendicular spread direction for follow hairs
	float3 guideDir0 = normalize(simPositions[guideBase + 1].xyz - simPositions[guideBase].xyz + 1e-6f);
	float3 orthoA = normalize(cross(guideDir0, abs(guideDir0.y) < 0.9f ? float3(0, 1, 0) : float3(1, 0, 0)));
	float3 orthoB = cross(guideDir0, orthoA);
	float angle = strandSeed * 6.2831853f;
	float3 spreadDir = cos(angle) * orthoA + sin(angle) * orthoB;

	float3 prevVertPos = float3(0, 0, 0);
	float prevVertRadius = rRoot;

	for (uint i = 0; i < numVerts; ++i)
	{
		float u = float(i) / float(numVerts - 1);
		float3 gp = simPositions[guideBase + i].xyz;
		float3 offset = rootOffset * (1.0f - u * 0.4f) + spreadDir * (tipSep * u * u);
		float3 wp = gp + offset;
		float radius = lerp(rRoot, rTip, u);

		float4 oldPos = resetSim ? float4(wp, radius) : renderPositions[renderBase + i];
		renderPrevPositions[renderBase + i] = oldPos;
		renderPositions[renderBase + i] = float4(wp, radius);

		// Compute smooth central-difference tangent
		float3 pPrev = (i == 0) ? wp : prevVertPos;
		float3 pNext = wp;
		if (i + 1 < numVerts)
		{
			float uNext = float(i + 1) / float(numVerts - 1);
			float3 gpNext = simPositions[guideBase + i + 1].xyz;
			pNext = gpNext + rootOffset * (1.0f - uNext * 0.4f) + spreadDir * (tipSep * uNext * uNext);
		}
		float3 tanVec = pNext - pPrev;
		float tanLen = length(tanVec);
		float3 tangent = tanLen > 1e-7f ? (tanVec / tanLen) : float3(0, -1, 0);
		renderTangents[renderBase + i] = float4(tangent, u);

		// Emit procedural AABB + segment record for hardware raytracing BLAS
		if (i > 0 && (xHairFlags & HAIRSTRAND_FLAG_RAYTRACED))
		{
			uint segIdx = segBase + (i - 1);
			float pad = max(prevVertRadius, radius) * 1.05f + 1e-4f;
			float3 bmin = min(prevVertPos, wp) - pad;
			float3 bmax = max(prevVertPos, wp) + pad;

			HairStrandAABB aabb;
			aabb.minX = bmin.x; aabb.minY = bmin.y; aabb.minZ = bmin.z;
			aabb.maxX = bmax.x; aabb.maxY = bmax.y; aabb.maxZ = bmax.z;
			segmentAABBs[segIdx] = aabb;

			HairStrandSegment seg;
			seg.v0 = renderBase + (i - 1);
			seg.v1 = renderBase + i;
			seg.strand_index = strandIdx;
			seg.u0 = float(i - 1) / float(numVerts - 1);
			segmentIndices[segIdx] = seg;
		}

		prevVertPos = wp;
		prevVertRadius = radius;
	}
}
