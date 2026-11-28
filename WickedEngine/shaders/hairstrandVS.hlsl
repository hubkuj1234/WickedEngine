#include "hairstrandHF.hlsli"

StructuredBuffer<float4> renderPositions : register(t0); // xyz = world pos, w = radius
StructuredBuffer<float4> renderTangents : register(t1);  // xyz = tangent, w = u (0..1)

PUSHCONSTANT(push, HairStrandPush);

// Draws all hair segments as camera-facing quads (6 vertices per segment, SV_VertexID).
HairStrandVSOut main(uint vid : SV_VertexID, out uint VPIndex : SV_ViewportArrayIndex)
{
	VPIndex = push.camera_index;
	ShaderCamera camera = GetCameraIndexed(push.camera_index);

	uint segsPerStrand = max(xHairVerticesPerStrand - 1u, 1u);
	uint segGlobal = vid / 6u;
	uint cornerIdx = vid % 6u;

	uint strandIdx = segGlobal / segsPerStrand;
	uint segLocal = segGlobal % segsPerStrand;

	// Quad triangulation: (0,1,2), (2,1,3)
	// bit 0 = side (-1 or +1), bit 1 = end (0 = v0, 1 = v1)
	static const uint cornerMap[6] = { 0u, 1u, 2u, 2u, 1u, 3u };
	uint c = cornerMap[cornerIdx];
	float side = (c & 1u) ? 1.0f : -1.0f;
	uint endOffset = (c >> 1u) & 1u;

	uint vIdx = strandIdx * xHairVerticesPerStrand + segLocal + endOffset;
	float4 posRad = renderPositions[vIdx];
	float4 tanU = renderTangents[vIdx];

	float3 worldPos = posRad.xyz;
	float radius = max(posRad.w, 1e-5f);
	float3 tangent = normalize(tanU.xyz + float3(0, 1e-6f, 0));

	float3 viewDir = normalize(camera.position.xyz - worldPos);
	float3 right = cross(viewDir, tangent);
	float rightLen = length(right);
	right = rightLen > 1e-6f ? (right / rightLen) : float3(1, 0, 0);

	// Coverage-preserving minimum pixel width so thin sub-pixel strands don't alias/disappear
	float4 clipCenter = mul(camera.view_projection, float4(worldPos, 1.0f));
	float pixelWorldSize = abs(clipCenter.w) * camera.internal_resolution_rcp.y * 2.0f / max(abs(camera.projection._22), 1e-4f);
	float minWorldRadius = clamp(0.5f * max(xHairMinPixelWidth, 1.0f) * pixelWorldSize, 0.0f, 0.01f);
	float expandedRadius = max(radius, minWorldRadius);
	float coverage = saturate(radius / max(expandedRadius, 1e-6f));

	float3 finalWorldPos = worldPos + right * (side * expandedRadius);

	HairStrandVSOut Out;
	Out.worldPos = finalWorldPos;
	Out.tangent = tangent;
	float strandSeed = frac(sin(float(strandIdx) * 12.9898f + 78.233f) * 43758.5453f);
	Out.strandData = float4(tanU.w, side, coverage, strandSeed);
	Out.clip = dot(float4(finalWorldPos, 1.0f), camera.clip_plane);
	Out.pos = mul(camera.view_projection, float4(finalWorldPos, 1.0f));
	return Out;
}
