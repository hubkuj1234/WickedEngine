#include "globals.hlsli"
#include "ShaderInterop_HairStrand.h"

// Full TressFX GPU Guide-Strand Simulation Compute Shader:
// - Skinning-driven root pinning (4 bone weights or world transform)
// - Fixed-timestep Verlet integration with arbitrary 3D gravity vector + turbulent 3D wind
// - Global shape constraint (root-to-tip stiffness falloff)
// - Local shape constraint (quaternion frame propagation preserving curls/waves)
// - Iterative length / inextensibility constraint
// - Analytical bone-attached sphere/capsule collisions + GPU 3D Head SDF collision

StructuredBuffer<HairStrandRestVertex> restVertices : register(t0);
StructuredBuffer<HairStrandSkinWeight> skinWeights : register(t1);
StructuredBuffer<float4x4> boneMatrices : register(t2);
Texture3D<float> headSDF : register(t3);

RWStructuredBuffer<float4> simPositions : register(u0);     // xyz = current world pos, w = invMass
RWStructuredBuffer<float4> simPrevPositions : register(u1); // xyz = previous world pos, w = radius

static const uint MAX_STRAND_VERTS = 32;

inline float4 QuatMul(float4 q1, float4 q2)
{
	return float4(
		q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
		q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
		q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
		q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
	);
}

inline float3 QuatRotate(float4 q, float3 v)
{
	float3 t = 2.0f * cross(q.xyz, v);
	return v + q.w * t + cross(q.xyz, t);
}

inline float4 QuatFromTwoVectors(float3 u, float3 v)
{
	float d = dot(u, v);
	if (d < -0.9999f)
	{
		float3 ortho = abs(u.x) < 0.9f ? float3(1, 0, 0) : float3(0, 1, 0);
		float3 axis = normalize(cross(u, ortho));
		return float4(axis, 0.0f);
	}
	float3 c = cross(u, v);
	float4 q = float4(c, 1.0f + d);
	return normalize(q);
}

inline float4x4 ComputeSkinMatrix(uint guideIdx)
{
	if (xHairBoneCount == 0)
		return xHairWorld;

	HairStrandSkinWeight sw = skinWeights[guideIdx];
	float totalW = sw.bone_weights.x + sw.bone_weights.y + sw.bone_weights.z + sw.bone_weights.w;
	if (totalW <= 1e-5f)
		return xHairWorld;

	float4x4 skin = (float4x4)0;
	float appliedW = 0.0f;
	if (sw.bone_weights.x > 0.0f && sw.bone_indices.x < xHairBoneCount)
	{
		skin += boneMatrices[sw.bone_indices.x] * sw.bone_weights.x;
		appliedW += sw.bone_weights.x;
	}
	if (sw.bone_weights.y > 0.0f && sw.bone_indices.y < xHairBoneCount)
	{
		skin += boneMatrices[sw.bone_indices.y] * sw.bone_weights.y;
		appliedW += sw.bone_weights.y;
	}
	if (sw.bone_weights.z > 0.0f && sw.bone_indices.z < xHairBoneCount)
	{
		skin += boneMatrices[sw.bone_indices.z] * sw.bone_weights.z;
		appliedW += sw.bone_weights.z;
	}
	if (sw.bone_weights.w > 0.0f && sw.bone_indices.w < xHairBoneCount)
	{
		skin += boneMatrices[sw.bone_indices.w] * sw.bone_weights.w;
		appliedW += sw.bone_weights.w;
	}
	if (appliedW <= 1e-5f)
		return xHairWorld;
	skin /= appliedW;

	return mul(xHairWorld, skin);
}

inline float SampleSDF(float3 localPos)
{
	float3 uvw = (localPos - xHairSDFBoundsMin) / max(xHairSDFBoundsMax - xHairSDFBoundsMin, 1e-5f);
	return headSDF.SampleLevel(sampler_linear_clamp, uvw, 0);
}

inline void ResolveCollisions(inout float3 pos, inout float3 prevPos)
{
	// 1. Analytical bone-attached sphere & capsule collisions
	if (xHairFlags & HAIRSTRAND_FLAG_USE_COLLIDERS)
	{
		uint count = min(xHairColliderCount, HAIRSTAND_MAX_COLLIDERS);
		for (uint i = 0; i < count; ++i)
		{
			HairStrandCollider col = xHairColliders[i];
			float3 closest = col.p0;
			if (col.type == 1) // capsule
			{
				float3 seg = col.p1 - col.p0;
				float segLenSq = dot(seg, seg);
				float t = segLenSq > 1e-7f ? clamp(dot(pos - col.p0, seg) / segLenSq, 0.0f, 1.0f) : 0.0f;
				closest = col.p0 + seg * t;
			}
			float3 diff = pos - closest;
			float dist = length(diff);
			float minDist = col.radius + xHairSDFMargin;
			if (dist < minDist)
			{
				float3 n = dist > 1e-6f ? (diff / dist) : float3(0, 1, 0);
				pos = closest + n * minDist;
				// Tangential friction
				float3 vel = pos - prevPos;
				float3 vn = n * dot(vel, n);
				float3 vt = vel - vn;
				prevPos = pos - (vn + vt * (1.0f - clamp(xHairFriction, 0.0f, 1.0f)));
			}
		}
	}

	// 2. GPU 3D Head/Upper-Body SDF collision
	if (xHairFlags & HAIRSTRAND_FLAG_USE_SDF)
	{
		float3 localPos = mul(xHairHeadInvWorld, float4(pos, 1.0f)).xyz;
		if (all(localPos >= xHairSDFBoundsMin) && all(localPos <= xHairSDFBoundsMax))
		{
			float dist = SampleSDF(localPos);
			if (dist < xHairSDFMargin)
			{
				float eps = max(xHairSDFCellSize, 1e-4f);
				float gx = SampleSDF(localPos + float3(eps, 0, 0)) - SampleSDF(localPos - float3(eps, 0, 0));
				float gy = SampleSDF(localPos + float3(0, eps, 0)) - SampleSDF(localPos - float3(0, eps, 0));
				float gz = SampleSDF(localPos + float3(0, 0, eps)) - SampleSDF(localPos - float3(0, 0, eps));
				float3 gradLocal = float3(gx, gy, gz);
				float gLen = length(gradLocal);
				if (gLen > 1e-6f)
				{
					gradLocal /= gLen;
					float push = xHairSDFMargin - dist;
					localPos += gradLocal * push;
					float3 pushedWorld = mul(xHairHeadWorld, float4(localPos, 1.0f)).xyz;
					float3 normalWorld = normalize(mul((float3x3)xHairHeadWorld, gradLocal));
					pos = pushedWorld;

					float3 vel = pos - prevPos;
					float3 vn = normalWorld * dot(vel, normalWorld);
					float3 vt = vel - vn;
					prevPos = pos - (vn + vt * (1.0f - clamp(xHairFriction, 0.0f, 1.0f)));
				}
			}
		}
	}
}

[numthreads(THREADCOUNT_HAIRSTAND_SIM, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	uint guideIdx = DTid.x;
	if (guideIdx >= xHairGuideCount)
		return;

	uint numVerts = min(xHairVerticesPerStrand, MAX_STRAND_VERTS);
	if (numVerts < 2)
		return;

	uint baseIdx = guideIdx * xHairVerticesPerStrand;
	float4x4 skinMat = ComputeSkinMatrix(guideIdx);

	bool resetSim = (xHairFlags & HAIRSTRAND_FLAG_RESET_SIM) != 0;
	bool simulate = (xHairFlags & HAIRSTRAND_FLAG_SIMULATE) != 0;

	float3 pos[MAX_STRAND_VERTS];
	float3 prevPos[MAX_STRAND_VERTS];
	float3 restWorld[MAX_STRAND_VERTS];
	float invMass[MAX_STRAND_VERTS];
	float restLen[MAX_STRAND_VERTS];

	for (uint i = 0; i < numVerts; ++i)
	{
		HairStrandRestVertex rv = restVertices[baseIdx + i];
		restWorld[i] = mul(skinMat, float4(rv.position_invmass.xyz, 1.0f)).xyz;
		invMass[i] = (i == 0) ? 0.0f : rv.position_invmass.w;
		restLen[i] = rv.rest_length;

		if (resetSim || !simulate)
		{
			pos[i] = restWorld[i];
			prevPos[i] = restWorld[i];
		}
		else
		{
			pos[i] = simPositions[baseIdx + i].xyz;
			prevPos[i] = simPrevPositions[baseIdx + i].xyz;
		}
	}

	if (simulate && !resetSim && xHairDt > 1e-6f)
	{
		// Fixed-timestep substepping (60 Hz, max 4 substeps)
		float totalDt = min(xHairDt, 1.0f / 15.0f);
		uint substeps = clamp((uint)ceil(totalDt * 60.0f), 1u, 4u);
		float dt = totalDt / float(substeps);
		float dt2 = dt * dt;
		float damp = clamp(1.0f - xHairDamping * dt * 10.0f, 0.7f, 0.999f);

		for (uint step = 0; step < substeps; ++step)
		{
			// 1. Pin root vertex to animated scalp
			pos[0] = restWorld[0];
			prevPos[0] = restWorld[0];

			// 2. Verlet integration with arbitrary 3D gravity & turbulent 3D wind
			for (uint i = 1; i < numVerts; ++i)
			{
				if (invMass[i] <= 0.0f)
				{
					pos[i] = restWorld[i];
					prevPos[i] = restWorld[i];
					continue;
				}

				float3 cur = pos[i];
				float3 vel = (cur - prevPos[i]) * damp;

				// Coherent spatial-temporal wind gust
				float3 windForce = xHairWind;
				if (xHairWindTurbulence > 1e-4f)
				{
					float phase = xHairTime * 3.5f + float(guideIdx) * 0.37f + float(i) * 0.23f;
					float3 gust = float3(
						sin(phase * 1.13f + cur.y * 4.0f),
						cos(phase * 0.87f + cur.x * 4.0f) * 0.4f,
						sin(phase * 1.41f + cur.z * 4.0f)
					);
					windForce += gust * xHairWindTurbulence;
				}

				float3 accel = xHairGravity + windForce;
				prevPos[i] = cur;
				pos[i] = cur + vel + accel * dt2;
			}

			// 3. Global shape constraint (pulls vertices toward skinned rest pose)
			if (xHairGlobalStiffness > 1e-4f)
			{
				float range = max(xHairGlobalStiffnessRange, 0.05f);
				for (uint i = 1; i < numVerts; ++i)
				{
					float u = float(i) / float(numVerts - 1);
					if (u <= range)
					{
						float factor = xHairGlobalStiffness * (1.0f - u / range);
						pos[i] += (restWorld[i] - pos[i]) * clamp(factor, 0.0f, 1.0f);
					}
				}
			}

			// 4. Local shape constraint (preserves curl/wave bend along strand)
			if (xHairLocalStiffness > 1e-4f && numVerts >= 3)
			{
				float localK = clamp(xHairLocalStiffness * 0.5f, 0.0f, 0.5f);
				float3 prevRestDir = normalize(restWorld[1] - restWorld[0] + 1e-6f);
				float3 prevSimDir = normalize(pos[1] - pos[0] + 1e-6f);
				float4 frameQ = QuatFromTwoVectors(prevRestDir, prevSimDir);

				for (uint i = 1; i + 1 < numVerts; ++i)
				{
					float3 restSeg = restWorld[i + 1] - restWorld[i];
					float3 targetPos = pos[i] + QuatRotate(frameQ, restSeg);
					float3 delta = (targetPos - pos[i + 1]) * localK;
					if (invMass[i] > 0.0f) pos[i] -= delta * 0.5f;
					pos[i + 1] += delta * 0.5f;

					float3 newSimDir = normalize(pos[i + 1] - pos[i] + 1e-6f);
					float3 newRestDir = normalize(restSeg + 1e-6f);
					frameQ = QuatFromTwoVectors(newRestDir, newSimDir);
				}
			}

			// 5. Length / inextensibility constraints + collisions
			uint iters = clamp(xHairLengthIterations, 1u, 8u);
			for (uint it = 0; it < iters; ++it)
			{
				pos[0] = restWorld[0];
				for (uint i = 1; i < numVerts; ++i)
				{
					float3 diff = pos[i] - pos[i - 1];
					float d = length(diff);
					float targetL = max(restLen[i], 1e-5f);
					if (d > 1e-7f)
					{
						float3 corr = diff * ((d - targetL) / d);
						if (invMass[i - 1] <= 0.0f)
						{
							pos[i] -= corr;
						}
						else
						{
							pos[i - 1] += corr * 0.25f;
							pos[i] -= corr * 0.75f;
						}
					}
				}
			}

			// 6. Collisions (capsules/spheres + GPU 3D Head SDF)
			for (uint i = 1; i < numVerts; ++i)
			{
				ResolveCollisions(pos[i], prevPos[i]);
			}

			// Final single length pass after collision so strands stay inextensible
			pos[0] = restWorld[0];
			for (uint i = 1; i < numVerts; ++i)
			{
				float3 diff = pos[i] - pos[i - 1];
				float d = length(diff);
				float targetL = max(restLen[i], 1e-5f);
				if (d > 1e-7f)
				{
					pos[i] = pos[i - 1] + diff * (targetL / d);
				}
			}
		}
	}

	for (uint i = 0; i < numVerts; ++i)
	{
		simPositions[baseIdx + i] = float4(pos[i], invMass[i]);
		simPrevPositions[baseIdx + i] = float4(prevPos[i], restLen[i]);
	}
}
