#ifndef WI_SHADERINTEROP_HAIRSTRAND_H
#define WI_SHADERINTEROP_HAIRSTRAND_H

#include "ShaderInterop.h"
#include "ShaderInterop_Renderer.h"

#define CBSLOT_OTHER_HAIRSTRAND CBSLOT_OTHER_HAIRPARTICLE

#define THREADCOUNT_HAIRSTAND_SIM 64
#define THREADCOUNT_HAIRSTAND_FOLLOW 64
#define THREADCOUNT_HAIRSTAND_SDF 4
#define HAIRSTAND_MAX_COLLIDERS 64
#define HAIRSTAND_SDF_RESOLUTION 32

enum HAIRSTRAND_FLAGS
{
	HAIRSTRAND_FLAG_NONE = 0,
	HAIRSTRAND_FLAG_RESET_SIM = 1 << 0,
	HAIRSTRAND_FLAG_SIMULATE = 1 << 1,
	HAIRSTRAND_FLAG_USE_SDF = 1 << 2,
	HAIRSTRAND_FLAG_USE_COLLIDERS = 1 << 3,
	HAIRSTRAND_FLAG_RAYTRACED = 1 << 4,
};

// Analytical sphere (type == 0) or capsule (type == 1) collider in world space
struct alignas(16) HairStrandCollider
{
	float3 p0;
	float radius;
	float3 p1;
	uint type; // 0 = sphere (p0, radius), 1 = capsule (p0..p1, radius)
};

// Root skinning attachment for a guide strand (up to 4 bone influences)
struct alignas(16) HairStrandSkinWeight
{
	uint4 bone_indices;
	float4 bone_weights;
};

// Per-vertex rest data for a guide strand
struct alignas(16) HairStrandRestVertex
{
	float4 position_invmass; // xyz = rest position in bind/local space, w = invMass (0 for root)
	float4 local_quat;       // xyzw = rest local rotation quaternion relative to parent segment
	float rest_length;       // rest segment length to previous vertex
	float root_radius;       // strand root radius (meters)
	float tip_radius;        // strand tip radius (meters)
	float length_scale;      // per-strand length multiplier
};

// Follow-hair deterministic offset descriptor (per follow strand)
struct alignas(16) HairStrandFollowDesc
{
	uint guide_index;
	float3 root_offset;      // local offset at root
	float tip_separation;    // tip spread multiplier
	float clump_strength;    // 0..1 pull toward guide
	float radius_scale;      // per-follow-strand radius multiplier
	float random_seed;       // deterministic [0,1] hash for melanin/roughness variation
};

// Chiang 2016 Hair BSDF material parameters (shared between Raster & Path Tracer via bindless SRV)
struct alignas(16) ShaderHairStrandMaterial
{
	float melanin;           // 0 = white/blonde, 0.5 = brown, 1.0 = black
	float melanin_redness;   // 0..1 pheomelanin ratio (ginger/auburn)
	float beta_m;            // longitudinal roughness (0.05..1.0)
	float beta_n;            // azimuthal roughness (0.05..1.0)

	float3 tint;             // artistic dye/tint reflectance
	float tint_strength;     // 0 = pure melanin, 1 = full dye tint

	float ior;               // index of refraction (default 1.55)
	float cuticle_tilt_rad;  // cuticle scale tilt angle in radians (default ~2 deg)
	float melanin_variation; // subtle per-strand melanin variation
	float roughness_variation;

	float3 root_color_shift; // subtle root darkening multiplier
	float padding0;
};

// Raytracing procedural AABB layout matching D3D12_RAYTRACING_AABB / VkAabbPositionsKHR / MTLPackedFloat3[2]
struct HairStrandAABB
{
	float minX, minY, minZ;
	float maxX, maxY, maxZ;
};

// Segment index pair for procedural raytracing hit lookup
struct HairStrandSegment
{
	uint v0;
	uint v1;
	uint strand_index;
	float u0; // normalized root-to-tip parameter at v0
};

CBUFFER(HairStrandCB, CBSLOT_OTHER_HAIRSTRAND)
{
	float4x4 xHairWorld;
	float4x4 xHairInvWorld;
	float4x4 xHairHeadWorld;
	float4x4 xHairHeadInvWorld;

	uint xHairFlags;
	uint xHairGuideCount;
	uint xHairVerticesPerStrand;
	uint xHairFollowPerGuide;

	uint xHairTotalRenderStrands;
	uint xHairColliderCount;
	uint xHairBoneCount;
	uint xHairLengthIterations;

	float3 xHairGravity;
	float xHairDt;

	float3 xHairWind;
	float xHairWindTurbulence;

	float xHairDamping;
	float xHairGlobalStiffness;
	float xHairGlobalStiffnessRange;
	float xHairLocalStiffness;

	float xHairSDFMargin;
	float xHairFriction;
	float xHairMinPixelWidth;
	float xHairTime;

	float3 xHairSDFBoundsMin;
	float xHairSDFCellSize;

	float3 xHairSDFBoundsMax;
	uint xHairSDFTriangleCount;

	uint xHairSDFJumpStep;
	uint xHairInstanceIndex;
	float xHairClumpRadius;
	float xHairTipSeparation;

	ShaderHairStrandMaterial xHairMaterial;
	HairStrandCollider xHairColliders[HAIRSTAND_MAX_COLLIDERS];
};

#endif // WI_SHADERINTEROP_HAIRSTRAND_H
