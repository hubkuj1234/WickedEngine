#pragma once
#include "CommonInclude.h"
#include "wiGraphicsDevice.h"
#include "wiEnums.h"
#include "wiMath.h"
#include "wiECS.h"
#include "wiPrimitive.h"
#include "wiVector.h"
#include "wiScene_Decl.h"
#include "wiScene_Components.h"
#include "shaders/ShaderInterop_HairStrand.h"

namespace wi
{
	class Archive;

	class HairStrandSystem
	{
	public:
		// CPU rest groom data
		uint32_t guideCount = 0;
		uint32_t verticesPerStrand = 12;
		uint32_t followPerGuide = 0;

		wi::vector<HairStrandRestVertex> restVertices;
		wi::vector<HairStrandSkinWeight> skinWeights;
		wi::vector<HairStrandFollowDesc> followDescs;

		// Dynamic per-frame pose & collision inputs (uploaded by Daisy's WickedSceneAdapter)
		wi::vector<XMFLOAT4X4> boneMatrices;
		wi::vector<HairStrandCollider> colliders;

		// Optional head/upper-body collision triangles in head-bone local space for GPU 3D SDF
		wi::vector<XMFLOAT4> sdfTriangleVertices; // 3 XMFLOAT4 per triangle
		XMFLOAT3 sdfBoundsMin = XMFLOAT3(-0.25f, -0.35f, -0.25f);
		XMFLOAT3 sdfBoundsMax = XMFLOAT3(0.25f, 0.30f, 0.25f);
		XMFLOAT4X4 headWorld = wi::math::IDENTITY_MATRIX;
		XMFLOAT4X4 headInvWorld = wi::math::IDENTITY_MATRIX;

		// TressFX simulation parameters
		bool enabled = true;
		bool simulatePhysics = true;
		bool useSDFCollision = true;
		bool useColliderShapes = true;
		XMFLOAT3 gravity = XMFLOAT3(0.0f, -9.81f, 0.0f);
		XMFLOAT3 wind = XMFLOAT3(0.0f, 0.0f, 0.0f);
		float windTurbulence = 0.25f;
		float damping = 0.08f;
		float globalStiffness = 0.25f;
		float globalStiffnessRange = 0.6f;
		float localStiffness = 0.45f;
		uint32_t lengthIterations = 4;
		float sdfCollisionMargin = 0.004f;
		float friction = 0.35f;
		float minPixelWidth = 1.25f;
		float clumpRadius = 0.008f;
		float tipSeparation = 0.015f;

		// Chiang 2016 Hair BSDF material parameters
		ShaderHairStrandMaterial material = {};

		// Transform & bounds
		XMFLOAT4X4 world = wi::math::IDENTITY_MATRIX;
		wi::primitive::AABB aabb;
		uint32_t layerMask = ~0u;

		// GPU resources
		wi::graphics::GPUBuffer constantBuffer;
		wi::graphics::GPUBuffer materialBuffer; // 1-element StructuredBuffer<ShaderHairStrandMaterial> for PT bindless lookup
		wi::graphics::GPUBuffer restVertexBuffer;
		wi::graphics::GPUBuffer skinWeightBuffer;
		wi::graphics::GPUBuffer followDescBuffer;
		wi::graphics::GPUBuffer boneMatrixBuffer;

		wi::graphics::GPUBuffer simPositionBuffer;
		wi::graphics::GPUBuffer simPrevPositionBuffer;

		wi::graphics::GPUBuffer renderPositionBuffer;
		wi::graphics::GPUBuffer renderPrevPositionBuffer;
		wi::graphics::GPUBuffer renderTangentBuffer;

		wi::graphics::GPUBuffer segmentAABBBuffer;
		wi::graphics::GPUBuffer segmentIndexBuffer;
		wi::graphics::RaytracingAccelerationStructure BLAS;

		wi::graphics::GPUBuffer sdfTriangleBuffer;
		wi::graphics::Texture sdfTexture[2];

		// State flags
		mutable bool dirtyBuffers = true;
		mutable bool dirtySDF = true;
		mutable bool resetSimulation = true;
		mutable bool mustRebuildBLAS = true;
		mutable float lastDt = 0.0f;
		mutable float elapsedTime = 0.0f;

		uint32_t GetTotalRenderStrands() const { return guideCount * (1u + followPerGuide); }
		uint32_t GetTotalRenderVertices() const { return GetTotalRenderStrands() * verticesPerStrand; }
		uint32_t GetTotalRenderSegments() const { return verticesPerStrand > 1 ? GetTotalRenderStrands() * (verticesPerStrand - 1u) : 0u; }

		void DeleteRenderData();
		void CreateRenderData();
		void CreateRaytracingRenderData();
		void CreateSDFData();

		void UpdateCPU(const wi::scene::TransformComponent& transform, float dt);

		struct UpdateGPUItem
		{
			const HairStrandSystem* hair = nullptr;
			uint32_t instanceIndex = 0;
		};
		static void UpdateGPU(
			const UpdateGPUItem* items,
			uint32_t itemCount,
			wi::graphics::CommandList cmd
		);

		void Draw(
			wi::enums::RENDERPASS renderPass,
			wi::graphics::CommandList cmd,
			uint16_t camera_mask = 1
		) const;

		void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);

		static void Initialize();
	};
}
