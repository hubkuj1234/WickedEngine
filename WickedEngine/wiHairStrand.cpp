#include "wiHairStrand.h"
#include "wiRenderer.h"
#include "wiScene.h"
#include "wiArchive.h"
#include "wiEventHandler.h"
#include "wiBacklog.h"
#include "shaders/ShaderInterop_HairStrand.h"

#include <algorithm>
#include <cstring>

using namespace wi::graphics;
using namespace wi::enums;

namespace wi
{
	static Shader cs_sdf_init;
	static Shader cs_sdf_smooth;
	static Shader cs_simulate;
	static Shader cs_follow;
	static Shader vs_strand;
	static Shader ps_strand;
	static Shader ps_strand_prepass;
	static Shader ps_strand_shadow;

	static DepthStencilState dss_default;
	static DepthStencilState dss_shadow;
	static RasterizerState rs_nocull;
	static RasterizerState rs_shadow;
	static BlendState bs_opaque;

	static PipelineState PSO_strand[RENDERPASS_COUNT];

	static void LoadHairStrandShaders()
	{
		wi::renderer::LoadShader(ShaderStage::CS, cs_sdf_init, "hairstrand_sdf_initCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, cs_sdf_smooth, "hairstrand_sdf_jfaCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, cs_simulate, "hairstrand_simulateCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, cs_follow, "hairstrand_followCS.cso");
		wi::renderer::LoadShader(ShaderStage::VS, vs_strand, "hairstrandVS.cso");
		wi::renderer::LoadShader(ShaderStage::PS, ps_strand, "hairstrandPS.cso");
		wi::renderer::LoadShader(ShaderStage::PS, ps_strand_prepass, "hairstrandPS_prepass.cso");
		wi::renderer::LoadShader(ShaderStage::PS, ps_strand_shadow, "hairstrandPS_shadow.cso");

		GraphicsDevice* device = wi::graphics::GetDevice();
		if (!device)
			return;

		for (int i = 0; i < RENDERPASS_COUNT; ++i)
		{
			PipelineStateDesc desc;
			desc.vs = &vs_strand;
			desc.bs = &bs_opaque;
			desc.rs = &rs_nocull;
			desc.dss = &dss_default;
			desc.pt = PrimitiveTopology::TRIANGLELIST;

			switch (i)
			{
			case RENDERPASS_MAIN:
				desc.ps = &ps_strand;
				break;
			case RENDERPASS_PREPASS:
			case RENDERPASS_PREPASS_DEPTHONLY:
				desc.ps = &ps_strand_prepass;
				break;
			case RENDERPASS_SHADOW:
				desc.ps = &ps_strand_shadow;
				desc.rs = &rs_shadow;
				desc.dss = &dss_shadow;
				break;
			default:
				desc.ps = &ps_strand;
				break;
			}
			device->CreatePipelineState(&desc, &PSO_strand[i]);
		}
	}

	void HairStrandSystem::Initialize()
	{
		rs_nocull.fill_mode = FillMode::SOLID;
		rs_nocull.cull_mode = CullMode::NONE;
		rs_nocull.front_counter_clockwise = true;
		rs_nocull.depth_clip_enable = true;

		rs_shadow = rs_nocull;
		rs_shadow.depth_bias = -1;
		rs_shadow.slope_scaled_depth_bias = -2.0f;

		dss_default.depth_enable = true;
		dss_default.depth_write_mask = DepthWriteMask::ALL;
		dss_default.depth_func = ComparisonFunc::GREATER_EQUAL;

		dss_shadow.depth_enable = true;
		dss_shadow.depth_write_mask = DepthWriteMask::ALL;
		dss_shadow.depth_func = ComparisonFunc::GREATER_EQUAL;

		bs_opaque.render_target[0].blend_enable = false;
		bs_opaque.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;

		static wi::eventhandler::Handle handle = wi::eventhandler::Subscribe(
			wi::eventhandler::EVENT_RELOAD_SHADERS,
			[](uint64_t userdata) { LoadHairStrandShaders(); }
		);
		LoadHairStrandShaders();

		wi::backlog::post("wi::HairStrandSystem Initialized");
	}

	void HairStrandSystem::DeleteRenderData()
	{
		constantBuffer = {};
		materialBuffer = {};
		restVertexBuffer = {};
		skinWeightBuffer = {};
		followDescBuffer = {};
		boneMatrixBuffer = {};
		simPositionBuffer = {};
		simPrevPositionBuffer = {};
		renderPositionBuffer = {};
		renderPrevPositionBuffer = {};
		renderTangentBuffer = {};
		segmentAABBBuffer = {};
		segmentIndexBuffer = {};
		BLAS = {};
		sdfTriangleBuffer = {};
		sdfTexture[0] = {};
		sdfTexture[1] = {};
	}

	void HairStrandSystem::CreateRenderData()
	{
		DeleteRenderData();
		dirtyBuffers = false;
		resetSimulation = true;
		mustRebuildBLAS = true;

		GraphicsDevice* device = wi::graphics::GetDevice();
		if (!device || guideCount == 0 || verticesPerStrand < 2)
			return;

		const uint32_t guideVertCount = guideCount * verticesPerStrand;
		const uint32_t totalStrands = GetTotalRenderStrands();
		const uint32_t renderVertCount = totalStrands * verticesPerStrand;
		const uint32_t renderSegCount = GetTotalRenderSegments();

		// Constant buffer
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::CONSTANT_BUFFER;
			bd.size = sizeof(HairStrandCB);
			device->CreateBuffer(&bd, nullptr, &constantBuffer);
			device->SetName(&constantBuffer, "HairStrand.constantBuffer");
		}

		// Material buffer (bindless SRV for path tracer)
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(ShaderHairStrandMaterial);
			bd.size = sizeof(ShaderHairStrandMaterial);
			device->CreateBuffer(&bd, &material, &materialBuffer);
			device->SetName(&materialBuffer, "HairStrand.materialBuffer");
		}

		// Rest vertices
		{
			wi::vector<HairStrandRestVertex> paddedRest = restVertices;
			paddedRest.resize(guideVertCount);
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(HairStrandRestVertex);
			bd.size = sizeof(HairStrandRestVertex) * guideVertCount;
			device->CreateBuffer(&bd, paddedRest.data(), &restVertexBuffer);
			device->SetName(&restVertexBuffer, "HairStrand.restVertices");
		}

		// Skin weights
		{
			wi::vector<HairStrandSkinWeight> paddedWeights = skinWeights;
			paddedWeights.resize(guideCount);
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(HairStrandSkinWeight);
			bd.size = sizeof(HairStrandSkinWeight) * guideCount;
			device->CreateBuffer(&bd, paddedWeights.data(), &skinWeightBuffer);
			device->SetName(&skinWeightBuffer, "HairStrand.skinWeights");
		}

		// Follow descriptors (always allocate at least 1 element so binding is valid)
		{
			const uint32_t followCount = std::max<uint32_t>(1u, totalStrands > guideCount ? (totalStrands - guideCount) : 0u);
			wi::vector<HairStrandFollowDesc> paddedFollow = followDescs;
			paddedFollow.resize(followCount);
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(HairStrandFollowDesc);
			bd.size = sizeof(HairStrandFollowDesc) * followCount;
			device->CreateBuffer(&bd, paddedFollow.data(), &followDescBuffer);
			device->SetName(&followDescBuffer, "HairStrand.followDescs");
		}

		// Bone matrices (at least 1 identity matrix)
		{
			const uint32_t boneCap = std::max<uint32_t>(1u, std::max<uint32_t>((uint32_t)boneMatrices.size(), 256u));
			wi::vector<XMFLOAT4X4> initBones(boneCap, wi::math::IDENTITY_MATRIX);
			for (size_t i = 0; i < boneMatrices.size() && i < boneCap; ++i)
				initBones[i] = boneMatrices[i];
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(XMFLOAT4X4);
			bd.size = sizeof(XMFLOAT4X4) * boneCap;
			device->CreateBuffer(&bd, initBones.data(), &boneMatrixBuffer);
			device->SetName(&boneMatrixBuffer, "HairStrand.boneMatrices");
		}

		// Simulation guide positions (current & previous)
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(XMFLOAT4);
			bd.size = sizeof(XMFLOAT4) * guideVertCount;
			device->CreateBuffer(&bd, nullptr, &simPositionBuffer);
			device->CreateBuffer(&bd, nullptr, &simPrevPositionBuffer);
			device->SetName(&simPositionBuffer, "HairStrand.simPositions");
			device->SetName(&simPrevPositionBuffer, "HairStrand.simPrevPositions");
		}

		// Interpolated render strand positions (current, previous) & tangents
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(XMFLOAT4);
			bd.size = sizeof(XMFLOAT4) * renderVertCount;
			device->CreateBuffer(&bd, nullptr, &renderPositionBuffer);
			device->CreateBuffer(&bd, nullptr, &renderPrevPositionBuffer);
			device->CreateBuffer(&bd, nullptr, &renderTangentBuffer);
			device->SetName(&renderPositionBuffer, "HairStrand.renderPositions");
			device->SetName(&renderPrevPositionBuffer, "HairStrand.renderPrevPositions");
			device->SetName(&renderTangentBuffer, "HairStrand.renderTangents");
		}

		// Procedural AABB & segment index buffers for raytracing
		if (renderSegCount > 0)
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			if (device->CheckCapability(GraphicsDeviceCapability::RAYTRACING))
				bd.misc_flags |= ResourceMiscFlag::RAY_TRACING;
			bd.stride = sizeof(HairStrandAABB);
			bd.size = sizeof(HairStrandAABB) * renderSegCount;
			device->CreateBuffer(&bd, nullptr, &segmentAABBBuffer);
			device->SetName(&segmentAABBBuffer, "HairStrand.segmentAABBs");

			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(HairStrandSegment);
			bd.size = sizeof(HairStrandSegment) * renderSegCount;
			device->CreateBuffer(&bd, nullptr, &segmentIndexBuffer);
			device->SetName(&segmentIndexBuffer, "HairStrand.segmentIndices");
		}

		CreateSDFData();
		CreateRaytracingRenderData();
	}

	void HairStrandSystem::CreateSDFData()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();
		if (!device)
			return;

		TextureDesc td;
		td.type = TextureDesc::Type::TEXTURE_3D;
		td.width = HAIRSTAND_SDF_RESOLUTION;
		td.height = HAIRSTAND_SDF_RESOLUTION;
		td.depth = HAIRSTAND_SDF_RESOLUTION;
		td.format = Format::R16_FLOAT;
		td.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
		td.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
		device->CreateTexture(&td, nullptr, &sdfTexture[0]);
		device->CreateTexture(&td, nullptr, &sdfTexture[1]);
		device->SetName(&sdfTexture[0], "HairStrand.sdfTexture0");
		device->SetName(&sdfTexture[1], "HairStrand.sdfTexture1");

		if (!sdfTriangleVertices.empty())
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(XMFLOAT4);
			bd.size = sizeof(XMFLOAT4) * sdfTriangleVertices.size();
			device->CreateBuffer(&bd, sdfTriangleVertices.data(), &sdfTriangleBuffer);
			device->SetName(&sdfTriangleBuffer, "HairStrand.sdfTriangles");
			dirtySDF = true;
		}
		else
		{
			sdfTriangleBuffer = {};
			dirtySDF = false;
		}
	}

	void HairStrandSystem::CreateRaytracingRenderData()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();
		if (!device || !device->CheckCapability(GraphicsDeviceCapability::RAYTRACING))
			return;
		const uint32_t segCount = GetTotalRenderSegments();
		if (segCount == 0 || !segmentAABBBuffer.IsValid())
			return;

		RaytracingAccelerationStructureDesc desc;
		desc.type = RaytracingAccelerationStructureDesc::Type::BOTTOMLEVEL;
		desc.flags = RaytracingAccelerationStructureDesc::FLAG_ALLOW_UPDATE |
		             RaytracingAccelerationStructureDesc::FLAG_PREFER_FAST_BUILD;

		desc.bottom_level.geometries.emplace_back();
		auto& geo = desc.bottom_level.geometries.back();
		geo.type = RaytracingAccelerationStructureDesc::BottomLevel::Geometry::Type::PROCEDURAL_AABBS;
		geo.flags = RaytracingAccelerationStructureDesc::BottomLevel::Geometry::FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;
		geo.aabbs.aabb_buffer = segmentAABBBuffer;
		geo.aabbs.offset = 0;
		geo.aabbs.count = segCount;
		geo.aabbs.stride = sizeof(HairStrandAABB);

		device->CreateRaytracingAccelerationStructure(&desc, &BLAS);
		device->SetName(&BLAS, "HairStrand.BLAS");
		mustRebuildBLAS = true;
	}

	void HairStrandSystem::UpdateCPU(const wi::scene::TransformComponent& transform, float dt)
	{
		world = transform.world;
		lastDt = dt;
		elapsedTime += dt;

		if (dirtyBuffers)
		{
			CreateRenderData();
		}

		// Conservative world-space AABB around character transform & rest vertices
		XMFLOAT3 bmin(FLT_MAX, FLT_MAX, FLT_MAX);
		XMFLOAT3 bmax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		XMMATRIX W = XMLoadFloat4x4(&world);
		if (!restVertices.empty())
		{
			const size_t stride = std::max<size_t>(1, restVertices.size() / 256);
			for (size_t i = 0; i < restVertices.size(); i += stride)
			{
				XMVECTOR p = XMVector3Transform(XMLoadFloat4(&restVertices[i].position_invmass), W);
				XMFLOAT3 wp;
				XMStoreFloat3(&wp, p);
				bmin.x = std::min(bmin.x, wp.x);
				bmin.y = std::min(bmin.y, wp.y);
				bmin.z = std::min(bmin.z, wp.z);
				bmax.x = std::max(bmax.x, wp.x);
				bmax.y = std::max(bmax.y, wp.y);
				bmax.z = std::max(bmax.z, wp.z);
			}
			// Expand for simulation sway
			const float pad = 0.6f;
			bmin.x -= pad; bmin.y -= pad; bmin.z -= pad;
			bmax.x += pad; bmax.y += pad; bmax.z += pad;
		}
		else
		{
			bmin = XMFLOAT3(-1, -1, -1);
			bmax = XMFLOAT3(1, 1, 1);
		}
		aabb = wi::primitive::AABB(bmin, bmax);
	}

	void HairStrandSystem::UpdateGPU(const UpdateGPUItem* items, uint32_t itemCount, CommandList cmd)
	{
		if (itemCount == 0)
			return;

		GraphicsDevice* device = wi::graphics::GetDevice();
		if (!device || !cs_simulate.IsValid() || !cs_follow.IsValid())
			return;

		device->EventBegin("HairStrandSystem - UpdateGPU", cmd);

		for (uint32_t idx = 0; idx < itemCount; ++idx)
		{
			const HairStrandSystem* hair = items[idx].hair;
			if (!hair || !hair->enabled || hair->guideCount == 0 || !hair->constantBuffer.IsValid())
				continue;

			// Upload bone matrices & material buffer
			if (!hair->boneMatrices.empty() && hair->boneMatrixBuffer.IsValid())
			{
				const size_t copyBytes = std::min<size_t>(
					hair->boneMatrices.size() * sizeof(XMFLOAT4X4),
					hair->boneMatrixBuffer.GetDesc().size
				);
				device->UpdateBuffer(&hair->boneMatrixBuffer, hair->boneMatrices.data(), cmd, copyBytes);
			}
			if (hair->materialBuffer.IsValid())
			{
				device->UpdateBuffer(&hair->materialBuffer, &hair->material, cmd, sizeof(ShaderHairStrandMaterial));
			}

			// Prepare constant buffer
			HairStrandCB cb = {};
			cb.xHairWorld = hair->world;
			XMMATRIX W = XMLoadFloat4x4(&hair->world);
			XMStoreFloat4x4(&cb.xHairInvWorld, XMMatrixInverse(nullptr, W));
			cb.xHairHeadWorld = hair->headWorld;
			cb.xHairHeadInvWorld = hair->headInvWorld;

			uint32_t flags = HAIRSTRAND_FLAG_NONE;
			if (hair->resetSimulation) flags |= HAIRSTRAND_FLAG_RESET_SIM;
			if (hair->simulatePhysics) flags |= HAIRSTRAND_FLAG_SIMULATE;
			if (hair->useSDFCollision && hair->sdfTexture[1].IsValid() && hair->sdfTriangleBuffer.IsValid())
				flags |= HAIRSTRAND_FLAG_USE_SDF;
			if (hair->useColliderShapes && !hair->colliders.empty())
				flags |= HAIRSTRAND_FLAG_USE_COLLIDERS;
			if (hair->BLAS.IsValid())
				flags |= HAIRSTRAND_FLAG_RAYTRACED;

			cb.xHairFlags = flags;
			cb.xHairGuideCount = hair->guideCount;
			cb.xHairVerticesPerStrand = hair->verticesPerStrand;
			cb.xHairFollowPerGuide = hair->followPerGuide;
			cb.xHairTotalRenderStrands = hair->GetTotalRenderStrands();
			cb.xHairColliderCount = std::min<uint32_t>((uint32_t)hair->colliders.size(), HAIRSTAND_MAX_COLLIDERS);
			cb.xHairBoneCount = (uint32_t)hair->boneMatrices.size();
			cb.xHairLengthIterations = hair->lengthIterations;
			cb.xHairGravity = hair->gravity;
			cb.xHairDt = hair->lastDt;
			cb.xHairWind = hair->wind;
			cb.xHairWindTurbulence = hair->windTurbulence;
			cb.xHairDamping = hair->damping;
			cb.xHairGlobalStiffness = hair->globalStiffness;
			cb.xHairGlobalStiffnessRange = hair->globalStiffnessRange;
			cb.xHairLocalStiffness = hair->localStiffness;
			cb.xHairSDFMargin = hair->sdfCollisionMargin;
			cb.xHairFriction = hair->friction;
			cb.xHairMinPixelWidth = hair->minPixelWidth;
			cb.xHairTime = hair->elapsedTime;
			cb.xHairSDFBoundsMin = hair->sdfBoundsMin;
			cb.xHairSDFBoundsMax = hair->sdfBoundsMax;
			const float extX = hair->sdfBoundsMax.x - hair->sdfBoundsMin.x;
			cb.xHairSDFCellSize = extX / float(HAIRSTAND_SDF_RESOLUTION);
			cb.xHairSDFTriangleCount = (uint32_t)(hair->sdfTriangleVertices.size() / 3);
			cb.xHairInstanceIndex = items[idx].instanceIndex;
			cb.xHairClumpRadius = hair->clumpRadius;
			cb.xHairTipSeparation = hair->tipSeparation;
			cb.xHairMaterial = hair->material;

			for (uint32_t c = 0; c < cb.xHairColliderCount; ++c)
				cb.xHairColliders[c] = hair->colliders[c];

			device->UpdateBuffer(&hair->constantBuffer, &cb, cmd, sizeof(HairStrandCB));
			{
				wi::vector<GPUBarrier> uploadBarriers;
				uploadBarriers.push_back(GPUBarrier::Buffer(&hair->constantBuffer, ResourceState::COPY_DST, ResourceState::CONSTANT_BUFFER));
				if (!hair->boneMatrices.empty() && hair->boneMatrixBuffer.IsValid())
					uploadBarriers.push_back(GPUBarrier::Buffer(&hair->boneMatrixBuffer, ResourceState::COPY_DST, ResourceState::SHADER_RESOURCE_COMPUTE));
				if (hair->materialBuffer.IsValid())
					uploadBarriers.push_back(GPUBarrier::Buffer(&hair->materialBuffer, ResourceState::COPY_DST, ResourceState::SHADER_RESOURCE_COMPUTE));
				device->Barrier(uploadBarriers.data(), (uint32_t)uploadBarriers.size(), cmd);
			}
			device->BindConstantBuffer(&hair->constantBuffer, CB_GETBINDSLOT(HairStrandCB), cmd);

			// 1. Rebuild GPU 3D Head SDF if dirty
			if (hair->dirtySDF && hair->sdfTriangleBuffer.IsValid() && cs_sdf_init.IsValid() && cs_sdf_smooth.IsValid())
			{
				const uint32_t groups = (HAIRSTAND_SDF_RESOLUTION + THREADCOUNT_HAIRSTAND_SDF - 1) / THREADCOUNT_HAIRSTAND_SDF;

				device->Barrier(GPUBarrier::Image(&hair->sdfTexture[0], ResourceState::SHADER_RESOURCE_COMPUTE, ResourceState::UNORDERED_ACCESS), cmd);
				device->BindComputeShader(&cs_sdf_init, cmd);
				device->BindResource(&hair->sdfTriangleBuffer, 0, cmd);
				device->BindUAV(&hair->sdfTexture[0], 0, cmd);
				device->Dispatch(groups, groups, groups, cmd);

				GPUBarrier b1[] = {
					GPUBarrier::Image(&hair->sdfTexture[0], ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE),
					GPUBarrier::Image(&hair->sdfTexture[1], ResourceState::SHADER_RESOURCE_COMPUTE, ResourceState::UNORDERED_ACCESS),
				};
				device->Barrier(b1, arraysize(b1), cmd);

				device->BindComputeShader(&cs_sdf_smooth, cmd);
				device->BindResource(&hair->sdfTexture[0], 0, cmd);
				device->BindUAV(&hair->sdfTexture[1], 0, cmd);
				device->Dispatch(groups, groups, groups, cmd);

				device->Barrier(GPUBarrier::Image(&hair->sdfTexture[1], ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE), cmd);
				hair->dirtySDF = false;
			}

			// 2. TressFX Guide-Strand Simulation
			{
				device->BindComputeShader(&cs_simulate, cmd);
				const GPUResource* srvs[] = {
					&hair->restVertexBuffer,
					&hair->skinWeightBuffer,
					&hair->boneMatrixBuffer,
					&hair->sdfTexture[1],
				};
				device->BindResources(srvs, 0, arraysize(srvs), cmd);
				const GPUResource* uavs[] = {
					&hair->simPositionBuffer,
					&hair->simPrevPositionBuffer,
				};
				device->BindUAVs(uavs, 0, arraysize(uavs), cmd);
				device->Dispatch((hair->guideCount + THREADCOUNT_HAIRSTAND_SIM - 1) / THREADCOUNT_HAIRSTAND_SIM, 1, 1, cmd);
				device->Barrier(GPUBarrier::Memory(), cmd);
			}

			// 3. Follow-hair interpolation, tangent/radius & procedural AABB build
			{
				device->BindComputeShader(&cs_follow, cmd);
				const GPUResource* srvs[] = {
					&hair->simPositionBuffer,
					&hair->restVertexBuffer,
					&hair->followDescBuffer,
				};
				device->BindResources(srvs, 0, arraysize(srvs), cmd);
				const GPUResource* uavs[] = {
					&hair->renderPositionBuffer,
					&hair->renderPrevPositionBuffer,
					&hair->renderTangentBuffer,
					&hair->segmentAABBBuffer,
					&hair->segmentIndexBuffer,
				};
				device->BindUAVs(uavs, 0, arraysize(uavs), cmd);
				const uint32_t totalStrands = hair->GetTotalRenderStrands();
				device->Dispatch((totalStrands + THREADCOUNT_HAIRSTAND_FOLLOW - 1) / THREADCOUNT_HAIRSTAND_FOLLOW, 1, 1, cmd);
				GPUBarrier postFollow[] = {
					GPUBarrier::Buffer(&hair->renderPositionBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Buffer(&hair->renderPrevPositionBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Buffer(&hair->renderTangentBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Buffer(&hair->segmentAABBBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Buffer(&hair->segmentIndexBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Memory(),
				};
				device->Barrier(postFollow, arraysize(postFollow), cmd);
			}

			hair->resetSimulation = false;
		}

		device->EventEnd(cmd);
	}

	void HairStrandSystem::Draw(RENDERPASS renderPass, CommandList cmd, uint16_t camera_mask) const
	{
		if (!enabled || guideCount == 0 || !constantBuffer.IsValid() || !renderPositionBuffer.IsValid())
			return;
		if (!PSO_strand[renderPass].IsValid())
			return;

		GraphicsDevice* device = wi::graphics::GetDevice();
		if (!device)
			return;

		const uint32_t segCount = GetTotalRenderSegments();
		if (segCount == 0)
			return;

		device->BindPipelineState(&PSO_strand[renderPass], cmd);
		device->BindConstantBuffer(&constantBuffer, CB_GETBINDSLOT(HairStrandCB), cmd);

		const GPUResource* srvs[] = {
			&renderPositionBuffer,
			&renderTangentBuffer,
		};
		device->BindResources(srvs, 0, arraysize(srvs), cmd);

		for (uint32_t camIdx = 0; camIdx < 16; ++camIdx)
		{
			if ((camera_mask & (1u << camIdx)) == 0)
				continue;
			uint32_t pushCam = camIdx;
			device->PushConstants(&pushCam, sizeof(pushCam), cmd);
			device->Draw(segCount * 6u, 0, cmd);
		}
	}

	void HairStrandSystem::Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri)
	{
		if (archive.IsReadMode())
		{
			archive >> guideCount;
			archive >> verticesPerStrand;
			archive >> followPerGuide;
			archive >> gravity;
			archive >> wind;
			dirtyBuffers = true;
		}
		else
		{
			archive << guideCount;
			archive << verticesPerStrand;
			archive << followPerGuide;
			archive << gravity;
			archive << wind;
		}
	}
}
