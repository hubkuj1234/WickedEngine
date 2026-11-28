#include "hairstrandHF.hlsli"
#include "hairBSDF_ChiangHF.hlsli"
#include "lightingHF.hlsli"

PUSHCONSTANT(push, HairStrandPush);

float4 main(HairStrandVSOut input) : SV_Target
{
	ShaderCamera camera = GetCameraIndexed(push.camera_index);

	// Stochastic alpha-hashed coverage for sub-pixel anti-aliasing
	float coverage = input.strandData.z;
	if (coverage < 0.99f)
	{
		float noise = dither(input.pos.xy + float2(GetFrame().frame_count * 1.618f, 0.0f));
		clip(coverage - noise);
	}

	float3 P = input.worldPos;
	float3 V = normalize(camera.position.xyz - P);
	float3 T = normalize(input.tangent);
	float u = input.strandData.x;
	float h = clamp(input.strandData.y, -0.995f, 0.995f);
	float strandSeed = input.strandData.w;

	ShaderHairStrandMaterial mat = xHairMaterial;
	// Subtle root darkening
	float rootAO = lerp(0.55f, 1.0f, saturate(u * 3.0f));

	float3 totalRadiance = float3(0, 0, 0);

	// Evaluate direct scene lights (directional, point, spot) with Chiang 2016 BSDF
	uint lightCount = lights().item_count();
	uint lightOffset = lights().first_item();
	for (uint i = 0; i < lightCount; ++i)
	{
		ShaderEntity light = load_entity(lightOffset + i);
		uint lightType = light.GetType();

		float3 L = float3(0, 1, 0);
		float3 lightColor = light.GetColor().rgb;
		float attenuation = 1.0f;

		if (lightType == ENTITY_TYPE_DIRECTIONALLIGHT)
		{
			L = normalize(light.GetDirection());
			if (light.IsCastingShadow())
			{
				// Simple cascade shadow lookup
				uint cascadeCount = light.GetShadowCascadeCount();
				for (uint c = 0; c < cascadeCount; ++c)
				{
					float4 shadowPos = mul(load_entitymatrix(light.GetMatrixIndex() + c), float4(P, 1.0f));
					shadowPos.xyz /= max(shadowPos.w, 1e-6f);
					float2 shadowUV = shadowPos.xy * float2(0.5f, -0.5f) + 0.5f;
					if (all(shadowUV >= 0.0f) && all(shadowUV <= 1.0f) && shadowPos.z >= 0.0f && shadowPos.z <= 1.0f)
					{
						attenuation = shadow_2D(light, shadowPos.z, shadowUV.xy, c, min16uint2(input.pos.xy)).x;
						break;
					}
				}
			}
		}
		else if (lightType == ENTITY_TYPE_POINTLIGHT || lightType == ENTITY_TYPE_SPOTLIGHT)
		{
			float3 toLight = light.position - P;
			float distSq = dot(toLight, toLight);
			float range = light.GetRange();
			if (distSq > range * range || distSq < 1e-6f)
				continue;
			float dist = sqrt(distSq);
			L = toLight / dist;
			float falloff = saturate(1.0f - pow(dist / max(range, 1e-4f), 4.0f));
			attenuation = (falloff * falloff) / max(distSq, 0.01f);
			if (lightType == ENTITY_TYPE_SPOTLIGHT)
			{
				float spotCos = dot(-L, normalize(light.GetDirection()));
				float cone = saturate((spotCos - light.GetConeAngleCos()) * light.GetAngleScale());
				attenuation *= cone * cone;
			}
		}
		else
		{
			continue;
		}

		if (attenuation <= 1e-5f)
			continue;

		float3 bsdf = HairBSDF_Eval(V, L, T, h, mat, strandSeed);
		totalRadiance += bsdf * lightColor * attenuation;
	}

	// Indirect / IBL dual-scattering approximation using cylindrical normal facing view
	float3 cylN = normalize(V - T * dot(V, T) + 1e-6f);
	float3 envUp = GetAmbient(cylN);
	float3 envDown = GetAmbient(-cylN);
	float3 approxAlbedo = HairApproximateAlbedo(mat, strandSeed);
	float3 indirectR = envUp * 0.12f;
	float3 indirectScatter = (envUp * 0.65f + envDown * 0.35f) * approxAlbedo * 0.85f;
	totalRadiance += (indirectR + indirectScatter);

	totalRadiance *= rootAO;
	totalRadiance *= lerp(mat.root_color_shift, float3(1, 1, 1), saturate(u * 2.5f));

	return float4(max(totalRadiance, 0.0f), 1.0f);
}
