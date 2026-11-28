#ifndef WI_HAIRBSDF_CHIANG_HF
#define WI_HAIRBSDF_CHIANG_HF

// Physically-based Hair BSDF based on:
// "A Practical and Controllable Hair and Fur Model for Production Path Tracing"
// Chiang et al. (2016), following pbrt-v4's HairBSDF implementation (Apache 2.0).

#include "ShaderInterop_HairStrand.h"

static const float HAIR_PI = 3.14159265359f;
static const float HAIR_TWO_PI = 6.28318530718f;
static const float HAIR_INV_PI = 0.31830988618f;
static const float HAIR_SQRT_PI_OVER_8 = 0.626657069f;
static const uint HAIR_P_MAX = 3;

inline float HairSqr(float x) { return x * x; }
inline float HairSafeSqrt(float x) { return sqrt(max(0.0f, x)); }

// Modified Bessel function of the first kind I0(x)
inline float HairI0(float x)
{
	float val = 0.0f;
	float x2i = 1.0f;
	float ifact = 1.0f;
	float i4 = 1.0f;
	// 10 terms is accurate to float32 precision for |x| <= 12
	[unroll]
	for (int i = 0; i < 10; ++i)
	{
		if (i > 1) ifact *= (float)i;
		val += x2i / (HairSqr(ifact) * i4);
		x2i *= x * x;
		i4 *= 4.0f;
	}
	return val;
}

inline float HairLogI0(float x)
{
	if (x > 12.0f)
		return x + 0.5f * (-log(HAIR_TWO_PI) + log(1.0f / x) + 1.0f / (8.0f * x));
	return log(max(HairI0(x), 1e-30f));
}

// Dielectric Fresnel reflectance for unpolarized light
inline float HairFrDielectric(float cosThetaI, float eta)
{
	cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
	float etaI = 1.0f;
	float etaT = eta;
	if (cosThetaI < 0.0f)
	{
		float tmp = etaI; etaI = etaT; etaT = tmp;
		cosThetaI = -cosThetaI;
	}
	float sinThetaI = HairSafeSqrt(1.0f - cosThetaI * cosThetaI);
	float sinThetaT = (etaI / etaT) * sinThetaI;
	if (sinThetaT >= 1.0f)
		return 1.0f;
	float cosThetaT = HairSafeSqrt(1.0f - sinThetaT * sinThetaT);
	float rParl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
	              max((etaT * cosThetaI) + (etaI * cosThetaT), 1e-6f);
	float rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
	              max((etaI * cosThetaI) + (etaT * cosThetaT), 1e-6f);
	return 0.5f * (rParl * rParl + rPerp * rPerp);
}

// Compute absorption coefficient sigma_a from melanin concentration & redness (Chiang 2016)
inline float3 HairSigmaAFromConcentration(float ce, float cp)
{
	const float3 eumelaninSigmaA = float3(0.419f, 0.697f, 1.37f);
	const float3 pheomelaninSigmaA = float3(0.187f, 0.4f, 1.05f);
	return ce * eumelaninSigmaA + cp * pheomelaninSigmaA;
}

// Compute absorption coefficient sigma_a from direct reflectance color & azimuthal roughness
inline float3 HairSigmaAFromReflectance(float3 c, float beta_n)
{
	float denom = 5.969f - 0.215f * beta_n + 2.532f * HairSqr(beta_n)
	            - 10.73f * pow(beta_n, 3.0f) + 5.574f * pow(beta_n, 4.0f)
	            + 0.245f * pow(beta_n, 5.0f);
	float3 logC = log(max(c, float3(1e-4f, 1e-4f, 1e-4f)));
	return HairSqr(logC.x / denom) * float3(1, 0, 0)
	     + HairSqr(logC.y / denom) * float3(0, 1, 0)
	     + HairSqr(logC.z / denom) * float3(0, 0, 1);
}

inline float3 HairComputeSigmaA(ShaderHairStrandMaterial mat, float strandSeed)
{
	float mel = clamp(mat.melanin + (strandSeed - 0.5f) * 2.0f * mat.melanin_variation, 0.0f, 0.999f);
	float melaninConc = -log(max(1.0f - mel, 1e-4f));
	float eumelanin = melaninConc * (1.0f - clamp(mat.melanin_redness, 0.0f, 1.0f));
	float pheomelanin = melaninConc * clamp(mat.melanin_redness, 0.0f, 1.0f);
	float3 sigma_melanin = HairSigmaAFromConcentration(eumelanin, pheomelanin);
	if (mat.tint_strength > 1e-4f)
	{
		float3 sigma_tint = HairSigmaAFromReflectance(clamp(mat.tint, 0.005f, 0.995f), clamp(mat.beta_n, 0.05f, 1.0f));
		return lerp(sigma_melanin, sigma_tint, clamp(mat.tint_strength, 0.0f, 1.0f));
	}
	return sigma_melanin;
}

// Approximate diffuse/albedo reflectance from Chiang material (for denoiser albedo & IBL fallback)
inline float3 HairApproximateAlbedo(ShaderHairStrandMaterial mat, float strandSeed)
{
	float3 sigma_a = HairComputeSigmaA(mat, strandSeed);
	float beta_n = clamp(mat.beta_n, 0.05f, 1.0f);
	float denom = 5.969f - 0.215f * beta_n + 2.532f * HairSqr(beta_n)
	            - 10.73f * pow(beta_n, 3.0f) + 5.574f * pow(beta_n, 4.0f)
	            + 0.245f * pow(beta_n, 5.0f);
	return clamp(exp(-sqrt(max(sigma_a, 0.0f)) * denom), 0.01f, 0.99f);
}

// Longitudinal scattering function Mp (Chiang 2016 Eq. 3)
inline float HairMp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
{
	float a = cosThetaI * cosThetaO / max(v, 1e-5f);
	float b = sinThetaI * sinThetaO / max(v, 1e-5f);
	if (v <= 0.1f)
	{
		return exp(HairLogI0(a) - b - 1.0f / max(v, 1e-5f) + 0.6931f + log(1.0f / (2.0f * max(v, 1e-5f))));
	}
	float sinhInvV = sinh(1.0f / v);
	return (exp(-b) * HairI0(a)) / max(2.0f * v * sinhInvV, 1e-30f);
}

// Logistic distribution & trimmed logistic for azimuthal scattering Np
inline float HairLogistic(float x, float s)
{
	x = abs(x);
	float e = exp(-x / max(s, 1e-5f));
	return e / max(s * HairSqr(1.0f + e), 1e-30f);
}

inline float HairLogisticCDF(float x, float s)
{
	return 1.0f / (1.0f + exp(-x / max(s, 1e-5f)));
}

inline float HairTrimmedLogistic(float x, float s, float a, float b)
{
	float denom = HairLogisticCDF(b, s) - HairLogisticCDF(a, s);
	return HairLogistic(x, s) / max(denom, 1e-6f);
}

inline float HairSampleTrimmedLogistic(float u, float s, float a, float b)
{
	float k = HairLogisticCDF(b, s) - HairLogisticCDF(a, s);
	float x = -s * log(max(1.0f / max(u * k + HairLogisticCDF(a, s), 1e-6f) - 1.0f, 1e-6f));
	return clamp(x, a, b);
}

inline float HairPhi(int p, float gammaO, float gammaT)
{
	return 2.0f * p * gammaT - 2.0f * gammaO + p * HAIR_PI;
}

inline float HairNp(float phi, int p, float s, float gammaO, float gammaT)
{
	float dphi = phi - HairPhi(p, gammaO, gammaT);
	// Remap dphi to [-pi, pi]
	while (dphi > HAIR_PI) dphi -= HAIR_TWO_PI;
	while (dphi < -HAIR_PI) dphi += HAIR_TWO_PI;
	return HairTrimmedLogistic(dphi, s, -HAIR_PI, HAIR_PI);
}

// Compute attenuation Ap for lobes p = 0 (R), 1 (TT), 2 (TRT), 3 (TRRT+)
inline void HairAp(float cosThetaO, float eta, float h, float3 T, out float3 ap[4])
{
	float cosGammaO = HairSafeSqrt(1.0f - h * h);
	float cosTheta = cosThetaO * cosGammaO;
	float f = HairFrDielectric(cosTheta, eta);

	ap[0] = float3(f, f, f);
	ap[1] = HairSqr(1.0f - f) * T;
	ap[2] = ap[1] * T * f;
	float3 denom = max(1.0f - T * f, float3(1e-4f, 1e-4f, 1e-4f));
	ap[3] = ap[2] * f * T / denom;
}

// Build local hair shading frame where X = strand tangent T, Y = N, Z = B
inline void HairBuildFrame(float3 tangent, out float3 X, out float3 Y, out float3 Z)
{
	X = normalize(tangent);
	float3 up = abs(X.y) < 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
	Z = normalize(cross(X, up));
	Y = cross(Z, X);
}

inline float3 HairWorldToLocal(float3 v, float3 X, float3 Y, float3 Z)
{
	return float3(dot(v, X), dot(v, Y), dot(v, Z));
}

inline float3 HairLocalToWorld(float3 v, float3 X, float3 Y, float3 Z)
{
	return v.x * X + v.y * Y + v.z * Z;
}

inline void HairComputeVariances(float beta_m, float beta_n, out float v[4], out float s)
{
	float bm = clamp(beta_m, 0.05f, 1.0f);
	float bn = clamp(beta_n, 0.05f, 1.0f);
	float v0 = HairSqr(0.726f * bm + 0.812f * bm * bm + 3.7f * pow(bm, 20.0f));
	v[0] = v0;
	v[1] = 0.25f * v0;
	v[2] = 4.0f * v0;
	v[3] = 4.0f * v0;
	s = HAIR_SQRT_PI_OVER_8 * (0.265f * bn + 1.194f * bn * bn + 5.372f * pow(bn, 22.0f));
}

// Evaluate pbrt-v4 Chiang 2016 Hair BSDF f(wo, wi) (returns radiance/irradiance * 1/cosThetaI absorbed into BSDF)
// Note: wo and wi are in world space pointing AWAY from the hair surface.
inline float3 HairBSDF_Eval(
	float3 woWorld,
	float3 wiWorld,
	float3 tangentWorld,
	float h,
	ShaderHairStrandMaterial mat,
	float strandSeed)
{
	float3 X, Y, Z;
	HairBuildFrame(tangentWorld, X, Y, Z);
	float3 wo = HairWorldToLocal(normalize(woWorld), X, Y, Z);
	float3 wi = HairWorldToLocal(normalize(wiWorld), X, Y, Z);

	float sinThetaO = clamp(wo.x, -0.9999f, 0.9999f);
	float cosThetaO = HairSafeSqrt(1.0f - sinThetaO * sinThetaO);
	float phiO = atan2(wo.z, wo.y);

	float sinThetaI = clamp(wi.x, -0.9999f, 0.9999f);
	float cosThetaI = HairSafeSqrt(1.0f - sinThetaI * sinThetaI);
	float phiI = atan2(wi.z, wi.y);

	float eta = max(mat.ior, 1.01f);
	float sinThetaT = sinThetaO / eta;
	float cosThetaT = HairSafeSqrt(1.0f - sinThetaT * sinThetaT);

	float etap = HairSafeSqrt(eta * eta - HairSqr(sinThetaO)) / max(cosThetaO, 1e-4f);
	float sinGammaT = clamp(h / max(etap, 1e-4f), -0.9999f, 0.9999f);
	float cosGammaT = HairSafeSqrt(1.0f - sinGammaT * sinGammaT);
	float gammaT = asin(sinGammaT);
	float gammaO = asin(clamp(h, -0.9999f, 0.9999f));

	float3 sigma_a = HairComputeSigmaA(mat, strandSeed);
	float3 T = exp(-sigma_a * (2.0f * cosGammaT / max(cosThetaT, 1e-4f)));

	float3 ap[4];
	HairAp(cosThetaO, eta, h, T, ap);

	float v[4];
	float s;
	float roughVar = (strandSeed - 0.5f) * 2.0f * mat.roughness_variation;
	HairComputeVariances(mat.beta_m + roughVar, mat.beta_n + roughVar, v, s);

	float alpha = mat.cuticle_tilt_rad;
	float sin2kAlpha[3];
	float cos2kAlpha[3];
	sin2kAlpha[0] = sin(alpha);
	cos2kAlpha[0] = HairSafeSqrt(1.0f - HairSqr(sin2kAlpha[0]));
	sin2kAlpha[1] = 2.0f * cos2kAlpha[0] * sin2kAlpha[0];
	cos2kAlpha[1] = HairSqr(cos2kAlpha[0]) - HairSqr(sin2kAlpha[0]);
	sin2kAlpha[2] = 2.0f * cos2kAlpha[1] * sin2kAlpha[1];
	cos2kAlpha[2] = HairSqr(cos2kAlpha[1]) - HairSqr(sin2kAlpha[1]);

	float phi = phiI - phiO;
	float3 fsum = float3(0, 0, 0);

	// p = 0 (R)
	{
		float sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
		float cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
		cosThetaOp = abs(cosThetaOp);
		fsum += HairMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[0]) * ap[0] * HairNp(phi, 0, s, gammaO, gammaT);
	}
	// p = 1 (TT)
	{
		float sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
		float cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
		cosThetaOp = abs(cosThetaOp);
		fsum += HairMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[1]) * ap[1] * HairNp(phi, 1, s, gammaO, gammaT);
	}
	// p = 2 (TRT)
	{
		float sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
		float cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
		cosThetaOp = abs(cosThetaOp);
		fsum += HairMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[2]) * ap[2] * HairNp(phi, 2, s, gammaO, gammaT);
	}
	// p = 3 (TRRT+)
	{
		fsum += HairMp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[3]) * ap[3] * (1.0f / HAIR_TWO_PI);
	}

	// Note: In pbrt-v4 HairBSDF::f divides by absCosTheta(wi) because the transport integral
	// multiplies by absCosTheta(wi). For cylindrical strands we return fsum directly so the
	// caller does NOT multiply by dot(N, L) — it multiplies by solid-angle radiance directly.
	return max(fsum, 0.0f);
}

// Evaluate PDF of HairBSDF_Sample for MIS
inline float HairBSDF_PDF(
	float3 woWorld,
	float3 wiWorld,
	float3 tangentWorld,
	float h,
	ShaderHairStrandMaterial mat,
	float strandSeed)
{
	float3 X, Y, Z;
	HairBuildFrame(tangentWorld, X, Y, Z);
	float3 wo = HairWorldToLocal(normalize(woWorld), X, Y, Z);
	float3 wi = HairWorldToLocal(normalize(wiWorld), X, Y, Z);

	float sinThetaO = clamp(wo.x, -0.9999f, 0.9999f);
	float cosThetaO = HairSafeSqrt(1.0f - sinThetaO * sinThetaO);
	float phiO = atan2(wo.z, wo.y);

	float sinThetaI = clamp(wi.x, -0.9999f, 0.9999f);
	float cosThetaI = HairSafeSqrt(1.0f - sinThetaI * sinThetaI);
	float phiI = atan2(wi.z, wi.y);

	float eta = max(mat.ior, 1.01f);
	float sinThetaT = sinThetaO / eta;
	float cosThetaT = HairSafeSqrt(1.0f - sinThetaT * sinThetaT);
	float etap = HairSafeSqrt(eta * eta - HairSqr(sinThetaO)) / max(cosThetaO, 1e-4f);
	float sinGammaT = clamp(h / max(etap, 1e-4f), -0.9999f, 0.9999f);
	float cosGammaT = HairSafeSqrt(1.0f - sinGammaT * sinGammaT);
	float gammaT = asin(sinGammaT);
	float gammaO = asin(clamp(h, -0.9999f, 0.9999f));

	float3 sigma_a = HairComputeSigmaA(mat, strandSeed);
	float3 T = exp(-sigma_a * (2.0f * cosGammaT / max(cosThetaT, 1e-4f)));
	float3 ap[4];
	HairAp(cosThetaO, eta, h, T, ap);

	float apPdf[4];
	float sumY = 0.0f;
	[unroll]
	for (int i = 0; i < 4; ++i)
	{
		apPdf[i] = dot(ap[i], float3(0.2126f, 0.7152f, 0.0722f));
		sumY += apPdf[i];
	}
	float invSumY = sumY > 1e-6f ? (1.0f / sumY) : 0.25f;
	[unroll]
	for (int i = 0; i < 4; ++i) apPdf[i] *= invSumY;

	float v[4];
	float s;
	float roughVar = (strandSeed - 0.5f) * 2.0f * mat.roughness_variation;
	HairComputeVariances(mat.beta_m + roughVar, mat.beta_n + roughVar, v, s);

	float alpha = mat.cuticle_tilt_rad;
	float sin2kAlpha[3];
	float cos2kAlpha[3];
	sin2kAlpha[0] = sin(alpha);
	cos2kAlpha[0] = HairSafeSqrt(1.0f - HairSqr(sin2kAlpha[0]));
	sin2kAlpha[1] = 2.0f * cos2kAlpha[0] * sin2kAlpha[0];
	cos2kAlpha[1] = HairSqr(cos2kAlpha[0]) - HairSqr(sin2kAlpha[0]);
	sin2kAlpha[2] = 2.0f * cos2kAlpha[1] * sin2kAlpha[1];
	cos2kAlpha[2] = HairSqr(cos2kAlpha[1]) - HairSqr(sin2kAlpha[1]);

	float phi = phiI - phiO;
	float pdf = 0.0f;

	{
		float sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
		float cosThetaOp = abs(cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1]);
		pdf += HairMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[0]) * apPdf[0] * HairNp(phi, 0, s, gammaO, gammaT);
	}
	{
		float sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
		float cosThetaOp = abs(cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0]);
		pdf += HairMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[1]) * apPdf[1] * HairNp(phi, 1, s, gammaO, gammaT);
	}
	{
		float sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
		float cosThetaOp = abs(cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2]);
		pdf += HairMp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[2]) * apPdf[2] * HairNp(phi, 2, s, gammaO, gammaT);
	}
	pdf += HairMp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[3]) * apPdf[3] * (1.0f / HAIR_TWO_PI);

	return max(pdf, 1e-6f);
}

// Importance sample Chiang 2016 Hair BSDF (pbrt-v4 HairBSDF::Sample_f)
// Returns BSDF throughput weight f / pdf, and outputs sampled wiWorld & pdf.
inline float3 HairBSDF_Sample(
	float3 woWorld,
	float3 tangentWorld,
	float h,
	ShaderHairStrandMaterial mat,
	float strandSeed,
	float4 uRand, // 4 uniform [0, 1) random numbers
	out float3 wiWorld,
	out float pdf)
{
	float3 X, Y, Z;
	HairBuildFrame(tangentWorld, X, Y, Z);
	float3 wo = HairWorldToLocal(normalize(woWorld), X, Y, Z);

	float sinThetaO = clamp(wo.x, -0.9999f, 0.9999f);
	float cosThetaO = HairSafeSqrt(1.0f - sinThetaO * sinThetaO);
	float phiO = atan2(wo.z, wo.y);

	float eta = max(mat.ior, 1.01f);
	float sinThetaT = sinThetaO / eta;
	float cosThetaT = HairSafeSqrt(1.0f - sinThetaT * sinThetaT);
	float etap = HairSafeSqrt(eta * eta - HairSqr(sinThetaO)) / max(cosThetaO, 1e-4f);
	float sinGammaT = clamp(h / max(etap, 1e-4f), -0.9999f, 0.9999f);
	float cosGammaT = HairSafeSqrt(1.0f - sinGammaT * sinGammaT);
	float gammaT = asin(sinGammaT);
	float gammaO = asin(clamp(h, -0.9999f, 0.9999f));

	float3 sigma_a = HairComputeSigmaA(mat, strandSeed);
	float3 T = exp(-sigma_a * (2.0f * cosGammaT / max(cosThetaT, 1e-4f)));
	float3 ap[4];
	HairAp(cosThetaO, eta, h, T, ap);

	float apPdf[4];
	float sumY = 0.0f;
	[unroll]
	for (int i = 0; i < 4; ++i)
	{
		apPdf[i] = dot(ap[i], float3(0.2126f, 0.7152f, 0.0722f));
		sumY += apPdf[i];
	}
	float invSumY = sumY > 1e-6f ? (1.0f / sumY) : 0.25f;
	[unroll]
	for (int i = 0; i < 4; ++i) apPdf[i] *= invSumY;

	// Choose lobe p in {0, 1, 2, 3}
	int p = 3;
	float uLobe = uRand.x;
	if (uLobe < apPdf[0]) p = 0;
	else if (uLobe < apPdf[0] + apPdf[1]) p = 1;
	else if (uLobe < apPdf[0] + apPdf[1] + apPdf[2]) p = 2;

	float v[4];
	float s;
	float roughVar = (strandSeed - 0.5f) * 2.0f * mat.roughness_variation;
	HairComputeVariances(mat.beta_m + roughVar, mat.beta_n + roughVar, v, s);

	float alpha = mat.cuticle_tilt_rad;
	float sin2kAlpha[3];
	float cos2kAlpha[3];
	sin2kAlpha[0] = sin(alpha);
	cos2kAlpha[0] = HairSafeSqrt(1.0f - HairSqr(sin2kAlpha[0]));
	sin2kAlpha[1] = 2.0f * cos2kAlpha[0] * sin2kAlpha[0];
	cos2kAlpha[1] = HairSqr(cos2kAlpha[0]) - HairSqr(sin2kAlpha[0]);
	sin2kAlpha[2] = 2.0f * cos2kAlpha[1] * sin2kAlpha[1];
	cos2kAlpha[2] = HairSqr(cos2kAlpha[1]) - HairSqr(sin2kAlpha[1]);

	// Rotate sinThetaO/cosThetaO by cuticle tilt for chosen lobe
	float sinThetaOp = sinThetaO;
	float cosThetaOp = cosThetaO;
	if (p == 0)
	{
		sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
		cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
	}
	else if (p == 1)
	{
		sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
		cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
	}
	else if (p == 2)
	{
		sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
		cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
	}
	cosThetaOp = abs(cosThetaOp);

	// Sample longitudinal Mp
	float u1 = max(uRand.y, 1e-5f);
	float u2 = uRand.z;
	float vp = v[p];
	float cosTheta = 1.0f + vp * log(u1 + (1.0f - u1) * exp(-2.0f / max(vp, 1e-5f)));
	float sinTheta = HairSafeSqrt(1.0f - HairSqr(cosTheta));
	float cosPhiM = cos(HAIR_TWO_PI * u2);
	float sinThetaI = clamp(-cosTheta * sinThetaOp + sinTheta * cosPhiM * cosThetaOp, -0.9999f, 0.9999f);
	float cosThetaI = HairSafeSqrt(1.0f - HairSqr(sinThetaI));

	// Sample azimuthal Np
	float dphi;
	if (p < 3)
	{
		dphi = HairPhi(p, gammaO, gammaT) + HairSampleTrimmedLogistic(uRand.w, s, -HAIR_PI, HAIR_PI);
	}
	else
	{
		dphi = HAIR_TWO_PI * uRand.w;
	}

	float phiI = phiO + dphi;
	float3 wiLocal = float3(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI));
	wiWorld = normalize(HairLocalToWorld(wiLocal, X, Y, Z));

	float3 f = HairBSDF_Eval(woWorld, wiWorld, tangentWorld, h, mat, strandSeed);
	pdf = HairBSDF_PDF(woWorld, wiWorld, tangentWorld, h, mat, strandSeed);
	return f / max(pdf, 1e-6f);
}

#endif // WI_HAIRBSDF_CHIANG_HF
