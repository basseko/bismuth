// ---- shaders (-Dmcopt.shaders=true; milestone 5): sun shadows and the light balance of the terrain, for the vanilla pipelines ----
// Entities, block entities, dropped items and armour keep vanilla's vertex and fragment shaders (SPIRV-Cross output); ShadePatch (java) inserts this
// source after `using namespace metal;` of their fragment library and adds three things to the entry points: a varying carrying the camera-relative
// position and the sky light (vertex stage), the extra arguments below (fragment stage), and one call that re-lights the finished colour.
// The maths mirror own_fs's shaded variant (terrain.metal): the ambient (the lightmap) is partly replaced by the sun, cooler where the sun does
// not reach, plus the sun's N.L x shadow and a sky fill. The shadow lookup is the same filtered cascade lookup (the shadow map is the terrain's).
// Nothing here is built unless the shaders are on.

// the per-frame block at fragment buffer 29: ShadeFrame (terrain.metal, 320 bytes) followed by the entity extras
struct MCOFrame {
	float4x4 clip[3];     // camera-relative world -> cascade clip
	float4 info[3];       // x blocks per texel, y 1 / depth range, z uv margin, w normal offset (blocks)
	float4 sunDir;        // xyz unit vector to the light, w its strength 0..1
	float4 sunColor;      // rgb direct light colour x amplitude, w cascade count
	float4 amb;           // x ambient attenuation at full sun and sky light, y filter radius (blocks), z hemisphere ambient, w sky fill
	float4 shade;         // rgb tint of surfaces the sun does not reach
	float4 fillc;         // rgb colour of the sky fill light
	float4 ent;           // x 1: re-light (world pass, direct light present), 0: leave the colour alone; y entity strength; z shadow bias scale; w fade-out distance
	float4 ent2;          // reserved
};

constexpr sampler mcoCmp(coord::normalized, filter::linear, address::clamp_to_edge, compare_func::less_equal);

static float mcoIgn(float2 p) {
	return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

// the flat normal of a model's faceted surface from the screen-space derivatives of the camera-relative position, turned toward the camera
static float3 mcoNormal(float3 p) {
	float3 n = cross(dfdx(p), dfdy(p));
	float l = length(n);
	n = l > 1e-9 ? n / l : float3(0, 1, 0);
	if (dot(n, p) > 0.0) n = -n;
	return n;
}

static float mcoCascade(constant MCOFrame &sf, depth2d_array<float> sm, uint c, float3 p, float3 n, float ndl, float rot, thread float &edge) {
	float4 info = sf.info[c];
	float slope = min(sqrt(max(1.0 - ndl * ndl, 0.0)) / max(ndl, 0.15), 4.0);
	float4 q = sf.clip[c] * float4(p + n * (info.w * sf.ent.z * (1.0 + 0.35 * slope)), 1.0);
	float2 uv = q.xy * 0.5 + 0.5;
	float2 e = min(uv, 1.0 - uv);
	edge = min(e.x, e.y);
	if (edge <= 0.0) return 0.0;
	if (q.z >= 1.0 || q.z <= 0.0) return 1.0;
	float z = q.z - info.x * (1.25 + 0.75 * slope) * sf.ent.z * info.y;
	float radius = max(sf.amb.y * info.z / info.x, 1.2 * info.z);
	// 8 Vogel-disc taps (each a hardware 2x2 compare), the first 2 spread over it: when those agree the rest is skipped
	const int TAPS = 8, STEP = 4;
	float lit = 0.0;
	for (int i = 0; i < TAPS; i += STEP) {
		float r = sqrt((float(i) + 0.5) / float(TAPS));
		float a = float(i) * 2.399963 + rot * 6.2831853;
		lit += sm.sample_compare(mcoCmp, uv + float2(cos(a), sin(a)) * (r * radius), c, z);
	}
	const float FIRST = float((TAPS + STEP - 1) / STEP);
	if (lit < 0.001 || lit > FIRST - 0.001) return lit / FIRST;
	for (int i = 0; i < TAPS; i++) {
		if (i % STEP == 0) continue;
		float r = sqrt((float(i) + 0.5) / float(TAPS));
		float a = float(i) * 2.399963 + rot * 6.2831853;
		lit += sm.sample_compare(mcoCmp, uv + float2(cos(a), sin(a)) * (r * radius), c, z);
	}
	return lit / float(TAPS);
}

static float mcoShadow(constant MCOFrame &sf, depth2d_array<float> sm, float3 p, float3 n, float ndl, float2 pix) {
	float rot = mcoIgn(pix);
	uint count = uint(sf.sunColor.w);
	float acc = 0.0, remaining = 1.0;
	float dist = length(p);
	uint c0 = 0;
	while (c0 + 1 < count && dist > 1.35 * 0.5 * sf.info[c0].x / sf.info[c0].z) c0++;
	for (uint c = c0; c < count; c++) {
		float edge;
		float s = mcoCascade(sf, sm, c, p, n, ndl, rot, edge);
		float t = saturate(edge / 0.08);
		acc += remaining * t * s;
		remaining *= 1.0 - t;
		if (remaining < 0.002) break;
	}
	return acc + remaining;
}

// col: the finished colour (vertex colour x texture x lightmap, fogged), lm: the lightmap colour it was multiplied by, p / sky: the camera-relative
// position and the sky light 0..1 from the vertex stage, n: the flat normal (computed in uniform control flow), pix: the pixel
static float3 mcoLit(float3 col, float3 lm, float3 p, float sky, float3 n, float2 pix, constant MCOFrame &f, depth2d_array<float> sm) {
	if (f.ent.x < 0.5) return col;
	float dist = length(p);
	float fade = 1.0 - smoothstep(f.ent.w * 0.7, f.ent.w, dist);  // (the shadow cascades end; the terrain's haze takes over)
	if (fade <= 0.0) return col;
	float access = saturate((sky - 0.35) / 0.55);
	float ndl = saturate(dot(n, f.sunDir.xyz));
	float S = f.sunDir.w * access;
	float dir = S * ndl;
	float lit = dir > 0.002 ? mcoShadow(f, sm, p, n, ndl, pix) : 1.0;
	float3 cool = mix(float3(1.0), f.shade.rgb, S * (1.0 - dir * lit) * 0.7);
	// the colour under the sun is col = albedo x lm: the sun and the fill act on the albedo, the ambient cut and the cool tint on the whole
	float3 albedo = col / max(lm, float3(0.06));
	float fill = f.amb.w * S * (0.5 + 0.5 * n.y);
	float3 relit = col * cool * (1.0 - f.amb.x * S) + albedo * (f.sunColor.rgb * (dir * lit) + fill * f.fillc.rgb);
	return mix(col, relit, fade * f.ent.y);
}
