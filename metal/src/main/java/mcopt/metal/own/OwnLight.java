package mcopt.metal.own;

/**
 * The time-of-day light and colour model of the shaders (-Dmcopt.shaders=true; milestone 2). One analytic table over the sun's
 * elevation (the sine of it, so the table is dense where the colour changes fastest: the horizon), blended with the rain and the
 * camera's medium. Three consumers read it:
 * <ul>
 * <li>the lightmap (mixin ShadeLightmapMixin): the sky light's colour and the block light's tint, so every vanilla pipeline (entities,
 * particles, hand) and the terrain's ambient are coloured by the time of day without any shader change;</li>
 * <li>the terrain's shaded variant (OwnShade, ShadeFrame): the sun's colour, the tint of unlit surfaces and the sky fill;</li>
 * <li>the colour response pass (OwnPost): exposure, tone curve, colour balance, lift, saturation and contrast of the finished picture.</li>
 * </ul>
 * Public techniques only: black-body-like sun colour by air mass, a bluish sky ambient, a warm block light, an ACES-style filmic curve.
 */
public final class OwnLight {
	/** -Dmcopt.shaders.light=false leaves the lightmap alone; -Dmcopt.shaders.grade=false skips the colour response pass. */
	public static final boolean LIGHTMAP = OwnShade.ON && !"false".equals(System.getProperty("mcopt.shaders.light"));
	public static final boolean GRADE = OwnShade.ON && !"false".equals(System.getProperty("mcopt.shaders.grade"));

	// columns of a row of TABLE
	private static final int EL = 0, AMB = 1, SUN = 4, SHADE = 7, EXPO = 10, CONTRAST = 11, SAT = 12, GAIN = 13, LIFT = 16, SPLIT = 19, TONE = 22, SKY = 23, N = 26;

	/**
	 * Rows by sun elevation (sin), ascending: night, deep dusk, dusk, sunset, golden hour, morning, noon.
	 * ambient rgb (the sky light's colour, the brightest component 1), sun rgb (relative; OwnShade scales it), shade tint rgb (what the sun does not
	 * reach), then the grade: exposure, contrast, saturation, gain rgb (linear), lift rgb (display), highlight split rgb, tone curve strength, and the sky disc's
	 * colour (the zenith, before the grade: chosen so that the graded zenith reads like a hazy daytime / sunset / night sky).
	 */
	private static final float[][] TABLE = {
		// el     ambient               sun                   shade tint            expo  contr sat   gain                 lift                    split              tone  sky
		{-0.3f, 0.42f, 0.62f, 1f, 0.45f, 0.60f, 1f, 0.45f, 0.6f, 1f, 0.70f, 1.0f, 1.3f, 0.43f, 1.06f, 0.78f, 0.02f, 0.04f, 0.06f, 0, 0, 0, 0.75f, 0.165f, 0.159f, 0.209f},
		{-0.1f, 0.62f, 0.62f, 1f, 0.8f, 0.6f, 0.45f, 0.7f, 0.7f, 1f, 0.85f, 0.94f, 1f, 0.95f, 1.12f, 0.85f, 0.005f, 0.035f, 0.06f, 0, 0, 0, 0.75f, 0.1f, 0.11f, 0.2f},
		{-0.02f, 0.85f, 0.65f, 0.85f, 1f, 0.42f, 0.18f, 0.9f, 0.72f, 0.82f, 1f, 0.95f, 1.12f, 1.25f, 1f, 0.75f, 0.02f, 0.01f, 0.04f, 0, 0, 0, 0.75f, 0.35f, 0.28f, 0.38f},
		{0.05f, 0.96f, 0.76f, 0.62f, 1f, 0.5f, 0.2f, 0.93f, 0.82f, 0.74f, 1.15f, 0.92f, 1.1f, 1.3f, 1.06f, 0.68f, 0.02f, 0.008f, 0.025f, 0, 0, 0, 0.75f, 0.5f, 0.38f, 0.36f},
		{0.14f, 0.97f, 0.8f, 0.66f, 1f, 0.56f, 0.26f, 0.95f, 0.84f, 0.76f, 1.08f, 0.92f, 1.1f, 1.32f, 1.06f, 0.66f, 0.02f, 0.012f, 0.025f, 0, 0, 0, 0.75f, 0.46f, 0.4f, 0.41f},
		{0.3f, 1f, 0.85f, 0.68f, 1f, 0.75f, 0.52f, 0.96f, 0.88f, 0.82f, 0.92f, 0.98f, 1f, 1.28f, 0.98f, 0.68f, 0.015f, 0.015f, 0.03f, 0, 0, 0, 0.75f, 0.45f, 0.43f, 0.47f},
		{0.47f, 0.95f, 0.93f, 1f, 1f, 0.88f, 0.74f, 0.92f, 0.94f, 1f, 0.8f, 1.0f, 0.85f, 1.2f, 0.92f, 0.72f, 0.022f, 0.026f, 0.04f, 0, 0, 0, 0.75f, 0.217f, 0.398f, 0.814f},
		{0.85f, 0.86f, 0.93f, 1f, 1f, 0.95f, 0.88f, 0.88f, 0.94f, 1f, 0.82f, 0.96f, 1.05f, 0.78f, 1.04f, 1.1f, 0.012f, 0.018f, 0.04f, 0, 0, 0, 0.75f, 0.386f, 0.5f, 0.754f},
	};

	/** The horizon / fog colour's tint per row (multiplies vanilla's fog colour, which is also the sky disc's horizon): warm at low sun, lighter at noon, bluer at night. */
	private static final float[][] FOGT = {
		{4.5f, 4.3f, 3.1f},  // night: the horizon haze is lifted (vanilla's night fog colour is near black)
		{1.0f, 1.0f, 1.0f},
		{1.0f, 1.0f, 1.0f},
		{1.0f, 1.0f, 1.0f},
		{1.0f, 1.0f, 1.0f},
		{1.12f, 0.96f, 0.82f},
		{1.2f, 0.94f, 0.72f},
		{1.1f, 1.08f, 1.04f},
	};

	static {
		for (float[] row : TABLE) if (row.length != N) throw new IllegalStateException("OwnLight.TABLE row of " + row.length);
	}

	// state of this frame (render thread)
	public static float el = 1.0f;           // sine of the sun's elevation
	public static float rain;                // 0..1
	public static boolean water;             // the camera is in water
	static final float[] ambient = {1, 1, 1}, sun = {1, 1, 1}, shade = {1, 1, 1}, grade = new float[N];
	/** The moon's illumination at night (sine of its elevation, 0 below the horizon): the night's light is bluish, not black. */
	static float moonEl;

	private OwnLight() {
	}

	/** The model for this frame: the sun's angle (radians, as SkyRenderState.sunAngle), the rain level 0..1, the camera in water. */
	public static void update(float sunAngle, float rainLevel, boolean inWater) {
		el = (float) Math.cos(sunAngle);
		rain = rainLevel;
		water = inWater;
		float x = Math.max(TABLE[0][EL], Math.min(TABLE[TABLE.length - 1][EL], el));
		int i = 0;
		while (i < TABLE.length - 2 && x > TABLE[i + 1][EL]) i++;
		float[] a = TABLE[i], b = TABLE[i + 1];
		float t = (x - a[EL]) / (b[EL] - a[EL]);
		for (int k = 1; k < N; k++) grade[k] = a[k] + (b[k] - a[k]) * t;
		for (int k = 0; k < 3; k++) fogTint[k] = 1 + (FOGT[i][k] + (FOGT[i + 1][k] - FOGT[i][k]) * t - 1) * (inWater ? 0 : 1 - Math.min(1, rainLevel));
		System.arraycopy(grade, AMB, ambient, 0, 3);
		System.arraycopy(grade, SUN, sun, 0, 3);
		System.arraycopy(grade, SHADE, shade, 0, 3);
		// rain: a grey, flat, overcast light
		if (rainLevel > 0) {
			float r = Math.min(1, rainLevel);
			for (int k = 0; k < 3; k++) {
				ambient[k] += (0.88f + 0.12f * ambient[k] - ambient[k]) * r * 0.6f;
				shade[k] += (1.0f - shade[k]) * r * 0.6f;
			}
			grade[SAT] *= 1 - 0.55f * r;
			grade[CONTRAST] *= 1 - 0.1f * r;
			grade[EXPO] *= 1 - 0.1f * r;  // BSL's rain sky is darker than the table's (sky 85 vs 111 of 255); more than this darkens the rain's terrain far below BSL's
			grade[GAIN] += (0.78f - grade[GAIN]) * r * 0.8f;
			grade[GAIN + 1] += (1.22f - grade[GAIN + 1]) * r * 0.8f;
			grade[GAIN + 2] += (1.16f - grade[GAIN + 2]) * r * 0.8f;
			grade[LIFT] = Math.max(grade[LIFT], 0.04f * r);
			grade[LIFT + 1] = Math.max(grade[LIFT + 1], 0.045f * r);
			grade[LIFT + 2] = Math.max(grade[LIFT + 2], 0.06f * r);
		}
		if (inWater) {
			grade[EXPO] *= 0.68f;
			grade[CONTRAST] *= 1.08f;
			grade[SAT] *= 0.85f;
			grade[GAIN] *= 0.9f;
			grade[GAIN + 1] *= 0.95f;
			grade[GAIN + 2] *= 0.9f;
		}
	}

	/** The horizon tint of this frame (FOGT by the same interpolation; faded out in rain and under water). */
	static final float[] fogTint = {1, 1, 1};

	/** The sky disc's (zenith) colour for this frame's sun elevation, before the grade. */
	public static float[] skyColor() {
		return new float[] {grade[SKY], grade[SKY + 1], grade[SKY + 2]};
	}

	/** Vanilla's fog colour (the horizon of the sky disc and the haze of the terrain) tinted by the time of day. */
	public static org.joml.Vector3fc tintFog(org.joml.Vector3fc c) {
		return new org.joml.Vector3f(Math.min(1, c.x() * fogTint[0]), Math.min(1, c.y() * fogTint[1]), Math.min(1, c.z() * fogTint[2]));
	}

	/**
	 * The brightness multiplier of translucent blue (water) in the shaded fragment path, which also desaturates it toward a dark slate:
	 * vanilla's water texture is much brighter than the land it lies in, and under the sun's added light and the grade's lift it reads as a vivid cyan
	 * (BSL's water is a dark, grey-blue slate). Darker still at night (the moon's light is faint). Rain keeps the day figure.
	 */
	static float waterDim() {
		float night = 1.0f - Math.max(0.0f, Math.min(1.0f, (el + 0.12f) / 0.2f));
		return WATER_DAY - (WATER_DAY - 0.4f) * night;
	}
	/** The grade's red / blue gain ratio: the water's slate tint is divided by it, so the graded water keeps its grey-blue by day and by sunset. */
	static float waterTintK() {
		return Math.max(0.5f, Math.min(2.5f, grade[GAIN] / Math.max(0.05f, grade[GAIN + 2])));
	}
	private static final float WATER_DAY = Float.parseFloat(System.getProperty("mcopt.shaders.waterDay", "0.62"));

	/** The sky light colour the lightmap gets (relative, the brightest component 1). */
	public static float[] ambient() {
		return ambient;
	}

	/** Bloom (-Dmcopt.shaders.bloom=0 off; 1 is the default strength) and light shafts (-Dmcopt.shaders.shafts, the same). */
	static final float BLOOM = Float.parseFloat(System.getProperty("mcopt.shaders.bloom", "1"));
	static final float SHAFTS = Float.parseFloat(System.getProperty("mcopt.shaders.shafts", "1.3"));
	static final float BLOOM_THR = Float.parseFloat(System.getProperty("mcopt.shaders.bloom.threshold", "0.84"));

	/**
	 * Grade polish (-Dmcopt.shaders.grade.log / .hue / .gain / .purkinje: strengths, 0 off, 1 default; -Dmcopt.shaders.bloomyfog: strength of the glow in rain and under water).
	 * Each is also an OwnProbe key (shaders.grade.log, shaders.grade.hue, shaders.grade.gain, shaders.grade.purkinje, shaders.bloomyfog: booleans, on = the configured strength).
	 */
	static final float GR_LOG = Float.parseFloat(System.getProperty("mcopt.shaders.grade.log", "1"));
	static final float GR_HUE = Float.parseFloat(System.getProperty("mcopt.shaders.grade.hue", "1"));
	static final float GR_GAIN = Float.parseFloat(System.getProperty("mcopt.shaders.grade.gain", "1"));
	static final float GR_PK = Float.parseFloat(System.getProperty("mcopt.shaders.grade.purkinje", "1"));
	static final float FOG = Float.parseFloat(System.getProperty("mcopt.shaders.bloomyfog", "1"));
	/** The share of the table's contrast done in log space about 0.18 (the rest stays the display-space matrix). */
	static final float LOG_SHARE = Float.parseFloat(System.getProperty("mcopt.shaders.grade.logShare", "0.35"));
	/** What is left of the table's contrast in the display-space matrix while the log contrast is on (BSL has less overall contrast than the table gave; the log curve is the stronger one in the darks). */
	static final float DISP_SHARE = Float.parseFloat(System.getProperty("mcopt.shaders.grade.dispShare", "0.45"));
	static final float HUE_ORANGE = Float.parseFloat(System.getProperty("mcopt.shaders.grade.orange", "0.12"));
	static final float HUE_TEAL = Float.parseFloat(System.getProperty("mcopt.shaders.grade.teal", "0.18"));
	/** True when any of the bloomy fog's inputs can matter: PostFrame.gr1.y > 0 turns the unthresholded chain on (OwnPost). */
	static boolean fogActive() {
		return FOG > 0 && (rain > 0.02f || water) && OwnProbe.bool("shaders.bloomyfog", true);
	}

	/** Writes the PostFrame (terrain.metal) of this frame: 10 float4 (160 bytes), then the grade track's float4s at byte 224. */
	static void postFrame(java.nio.FloatBuffer b, boolean nop, int w, int h, boolean bloomOn) {
		float e = grade[EXPO], sat = grade[SAT], ct = grade[CONTRAST];
		b.put(grade[GAIN] * e).put(grade[GAIN + 1] * e).put(grade[GAIN + 2] * e).put(grade[TONE]);
		b.put(1 - grade[LIFT]).put(1 - grade[LIFT + 1]).put(1 - grade[LIFT + 2]).put(1.0f);
		b.put(grade[LIFT]).put(grade[LIFT + 1]).put(grade[LIFT + 2]).put(nop ? 1f : 0f);
		// saturation about the luma, then contrast about mid grey, as one matrix: ct * (sat * g + (1 - sat) * luma(g)) + 0.5 * (1 - ct)
		float lc = 1.0f;
		if (GR_LOG > 0 && OwnProbe.bool("shaders.grade.log", true)) {
			lc = 1 + (ct - 1) * LOG_SHARE * GR_LOG;
			ct = 1 + (ct - 1) * DISP_SHARE;
		}
		float[] luma = {0.2126f, 0.7152f, 0.0722f};
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) b.put(ct * ((i == j ? sat : 0) + (1 - sat) * luma[j]));
			b.put(0.5f * (1 - ct));
		}
		// bloom: a little more in the dark of dusk and night (the glow of the sun and the moon), none of the shafts in rain or under water
		float lowSun = 1.0f - Math.max(0f, Math.min(1f, (el - 0.05f) / 0.55f));  // 1 at the horizon, 0 high
		float up = Math.max(0f, Math.min(1f, (el + 0.05f) / 0.12f));             // 0 with the sun well below the horizon
		float shaft = SHAFTS * up * (0.35f + 0.65f * lowSun) * (1 - Math.min(1, rain) * 0.9f) * (water ? (OwnWater.UW_SHAFTS && OwnProbe.bool("shaders.water.uwshafts", true) ? OwnWater.UW_SHAFT * up : 0f) : 1f);
		float[] sc = OwnSky.sunTrans;
		float m = Math.max(1e-3f, Math.max(sc[0], Math.max(sc[1], sc[2])));
		b.put(BLOOM * (1.0f - 0.45f * Math.min(1, rain))).put(shaft).put(BLOOM_THR).put(bloomOn ? 1f : 0f);
		b.put(sc[0] / m).put(sc[1] / m).put(sc[2] / m).put(1.0f);
		b.put(1.0f / w).put(1.0f / h).put((float) w / h).put(OwnProbe.bool("shaders.bloom.chain", true) ? 0f : 1f);
		b.put(OwnSky.sunScreen[0]).put(OwnSky.sunScreen[1]).put(OwnSky.sunScreen[2]).put(Float.parseFloat(System.getProperty("mcopt.shaders.bloom.debug", "0")));
		b.put(0f).put(0f).put(0f).put(0f);  // ws (set natively)
		// track water (PostFrame.uw): under water, the shaft noise's time, the HDR glare's gain into the bloom
		b.put(water && OwnWater.UW_SHAFTS ? 1f : 0f).put((float) ((System.nanoTime() * 1e-9) % 4096.0)).put(OwnWater.hdrOn() ? OwnWater.GLARE_BLOOM : 0f).put(0f);
		// bytes 192 .. 223 belong to the volumetric track (set natively, untouched here); 224: gr0, gr1
		b.position(56);
		boolean hue = GR_HUE > 0 && OwnProbe.bool("shaders.grade.hue", true);
		boolean gn = GR_GAIN > 0 && OwnProbe.bool("shaders.grade.gain", true);
		b.put(lc).put(hue ? HUE_ORANGE * GR_HUE : 0f).put(hue ? HUE_TEAL * GR_HUE : 0f).put(gn ? 0.05f * GR_GAIN : 0f);
		// Purkinje: rods take over when the sun is well below the horizon (Photon: none in daylight; smoothstep(-0.12, -0.06) of the sun's height)
		float nt = Math.max(0f, Math.min(1f, (el + 0.02f) / -0.10f));
		nt = nt * nt * (3 - 2 * nt);
		float pk = GR_PK > 0 && OwnProbe.bool("shaders.grade.purkinje", true) ? 0.05f * GR_PK * nt : 0f;
		// bloomy fog: rain a thin haze with distance, under water a denser one (the weight goes by the square root there)
		float fogK = 0f, fogD = 0.025f;
		if (fogActive()) {
			if (water) { fogK = 0.5f * FOG; fogD = 0.08f; }
			else fogK = 0.45f * FOG * Math.min(1f, rain);
		}
		b.put(pk).put(fogK).put(fogD).put(water ? 1f : 0f);
		// gr2.x: ask for the grade table (mcs_post bakes post_fs's colour maths into 33^3 and clears this if it cannot); -Dmcopt.shaders.grade.lut=false, probe shaders.grade.lut
		b.put(GRADE_LUT && OwnProbe.bool("shaders.grade.lut", true) ? 1f : 0f).put(0f).put(0f).put(0f);
	}

	/** post_fs's colour maths baked into a 3D table once per frame (one sample per pixel instead of the logs and exps). */
	static final boolean GRADE_LUT = !"false".equals(System.getProperty("mcopt.shaders.grade.lut"));
}
