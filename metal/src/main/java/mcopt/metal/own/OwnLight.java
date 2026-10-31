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
		{-0.3f, 0.5f, 0.64f, 1f, 0.55f, 0.68f, 1f, 0.55f, 0.7f, 1f, 0.75f, 0.92f, 0.95f, 0.95f, 1.2f, 0.8f, 0, 0.05f, 0.07f, 0, 0, 0, 0.75f, 0.06f, 0.09f, 0.15f},
		{-0.1f, 0.62f, 0.62f, 1f, 0.8f, 0.6f, 0.45f, 0.7f, 0.7f, 1f, 0.85f, 0.94f, 1f, 0.95f, 1.12f, 0.85f, 0.005f, 0.035f, 0.06f, 0, 0, 0, 0.75f, 0.1f, 0.11f, 0.2f},
		{-0.02f, 0.85f, 0.65f, 0.85f, 1f, 0.42f, 0.18f, 0.9f, 0.72f, 0.82f, 1f, 0.95f, 1.12f, 1.25f, 1f, 0.75f, 0.02f, 0.01f, 0.04f, 0, 0, 0, 0.75f, 0.35f, 0.28f, 0.38f},
		{0.05f, 1f, 0.7f, 0.46f, 1f, 0.5f, 0.2f, 1f, 0.8f, 0.62f, 1.05f, 0.9f, 1.2f, 1.4f, 1.1f, 0.62f, 0.035f, 0.01f, 0.04f, 0, 0, 0, 0.75f, 0.5f, 0.38f, 0.36f},
		{0.14f, 1f, 0.72f, 0.5f, 1f, 0.56f, 0.26f, 1f, 0.82f, 0.66f, 1f, 0.9f, 1.15f, 1.4f, 1.08f, 0.62f, 0.04f, 0.02f, 0.035f, 0, 0, 0, 0.75f, 0.46f, 0.4f, 0.41f},
		{0.3f, 1f, 0.85f, 0.68f, 1f, 0.75f, 0.52f, 0.96f, 0.88f, 0.82f, 0.92f, 0.93f, 1f, 1.28f, 0.98f, 0.68f, 0.035f, 0.03f, 0.05f, 0, 0, 0, 0.75f, 0.45f, 0.43f, 0.47f},
		{0.47f, 0.95f, 0.93f, 1f, 1f, 0.88f, 0.74f, 0.92f, 0.94f, 1f, 0.8f, 0.96f, 0.78f, 1.2f, 0.92f, 0.72f, 0.03f, 0.03f, 0.05f, 0, 0, 0, 0.75f, 0.42f, 0.42f, 0.5f},
		{0.85f, 0.86f, 0.93f, 1f, 1f, 0.95f, 0.88f, 0.88f, 0.94f, 1f, 0.82f, 0.92f, 0.95f, 0.78f, 1.04f, 1.1f, 0, 0.015f, 0.06f, 0, 0, 0, 0.75f, 0.4f, 0.54f, 0.84f},
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
			grade[SAT] *= 1 - 0.5f * r;
			grade[CONTRAST] *= 1 - 0.1f * r;
			grade[EXPO] *= 1 + 0.0f * r;
			grade[GAIN] += (0.78f - grade[GAIN]) * r * 0.8f;
			grade[GAIN + 1] += (1.12f - grade[GAIN + 1]) * r * 0.8f;
			grade[GAIN + 2] += (1.18f - grade[GAIN + 2]) * r * 0.8f;
			grade[LIFT] = Math.max(grade[LIFT], 0.04f * r);
			grade[LIFT + 1] = Math.max(grade[LIFT + 1], 0.045f * r);
			grade[LIFT + 2] = Math.max(grade[LIFT + 2], 0.06f * r);
		}
		if (inWater) {
			grade[EXPO] *= 0.75f;
			grade[SAT] *= 0.72f;
			grade[GAIN] *= 0.72f;
			grade[GAIN + 1] *= 0.72f;
			grade[GAIN + 2] *= 0.40f;
		}
	}

	/** The sky disc's (zenith) colour for this frame's sun elevation, before the grade. */
	public static float[] skyColor() {
		return new float[] {grade[SKY], grade[SKY + 1], grade[SKY + 2]};
	}

	/** The sky light colour the lightmap gets (relative, the brightest component 1). */
	public static float[] ambient() {
		return ambient;
	}

	/** Writes the PostFrame (terrain.metal) of this frame: 7 float4 (112 bytes). */
	static void postFrame(java.nio.FloatBuffer b, boolean nop) {
		float e = grade[EXPO], sat = grade[SAT], ct = grade[CONTRAST];
		b.put(grade[GAIN] * e).put(grade[GAIN + 1] * e).put(grade[GAIN + 2] * e).put(grade[TONE]);
		b.put(1 - grade[LIFT]).put(1 - grade[LIFT + 1]).put(1 - grade[LIFT + 2]).put(1.0f);
		b.put(grade[LIFT]).put(grade[LIFT + 1]).put(grade[LIFT + 2]).put(nop ? 1f : 0f);
		// saturation about the luma, then contrast about mid grey, as one matrix: ct * (sat * g + (1 - sat) * luma(g)) + 0.5 * (1 - ct)
		float[] luma = {0.2126f, 0.7152f, 0.0722f};
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) b.put(ct * ((i == j ? sat : 0) + (1 - sat) * luma[j]));
			b.put(0.5f * (1 - ct));
		}
	}
}
