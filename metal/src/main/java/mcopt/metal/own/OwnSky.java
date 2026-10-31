package mcopt.metal.own;

import net.minecraft.client.CloudStatus;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.world.level.dimension.DimensionType;
import net.minecraft.world.level.material.FogType;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * The sky of the shaders (-Dmcopt.shaders=true, milestone 3): a physically based single-scattering sky (Rayleigh + Mie + ozone) instead of vanilla's flat
 * sky disc. Per frame a small look-up table (azimuth from the sun x elevation, 192 x 108 RGBA16F) is computed by a compute kernel in the pre command buffer
 * (terrain.metal sky_lut), and the dome (one full-screen triangle at the far plane, opaque, in the sky pass: terrain drawn later hides it) reads it, adds the
 * sun's disc and glow, the night sky, the overcast of rain and the clouds. The same table gives the terrain's aerial perspective its colour (the shaded
 * variant reads it at texture 25), and a CPU twin of the atmosphere ({@link OwnAtmo}) gives the sun's colour and the horizon colour (the fog colour of
 * everything else). Vanilla's sun quad, sunrise fan and sky disc are skipped while the dome draws (mixin ShadeSkyDomeMixin); the moon and stars stay vanilla's.
 */
public final class OwnSky {
	public static final boolean ON = OwnShade.ON && !"false".equals(System.getProperty("mcopt.shaders.sky"));
	static final int LW = 192, LH = 108;
	/** The table's exposure (its radiance is in sun-intensity units), the sun disc's angular radius (degrees) and the haze: its length (blocks) clear and in rain. */
	static final float EXPOSURE = Float.parseFloat(System.getProperty("mcopt.shaders.sky.exposure", "0.8"));
	static final float SUN_DEG = Float.parseFloat(System.getProperty("mcopt.shaders.sky.sunDeg", "2.8"));
	static final float HAZE_CLEAR = Float.parseFloat(System.getProperty("mcopt.shaders.haze", "700"));
	static final float HAZE_RAIN = Float.parseFloat(System.getProperty("mcopt.shaders.hazeRain", "170"));
	static final float HAZE_STRENGTH = Float.parseFloat(System.getProperty("mcopt.shaders.hazeStrength", "1.0"));
	private static final int PARAM_BYTES = 512, LUT_BYTES = 32, SH_BYTES = 112;

	/** The dome drew this frame (the shaded terrain may read the table, vanilla's sun and sunrise fan are skipped). */
	static boolean active;
	private static boolean failed, said;
	private static int ring;

	// the frame's sky state
	static float sinEl, rain;
	/** The sun's screen position (uv, v down) and how much of it counts (0 behind the camera or far off screen). */
	static final float[] sunScreen = new float[3];
	static final float[] sunDir = {0, 1, 0}, sunTrans = {1, 1, 1}, horizon = {0.5f, 0.6f, 0.8f};
	/** The old (table) horizon colour of the fog at night, remembered by the fog mixin: the night sky's horizon. */
	static final float[] nightFog = {0.05f, 0.05f, 0.07f};

	private OwnSky() {
	}

	/**
	 * The dome's flag is only reset by draw(), which does not run when vanilla skips the sky pass (lava, powder snow, blindness / darkness): so every reader
	 * also checks the live camera (air only, no sky-blocking effect) and never trusts a stale "drew last frame".
	 */
	private static boolean live() {
		if (!active) return false;
		CameraRenderState cam = Minecraft.getInstance().gameRenderer.gameRenderState().levelRenderState.cameraRenderState;
		boolean air = cam.fogType == FogType.ATMOSPHERIC || cam.fogType == FogType.NONE;
		if (!air || cam.entityRenderState.doesMobEffectBlockSky) {
			active = false;
			return false;
		}
		return true;
	}

	public static boolean isActive() {
		return live();
	}

	/** Our clouds replace vanilla's this frame. */
	public static boolean ownClouds() {
		return live() && cloudsWanted();
	}

	/**
	 * Clouds are drawn only when vanilla would draw its own: the video option Clouds is not Off and the dimension's cloud colour is visible (alpha > 0;
	 * vanilla's LevelRenderer tests exactly this), on top of our switches.
	 */
	private static boolean cloudsWanted() {
		if (!CLOUDS || !OwnProbe.bool("shaders.clouds", true)) return false;
		var gs = Minecraft.getInstance().gameRenderer.gameRenderState();
		return gs.optionsRenderState.cloudStatus != CloudStatus.OFF && ((gs.levelRenderState.cloudColor >>> 24) > 0);
	}

	/** The shaded terrain applies the haze this frame: the dome drew and the camera is in air. */
	static boolean hazeActive() {
		return live() && sinEl > -0.2f;
	}

	/** Sky colour state of this frame from the sky render state: sun direction and colour, the horizon colour, rain. */
	private static void update(SkyRenderState st) {
		double a = st.sunAngle;
		sunDir[0] = (float) -Math.sin(a);
		sunDir[1] = (float) Math.cos(a);
		sunDir[2] = 0;
		sinEl = sunDir[1];
		rain = 1.0f - Math.max(0f, Math.min(1f, st.rainBrightness));
		// (the CPU model and the table only change when the sun has moved: a few hundredths of a degree)
		if (Math.abs(sinEl - modelSin) > 0.0004f) {
			modelSin = sinEl;
			version++;
			float[] t = OwnAtmo.sunTransmittance(sinEl);
			System.arraycopy(t, 0, sunTrans, 0, 3);
			float[] r = new float[3];
			java.util.Arrays.fill(hAcc, 0f);
			double[] wgt = {0.5, 1, 1, 1, 0.5};
			for (int i = 0; i < 5; i++) {
				OwnAtmo.radiance(0.05, Math.PI * i / 4.0, sinEl, r);
				for (int k = 0; k < 3; k++) hAcc[k] += r[k] * wgt[i] * 0.25f;
			}
		}
		float[] acc = hAcc;
		if (LOG && (frameNo % 600) == 1) System.out.println("mcopt-shaders: sky horizon radiance*exposure " + acc[0] * 0.8f + " " + acc[1] * 0.8f + " " + acc[2] * 0.8f + " exp " + (EXPOSURE * (0.55f + 0.45f * smooth(0.05f, 0.5f, sinEl))) + " sinEl " + sinEl);
		exposure = EXPOSURE * (0.55f + 0.45f * smooth(0.05f, 0.5f, sinEl)) * (1.0f - 0.3f * rain);
		float mx = Math.max(sunTrans[0], Math.max(sunTrans[1], Math.max(sunTrans[2], 1e-3f)));
		for (int k = 0; k < 3; k++) warmTint[k] = sunTrans[k] / mx;
		warm = WARM * (1.0f - smooth(0.0f, 0.6f, sinEl)) * smooth(-0.12f, 0.03f, sinEl);
		float[] d = new float[3];
		// the daytime grade (terrain.metal skLift) at the horizon's elevation of the table sample (0.05)
		float day = smooth(0.12f, 0.55f, sinEl), lw = (float) Math.pow(0.95, 9);
		for (int k = 0; k < 3; k++) {
			float add = (LIFT_Z[k] + (LIFT_H[k] - LIFT_Z[k]) * lw) * day;
			d[k] = 1.0f - (float) Math.exp(-(acc[k] * exposure + add));
		}
		float lum = d[0] * 0.3f + d[1] * 0.59f + d[2] * 0.11f;
		for (int k = 0; k < 3; k++) horizon[k] = lum + (d[k] - lum) * SKY_SAT;
		float hl = horizon[0] * 0.3f + horizon[1] * 0.59f + horizon[2] * 0.11f;
		for (int k = 0; k < 3; k++) horizon[k] += (Math.min(hl * warmTint[k] * 1.25f, 1.0f) - horizon[k]) * warm;
	}

	/** The exposure of this frame (lower at a low sun, whose table is much brighter), and the table's display desaturation (terrain.metal SK_SAT). */
	private static float modelSin = -9f;
	private static int version;
	private static final int[] slotVersion = new int[4];
	private static final boolean[] slotMs = new boolean[4];
	private static final float[] hAcc = new float[3];
	/** mcopt.shaders.ambient.sh: the terrain's ambient light from the sky's own colour (L2 spherical harmonics of the displayed sky, sky_sh); shReady: this frame's coefficients exist. */
	static final boolean AMBIENT_SH = Boolean.parseBoolean(System.getProperty("mcopt.shaders.ambient.sh", "true"));
	static final float SH_GROUND = Float.parseFloat(System.getProperty("mcopt.shaders.ambient.ground", "0.3"));
	static boolean shReady;
	private static final int[] shKey = new int[4];
	private static final boolean[] shUnset = {true, true, true, true};
	static float exposure = EXPOSURE, warm;
	static final float[] warmTint = {1, 1, 1};
	static final float WARM = Float.parseFloat(System.getProperty("mcopt.shaders.sky.warm", "0.55"));
	static final float SKY_SAT = 0.95f;
	private static final float[] LIFT_Z = {0.26f, 0.17f, 0.0f}, LIFT_H = {0.10f, 0.80f, 4.0f};

	/** The fog colour of this frame (ShadeFogMixin): the sky's horizon by day, the old table's tint at night, grey in rain. */
	public static org.joml.Vector3fc fogColor(org.joml.Vector3fc old) {
		if (!ON || !live()) return old;
		System.arraycopy(new float[] {old.x(), old.y(), old.z()}, 0, nightFog, 0, 3);
		float day = smooth(-0.14f, 0.06f, sinEl);
		float lum = horizon[0] * 0.3f + horizon[1] * 0.59f + horizon[2] * 0.11f;
		float[] c = new float[3];
		for (int k = 0; k < 3; k++) {
			float hz = horizon[k] + (lum * (k == 0 ? 0.86f : k == 1 ? 0.97f : 1.1f) * 0.62f - horizon[k]) * rain * 0.9f;
			c[k] = old.get(k) + (hz - old.get(k)) * day;
		}
		return new org.joml.Vector3f(c[0], c[1], c[2]);
	}

	private static float smooth(float a, float b, float x) {
		float t = Math.max(0, Math.min(1, (x - a) / (b - a)));
		return t * t * (3 - 2 * t);
	}

	/** Draws the dome into the sky pass (called by the mixin at vanilla's sky disc). True: drawn, vanilla's disc, sunrise fan and sun are skipped. */
	public static boolean draw() {
		active = false;
		shReady = false;
		boolean cloudsBefore = cloudPrev;
		shadowK = 0f;
		cloudPrev = false;  // (set again below once this frame's clouds are drawn: a frame without them breaks the history)
		if (!ON || failed || !OwnProbe.bool("shaders", true) || !OwnProbe.bool("shaders.sky", true)) return false;
		OwnTerrain t = OwnShade.lastOwner;
		if (t == null) return why("no terrain renderer yet");
		LevelRenderState level = Minecraft.getInstance().gameRenderer.gameRenderState().levelRenderState;
		SkyRenderState st = level.skyRenderState;
		CameraRenderState cam = level.cameraRenderState;
		if (st.skybox != DimensionType.Skybox.OVERWORLD || (cam.fogType != FogType.ATMOSPHERIC && cam.fogType != FogType.NONE)) return why("skybox " + st.skybox + " fog " + cam.fogType);
		try (MemoryStack stack = MemoryStack.stackPush()) {
			update(st);
			long own = t.shadeOwn(), enc = t.shadeEnc();
			if (!noiseDone) {
				long nb = MemoryUtil.nmemAlloc(NOISE * NOISE * 8L);
				try {
					fillNoise(nb);
					if (!OwnNative.skyInit(own, LW, LH, nb, NOISE)) throw new IllegalStateException("sky table allocation failed");
				} finally {
					MemoryUtil.nmemFree(nb);
				}
				noiseDone = true;
			}
			if (!OwnNative.skyInit(own, LW, LH, 0, 0)) throw new IllegalStateException("sky table allocation failed");
			long lp = stack.ncalloc(16, 1, LUT_BYTES);
			MemoryUtil.memPutFloat(lp, sinEl);
			MemoryUtil.memPutFloat(lp + 4, (float) Math.sqrt(Math.max(0, 1 - sinEl * sinEl)));
			MemoryUtil.memPutFloat(lp + 8, OwnAtmo.CAM_KM);
			MemoryUtil.memPutFloat(lp + 12, OwnAtmo.SUN_I);
			boolean fog = OwnFog.ON && OwnProbe.bool("shaders.skyfog", true);
			boolean ms = fog && OwnFog.ms();
			MemoryUtil.memPutFloat(lp + 16, ms ? 1.0f : OwnAtmo.MS);
			MemoryUtil.memPutFloat(lp + 20, ms ? 1.0f : 0.0f);
			int k = ring++ & 3;
			boolean compute = slotVersion[k] != version || slotMs[k] != ms || !OwnProbe.bool("shaders.lutCache", true);
			slotVersion[k] = version;
			slotMs[k] = ms;
			if (!OwnNative.skyLut(own, enc, k, lp, LUT_BYTES, compute)) throw new IllegalStateException("sky_lut missing");
			// the sky ambient (mcopt.shaders.ambient.sh): the table projected onto L2 SH when it is rebuilt or the rain (which dims the displayed sky) has changed
			// the new sky and fog (OwnFog): this frame's values, before the sky ambient (which then follows the same display path as the dome) and the dome
			if (fog) OwnFog.update();
			if (AMBIENT_SH && OwnProbe.bool("shaders.ambient.sh", true)) {
				// (merge: with the skyfog sky the SH samples its display path, skfLin / analytic fog / skfDisp, so the key also follows the fog's state)
				int key = (version * 64 + Math.round(rain * 32f)) * 2 + (fog ? 1 : 0);
				if (fog) key = key * 31 + Math.round(OwnFog.v[7] * 2e4f) * 1009 + Math.round(OwnFog.v[6] * 2e4f) * 17 + Math.round(OwnFog.v[1] / 8f);
				boolean sh = compute || shKey[k] != key || shUnset[k];
				shKey[k] = key;
				shUnset[k] = false;
				long sp = stack.ncalloc(16, 1, SH_BYTES);
				MemoryUtil.memPutFloat(sp, sinEl);
				MemoryUtil.memPutFloat(sp + 4, (float) Math.sqrt(Math.max(0, 1 - sinEl * sinEl)));
				MemoryUtil.memPutFloat(sp + 16, exposure);
				MemoryUtil.memPutFloat(sp + 20, rain);
				MemoryUtil.memPutFloat(sp + 24, SH_GROUND);
				for (int c = 0; c < 3; c++) MemoryUtil.memPutFloat(sp + 32 + c * 4L, warmTint[c]);
				MemoryUtil.memPutFloat(sp + 44, warm);
				if (fog) {
					MemoryUtil.memPutFloat(sp + 28, 1f);  // (SkyShP.misc.w: the skyfog display path)
					OwnFog.put(sp + 48);                  // (SkyShP.fogA .. fogD)
				}
				if (!OwnNative.skySh(own, enc, k, sp, SH_BYTES, sh)) throw new IllegalStateException("sky_sh missing");
				shReady = true;
			} else {
				shReady = false;
			}
			long p = stack.ncalloc(16, 1, PARAM_BYTES);
			Matrix4f proj = cam.projectionMatrix, view = cam.viewRotationMatrix;
			float tx = 1.0f / proj.m00(), ty = 1.0f / proj.m11();
			// the view rotation's rows are the camera axes in world space
			put3(p, 0, -view.m02(), -view.m12(), -view.m22());
			put3(p, 16, view.m00() * tx, view.m10() * tx, view.m20() * tx);
			put3(p, 32, view.m01() * ty, view.m11() * ty, view.m21() * ty);
			if (LOG && (frameNo++ % 600) == 0) System.out.println("mcopt-shaders: sky fwd " + (-view.m02()) + " " + (-view.m12()) + " " + (-view.m22()) + " tan " + tx + " " + ty + " proj " + proj + " view " + view + " sun " + sunDir[1]);
			// the sun's place on the screen (uv, v down) for the light shafts: the sky shader's own mapping, inverted (dir = F + R tx x - D ty y, D the view's second row)
			{
				float sf = sunDir[0] * -view.m02() + sunDir[1] * -view.m12() + sunDir[2] * -view.m22();
				float sr = sunDir[0] * view.m00() + sunDir[1] * view.m10() + sunDir[2] * view.m20();
				float sd = sunDir[0] * view.m01() + sunDir[1] * view.m11() + sunDir[2] * view.m21();
				if (sf > 0.02f) {
					sunScreen[0] = 0.5f + 0.5f * sr / (sf * tx);
					sunScreen[1] = 0.5f + 0.5f * sd / (sf * ty);
					float off = Math.max(Math.abs(sunScreen[0] - 0.5f), Math.abs(sunScreen[1] - 0.5f)) * 2.0f;  // 1 at the screen's edge
					sunScreen[2] = Math.min(1.0f, sf * 4.0f) * (1.0f - smooth(1.2f, 3.2f, off));
					if (LOG && (frameNo % 600) == 1) System.out.println("mcopt-shaders: sun on screen " + sunScreen[0] + " " + sunScreen[1] + " w " + sunScreen[2]);
				} else {
					sunScreen[2] = 0;
				}
			}
			float night = 1.0f - smooth(-0.32f, -0.05f, sinEl);
			// clouds
			boolean clouds = cloudsWanted();
			float cover = CLOUD_COVER * (1.0f - MORN_CLOUD * morning() * (1.0f - rain)) + (1.0f - CLOUD_COVER) * rain * 0.9f;
			MemoryUtil.memPutFloat(p + 112, CLOUD_Y0);
			MemoryUtil.memPutFloat(p + 116, CLOUD_Y1);
			MemoryUtil.memPutFloat(p + 120, cover);
			MemoryUtil.memPutFloat(p + 124, clouds ? 1.0f + rain : 0f);
			double time = level.gameTime + level.worldPartialTicks;
			MemoryUtil.memPutFloat(p + 128, (float) (time * WIND));
			MemoryUtil.memPutFloat(p + 132, (float) (time * WIND * 0.3));
			MemoryUtil.memPutFloat(p + 136, (float) cam.pos.x);
			MemoryUtil.memPutFloat(p + 140, (float) cam.pos.z);
			MemoryUtil.memPutFloat(p + 144, (float) cam.pos.y);
			MemoryUtil.memPutFloat(p + 148, (float) time);
			MemoryUtil.memPutFloat(p + 152, CLOUD_SOFT);
			float sl = 1.0f - rain * 0.85f;
			put3(p, 160, sunTrans[0] * sl, sunTrans[1] * sl, sunTrans[2] * sl);
			MemoryUtil.memPutFloat(p + 172, sinEl);
			float nl = 1.0f - 0.82f * night;  // (clouds are dark at night)
			put3(p, 176, (1.0f - rain * 0.25f) * nl * (1 - 0.3f * night), (1.0f - rain * 0.25f) * nl * (1 - 0.1f * night), (1.0f - rain * 0.2f) * nl);
			MemoryUtil.memPutFloat(p + 188, warm);
			put3(p, 48, sunDir[0], sunDir[1], sunDir[2]);
			MemoryUtil.memPutFloat(p + 60, (float) Math.cos(Math.toRadians(SUN_DEG)));
			float vis = 1.0f;
			put3(p, 64, sunTrans[0] * vis, sunTrans[1] * vis, sunTrans[2] * vis);
			MemoryUtil.memPutFloat(p + 76, rain);
			float[] z = {0.165f, 0.159f, 0.209f};
			put3(p, 80, z[0], z[1], z[2]);
			MemoryUtil.memPutFloat(p + 92, night);
			put3(p, 96, nightFog[0], nightFog[1], nightFog[2]);
			MemoryUtil.memPutFloat(p + 108, exposure);
			if (clouds) {
				// the clouds track: the coverage map (one bilinear sample per density evaluation, and the terrain's cloud shadows), shaped and lit clouds, accumulated history
				boolean wantMap = OwnProbe.bool("shaders.clouds.map", MAP) && (OwnProbe.bool("shaders.clouds.shape", SHAPE) || OwnProbe.bool("shaders.clouds.shadows", SHADOWS));
				boolean shape = OwnProbe.bool("shaders.clouds.shape", SHAPE), accum = ACCUM && OwnProbe.bool("shaders.clouds.accum", true);
				// the camera's turn since the last cloud frame: fast turns shorten the history and march more pixels per frame
				float turn = cloudView[0] * -view.m02() + cloudView[1] * -view.m12() + cloudView[2] * -view.m22();
				float fast = cloudPrev0 ? smooth(0.002f, 0.02f, (float) Math.acos(Math.max(-1f, Math.min(1f, turn)))) : 0f;
				int div = accum ? ACCUM_DIV : CLOUD_DIV;
				int cw = Math.max(64, (int) Math.ceil(screenW() / (double) div)), ch = Math.max(64, (int) Math.ceil(screenH() / (double) div));
				MemoryUtil.memPutFloat(p + 256, COV_R);
				MemoryUtil.memPutFloat(p + 272, LIGHT_GAIN);
				MemoryUtil.memPutFloat(p + 276, ACCUM_AGE + (3.0f - ACCUM_AGE) * fast);
				MemoryUtil.memPutFloat(p + 280, (float) ((frameCtr++ * 0.6180339887) % 1.0));
				MemoryUtil.memPutFloat(p + 288, (float) (cam.pos.x - cloudPos[0]));
				MemoryUtil.memPutFloat(p + 292, (float) (cam.pos.y - cloudPos[1]));
				MemoryUtil.memPutFloat(p + 296, (float) (cam.pos.z - cloudPos[2]));
				MemoryUtil.memPutFloat(p + 304, (float) (time * WIND - cloudWind[0]));
				MemoryUtil.memPutFloat(p + 308, (float) (time * WIND * 0.3 - cloudWind[1]));
				cloudWind[0] = time * WIND;
				cloudWind[1] = time * WIND * 0.3;
				boolean covOk = false;
				if (wantMap) {
					MemoryUtil.memPutFloat(p + 260, 0f);
					covOk = OwnNative.skyCov(own, enc, p, PARAM_BYTES);
				}
				shape &= covOk;
				MemoryUtil.memPutFloat(p + 260, (shape ? 1f : 0f) + (accum ? 2f : 0f));
				MemoryUtil.memPutFloat(p + 264, fast > 0.5f ? 2f : ACCUM_N);
				if (covOk && OwnProbe.bool("shaders.clouds.shadows", SHADOWS)) {
					shadowK = SHADOW_STRENGTH * (1.0f - rain);
					shadowCover = cover;
					shadowDens = 1.0f + rain;
				}
				// temporal: a quarter of the pixels marched per frame, the rest reprojected from the previous frame's target (same size, drawn last frame, no jump)
				boolean hist = TEMPORAL && OwnProbe.bool("shaders.clouds.temporal", true) && cloudsBefore && cw == cloudW && ch == cloudH && accum == cloudAccum
					&& Math.abs(cam.pos.x - cloudPos[0]) + Math.abs(cam.pos.y - cloudPos[1]) + Math.abs(cam.pos.z - cloudPos[2]) < 16.0;
				for (int i = 0; i < 12; i++) MemoryUtil.memPutFloat(p + 192 + i * 4 + (i / 3) * 4, cloudView[i]);
				MemoryUtil.memPutFloat(p + 240, (float) (cloudPhase++ % (accum ? 144 : 4)));
				MemoryUtil.memPutFloat(p + 244, hist ? 1f : 0f);
				if (!OwnNative.skyClouds(own, enc, k, p, PARAM_BYTES, cw, ch)) throw new IllegalStateException("cloud target missing");
				for (int i = 0; i < 12; i++) cloudView[i] = MemoryUtil.memGetFloat(p + i * 4 + (i / 3) * 4);
				cloudW = cw;
				cloudH = ch;
				cloudAccum = accum;
				cloudPrev0 = true;
				cloudPos[0] = cam.pos.x;
				cloudPos[1] = cam.pos.y;
				cloudPos[2] = cam.pos.z;
			}
			cloudPrev = clouds;
			if (fog) {
				// the new sky and fog (OwnFog; SkyFrame bytes 384..447, updated above; the sun disc's boost at sunrise and sunset 448)
				OwnFog.put(p + 384);
				MemoryUtil.memPutFloat(p + 448, OwnFog.SUN_BOOST);
			}
			if (!OwnNative.skyDraw(own, enc, p, PARAM_BYTES)) return why("no open render encoder for the dome");
		} catch (RuntimeException e) {
			System.out.println("mcopt-shaders: sky disabled: " + e);
			failed = true;
			return false;
		}
		if (!said) {
			said = true;
			System.out.println("mcopt-shaders: sky dome on (single scattering table " + LW + "x" + LH + ")");
		}
		active = true;
		return true;
	}

	static final boolean CLOUDS = !"false".equals(System.getProperty("mcopt.shaders.clouds"));
	static final float CLOUD_Y0 = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.base", "230"));
	static final float CLOUD_Y1 = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.top", "340"));
	static final float CLOUD_COVER = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.cover", "0.55"));
	static final float CLOUD_SOFT = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.soft", "2.4"));
	/** Wind: blocks per tick. */
	static final float WIND = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.wind", "0.08"));
	static final int NOISE = 256;
	/** The cloud target is the screen divided by this in each direction. */
	static final int CLOUD_DIV = Integer.getInteger("mcopt.shaders.clouds.div", 6);
	/** Temporal clouds: one pixel of each 2 x 2 marched per frame, the others reprojected from the previous frame (-Dmcopt.shaders.clouds.temporal=false: all marched). */
	static final boolean TEMPORAL = !"false".equals(System.getProperty("mcopt.shaders.clouds.temporal"));
	/** The previous cloud frame: its view (fwd, right, up as xyz triples), target size, camera position, whether clouds were drawn. */
	private static final float[] cloudView = new float[12];
	private static final double[] cloudPos = new double[3];
	private static int cloudW, cloudH, cloudPhase;
	private static boolean cloudAccum, cloudPrev0;
	private static final double[] cloudWind = new double[2];
	private static long frameCtr;

	// ---- the clouds track (sub-flags: .map the coverage map, .shape shaped and lit clouds, .accum the accumulated higher-resolution target, .shadows cloud shadows on terrain) ----
	static final boolean MAP = !"false".equals(System.getProperty("mcopt.shaders.clouds.map"));
	static final boolean SHAPE = !"false".equals(System.getProperty("mcopt.shaders.clouds.shape"));
	static final boolean ACCUM = !"false".equals(System.getProperty("mcopt.shaders.clouds.accum"));
	static final boolean SHADOWS = !"false".equals(System.getProperty("mcopt.shaders.clouds.shadows"));
	static final float COV_R = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.mapRadius", "9000"));
	static final float LIGHT_GAIN = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.gain", "0.8"));
	static final float SHADOW_STRENGTH = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.shadowStrength", "0.8"));
	/** The accumulated target is the screen divided by this; one pixel of each N x N block is marched per frame; the history holds up to ACCUM_AGE samples. */
	static final int ACCUM_DIV = Integer.getInteger("mcopt.shaders.clouds.divAccum", 3);
	static final int ACCUM_N = Integer.getInteger("mcopt.shaders.clouds.check", 3);
	static final float ACCUM_AGE = Float.parseFloat(System.getProperty("mcopt.shaders.clouds.age", "12"));
	/** This frame's cloud shadow state for the terrain (OwnShade.writeFrame): strength 0 = none. */
	static float shadowK, shadowCover, shadowDens;

	static float cloudMid() {
		return 0.5f * (CLOUD_Y0 + CLOUD_Y1);
	}
	private static boolean cloudPrev;

	private static int screenW() {
		return Minecraft.getInstance().gameRenderer.mainRenderTarget().width;
	}

	private static int screenH() {
		return Minecraft.getInstance().gameRenderer.mainRenderTarget().height;
	}
	private static boolean noiseDone;

	/** A tileable noise texture (RGBA8): r a low-frequency fractal value noise, g a finer one, b and a two more. Periodic lattices, so it repeats seamlessly. */
	private static void fillNoise(long out) {
		java.util.Random rnd = new java.util.Random(7);
		int[] base = {4, 8, 16, 32, 64};
		float[][] chan = new float[4][NOISE * NOISE];
		int[][] start = {{0, 4}, {1, 5}, {2, 5}, {3, 5}};  // octave ranges per channel: first and past-last index into base
		for (int c = 0; c < 4; c++) {
			float amp = 1, sum = 0;
			for (int o = start[c][0]; o < start[c][1] && o < base.length; o++) {
				int n = base[o];
				float[] lat = new float[n * n];
				for (int i = 0; i < lat.length; i++) lat[i] = rnd.nextFloat();
				for (int y = 0; y < NOISE; y++) for (int x = 0; x < NOISE; x++) {
					float fx = (float) x / NOISE * n, fy = (float) y / NOISE * n;
					int ix = (int) fx, iy = (int) fy;
					float tx = fx - ix, ty = fy - iy;
					tx = tx * tx * tx * (tx * (tx * 6 - 15) + 10);
					ty = ty * ty * ty * (ty * (ty * 6 - 15) + 10);
					float a = lat[iy % n * n + ix % n], b = lat[iy % n * n + (ix + 1) % n], cc = lat[(iy + 1) % n * n + ix % n], d = lat[(iy + 1) % n * n + (ix + 1) % n];
					chan[c][y * NOISE + x] += amp * ((a + (b - a) * tx) + ((cc + (d - cc) * tx) - (a + (b - a) * tx)) * ty);
				}
				sum += amp;
				amp *= 0.5f;
			}
			// normalise to the channel's own range so coverage thresholds mean the same for every channel
			float lo = 1e9f, hi = -1e9f;
			for (float v : chan[c]) { lo = Math.min(lo, v / sum); hi = Math.max(hi, v / sum); }
			for (int i = 0; i < chan[c].length; i++) chan[c][i] = (chan[c][i] / sum - lo) / (hi - lo);
		}
		for (int i = 0; i < NOISE * NOISE; i++)
			for (int c = 0; c < 4; c++) MemoryUtil.memPutShort(out + i * 8L + c * 2L, (short) Math.round(chan[c][i] * 65535f));
	}

	private static long frameNo;
	private static String lastWhy;

	private static boolean why(String reason) {
		if (LOG && !reason.equals(lastWhy)) System.out.println("mcopt-shaders: no sky dome this frame: " + reason);
		lastWhy = reason;
		return false;
	}

	private static final boolean LOG = Boolean.getBoolean("mcopt.shaders.log");

	private static void put3(long p, int off, float x, float y, float z) {
		MemoryUtil.memPutFloat(p + off, x);
		MemoryUtil.memPutFloat(p + off + 4, y);
		MemoryUtil.memPutFloat(p + off + 8, z);
	}

	/** Haze density of this frame (1 / blocks). */
	static float hazeDensity() {
		float len = HAZE_CLEAR + (HAZE_RAIN - HAZE_CLEAR) * rain;
		return 1.0f / (len * (1.0f + MORN_HAZE * morning() * (1.0f - rain) + DUSK_HAZE * dusk() * (1.0f - rain)));
	}

	/** A clear mid-morning sun (terrain.metal skMorn): 0..1. */
	static float morning() {
		return smooth(0.08f, 0.2f, sinEl) * (1.0f - smooth(0.5f, 0.8f, sinEl));
	}
	static final float MORN_CLOUD = Float.parseFloat(System.getProperty("mcopt.shaders.sky.mornCloud", "0.55"));
	/** A low sun (sunset, golden hour): the haze thins so that the far land keeps its contrast against the glowing sky (-Dmcopt.shaders.sky.duskHaze, 0 = as before). */
	static final float DUSK_HAZE = Float.parseFloat(System.getProperty("mcopt.shaders.sky.duskHaze", "1.0"));
	static float dusk() {
		return (1.0f - smooth(0.0f, 0.3f, sinEl)) * smooth(-0.08f, 0.02f, sinEl);
	}
	static final float MORN_HAZE = Float.parseFloat(System.getProperty("mcopt.shaders.sky.mornHaze", "0.8"));
}
