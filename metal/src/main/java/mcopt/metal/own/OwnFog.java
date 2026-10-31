package mcopt.metal.own;

import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.LightLayer;
import org.lwjgl.system.MemoryUtil;

/**
 * The sky model and fog of the shaders (-Dmcopt.shaders.skyfog; track B): per-frame uniforms of the analytic height fog (terrain.metal skfT / skfHaze: two
 * exponential height falloffs, Rayleigh from sea level + 30 with a 30 block half-life and Mie from sea level + 7 with 7), written both into the ShadeFrame
 * (bytes 368..431) for the terrain's pixels and into the SkyFrame (bytes 384..447) for the dome. Mie density by time of day (a morning's mist, a quiet noon, an
 * evening haze, the blue hour's extra, rain) after Photon Shaders' settings (Photon licence); the colours are the sky table's own (the shader samples it).
 * -Dmcopt.shaders.skyfog=false: the old sky and haze; -Dmcopt.shaders.skyfog.ms=false: the isotropic multiple scattering term instead of Hillaire's table;
 * -Dmcopt.shaders.skyfog.fog=false: the new sky colour model without the fog. Probe keys shaders.skyfog, shaders.skyfog.ms.
 */
final class OwnFog {
	static final boolean ON = OwnShade.ON && !"false".equals(System.getProperty("mcopt.shaders.skyfog"));
	static final boolean MS = ON && !"false".equals(System.getProperty("mcopt.shaders.skyfog.ms"));
	private static final boolean FOG = !"false".equals(System.getProperty("mcopt.shaders.skyfog.fog"));
	static final float MIE = Float.parseFloat(System.getProperty("mcopt.shaders.skyfog.mie", "0.45"));
	static final float RAY = Float.parseFloat(System.getProperty("mcopt.shaders.skyfog.ray", "3.0"));
	static final float HALO = Float.parseFloat(System.getProperty("mcopt.shaders.skyfog.halo", "0.5"));
	static final float GAIN = Float.parseFloat(System.getProperty("mcopt.shaders.skyfog.gain", "0.4"));
	static final float SUN_BOOST = Float.parseFloat(System.getProperty("mcopt.shaders.skyfog.sun", "0.5"));
	static final float SEA = 63.0f;

	/** fogA, fogB, fogC, fogD (16 floats), of the last update(). */
	static final float[] v = new float[16];
	private static float skyLight = 1f;

	private OwnFog() {
	}

	/** The new sky and fog apply this frame. */
	static boolean active() {
		return ON && OwnProbe.bool("shaders.skyfog", true) && OwnSky.hazeActive();
	}

	static boolean ms() {
		return MS && OwnProbe.bool("shaders.skyfog.ms", true);
	}

	private static float smooth(float a, float b, float x) {
		float t = Math.max(0, Math.min(1, (x - a) / (b - a)));
		return t * t * (3 - 2 * t);
	}

	/** Mie extinction (1 / block) of this frame's sun and weather, Photon's morning / noon / evening / blue hour table. */
	static float mieDensity() {
		float y = OwnSky.sinEl, rain = OwnSky.rain;
		boolean morning = OwnSky.sunDir[0] > 0;
		float noon = 0.0001f;
		float d = morning ? noon + (0.0070f - noon) * (1f - smooth(0.05f, 0.5f, y)) : noon + (0.0050f - noon) * (1f - smooth(0.0f, 0.35f, y));
		float night = 1f - smooth(-0.3f, -0.1f, y);
		d += (0.0010f - d) * night;  // (midnight)
		d += 0.0020f * (float) Math.exp(-190.0 * (y + 0.07283) * (y + 0.07283));  // the blue hour
		d += (0.03f - d) * rain;
		return d * 0.5f * MIE;
	}

	/** The frame's values (call once per frame before put()). */
	static void update() {
		Minecraft mc = Minecraft.getInstance();
		CameraRenderState cam = mc.gameRenderer.gameRenderState().levelRenderState.cameraRenderState;
		float sinEl = OwnSky.sinEl, rain = OwnSky.rain;
		// the camera's sky light (a cave has no fog, a clearing in a forest less): smoothed so that walking in does not pop
		try {
			if (mc.level != null) {
				float target = mc.level.getBrightness(LightLayer.SKY, BlockPos.containing(cam.pos.x, cam.pos.y, cam.pos.z)) / 15f;
				skyLight += (target - skyLight) * 0.04f;
			}
		} catch (RuntimeException e) {
			skyLight = 1f;
		}
		float k = FOG ? Math.max(0.0f, Math.min(1.0f, (skyLight - 0.1f) / 0.6f)) : 0f;
		float mie = mieDensity() * k, ray = 0.0005f * 0.5f * RAY * k;
		float rd = mc.options.getEffectiveRenderDistance() * 16f;
		v[0] = (float) cam.pos.y - (SEA + 30f);
		v[1] = (float) cam.pos.y - (SEA + 7f);
		v[2] = 1f;
		v[3] = rd;
		v[4] = ray * 0.31f;
		v[5] = ray * 0.67f;
		v[6] = ray;
		v[7] = mie;
		// the sun's halo (the Mie phase x this): the sun's colour at the camera, brighter in the evening's glow, gone with the sun and in rain
		float glow = 1f + 0.75f * (float) Math.exp(-300.0 * (sinEl + 0.02) * (sinEl + 0.02));
		float h = HALO * glow * (1f - 0.9f * rain) * smooth(-0.08f, 0.06f, sinEl);
		for (int i = 0; i < 3; i++) v[8 + i] = h * OwnSky.warmTint[i];
		v[11] = 0f;
		v[12] = v[13] = v[14] = 0f;
		v[15] = GAIN;
	}

	/** Writes fogA..fogD at p. */
	static void put(long p) {
		for (int i = 0; i < 16; i++) MemoryUtil.memPutFloat(p + i * 4L, v[i]);
	}
}
