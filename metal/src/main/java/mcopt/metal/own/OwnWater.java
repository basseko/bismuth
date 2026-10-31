package mcopt.metal.own;

import com.mojang.blaze3d.pipeline.RenderTarget;
import mcopt.metal.MetalBridge;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.block.FluidModel;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.world.level.material.FogType;
import net.minecraft.world.level.material.Fluids;
import org.joml.Matrix4f;
import org.joml.Vector3d;
import org.joml.Vector3f;
import org.joml.Vector3fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * Water of the shaders (-Dmcopt.shaders=true and not -Dmcopt.shaders.water=false; milestone 4).
 * <p>
 * Surface: our translucent terrain draw (kind bit 10, K_WATER, function constant 35) shades the quads that carry the water sprites (terrain.metal
 * waterSurface): procedural scrolling wave normals, Fresnel, the sky's reflection through {@code skyRadiance} (the one function the milestone 3
 * sky model replaces), a screen-space reflection of the opaque scene, the refracted bed absorbed by the water between it and the surface, and the sun's
 * glitter. Under the surface the same pixels show the sky through a window and total internal reflection of the sea bed.
 * <p>
 * Scene behind the water, and the pass order: the world's one render pass draws the opaque terrain, entities and the translucent features, then (vanilla's
 * classic transparency) the translucent terrain. Right before our translucent draw the pass is split once ({@link #beforeTranslucent}: mcown.m
 * mcs_water_split, the same split the terrain occlusion makes): the encoder ends with colour and depth stored and reopens with both loaded. The stored
 * textures are then read by the water's pixels (the stale opaque scene: tile memory holds the later writes until the pass ends), so there is no copy;
 * {@code -Dmcopt.shaders.water.copy=true} blits both first (if reading an attachment misbehaves). The split is only paid when the translucent terrain has
 * something to draw this frame; without it (the pass has an unexpected shape, {@code -Dmcopt.shaders.water.split=false}) the water falls back to a
 * sky-only reflection over a constant depth of absorption, drawn with the ordinary blend.
 * <p>
 * Underwater medium: while the camera is in water every terrain fragment kind draws the medium instead of vanilla's fog (per channel absorption toward the
 * water's fog colour, {@code waterMedium} in terrain.metal), with caustics on the sea bed; the fog colour itself is darkened and tinted by a mixin
 * (ShadeWaterFogMixin) so the clear colour and the vanilla pipelines (entities, particles, sky) agree; the vanilla blue screen overlay is dropped
 * (ShadeWaterOverlayMixin).
 * <p>
 * Everything is data for the one frame block (terrain.metal WaterFrame, fragment buffer 12) written by {@link #write}. Probe keys (OwnProbe, in-run A/B):
 * {@code shaders.water} (all of it), {@code shaders.water.ssr}, {@code shaders.water.split}.
 */
public final class OwnWater {
	/** Water on (with the shaders). */
	public static final boolean ON = OwnShade.ON && !"false".equals(System.getProperty("mcopt.shaders.water"));
	/** Screen-space reflection steps (0 turns the SSR off). */
	static final int STEPS = !"false".equals(System.getProperty("mcopt.shaders.water.ssr")) ? Math.max(0, Math.min(48, Integer.getInteger("mcopt.shaders.water.steps", 14))) : 0;
	/** Split the world pass for the scene's colour and depth (false: no scene, the fallback look). */
	static final boolean SPLIT = !"false".equals(System.getProperty("mcopt.shaders.water.split"));
	/** Blit colour and depth at the split instead of reading the attachments themselves. */
	static final int COPY = "false".equals(System.getProperty("mcopt.shaders.water.copy")) ? 0 : "both".equals(System.getProperty("mcopt.shaders.water.copy")) ? 3 : "depth".equals(System.getProperty("mcopt.shaders.water.copy")) ? 2 : 1;
	/** Keep vanilla's blue screen overlay under water. */
	public static final boolean OVERLAY = Boolean.getBoolean("mcopt.shaders.water.overlay");
	static final float WAVES = prop("mcopt.shaders.water.waves", 0.6f), ABSORB = prop("mcopt.shaders.water.absorb", 1.0f), GLITTER = prop("mcopt.shaders.water.glitter", 0.7f),
		BODY = prop("mcopt.shaders.water.body", 1.0f), SKY = prop("mcopt.shaders.water.sky", 1.0f), MURK = prop("mcopt.shaders.water.murk", 1.0f),
		CAUSTIC = prop("mcopt.shaders.water.caustic", 0.5f), THICK = prop("mcopt.shaders.water.thick", 0.6f);
	static final float[] FOG = parse3(System.getProperty("mcopt.shaders.water.fog", "0.20,0.27,0.28"));

	private static float[] parse3(String s) {
		String[] p = s.split(",");
		return new float[] {Float.parseFloat(p[0]), Float.parseFloat(p[1]), Float.parseFloat(p[2])};
	}
	private static final int FRAME_BYTES = 320;
	/** kind bit 10 */
	static final int K_WATER = 1 << 10;

	/** This frame is shaded for water (beginFrame), and its translucent draw shades water surfaces (beforeTranslucent). */
	private static boolean frameOn, surface, underwater, failed, said, splitSaid;
	private static float[] rects;
	private static boolean rectsTried;

	private OwnWater() {
	}

	private static float prop(String key, float dflt) {
		String v = System.getProperty(key);
		return v == null ? dflt : Float.parseFloat(v);
	}

	private static float smooth(float a, float b, float x) {
		float t = Math.max(0, Math.min(1, (x - a) / (b - a)));
		return t * t * (3 - 2 * t);
	}

	/** The frame is not shaded (other dimension, shaders off by probe): no water variants. */
	static void off() {
		frameOn = false;
		surface = false;
	}

	/** Whether this frame's terrain draws may be water variants (the medium under water, the surface in the translucent draw). */
	static boolean frameOn() {
		return frameOn;
	}

	/**
	 * Called by OwnShade.frame at the start of each world frame (overworld, after OwnLight.update): decides whether water is shaded this frame and publishes the
	 * frame block (without the matrices and the scene) so the opaque draws can already draw the underwater medium.
	 */
	static void beginFrame(OwnTerrain t, CameraRenderState camera, SkyRenderState sky) {
		frameOn = false;
		surface = false;
		if (!ON || failed || !OwnProbe.bool("shaders.water", true)) return;
		underwater = camera.fogType == FogType.WATER;
		try {
			write(t, camera, sky, null, null, false);
		} catch (RuntimeException | LinkageError e) {
			fail(e);
			return;
		}
		frameOn = true;
	}

	/**
	 * Called by OwnTerrain.drawTranslucent when it has translucent terrain to draw and before it binds anything: splits the world pass for the scene behind the
	 * water and publishes the full frame block. proj: the projection the draw uses, clip: projection x view rotation (camera-relative world to clip, GL row order).
	 */
	static void beforeTranslucent(OwnTerrain t, Object encoder, CameraRenderState camera, Matrix4f proj, Matrix4f clip) {
		surface = false;
		if (!frameOn) return;
		try {
			SkyRenderState sky = Minecraft.getInstance().gameRenderer.gameRenderState().levelRenderState.skyRenderState;
			boolean scene = false;
			if (SPLIT && OwnProbe.bool("shaders.water.split", true)) {
				RenderTarget main = Minecraft.getInstance().gameRenderer.mainRenderTarget();
				int r = OwnNative.waterSplit(t.shadeOwn(), t.shadeEnc(), MetalBridge.textureHandle(main.getColorTexture()), COPY);
				if (r > 0) {
					scene = true;
					MetalBridge.restorePass(encoder);  // (the reopened encoder has none of the pass's state)
				} else if (!splitSaid) {
					splitSaid = true;
					System.out.println("mcopt-shaders: water: no split (" + (r < 0 ? "the main target's attachments are not shader readable" : "the open pass is not the plain main pass")
						+ "): the water has no scene to refract or reflect");
				}
			}
			write(t, camera, sky, proj, clip, scene);
			surface = true;
			if (!said) {
				said = true;
				System.out.println("mcopt-shaders: water on (" + (scene ? "split pass, " + (COPY == 0 ? "scene read in place" : "scene copied: " + (COPY == 1 ? "colour" : COPY == 2 ? "depth" : "colour and depth")) + ", " + (STEPS > 0 ? "SSR " + STEPS + " steps" : "no SSR")
					: "no scene") + ")");
			}
		} catch (RuntimeException | LinkageError e) {
			fail(e);
		}
	}

	/** The kind of a terrain draw with water applied: the medium on every fragment kind under water, the surface on translucent ones. */
	static int kind(int kind) {
		if (!frameOn) return kind;
		int f = kind & 15;
		if (f != 0 && f != 1 && f != 3) return kind;
		if ((kind & ((1 << 21) | (1 << 29) | (1 << 30) | (1 << 9) | K_WATER)) != 0) return kind;  // mesh stages, the oracle, shadow draws: not here
		return underwater || (f == 3 && surface) ? kind | K_WATER : kind;
	}

	/** A water draw could not build its pipeline (OwnNative.draw drew it without): no water from now on. */
	static void drawFailed(RuntimeException e) {
		fail(e);
	}

	private static void fail(Throwable e) {
		if (!failed) System.out.println("mcopt-shaders: water disabled: " + e);
		failed = true;
		frameOn = false;
		surface = false;
	}

	/** The water and flowing water sprites' uv rectangles (atlas coordinates), or null. */
	private static float[] rects() {
		if (rectsTried) return rects;
		rectsTried = true;
		try {
			FluidModel m = Minecraft.getInstance().getModelManager().getFluidStateModelSet().get(Fluids.WATER.defaultFluidState());
			TextureAtlasSprite a = m.stillMaterial().sprite(), b = m.flowingMaterial().sprite();
			rects = new float[] {a.getU0(), a.getV0(), a.getU1(), a.getV1(), b.getU0(), b.getV0(), b.getU1(), b.getV1()};
		} catch (RuntimeException | LinkageError e) {
			System.out.println("mcopt-shaders: water sprites not found (" + e + "): water is told by its colour");
		}
		return rects;
	}

	/** Writes this frame's WaterFrame (terrain.metal) and hands it to the native side. */
	private static void write(OwnTerrain t, CameraRenderState camera, SkyRenderState sky, Matrix4f proj, Matrix4f clip, boolean scene) {
		try (MemoryStack stack = MemoryStack.stackPush()) {
			long b = stack.ncalloc(16, 1, FRAME_BYTES);
			RenderTarget main = Minecraft.getInstance().gameRenderer.mainRenderTarget();
			int w = main.width, h = main.height;
			float projA = 0, projB = 1;
			if (clip != null) {
				// the matrix the vertex stage multiplies by, with y flipped as it flips it
				Matrix4f vp = new Matrix4f(clip);
				vp.m01(-vp.m01()).m11(-vp.m11()).m21(-vp.m21()).m31(-vp.m31());
				vp.getToAddress(b);
				new Matrix4f(vp).invert().getToAddress(b + 64);
				// z_ndc = A + B / w for a perspective projection (reversed Z or not)
				projA = proj.m22() / proj.m23();
				projB = proj.m32();
			}
			// the light that casts: the sun, or the moon at night (as OwnShade picks it)
			double sa = sky.sunAngle, ma = sky.moonAngle;
			Vector3d sun = new Vector3d(-Math.sin(sa), Math.cos(sa), 0), moon = new Vector3d(-Math.sin(ma), Math.cos(ma), 0);
			float clear = Math.max(0, Math.min(1, sky.rainBrightness));
			float sunStr = smooth(0.0f, 0.12f, (float) sun.y) * clear * clear;
			float moonStr = 0.22f * smooth(0.0f, 0.2f, (float) moon.y) * (1 - smooth(0.0f, 0.12f, (float) sun.y)) * clear;
			boolean useSun = sunStr >= moonStr;
			Vector3d l = useSun ? sun : moon;
			float strength = OwnProbe.bool("shaders.sun", true) ? Math.min(1, (useSun ? sunStr : moonStr) * (useSun ? 1.0f : 3.0f)) : 0f;
			float[] lc = useSun ? OwnLight.sun : new float[] {0.45f, 0.60f, 1.0f};
			put4(b + 128, (float) l.x, (float) l.y, (float) l.z, strength);
			put4(b + 144, lc[0], lc[1], lc[2], GLITTER);
			// the sky's zenith colour as the sky disc has it (OwnLightmap.sky): greyed and dimmed in rain
			float[] top = OwnLight.skyColor();
			float rb = clear, grey = (top[0] * 0.3f + top[1] * 0.59f + top[2] * 0.11f) * (0.75f + 0.25f * rb);
			for (int c = 0; c < 3; c++) top[c] = (grey + (top[c] - grey) * (0.4f + 0.6f * rb)) * (0.7f + 0.3f * rb);
			put4(b + 160, top[0], top[1], top[2], SKY);
			put4(b + 176, 0.50f * ABSORB, 0.24f * ABSORB, 0.18f * ABSORB, 48.0f);
			put4(b + 192, 0.20f, 0.27f, 0.31f, BODY);
			put4(b + 208, (float) mod(camera.pos.x, 1024.0), (float) mod(camera.pos.y, 1024.0), (float) mod(camera.pos.z, 1024.0), (float) mod(System.nanoTime() * 1e-9, 4096.0));
			put4(b + 224, w, h, 1.0f / w, 1.0f / h);
			float[] rc = rects();
			int flags = (scene ? 1 : 0) | (rc != null ? 2 : 0);
			put4(b + 240, underwater ? 1f : 0f, WAVES * (1.0f + 0.8f * OwnLight.rain), scene && OwnProbe.bool("shaders.water.ssr", true) ? STEPS : 0, flags);
			put4(b + 256, projA, projB, THICK, underwater ? CAUSTIC : 0f);
			if (rc != null) {
				put4(b + 272, rc[0], rc[1], rc[2], rc[3]);
				put4(b + 288, rc[4], rc[5], rc[6], rc[7]);
			}
			put4(b + 304, 0.20f * MURK, 0.075f * MURK, 0.05f * MURK, 0f);
			OwnNative.waterFrame(t.shadeOwn(), b, FRAME_BYTES);
		}
	}

	private static double mod(double x, double m) {
		return x - Math.floor(x / m) * m;
	}

	private static void put4(long a, float x, float y, float z, float w) {
		MemoryUtil.memPutFloat(a, x);
		MemoryUtil.memPutFloat(a + 4, y);
		MemoryUtil.memPutFloat(a + 8, z);
		MemoryUtil.memPutFloat(a + 12, w);
	}

	/**
	 * The underwater fog colour (data: the water fog environment's base colour, which is also the clear colour and what vanilla's fog blends toward): murkier,
	 * greener and darker than vanilla's bright blue, and darker still at night.
	 */
	public static Vector3fc fogColor(Vector3fc c) {
		float day = Math.max(0.0f, Math.min(1.0f, (OwnLight.el + 0.15f) / 0.5f));
		float k = (0.35f + 0.65f * day) * (1.0f - 0.35f * Math.min(1, OwnLight.rain));
		// a teal-grey (BSL's murk) taking a little of the biome's water colour (vanilla's is a saturated (0.2, 0.2, 1)); the grade's underwater row is mild
		float m = Math.max(0.05f, (c.x() + c.y() + c.z()) / 3.0f);
		float tr = 0.90f + 0.10f * c.x() / m, tg = 0.90f + 0.10f * c.y() / m, tb = 0.90f + 0.10f * c.z() / m;
		return new Vector3f(FOG[0] * tr * k, FOG[1] * tg * k, FOG[2] * tb * k);
	}

	/** Whether the data mixins (fog colour, the screen overlay) apply this frame. */
	public static boolean dataOn() {
		return ON && !failed && OwnProbe.bool("shaders", true) && OwnProbe.bool("shaders.water", true);
	}
}
