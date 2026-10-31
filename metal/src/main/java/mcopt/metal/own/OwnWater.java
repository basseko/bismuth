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
 * mcs_water_split, the same split the terrain occlusion makes): the encoder ends with colour and depth stored, a compute kernel (terrain.metal water_copy)
 * reads them and writes a half-resolution copy of each (colour; depth as R32Float, the nearest of each 2 x 2 block), and the encoder reopens with both loaded.
 * The water's pixels read only those copies, never an attachment of the pass they are drawn in (reading an attachment as a texture is undefined: it corrupted
 * the colour). {@code -Dmcopt.shaders.water.copy=full} copies at full resolution instead. The split and the copy are only paid when the frame lists a section
 * that holds water quads (OwnMesh.hasWater, found when the section's translucent layer is stored), not for glass or ice alone; without a scene (no water listed,
 * the pass has an unexpected shape, {@code -Dmcopt.shaders.water.split=false}) the water falls back to a sky-only reflection over a constant depth of absorption,
 * drawn with the ordinary blend.
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
	/**
	 * Tile water (default; {@code -Dmcopt.shaders.water.tile=false} for the split): no split of the world pass. The water's pixels read the colour behind them in
	 * place (programmable blending) and the distance behind them from that colour's alpha, which the shaded opaque terrain writes as a code; the reflection marches
	 * the previous frame's half-size scene, which the colour pass writes (colour and distance), reprojected to this frame's camera. Costs: no refraction offset
	 * (the pixel behind is all a pass can read), reflections one frame late, the bed's depth to about 3 % (an 8-bit code, dithered).
	 */
	static final boolean TILE = ON && SPLIT && !"false".equals(System.getProperty("mcopt.shaders.water.tile"));
	/** This frame's and the previous frame's camera-relative clip matrix (projection x view rotation) and camera position (tile water). */
	private static Matrix4f viewNow, viewPrev;
	private static final Vector3d posNow = new Vector3d(), posPrev = new Vector3d();
	/** The scene copy's size at the split: 1 half resolution (default), 2 full resolution. */
	static final int COPY = "full".equals(System.getProperty("mcopt.shaders.water.copy")) ? 2 : 1;
	/** Keep vanilla's blue screen overlay under water. */
	public static final boolean OVERLAY = Boolean.getBoolean("mcopt.shaders.water.overlay");
	static final float WAVES = prop("mcopt.shaders.water.waves", 0.6f), ABSORB = prop("mcopt.shaders.water.absorb", 0.7f), GLITTER = prop("mcopt.shaders.water.glitter", 0.7f),
		BODY = prop("mcopt.shaders.water.body", 0.7f), SKY = prop("mcopt.shaders.water.sky", 1.0f), MURK = prop("mcopt.shaders.water.murk", 1.0f),
		CAUSTIC = prop("mcopt.shaders.water.caustic", 0.5f), THICK = prop("mcopt.shaders.water.thick", 0.6f);
	static final float[] FOG = parse3(System.getProperty("mcopt.shaders.water.fog", "0.26,0.35,0.36"));

	private static float[] parse3(String s) {
		String[] p = s.split(",");
		return new float[] {Float.parseFloat(p[0]), Float.parseFloat(p[1]), Float.parseFloat(p[2])};
	}
	private static final int FRAME_BYTES = 448;
	/** Track water sub-flags (each its own -Dmcopt.shaders.water.<name>=false and OwnProbe key shaders.water.<name>). */
	static final boolean GLARE_ON = ON && !"false".equals(System.getProperty("mcopt.shaders.water.glare")),
		REFRACT_ON = ON && !"false".equals(System.getProperty("mcopt.shaders.water.refract")),
		SHORE_ON = ON && !"false".equals(System.getProperty("mcopt.shaders.water.shore")),
		UW_SCATTER_ON = ON && !"false".equals(System.getProperty("mcopt.shaders.water.uwscatter")),
		UW_SHAFTS = ON && !"false".equals(System.getProperty("mcopt.shaders.water.uwshafts"));
	static final float GLARE_GAIN = prop("mcopt.shaders.water.glare.gain", 10.0f), GLARE_BLOOM = prop("mcopt.shaders.water.glare.bloom", 1.0f),
		GLARE_ROUGH = prop("mcopt.shaders.water.glare.rough", 0.3f), GLARE_SWELL = prop("mcopt.shaders.water.glare.swell", 0.2f),
		REFRACT_W = prop("mcopt.shaders.water.refract.weight", 0.4f), SHORE_GAIN = prop("mcopt.shaders.water.shore.gain", 0.5f),
		UW_SCAT_GAIN = prop("mcopt.shaders.water.uwscatter.gain", 1.0f), UW_MS = prop("mcopt.shaders.water.uwscatter.ms", 0.5f),
		UW_SKY = prop("mcopt.shaders.water.uwscatter.sky", 1.0f), UW_SHAFT = prop("mcopt.shaders.water.uwshafts.gain", 1.0f);
	/** The camera's depth below the water surface this frame (blocks), found by scanning up the water column. */
	private static float uwDepth;

	/** The HDR side channel is written this frame (the glare model on and the bloom chain there to read it). */
	static boolean hdrOn() {
		return GLARE_ON && OwnPost.BLOOM_ON && OwnProbe.bool("shaders.water.glare", true) && TILE && frameOn;
	}
	/** kind bit 10 */
	static final int K_WATER = 1 << 10;

	/** This frame is shaded for water (beginFrame), and its translucent draw shades water surfaces (beforeTranslucent). */
	private static boolean frameOn, surface, underwater, failed, said, splitSaid;
	private static float[] rects;
	private static boolean rectsTried;
	/** The underwater medium's distance absorption per block (rgb), before MURK: thin enough that far terrain shows through a gradual fade. */
	static final float[] MEDIUM = {0.11f, 0.045f, 0.03f};

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
		boolean on = ON && !failed && OwnProbe.bool("shaders.water", true);
		if (TILE) OwnNative.waterTile(t.shadeOwn(), 1, on ? 1 : 0);
		if (!on) return;
		underwater = camera.fogType == FogType.WATER;
		uwDepth = underwater ? waterDepthAt(camera) : 0f;
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
	static void beforeTranslucent(OwnTerrain t, Object encoder, CameraRenderState camera, Matrix4f proj, Matrix4f clip, boolean waterListed) {
		surface = false;
		if (!frameOn || !waterListed) return;  // (no water quads in the list: glass and ice draw through the plain shaded variant, no split, no copy)
		try {
			SkyRenderState sky = Minecraft.getInstance().gameRenderer.gameRenderState().levelRenderState.skyRenderState;
			boolean scene = false;
			if (TILE) {
				scene = viewPrev != null && OwnNative.waterTileScene(t.shadeOwn());
			} else if (SPLIT && OwnProbe.bool("shaders.water.split", true)) {
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
				System.out.println("mcopt-shaders: water on (" + (TILE ? "in the pass, the bed read in place, " + (STEPS > 0 ? "SSR " + STEPS + " steps over the previous frame" : "no SSR") : scene ? "split pass, scene copied at " + (COPY == 1 ? "half" : "full") + " resolution, " + (STEPS > 0 ? "SSR " + STEPS + " steps" : "no SSR")
					: "no scene") + ")");
			}
		} catch (RuntimeException | LinkageError e) {
			fail(e);
		}
	}

	/** Tile water: this frame's view (OwnTerrain.drawOpaque, every world frame; clip = projection x view rotation, camera-relative). */
	static void noteView(Matrix4f clip, CameraRenderState camera) {
		if (viewNow != null) {
			viewPrev = viewNow;
			posPrev.set(posNow);
		}
		viewNow = new Matrix4f(clip);
		posNow.set(camera.pos.x, camera.pos.y, camera.pos.z);
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
	private static synchronized float[] rects() {
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

	/**
	 * A meshing worker's translucent layer holds water when one of its vertices' uv lies in the water sprites' rectangles (any thread; conservatively true when the
	 * sprites cannot be found yet). v: the address of the layer's first vertex, n quads of QUAD_BYTES (4 vertices of VERTEX_BYTES, uv at +16).
	 */
	static boolean hasWater(long v, int n, int quadBytes) {
		if (!ON) return false;
		float[] r = rects();
		if (r == null) return true;
		for (int q = 0; q < n; q++) {
			long a = v + (long) q * quadBytes + 16;
			float u = MemoryUtil.memGetFloat(a), w = MemoryUtil.memGetFloat(a + 4);
			if (u >= r[0] - 1e-4f && u <= r[2] + 1e-4f && w >= r[1] - 1e-4f && w <= r[3] + 1e-4f
				|| u >= r[4] - 1e-4f && u <= r[6] + 1e-4f && w >= r[5] - 1e-4f && w <= r[7] + 1e-4f) return true;
		}
		return false;
	}

	/** Writes this frame's WaterFrame (terrain.metal) and hands it to the native side. */
	private static void write(OwnTerrain t, CameraRenderState camera, SkyRenderState sky, Matrix4f proj, Matrix4f clip, boolean scene) {
		try (MemoryStack stack = MemoryStack.stackPush()) {
			long b = stack.ncalloc(16, 1, FRAME_BYTES);
			RenderTarget main = Minecraft.getInstance().gameRenderer.mainRenderTarget();
			int w = main.width, h = main.height;
			float projA = 0, projB = 1;
			if (clip != null) {
				// the matrix the vertex stage multiplies by, with y flipped as it flips it (tile water: the previous frame's, from this frame's camera-relative positions)
				Matrix4f vp = new Matrix4f(TILE && viewPrev != null ? viewPrev : clip);
				if (TILE && viewPrev != null) vp.translate((float) (camera.pos.x - posPrev.x), (float) (camera.pos.y - posPrev.y), (float) (camera.pos.z - posPrev.z));
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
			put4(b + 176, 0.20f * ABSORB, 0.115f * ABSORB, 0.09f * ABSORB, 48.0f);
			put4(b + 192, 0.26f, 0.30f, 0.31f, BODY);
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
			put4(b + 304, MEDIUM[0] * MURK, MEDIUM[1] * MURK, MEDIUM[2] * MURK, COPY == 1 ? 0.5f : 1.0f);  // (w: the scene depth copy's pixels per screen pixel)
			if (TILE && viewPrev != null) put4(b + 320, (float) (camera.pos.x - posPrev.x), (float) (camera.pos.y - posPrev.y), (float) (camera.pos.z - posPrev.z), 0f);
			// track water (bytes 336-447)
			int tf = (GLARE_ON && OwnProbe.bool("shaders.water.glare", true) ? 1 : 0) | (hdrOn() ? 2 : 0) | (REFRACT_ON && OwnProbe.bool("shaders.water.refract", true) ? 4 : 0)
				| (SHORE_ON && OwnProbe.bool("shaders.water.shore", true) ? 8 : 0) | (UW_SCATTER_ON && OwnProbe.bool("shaders.water.uwscatter", true) ? 16 : 0);
			put4(b + 336, uwDepth, GLARE_GAIN, tf, REFRACT_W);
			put4(b + 352, UW_SCAT_GAIN, SHORE_GAIN, GLARE_ROUGH, GLARE_SWELL);
			put4(b + 368, UW_MS, UW_SKY, 0f, 0f);
			OwnNative.waterFrame(t.shadeOwn(), b, FRAME_BYTES);
		}
	}

	/** Blocks from the camera to the top of the water column above it (the lit surface), capped at 64. */
	private static float waterDepthAt(CameraRenderState camera) {
		try {
			net.minecraft.client.multiplayer.ClientLevel level = Minecraft.getInstance().level;
			if (level == null) return 0f;
			net.minecraft.core.BlockPos.MutableBlockPos bp = new net.minecraft.core.BlockPos.MutableBlockPos(Math.floor(camera.pos.x), Math.floor(camera.pos.y), Math.floor(camera.pos.z));
			int n = 0;
			while (n < 64 && level.getFluidState(bp).is(net.minecraft.tags.FluidTags.WATER)) {
				bp.move(0, 1, 0);
				n++;
			}
			return Math.max(0f, (float) (bp.getY() - 0.12 - camera.pos.y));
		} catch (RuntimeException | LinkageError e) {
			return 0f;
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
