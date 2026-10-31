package mcopt.metal.own;

import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.world.level.dimension.DimensionType;
import org.joml.Matrix4f;
import org.joml.Vector3d;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * Our own shader effects (-Dmcopt.shaders=true, default off). Milestone 1: sun shadows and the directional light on terrain.
 * <p>
 * Per frame, from {@link OwnTerrain#drawOpaque} (after the camera cull, in the pre command buffer's time): the sun or moon direction and
 * strength from the sky state, then per cascade a light-space cull of our draw units (terrain.metal shade_cull: the cascade's box and the
 * facing bucket against the light), and a depth-only render of the selected solid and cutout units into a slice of a 2D-array depth map
 * (mcown.m mcs_shadow_pass, an encoder on the pre command buffer, so the world encoder is never split). The terrain's shaded fragment
 * variant (kind bit 8, function constant 30) reads the map and the frame block (ShadeFrame) and adds the sun with soft PCF shadows.
 * <p>
 * Stabilised: each cascade is an orthographic box of constant size whose centre is snapped to shadow texels in absolute world light space
 * (double precision), so shadows do not shimmer when the camera moves or turns.
 */
public final class OwnShade {
	/** -Dmcopt.shaders=true. */
	public static final boolean ON = Boolean.getBoolean("mcopt.shaders");
	/** Shadow map side, cascades and the farthest cascade's half extent (blocks). */
	static final int RES = Integer.getInteger("mcopt.shaders.shadow.res", 2048);
	static final int CASCADES = Math.max(1, Math.min(3, Integer.getInteger("mcopt.shaders.shadow.cascades", 2)));
	static final double DISTANCE = Double.parseDouble(System.getProperty("mcopt.shaders.shadow.distance", "128"));
	/** Soft edge radius (blocks) and the ambient light the sun replaces at full sun and sky light. */
	static final int TAPS = Integer.getInteger("mcopt.shaders.shadow.taps", 12);
	static final float SOFT = Float.parseFloat(System.getProperty("mcopt.shaders.shadow.soft", "0.10"));
	static final float AMBIENT_CUT = Float.parseFloat(System.getProperty("mcopt.shaders.ambientCut", "0.48"));
	static final float SUN_AMP = Float.parseFloat(System.getProperty("mcopt.shaders.sunAmp", "0.62"));
	/** The light-space depth range: blocks toward the light that still cast, and beyond the box's centre plane. */
	private static final double CASTER = 320.0, BEYOND = 192.0, DEPTH = CASTER + BEYOND;
	private static final int FRAME_BYTES = 288, CULL_BYTES = 160, CULLFRAME_BYTES = 144, DRAW_BYTES = OwnTerrain.DRAW_FRAME_BYTES;

	/** This frame draws the shaded terrain variant (set by {@link #frame}; read at the one choke point, OwnNative.draw). */
	static boolean frameActive;
	private static boolean said;

	// kind bits (mcown.m)
	static final int K_SHADE = 1 << 8;

	private OwnShade() {
	}

	/** The kind of a terrain draw with shading applied: the shaded variant for solid, cutout and translucent fragment kinds. */
	static int kind(int kind) {
		if (!frameActive) return kind;
		int f = kind & 15;
		if (f != 0 && f != 1 && f != 3) return kind;
		if ((kind & ((1 << 21) | (1 << 29) | (1 << 30) | (1 << 9))) != 0) return kind;  // mesh stages, the oracle, shadow draws: not shaded
		return kind | K_SHADE;
	}

	// ---- per terrain state ----
	private static long[][] args, lists, proj, terr;   // [ring][cascade]
	private static int listCap, ring;
	private static boolean failed;
	private static OwnTerrain owner;

	private static float smooth(float a, float b, float x) {
		float t = Math.max(0, Math.min(1, (x - a) / (b - a)));
		return t * t * (3 - 2 * t);
	}

	/**
	 * Called once per frame by drawOpaque before its draws: decides whether this frame is shaded, renders the shadow cascades into the
	 * pre command buffer and publishes the frame block. u: the draw's uniform handles (OwnTerrain.drawOpaque).
	 */
	static void frame(OwnTerrain t, CameraRenderState camera, long u, long d, long atlas, long atlasSampler, long light, long lightSampler, int k, int atlasW, int atlasH,
		boolean cpuClip, boolean fat) {
		frameActive = false;
		if (failed) return;
		Minecraft mc = Minecraft.getInstance();
		SkyRenderState sky = mc.gameRenderer.gameRenderState().levelRenderState.skyRenderState;
		if (sky.skybox != DimensionType.Skybox.OVERWORLD || !OwnProbe.bool("shaders", true)) return;
		if (fat) {
			if (!said) System.out.println("mcopt-shaders: not with -Dmcopt.own.fat (lists of instances), off");
			said = true;
			failed = true;
			return;
		}
		try {
			frame0(t, camera, sky, u, d, atlas, atlasSampler, light, lightSampler, k, atlasW, atlasH, cpuClip);
		} catch (RuntimeException e) {
			System.out.println("mcopt-shaders: disabled: " + e);
			e.printStackTrace(System.out);
			failed = true;
			frameActive = false;
		}
	}

	private static void frame0(OwnTerrain t, CameraRenderState camera, SkyRenderState sky, long u, long d, long atlas, long atlasSampler, long light, long lightSampler, int k,
		int atlasW, int atlasH, boolean cpuClip) {
		long own = t.shadeOwn(), enc = t.shadeEnc(), ctx = t.shadeCtx();
		int frames = OwnTerrain.FRAMES;
		if (owner != t) {
			owner = t;
			args = null;  // (a new terrain renderer: its own buffers)
		}
		if (!OwnNative.shadeInit(own, RES, CASCADES, frames)) throw new IllegalStateException("shadow map allocation failed");
		if (ring != frames || args == null) {
			args = new long[frames][CASCADES];
			lists = new long[frames][CASCADES];
			proj = new long[frames][CASCADES];
			terr = new long[frames][CASCADES];
			for (int i = 0; i < frames; i++)
				for (int c = 0; c < CASCADES; c++) {
					args[i][c] = OwnNative.buffer(ctx, 128, 0);
					proj[i][c] = OwnNative.buffer(ctx, 256, true);
					terr[i][c] = OwnNative.buffer(ctx, 256, true);
				}
			ring = frames;
			listCap = 0;
		}
		int cap = t.shadeListCap();
		if (cap != listCap) {
			for (int i = 0; i < frames; i++)
				for (int c = 0; c < CASCADES; c++) {
					long old = lists[i][c];
					if (old != 0) t.later(() -> OwnNative.release(old));
					lists[i][c] = OwnNative.buffer(ctx, 4L * cap * 2, false);
				}
			listCap = cap;
		}
		if (!said) {
			said = true;
			System.out.println("mcopt-shaders: on, shadow map " + RES + "x" + RES + " x" + CASCADES + " cascades, distance " + DISTANCE + " blocks");
		}

		// ---- the light ----
		double sa = sky.sunAngle, ma = sky.moonAngle;
		Vector3d sun = new Vector3d(-Math.sin(sa), Math.cos(sa), 0), moon = new Vector3d(-Math.sin(ma), Math.cos(ma), 0);
		float clear = Math.max(0, Math.min(1, sky.rainBrightness));
		float sunStr = smooth(0.0f, 0.12f, (float) sun.y) * clear * clear;
		float moonStr = 0.22f * smooth(0.0f, 0.2f, (float) moon.y) * (1 - smooth(0.0f, 0.12f, (float) sun.y)) * clear;
		boolean useSun = sunStr >= moonStr;
		Vector3d L = useSun ? sun : moon;
		float strength = useSun ? sunStr : moonStr;
		float el = (float) L.y;
		float cr, cg, cb;
		if (useSun) {
			float w = smooth(0.04f, 0.45f, el);
			cr = 1.0f;
			cg = 0.46f + (0.95f - 0.46f) * w;
			cb = 0.20f + (0.88f - 0.20f) * w;
		} else {
			cr = 0.55f;
			cg = 0.68f;
			cb = 1.0f;
		}
		float amp = useSun ? SUN_AMP : 0.45f;

		try (MemoryStack stack = MemoryStack.stackPush()) {
			long fb = stack.ncalloc(16, 1, FRAME_BYTES);
			boolean render = OwnProbe.bool("shaders.render", true), far = OwnProbe.bool("shaders.far", true);
			if (strength > 0.01f) {
				// light basis: F away from the light
				Vector3d F = new Vector3d(L).negate();
				Vector3d up = Math.abs(F.y) > 0.98 ? new Vector3d(1, 0, 0) : new Vector3d(0, 1, 0);
				Vector3d R = up.cross(F, new Vector3d()).normalize();
				Vector3d U = F.cross(R, new Vector3d()).normalize();
				double[] radius = CASCADES == 1 ? new double[] {DISTANCE} : CASCADES == 2 ? new double[] {DISTANCE * 0.3, DISTANCE}
					: new double[] {DISTANCE * 0.12, DISTANCE * 0.35, DISTANCE};
				double yaw = Math.toRadians(camera.yRot);
				double fx = -Math.sin(yaw), fz = Math.cos(yaw);
				Vector3d C = new Vector3d(camera.pos.x, camera.pos.y, camera.pos.z);
				double cR = C.dot(R), cU = C.dot(U);
				long cf = stack.ncalloc(16, 1, CULLFRAME_BYTES), sc = stack.ncalloc(16, 1, CULL_BYTES), df = stack.ncalloc(16, 1, DRAW_BYTES), uu = stack.ncalloc(8, 1, 64);
				MemoryUtil.memPutInt(cf + 120, listCap);
				MemoryUtil.memPutInt(cf + 128, OwnTerrain.RUN);
				Matrix4f m = new Matrix4f();
				for (int c = 0; c < CASCADES; c++) {
					double r = radius[c], tw = 2 * r / RES;
					double wx = camera.pos.x + fx * 0.3 * r, wz = camera.pos.z + fz * 0.3 * r;
					double sx = Math.floor((wx * R.x + camera.pos.y * R.y + wz * R.z) / tw) * tw, sy = Math.floor((wx * U.x + camera.pos.y * U.y + wz * U.z) / tw) * tw;
					double tx = cR - sx, ty = cU - sy;
					m.identity();
					m.m00((float) (R.x / r)).m01((float) (U.x / r)).m02((float) (F.x / DEPTH)).m03(0);
					m.m10((float) (R.y / r)).m11((float) (U.y / r)).m12((float) (F.y / DEPTH)).m13(0);
					m.m20((float) (R.z / r)).m21((float) (U.z / r)).m22((float) (F.z / DEPTH)).m23(0);
					m.m30((float) (tx / r)).m31((float) (ty / r)).m32((float) (CASTER / DEPTH)).m33(1);
					m.getToAddress(fb + c * 64L);
					// info: blocks per texel, 1 / depth range, 1 / res, normal offset
					MemoryUtil.memPutFloat(fb + 192 + c * 16L, (float) tw);
					MemoryUtil.memPutFloat(fb + 192 + c * 16L + 4, (float) (1.0 / DEPTH));
					MemoryUtil.memPutFloat(fb + 192 + c * 16L + 8, 1.0f / RES);
					MemoryUtil.memPutFloat(fb + 192 + c * 16L + 12, (float) (tw * 1.2));
					// cull volume: p.R + tx in [-r, r], p.U + ty in [-r, r], p.F + CASTER in [0, DEPTH] (+ margin)
					double mg = 1.0 + SOFT * 4;
					plane(sc, 0, R.x, R.y, R.z, tx + r + mg);
					plane(sc, 1, -R.x, -R.y, -R.z, -tx + r + mg);
					plane(sc, 2, U.x, U.y, U.z, ty + r + mg);
					plane(sc, 3, -U.x, -U.y, -U.z, -ty + r + mg);
					plane(sc, 4, F.x, F.y, F.z, CASTER + mg);
					plane(sc, 5, -F.x, -F.y, -F.z, DEPTH - CASTER + mg);
					int bx = camera.blockPos.getX(), by = camera.blockPos.getY(), bz = camera.blockPos.getZ();
					MemoryUtil.memPutInt(sc + 96, bx);
					MemoryUtil.memPutInt(sc + 100, by);
					MemoryUtil.memPutInt(sc + 104, bz);
					MemoryUtil.memPutFloat(sc + 112, (float) (bx - camera.pos.x));
					MemoryUtil.memPutFloat(sc + 116, (float) (by - camera.pos.y));
					MemoryUtil.memPutFloat(sc + 120, (float) (bz - camera.pos.z));
					MemoryUtil.memPutFloat(sc + 128, (float) L.x);
					MemoryUtil.memPutFloat(sc + 132, (float) L.y);
					MemoryUtil.memPutFloat(sc + 136, (float) L.z);
					MemoryUtil.memPutInt(sc + 144, t.shadeSectionCount());
					MemoryUtil.memPutInt(sc + 148, listCap);
					MemoryUtil.memPutInt(sc + 152, t.shadeRecCapacity());
					if (render && (far || c == 0) && !OwnNative.shadeCull(own, enc, cf, CULLFRAME_BYTES, sc, CULL_BYTES, t.shadeSections(k), t.shadeRecs(), args[k][c], lists[k][c], t.shadeRecCapacity()))
						throw new IllegalStateException("shade_cull missing");
					// the cascade's Projection and Terrain blocks for its draws
					long pa = OwnNative.contents(proj[k][c]), ta = OwnNative.contents(terr[k][c]);
					m.getToAddress(pa);
					new Matrix4f().getToAddress(ta);
					MemoryUtil.memPutInt(ta + 64, atlasW);
					MemoryUtil.memPutInt(ta + 68, atlasH);
				}
				for (int c = 0; c < CASCADES; c++) {
					MemoryUtil.memCopy(d, df, DRAW_BYTES);
					if (cpuClip) {
						MemoryUtil.memPutInt(df + 12, 1);
						m.setFromAddress(OwnNative.contents(proj[k][c]));
						m.getToAddress(df + 16);
					}
					MemoryUtil.memPutLong(uu, proj[k][c]);
					MemoryUtil.memPutLong(uu + 8, 0);
					MemoryUtil.memPutLong(uu + 16, MemoryUtil.memGetLong(u + 16));
					MemoryUtil.memPutLong(uu + 24, MemoryUtil.memGetLong(u + 24));
					MemoryUtil.memPutLong(uu + 32, terr[k][c]);
					MemoryUtil.memPutLong(uu + 40, 0);
					MemoryUtil.memPutLong(uu + 48, MemoryUtil.memGetLong(u + 48));
					MemoryUtil.memPutLong(uu + 56, MemoryUtil.memGetLong(u + 56));
					if (render && (far || c == 0)) OwnNative.shadePass(own, enc, k, c, df, DRAW_BYTES, t.shadeArena(), t.shadeSections(k), t.shadeRecs(), lists[k][c], args[k][c], listCap, uu, atlas,
						atlasSampler, light, lightSampler);
				}
			}
			// sunDir, sunColor, amb
			MemoryUtil.memPutFloat(fb + 240, (float) L.x);
			MemoryUtil.memPutFloat(fb + 244, (float) L.y);
			MemoryUtil.memPutFloat(fb + 248, (float) L.z);
			MemoryUtil.memPutFloat(fb + 252, strength > 0.01f && OwnProbe.bool("shaders.sun", true) ? strength : 0f);
			MemoryUtil.memPutFloat(fb + 256, cr * amp);
			MemoryUtil.memPutFloat(fb + 260, cg * amp);
			MemoryUtil.memPutFloat(fb + 264, cb * amp);
			MemoryUtil.memPutFloat(fb + 268, CASCADES);
			MemoryUtil.memPutFloat(fb + 272, AMBIENT_CUT * (useSun ? smooth(0.02f, 0.45f, el) : 0.5f));
			MemoryUtil.memPutFloat(fb + 276, SOFT);
			MemoryUtil.memPutFloat(fb + 280, TAPS);
			OwnNative.shadeFrame(own, k, fb, FRAME_BYTES, 0f, 0f);
		}
		frameActive = true;
	}

	private static void plane(long base, int i, double x, double y, double z, double w) {
		MemoryUtil.memPutFloat(base + i * 16L, (float) x);
		MemoryUtil.memPutFloat(base + i * 16L + 4, (float) y);
		MemoryUtil.memPutFloat(base + i * 16L + 8, (float) z);
		MemoryUtil.memPutFloat(base + i * 16L + 12, (float) w);
	}
}
