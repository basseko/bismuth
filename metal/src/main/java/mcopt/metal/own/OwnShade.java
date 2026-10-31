package mcopt.metal.own;

import mcopt.metal.EntityShade;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.world.level.dimension.DimensionType;
import net.minecraft.world.level.material.FogType;
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
	static final int CASCADES = Math.max(1, Math.min(3, Integer.getInteger("mcopt.shaders.shadow.cascades", 3)));
	static final double DISTANCE = Double.parseDouble(System.getProperty("mcopt.shaders.shadow.distance", "128"));
	/** The third (far, cached) cascade's half extent: the whole loaded terrain at RD16 and beyond (blocks). */
	static final double FAR = Double.parseDouble(System.getProperty("mcopt.shaders.shadow.far", "512"));
	/** Frames between rebuilds of the far cascade after a geometry change (the camera leaving it or the sun moving rebuilds at once). */
	static final int FAR_GAP = Integer.getInteger("mcopt.shaders.shadow.farGap", 30);
	private static final boolean LOG = Boolean.getBoolean("mcopt.shaders.log");
	/** Soft edge radius (blocks) and the ambient light the sun replaces at full sun and sky light. */
	static final int TAPS = Integer.getInteger("mcopt.shaders.shadow.taps", 12);
	static final float SOFT = Float.parseFloat(System.getProperty("mcopt.shaders.shadow.soft", "0.10"));
	static final float AMBIENT_CUT = Float.parseFloat(System.getProperty("mcopt.shaders.ambientCut", "0.25"));
	static final float FILL = Float.parseFloat(System.getProperty("mcopt.shaders.fill", "0.09"));
	/** 1: the baked vanilla face shade is replaced by a hemisphere ambient (up 1, side 0.6, down 0.4) wherever the sun applies; 0: kept. */
	static final float HEMI = Float.parseFloat(System.getProperty("mcopt.shaders.hemi", "1.0"));
	static final float SUN_AMP = Float.parseFloat(System.getProperty("mcopt.shaders.sunAmp", "0.72"));
	/** The light-space depth range: blocks toward the light that still cast, and beyond the box's centre plane. */
	private static final double CASTER = 320.0, BEYOND = 192.0;
	private static final int FRAME_BYTES = 672, CULL_BYTES = 160, CULLFRAME_BYTES = 144, DRAW_BYTES = OwnTerrain.DRAW_FRAME_BYTES;

	/** This frame draws the shaded terrain variant (set by {@link #frame}; read at the one choke point, OwnNative.draw). */
	static boolean frameActive;
	private static boolean said;
	/** The terrain renderer of the last frame (OwnPost needs its native handles). */
	static OwnTerrain postOwner;
	/** The terrain renderer of the last shaded frame, kept (postOwner is consumed by the colour pass each frame): the sky pass of the next frame draws with it. */
	static OwnTerrain lastOwner;

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
		if (OwnWater.ON) OwnWater.off();
		if (EntityShade.ON) EntityShade.begin();  // (milestone 5: entities are re-lit only in the frames where the block below is published)
		if (t.changedCount() > 0) geomDirty = true;
		if (failed) return;
		Minecraft mc = Minecraft.getInstance();
		SkyRenderState sky = mc.gameRenderer.gameRenderState().levelRenderState.skyRenderState;
		if (sky.skybox != DimensionType.Skybox.OVERWORLD || !OwnProbe.bool("shaders", true)) return;
		postOwner = t;
		lastOwner = t;
		OwnLight.update(sky.sunAngle, 1.0f - sky.rainBrightness, camera.fogType == FogType.WATER);
		if (OwnWater.ON && !fat) OwnWater.beginFrame(t, camera, sky);  // (the water's frame block: independent of the sun, so before the no-sun return below)
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

	/** A cascade's light-space frame: the basis and the snapped origin it was last (re)built for, and which ring slots hold a map rendered for it. */
	private static final class Casc {
		final Vector3d L = new Vector3d(), R = new Vector3d(), U = new Vector3d(), F = new Vector3d();
		double sx, sy, sf;
		boolean valid, dirty;
		long built;
		int version = 1;
		int[] slot;
	}

	private static Casc[] casc;
	/** The geometry (a section table entry) changed since the cached cascades were last invalidated for it. */
	private static boolean geomDirty;
	private static long frameNo;
	/** The cached cascades (all but the nearest) are rebuilt when the sun moves by more than this (cosine of 0.3 degrees), the camera drifts from the centre by more than DRIFT x radius. */
	private static final double REBUILD_COS = Math.cos(Math.toRadians(0.3)), DRIFT = 0.12, DRIFT_DEPTH = 64.0;
	/** The snapping quantum of a cached cascade's origin, in texels. */
	private static final int CACHE_QUANTUM = 16;
	static final boolean CACHE = !"false".equals(System.getProperty("mcopt.shaders.shadow.cache"));
	private static final boolean ALWAYS = Boolean.getBoolean("mcopt.shaders.always");

	private static void basis(Casc s, Vector3d L) {
		s.L.set(L);
		s.F.set(L).negate();
		Vector3d up = Math.abs(s.F.y) > 0.98 ? new Vector3d(1, 0, 0) : new Vector3d(0, 1, 0);
		up.cross(s.F, s.R).normalize();
		s.F.cross(s.R, s.U).normalize();
	}

	/** A shaded draw could not build its pipeline (OwnNative.draw already fell back to the plain kind for it): no shading from now on. */
	static void drawFailed(RuntimeException e) {
		if (!failed) System.out.println("mcopt-shaders: disabled: shaded terrain variant failed: " + e.getMessage());
		failed = true;
		frameActive = false;
	}

	private static void frame0(OwnTerrain t, CameraRenderState camera, SkyRenderState sky, long u, long d, long atlas, long atlasSampler, long light, long lightSampler, int k,
		int atlasW, int atlasH, boolean cpuClip) {
		// ---- the light ----
		double sa = sky.sunAngle, ma = sky.moonAngle;
		Vector3d sun = new Vector3d(-Math.sin(sa), Math.cos(sa), 0), moon = new Vector3d(-Math.sin(ma), Math.cos(ma), 0);
		float clear = Math.max(0, Math.min(1, sky.rainBrightness));
		float sunStr = smooth(0.0f, 0.12f, (float) sun.y) * clear * clear;
		float moonStr = 0.22f * smooth(0.0f, 0.2f, (float) moon.y) * (1 - smooth(0.0f, 0.12f, (float) sun.y)) * clear;
		boolean useSun = sunStr >= moonStr;
		Vector3d L = useSun ? sun : moon;
		float strength = useSun ? sunStr : moonStr;
		boolean sunOn = strength > 0.01f && OwnProbe.bool("shaders.sun", true);
		// nothing to add without direct light (night, heavy rain): the plain terrain variants draw, at no cost (unless the sky's haze needs the shaded variant)
		boolean haze = OwnSky.hazeActive() && OwnProbe.bool("shaders.haze", true);
		if (!sunOn && !ALWAYS && !haze) return;
		float el = (float) L.y;
		float cr, cg, cb;
		if (useSun) {
			cr = OwnLight.sun[0];
			cg = OwnLight.sun[1];
			cb = OwnLight.sun[2];
		} else {
			cr = 0.45f;
			cg = 0.60f;
			cb = 1.0f;
		}
		float amp = useSun ? SUN_AMP : 0.45f;

		long own = t.shadeOwn(), enc = t.shadeEnc(), ctx = t.shadeCtx();
		int frames = OwnTerrain.FRAMES;
		if (owner != t) {
			owner = t;
			args = null;  // (a new terrain renderer: its own buffers)
			casc = null;
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
			casc = null;
		}
		if (casc == null) {
			casc = new Casc[CASCADES];
			for (int c = 0; c < CASCADES; c++) {
				casc[c] = new Casc();
				casc[c].slot = new int[frames];
			}
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
			System.out.println("mcopt-shaders: on, shadow map " + RES + "x" + RES + " x" + CASCADES + " cascades, distance " + DISTANCE + " blocks, far cascades "
				+ (CACHE ? "cached" : "every frame"));
		}
		frameNo++;
		if (geomDirty) {
			for (Casc s : casc) s.dirty = true;
			geomDirty = false;
		}
		if (LOG && frameNo % 240 == 1) System.out.println("mcopt-shaders: sun " + (useSun ? "sun" : "moon") + " elevation " + Math.round(Math.toDegrees(Math.asin(L.y))) + " deg, strength " + strength);

		try (MemoryStack stack = MemoryStack.stackPush()) {
			long fb = stack.ncalloc(16, 1, FRAME_BYTES);
			boolean render = OwnProbe.bool("shaders.render", true), far = OwnProbe.bool("shaders.far", true);
			if (sunOn) {
				double[] radius = CASCADES == 1 ? new double[] {DISTANCE} : CASCADES == 2 ? new double[] {DISTANCE * 0.3, DISTANCE}
					: new double[] {DISTANCE * 0.3, DISTANCE, FAR};
				double yaw = Math.toRadians(camera.yRot);
				double fx = -Math.sin(yaw), fz = Math.cos(yaw);
				double cx = camera.pos.x, cy = camera.pos.y, cz = camera.pos.z;
				long cf = stack.ncalloc(16, 1, CULLFRAME_BYTES), sc = stack.ncalloc(16, 1, CULL_BYTES), df = stack.ncalloc(16, 1, DRAW_BYTES), uu = stack.ncalloc(8, 1, 64);
				MemoryUtil.memPutInt(cf + 120, listCap);
				MemoryUtil.memPutInt(cf + 128, OwnTerrain.RUN);
				Matrix4f m = new Matrix4f();
				boolean[] need = new boolean[CASCADES];
				for (int c = 0; c < CASCADES; c++) {
					Casc s = casc[c];
					double r = radius[c], tw = 2 * r / RES;
					double DEPTH = CASTER + Math.max(BEYOND, r * 1.05);
					boolean cached = CACHE && c > 0;
					if (!cached) {
						// the nearest cascade: rebuilt every frame, centred ahead of the camera, its origin snapped to texels
						basis(s, L);
						double wx = cx + fx * 0.3 * r, wz = cz + fz * 0.3 * r;
						s.sx = Math.floor((wx * s.R.x + cy * s.R.y + wz * s.R.z) / tw) * tw;
						s.sy = Math.floor((wx * s.U.x + cy * s.U.y + wz * s.U.z) / tw) * tw;
						s.sf = cx * s.F.x + cy * s.F.y + cz * s.F.z;
						s.valid = true;
					} else {
						double dR = cx * s.R.x + cy * s.R.y + cz * s.R.z - s.sx, dU = cx * s.U.x + cy * s.U.y + cz * s.U.z - s.sy;
						double dF = cx * s.F.x + cy * s.F.y + cz * s.F.z - s.sf;
						if (!s.valid || s.L.dot(L) < REBUILD_COS || Math.abs(dR) > DRIFT * r || Math.abs(dU) > DRIFT * r || Math.abs(dF) > Math.max(DRIFT_DEPTH, DRIFT * r)) {
							// the sun moved or the camera drifted: a new frame, snapped to a multiple of CACHE_QUANTUM texels (centred on the camera: turning does not matter)
							basis(s, L);
							double q = tw * CACHE_QUANTUM;
							s.sx = Math.floor((cx * s.R.x + cy * s.R.y + cz * s.R.z) / q) * q;
							s.sy = Math.floor((cx * s.U.x + cy * s.U.y + cz * s.U.z) / q) * q;
							s.sf = Math.floor((cx * s.F.x + cy * s.F.y + cz * s.F.z) / q) * q;
							s.valid = true;
							s.version++;
							s.dirty = false;
							s.built = frameNo;
						} else if (s.dirty && frameNo - s.built >= (c == CASCADES - 1 && c >= 2 ? FAR_GAP : 0)) {
							s.version++;
							s.dirty = false;
							s.built = frameNo;
						}
					}
					double cR = cx * s.R.x + cy * s.R.y + cz * s.R.z, cU = cx * s.U.x + cy * s.U.y + cz * s.U.z, cF = cx * s.F.x + cy * s.F.y + cz * s.F.z;
					double tx = cR - s.sx, ty = cU - s.sy, tf = cF - s.sf + CASTER;
					need[c] = !cached || s.slot[k] != s.version;
					m.identity();
					m.m00((float) (s.R.x / r)).m01((float) (s.U.x / r)).m02((float) (s.F.x / DEPTH)).m03(0);
					m.m10((float) (s.R.y / r)).m11((float) (s.U.y / r)).m12((float) (s.F.y / DEPTH)).m13(0);
					m.m20((float) (s.R.z / r)).m21((float) (s.U.z / r)).m22((float) (s.F.z / DEPTH)).m23(0);
					m.m30((float) (tx / r)).m31((float) (ty / r)).m32((float) (tf / DEPTH)).m33(1);
					m.getToAddress(fb + c * 64L);
					// info: blocks per texel, 1 / depth range, 1 / res, normal offset
					MemoryUtil.memPutFloat(fb + 192 + c * 16L, (float) tw);
					MemoryUtil.memPutFloat(fb + 192 + c * 16L + 4, (float) (1.0 / DEPTH));
					MemoryUtil.memPutFloat(fb + 192 + c * 16L + 8, 1.0f / RES);
					MemoryUtil.memPutFloat(fb + 192 + c * 16L + 12, (float) (tw * 1.2));
					if (!need[c] || !render || (!far && c > 0)) continue;
					// cull volume: p.R + tx in [-r, r], p.U + ty in [-r, r], p.F + tf in [0, DEPTH] (+ margin)
					double mg = 1.0 + SOFT * 4;
					plane(sc, 0, s.R.x, s.R.y, s.R.z, tx + r + mg);
					plane(sc, 1, -s.R.x, -s.R.y, -s.R.z, -tx + r + mg);
					plane(sc, 2, s.U.x, s.U.y, s.U.z, ty + r + mg);
					plane(sc, 3, -s.U.x, -s.U.y, -s.U.z, -ty + r + mg);
					plane(sc, 4, s.F.x, s.F.y, s.F.z, tf + mg);
					plane(sc, 5, -s.F.x, -s.F.y, -s.F.z, DEPTH - tf + mg);
					int bx = camera.blockPos.getX(), by = camera.blockPos.getY(), bz = camera.blockPos.getZ();
					MemoryUtil.memPutInt(sc + 96, bx);
					MemoryUtil.memPutInt(sc + 100, by);
					MemoryUtil.memPutInt(sc + 104, bz);
					MemoryUtil.memPutFloat(sc + 112, (float) (bx - camera.pos.x));
					MemoryUtil.memPutFloat(sc + 116, (float) (by - camera.pos.y));
					MemoryUtil.memPutFloat(sc + 120, (float) (bz - camera.pos.z));
					MemoryUtil.memPutFloat(sc + 128, (float) s.L.x);
					MemoryUtil.memPutFloat(sc + 132, (float) s.L.y);
					MemoryUtil.memPutFloat(sc + 136, (float) s.L.z);
					MemoryUtil.memPutInt(sc + 144, t.shadeSectionCount());
					MemoryUtil.memPutInt(sc + 148, listCap);
					MemoryUtil.memPutInt(sc + 152, t.shadeRecCapacity());
					if (!OwnNative.shadeCull(own, enc, cf, CULLFRAME_BYTES, sc, CULL_BYTES, t.shadeSections(k), t.shadeRecs(), args[k][c], lists[k][c], t.shadeRecCapacity()))
						throw new IllegalStateException("shade_cull missing");
					// the cascade's Projection and Terrain blocks for its draws
					long pa = OwnNative.contents(proj[k][c]), ta = OwnNative.contents(terr[k][c]);
					m.getToAddress(pa);
					new Matrix4f().getToAddress(ta);
					MemoryUtil.memPutInt(ta + 64, atlasW);
					MemoryUtil.memPutInt(ta + 68, atlasH);
				}
				for (int c = 0; c < CASCADES; c++) {
					if (!need[c] || !render || (!far && c > 0)) continue;
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
					OwnNative.shadePass(own, enc, k, c, df, DRAW_BYTES, t.shadeArena(), t.shadeSections(k), t.shadeRecs(), lists[k][c], args[k][c], listCap, uu, atlas,
						atlasSampler, light, lightSampler);
					casc[c].slot[k] = casc[c].version;
				}
			}
			// sunDir, sunColor, amb
			MemoryUtil.memPutFloat(fb + 240, (float) L.x);
			MemoryUtil.memPutFloat(fb + 244, (float) L.y);
			MemoryUtil.memPutFloat(fb + 248, (float) L.z);
			MemoryUtil.memPutFloat(fb + 252, sunOn ? strength : 0f);
			MemoryUtil.memPutFloat(fb + 256, cr * amp);
			MemoryUtil.memPutFloat(fb + 260, cg * amp);
			MemoryUtil.memPutFloat(fb + 264, cb * amp);
			MemoryUtil.memPutFloat(fb + 268, CASCADES);
			MemoryUtil.memPutFloat(fb + 272, AMBIENT_CUT * (useSun ? smooth(0.02f, 0.45f, el) : 0.5f));
			MemoryUtil.memPutFloat(fb + 276, SOFT);
			MemoryUtil.memPutFloat(fb + 280, HEMI);
			MemoryUtil.memPutFloat(fb + 284, FILL);
			MemoryUtil.memPutFloat(fb + 300, OwnLight.waterDim());
			MemoryUtil.memPutFloat(fb + 316, OwnLight.waterTintK());
			// shade tint (what the sun does not reach) and the sky fill's colour
			for (int c = 0; c < 3; c++) {
				MemoryUtil.memPutFloat(fb + 288 + c * 4L, useSun ? OwnLight.shade[c] : 0.84f + 0.08f * c);
				MemoryUtil.memPutFloat(fb + 304 + c * 4L, useSun ? OwnLight.ambient[c] : 0.84f + 0.08f * c);
			}
			// the sky's haze: density, strength (0: vanilla's fog colour), the table's exposure; the true sun's direction and the rain
			if (haze) {
				MemoryUtil.memPutFloat(fb + 320, OwnSky.hazeDensity());
				MemoryUtil.memPutFloat(fb + 324, OwnSky.HAZE_STRENGTH);
				MemoryUtil.memPutFloat(fb + 328, OwnSky.exposure);
			}
			MemoryUtil.memPutFloat(fb + 336, OwnSky.sunDir[0]);
			MemoryUtil.memPutFloat(fb + 340, OwnSky.sunDir[1]);
			MemoryUtil.memPutFloat(fb + 344, OwnSky.sunDir[2]);
			MemoryUtil.memPutFloat(fb + 348, OwnSky.rain);
			for (int c = 0; c < 3; c++) MemoryUtil.memPutFloat(fb + 352 + c * 4L, OwnSky.warmTint[c]);
			MemoryUtil.memPutFloat(fb + 364, OwnSky.warm);
			// the clouds track (640..671): the cloud shadows
			if (OwnSky.shadowK > 0f && OwnSky.hazeActive()) {
				MemoryUtil.memPutFloat(fb + 640, OwnSky.cloudMid() - (float) camera.pos.y);
				MemoryUtil.memPutFloat(fb + 644, OwnSky.shadowK);
				MemoryUtil.memPutFloat(fb + 648, OwnSky.COV_R);
				MemoryUtil.memPutFloat(fb + 652, OwnSky.shadowCover);
				MemoryUtil.memPutFloat(fb + 656, OwnSky.CLOUD_SOFT);
				MemoryUtil.memPutFloat(fb + 660, OwnSky.shadowDens);
			}
			OwnNative.shadeFrame(own, k, fb, FRAME_BYTES, 0f, 0f);
			// milestone 5: entities and items (vanilla pipelines) take the same block and shadow map; cascade 0 also takes their casters
			if (EntityShade.ON && OwnProbe.bool("shaders.entities", true)) EntityShade.publish(fb, k, sunOn && render);
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
