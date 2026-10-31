package mcopt.metal.own;

import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.tags.BiomeTags;
import net.minecraft.world.level.material.FogType;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * Volumetric light of the shaders (-Dmcopt.shaders.volumetric, default on with the shaders): sun rays through trees and a haze that glows toward the sun, without a
 * pass split. A compute kernel in the pre command buffer (terrain.metal vol_march, started from mcs_post) marches each pixel of a quarter-resolution grid from the
 * camera to the scene distance of the previous frame (the half-size scene copy the colour pass writes; it is written whenever this is on too), in an exponential
 * height fog, with one hard shadow compare per step against the frame's cascades; post_fs composites the result in linear light with a depth-aware upsample.
 * One frame of latency in the distances. This class decides whether the frame has it and fills the kernel's VolFrame (128 bytes).
 * <p>
 * Strength: subtle at noon, strong at a low sun (sunrise, sunset), more in forests (the camera's biome is in a forest / taiga / jungle tag), the moon's faint
 * rays at night, none under water, in rain (the light's strength falls with the clear weather) or without a sun. Probe key: {@code shaders.volumetric}.
 */
public final class OwnVolume {
	public static final boolean ON = OwnShade.ON && !"false".equals(System.getProperty("mcopt.shaders.volumetric"));
	/** The extinction per block at the reference height at noon, the height scale (blocks), the reference height and the march's range. */
	static final float DENSITY = prop("mcopt.shaders.volumetric.density", 0.0012f), SCALE = prop("mcopt.shaders.volumetric.scale", 45f),
		BASE_Y = prop("mcopt.shaders.volumetric.baseY", 64f), RANGE = prop("mcopt.shaders.volumetric.range", 110f);
	/** The light's in-scatter intensity and the ambient in-scatter's. */
	static final float LIGHT = prop("mcopt.shaders.volumetric.light", 0.55f), AMBIENT = prop("mcopt.shaders.volumetric.ambient", 0.08f);
	/** Extra density at a low sun, in forests. */
	static final float LOW_SUN = prop("mcopt.shaders.volumetric.lowSun", 1.0f), FOREST = prop("mcopt.shaders.volumetric.forest", 0.8f);
	/** The screen-space shafts (bloom_up) keep this fraction of their strength while the volumetric light is on. */
	static final float SHAFT_KEEP = prop("mcopt.shaders.volumetric.shafts", 0.25f);
	/** The composite's strength over the sky pixels (the dome has its own atmosphere). */
	static final float SKY = prop("mcopt.shaders.volumetric.sky", 0.3f);
	static final int BYTES = 128;

	private static boolean failed, said, lastOn;
	private static float forest;
	private static int biomeTick;
	private static double lx, ly, lz;

	private OwnVolume() {
	}

	private static float prop(String key, float dflt) {
		String v = System.getProperty(key);
		return v == null ? dflt : Float.parseFloat(v);
	}

	private static float smooth(float a, float b, float x) {
		float t = Math.max(0, Math.min(1, (x - a) / (b - a)));
		return t * t * (3 - 2 * t);
	}

	/**
	 * Called by OwnPost.finish before the colour pass: publishes this frame's request to the native side. Returns whether the frame has volumetric light (the
	 * screen-space shafts are then toned down).
	 */
	static boolean prepare(OwnTerrain t, CameraRenderState cam) {
		if (!ON || failed) return false;
		boolean air = cam.fogType == FogType.ATMOSPHERIC || cam.fogType == FogType.NONE;
		float str = OwnShade.lightStrength;
		boolean on = OwnShade.frameActive && air && str > 0.01f && OwnProbe.bool("shaders.volumetric", true);
		try {
			if (!on) {
				lastOn = false;
				OwnNative.volFrame(t.shadeOwn(), 0, 0, 0);
				return false;
			}
			boolean jump = Math.abs(cam.pos.x - lx) + Math.abs(cam.pos.y - ly) + Math.abs(cam.pos.z - lz) > 8.0;
			lx = cam.pos.x;
			ly = cam.pos.y;
			lz = cam.pos.z;
			boolean hist = lastOn && !jump;
			lastOn = true;
			float el = OwnSky.sinEl;
			boolean sun = OwnShade.lightIsSun;
			float lowSun = 1.0f - smooth(0.05f, 0.6f, el);
			// the forest: eased toward the camera's biome (checked a few times a second)
			if ((biomeTick++ & 7) == 0) {
				float target = 0f;
				try {
					var level = Minecraft.getInstance().level;
					if (level != null) {
						var b = level.getBiome(BlockPos.containing(cam.pos.x, cam.pos.y, cam.pos.z));
						target = b.is(BiomeTags.IS_FOREST) || b.is(BiomeTags.IS_TAIGA) || b.is(BiomeTags.IS_JUNGLE) ? 1f : 0f;
					}
				} catch (RuntimeException | LinkageError e) {
					target = 0f;
				}
				forest += (target - forest) * 0.15f;
			}
			float dens = DENSITY * (1.0f + LOW_SUN * lowSun * (sun ? 1f : 0f)) * (1.0f + FOREST * forest);
			float[] tint = OwnSky.sunTrans;
			float m = Math.max(1e-3f, Math.max(tint[0], Math.max(tint[1], tint[2])));
			float[] lc = sun ? new float[] {tint[0] / m, tint[1] / m, tint[2] / m} : new float[] {0.45f, 0.60f, 1.0f};
			// the light: the shadow block's strength (the sun's 1, the moon's 0.22; weather folded in), a little stronger when low (its light crosses the same fog)
			float li = LIGHT * str * (sun ? 1.0f + 0.6f * lowSun : 1.6f);
			float ai = AMBIENT * smooth(-0.15f, 0.2f, el) * str;
			float[] amb = OwnLight.ambient;
			Matrix4f proj = cam.projectionMatrix, view = cam.viewRotationMatrix;
			float tx = 1.0f / proj.m00(), ty = 1.0f / proj.m11();
			try (MemoryStack stack = MemoryStack.stackPush()) {
				long p = stack.ncalloc(16, 1, BYTES);
				put(p, 0, -view.m02(), -view.m12(), -view.m22(), (float) cam.pos.y);
				put(p, 16, view.m00() * tx, view.m10() * tx, view.m20() * tx, 0f);
				put(p, 32, view.m01() * ty, view.m11() * ty, view.m21() * ty, 0f);
				put(p, 48, dens, SCALE, BASE_Y, RANGE);
				put(p, 64, 14f, (float) ((frame++ & 15) * 0.0625), SKY, 0f);
				put(p, 80, lc[0] * li, lc[1] * li, lc[2] * li, 0f);
				// (in the sky look of the ambient: the brightest component 1; squared to the linear light the composite adds into)
				put(p, 96, amb[0] * amb[0] * ai, amb[1] * amb[1] * ai, amb[2] * amb[2] * ai, 0f);
				OwnNative.volFrame(t.shadeOwn(), p, BYTES, hist ? 1 : 0);
			}
		} catch (RuntimeException | LinkageError e) {
			System.out.println("mcopt-shaders: volumetric light disabled: " + e);
			failed = true;
			return false;
		}
		if (!said) {
			said = true;
			System.out.println("mcopt-shaders: volumetric light on (quarter-resolution march of the previous frame's scene distance, shadow-mapped, composited in the colour pass)");
		}
		return true;
	}

	private static int frame;

	private static void put(long p, int off, float a, float b, float c, float d) {
		MemoryUtil.memPutFloat(p + off, a);
		MemoryUtil.memPutFloat(p + off + 4, b);
		MemoryUtil.memPutFloat(p + off + 8, c);
		MemoryUtil.memPutFloat(p + off + 12, d);
	}
}
