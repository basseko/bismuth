package mcopt.metal.own;

import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.util.ARGB;
import net.minecraft.world.level.dimension.DimensionType;

/**
 * Vanilla clouds in the shaders' light (milestone 5): the cloud colour of the frame's render state is multiplied by a tint taken from the time of day
 * (the sun's colour and the sky ambient, OwnLight), so clouds are warm at a low sun, cool at night and grey in rain like the rest of the picture. The
 * colour response pass grades the clouds with everything else; this only moves them toward the light that falls on them. Our own cloud layer (m3) may
 * replace the vanilla one; this is then unused.
 * <p>
 * -Dmcopt.shaders.cloudTint=K (0.5; 0 off) is how far the tint is applied.
 */
public final class OwnClouds {
	private static final float K = Float.parseFloat(System.getProperty("mcopt.shaders.cloudTint", "0.5"));

	private OwnClouds() {
	}

	/** After LevelExtractor.extract: recolours state.cloudColor (ARGB, alpha untouched). */
	public static void tint(LevelRenderState state) {
		if (K <= 0 || state.skyRenderState.skybox != DimensionType.Skybox.OVERWORLD || !OwnProbe.bool("shaders", true)) return;
		int argb = state.cloudColor;
		if (ARGB.alpha(argb) <= 0) return;
		float[] s = OwnLight.sun, a = OwnLight.ambient;
		float sm = Math.max(0.05f, Math.max(s[0], Math.max(s[1], s[2]))), am = Math.max(0.05f, Math.max(a[0], Math.max(a[1], a[2])));
		float r = ARGB.red(argb), g = ARGB.green(argb), b = ARGB.blue(argb);
		float[] c = {r, g, b};
		for (int i = 0; i < 3; i++) {
			float t = 0.6f * (s[i] / sm) + 0.4f * (a[i] / am);
			c[i] = Math.min(255f, c[i] * (1.0f + (t - 1.0f) * K));
		}
		state.cloudColor = ARGB.color(ARGB.alpha(argb), Math.round(c[0]), Math.round(c[1]), Math.round(c[2]));
	}
}
