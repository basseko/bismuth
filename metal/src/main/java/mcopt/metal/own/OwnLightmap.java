package mcopt.metal.own;

import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.LightmapRenderState;
import net.minecraft.world.attribute.EnvironmentAttributes;
import net.minecraft.world.level.dimension.DimensionType;
import net.minecraft.world.level.material.FogType;
import org.joml.Vector3f;
import org.joml.Vector3fc;

/**
 * The lightmap's inputs, coloured by the time of day (milestone 2): the lightmap texture is data-driven (vanilla's lightmap.fsh from the
 * sky light colour and factor, the block light tint and factor, the ambient colour), so changing these inputs re-colours the terrain's
 * ambient, entities, particles and the hand at once, with no shader change. Called after each extraction of the lightmap state.
 */
public final class OwnLightmap {
	private static final boolean LOG = Boolean.getBoolean("mcopt.shaders.log");
	/** The block light's tint: a warm torch colour (vanilla's is much closer to white). */
	private static final Vector3fc WARM = new Vector3f(1.0f, 0.66f, 0.36f);
	/** Night: how much of the vanilla sky factor stays at the moon's level, and the extra factor that keeps moonlit terrain readable. */
	private static final float NIGHT_BOOST = Float.parseFloat(System.getProperty("mcopt.shaders.nightBoost", "1.35"));
	private static long n;

	private OwnLightmap() {
	}

	/** Sky state of this frame (SkyRenderer.extractRenderState RETURN): the zenith colour by the time of day; the glow of a low sun is kept as vanilla has it. */
	public static void sky(net.minecraft.client.renderer.state.level.SkyRenderState state) {
		if (state.skybox != DimensionType.Skybox.OVERWORLD || state.skyColor == null || !OwnProbe.bool("shaders", true)) return;
		OwnLight.update(state.sunAngle, 1.0f - state.rainBrightness, OwnLight.water);
		float[] t = OwnLight.skyColor();
		// a clear sky takes the table's colour; in rain it is dimmed and greyed
		float rb = Math.max(0.0f, Math.min(1.0f, state.rainBrightness)), grey = (t[0] * 0.3f + t[1] * 0.59f + t[2] * 0.11f) * (0.75f + 0.25f * rb);
		float s = 1.0f;
		for (int c = 0; c < 3; c++) t[c] = (grey + (t[c] - grey) * (0.4f + 0.6f * rb)) * (0.7f + 0.3f * rb);
		state.skyColor = new Vector3f(t[0] * s, t[1] * s, t[2] * s);
	}

	public static void apply(LightmapRenderState st, Minecraft mc, GameRenderer renderer, float partial) {
		if (!st.needsUpdate || !OwnProbe.bool("shaders", true)) return;
		ClientLevel level = mc.level;
		if (level == null || level.dimensionType().skybox() != DimensionType.Skybox.OVERWORLD) return;
		Camera cam = renderer.mainCamera();
		float sunAngle = cam.attributeProbe().getValue(EnvironmentAttributes.SUN_ANGLE, partial) * (float) (Math.PI / 180.0);
		OwnLight.update(sunAngle, level.getRainLevel(partial), cam.getFluidInCamera() == FogType.WATER);
		if (LOG && (n++ % 100) == 0) {
			System.out.println("mcopt-shaders: lightmap in: skyFactor " + st.skyFactor + " skyColor " + st.skyLightColor + " ambient " + st.ambientColor + " blockTint "
				+ st.blockLightTint + " blockFactor " + st.blockFactor + " sun el " + OwnLight.el);
		}
		float[] a = OwnLight.ambient();
		Vector3fc c = st.skyLightColor;
		st.skyLightColor = new Vector3f(c.x() * a[0], c.y() * a[1], c.z() * a[2]);
		st.blockLightTint = WARM;
		// the moon's light is faint but not nothing: a lift of the sky factor when the sun is down
		float night = 1.0f - Math.max(0.0f, Math.min(1.0f, (OwnLight.el + 0.12f) / 0.2f));
		if (night > 0) st.skyFactor *= 1.0f + (NIGHT_BOOST - 1.0f) * night;
	}
}
