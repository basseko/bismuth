package mcopt.metal.own;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.commands.RenderPass;
import mcopt.metal.MetalBridge;
import java.util.Optional;
import java.util.OptionalDouble;
import net.minecraft.client.renderer.GameRenderer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * The colour response pass of the shaders (-Dmcopt.shaders=true; milestone 2): exposure, an ACES-style tone curve, colour balance by the
 * time of day (OwnLight), shadow lift, saturation and contrast on the finished 3D picture, after the hand and before the GUI, so the
 * HUD and the menus are not touched.
 * <p>
 * It is one full-screen triangle drawn into the world's render encoder (a merged pass on the main target, which is still open at that
 * point), reading the pixel it replaces through programmable blending: no copy of the 6.6 MP target, no pass split, no extra store or load.
 * The cost is the fragment ALU over the screen. If the encoder was closed in between (a post effect, the entity outline) the vanilla pass
 * API reopens it with the colour loaded, which is the one split a plain post pass would always pay.
 */
public final class OwnPost {
	private static boolean failed, said, splitSaid;
	/** PostFrame's size (terrain.metal). */
	static final int BYTES = 160;
	/** -Dmcopt.shaders.bloom=0 turns bloom and the light shafts off for good (and the extra texture writes of the pass with them). */
	static final boolean BLOOM_ON = OwnLight.BLOOM > 0 || OwnLight.SHAFTS > 0;

	private OwnPost() {
	}

	/** Called by GameRenderer.render after the world and before the GUI. */
	public static void finish(GameRenderer renderer) {
		if (!OwnLight.GRADE || failed) return;
		OwnTerrain t = OwnShade.postOwner;
		OwnShade.postOwner = null;  // (one grade per world frame: OwnShade.frame arms it)
		if (t == null || !OwnProbe.bool("shaders.grade", true)) return;
		try (MemoryStack stack = MemoryStack.stackPush()) {
			long p = stack.ncalloc(16, 1, BYTES);
			java.nio.FloatBuffer b = MemoryUtil.memFloatBuffer(p, BYTES / 4);
			RenderTarget main = renderer.mainRenderTarget();
			OwnLight.postFrame(b, OwnProbe.bool("shaders.postnop", false), main.width, main.height, BLOOM_ON && OwnProbe.bool("shaders.bloom", true));
			// volumetric light (OwnVolume): the pass composites it; the screen-space shafts (PostFrame.bl.y) are toned down so they do not double up
			if (OwnVolume.ON && OwnVolume.prepare(t, renderer.gameRenderState().levelRenderState.cameraRenderState)) b.put(25, b.get(25) * OwnVolume.SHAFT_KEEP);
			long target = MetalBridge.textureHandle(main.getColorTexture());
			// the world's encoder is normally still open on the main target: draw straight into it. (If it is not, the vanilla pass API
			// reopens one with the colour loaded: the split a plain post pass would always pay.)
			boolean drawn = OwnNative.post(t.shadeOwn(), t.shadeEnc(), target, p, BYTES);
			if (!drawn) {
				try (RenderPass pass = RenderSystem.getDevice().createCommandEncoder().createRenderPass(() -> "mcopt shade post", main.getColorTextureView(), Optional.empty(),
					main.getDepthTextureView(), OptionalDouble.empty())) {
					OwnNative.post(t.shadeOwn(), t.shadeEnc(), target, p, BYTES);
				}
				if (!splitSaid) System.out.println("mcopt-shaders: the colour response pass had to reopen the world pass (a split)");
				splitSaid = true;
			}
			if (!said) {
				said = true;
				System.out.println("mcopt-shaders: colour response pass on (programmable blending, no pass split)" + (BLOOM_ON ? "; bloom + light shafts on (half-resolution source written by the pass, blurred next frame)" : ""));
			}
		} catch (RuntimeException e) {
			System.out.println("mcopt-shaders: colour response pass disabled: " + e);
			failed = true;
		}
	}
}
