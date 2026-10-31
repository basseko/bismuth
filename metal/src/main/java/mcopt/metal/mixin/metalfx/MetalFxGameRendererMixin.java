package mcopt.metal.mixin.metalfx;

import mcopt.metal.MetalFx;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** -Dmcopt.metalfx: the world renders into the reduced target and is upscaled before the GUI (see {@link MetalFx}). */
@Mixin(value = GameRenderer.class, priority = 1100)
abstract class MetalFxGameRendererMixin {
	@Inject(method = "render", at = @At("HEAD"))
	private void mcopt$fxHead(CallbackInfo ci) {
		MetalFx.end(((GameRenderer) (Object) this).mainRenderTarget());  // (a frame that was cut short leaves the reduced set in place)
	}

	@Inject(method = "renderLevel", at = @At("HEAD"))
	private void mcopt$fxBegin(CallbackInfo ci) {
		MetalFx.begin(((GameRenderer) (Object) this).mainRenderTarget());
	}

	/** After the shader grade (ShadePostMixin, priority 1000) at the same point, before the final depth clear and the GUI. */
	@Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/fog/FogRenderer;endFrame()V"))
	private void mcopt$fxEnd(CallbackInfo ci) {
		MetalFx.end(((GameRenderer) (Object) this).mainRenderTarget());
	}
}
