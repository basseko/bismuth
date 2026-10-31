package mcopt.metal.mixin.shaders;

import mcopt.metal.EntityShade;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Phase boundaries of the entity shading (EntityShade): the first-person hand receives the sun and shadows (its positions are camera-relative and
 * world-aligned) but casts nothing; the GUI that follows (3D items, entity previews: their own spaces) is not shaded at all.
 */
@Mixin(GameRenderer.class)
abstract class ShadeHandMixin {
	@Inject(method = "render3dHud", at = @At("HEAD"))
	private void mcopt$shadeHand(CallbackInfo ci) {
		EntityShade.hand();
	}

	@Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/fog/FogRenderer;endFrame()V"))
	private void mcopt$shadeGui(CallbackInfo ci) {
		EntityShade.gui();
	}
}
