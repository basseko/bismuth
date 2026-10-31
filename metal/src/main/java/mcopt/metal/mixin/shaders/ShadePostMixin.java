package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnPost;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** The colour response pass runs after the world (hand, post effects) and before the GUI. */
@Mixin(GameRenderer.class)
abstract class ShadePostMixin {
	@Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/fog/FogRenderer;endFrame()V"))
	private void mcopt$shadePost(CallbackInfo ci) {
		OwnPost.finish((GameRenderer) (Object) this);
	}
}
