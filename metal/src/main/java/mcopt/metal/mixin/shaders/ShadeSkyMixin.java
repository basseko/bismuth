package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnLightmap;
import net.minecraft.client.Camera;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.SkyRenderer;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** The sky's colours by the time of day (data: the sky disc's colour and the sunrise / sunset glow). */
@Mixin(SkyRenderer.class)
abstract class ShadeSkyMixin {
	@Inject(method = "extractRenderState", at = @At("RETURN"))
	private void mcopt$shadeSky(ClientLevel level, float partialTicks, Camera camera, SkyRenderState state, CallbackInfo ci) {
		OwnLightmap.sky(state);
	}
}
