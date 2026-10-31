package mcopt.metal.mixin.shaders;

import com.mojang.blaze3d.vertex.PoseStack;
import mcopt.metal.own.OwnWater;
import net.minecraft.client.renderer.ScreenEffectRenderer;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.state.level.PlayerRenderState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** The blue screen overlay vanilla draws over the whole view under water is dropped: the underwater medium of the terrain and the fog colour do that job (-Dmcopt.shaders.water.overlay=true keeps it). */
@Mixin(ScreenEffectRenderer.class)
abstract class ShadeWaterOverlayMixin {
	@Inject(method = "submitWater", at = @At("HEAD"), cancellable = true)
	private static void mcopt$noWaterOverlay(PlayerRenderState.WaterOverlay waterOverlay, PoseStack poseStack, SubmitNodeCollector submitNodeCollector, CallbackInfo ci) {
		if (!OwnWater.OVERLAY && OwnWater.dataOn()) ci.cancel();
	}
}
