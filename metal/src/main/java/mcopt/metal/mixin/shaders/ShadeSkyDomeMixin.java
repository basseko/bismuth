package mcopt.metal.mixin.shaders;

import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.renderpearl.api.commands.RenderPass;
import mcopt.metal.own.OwnSky;
import net.minecraft.client.renderer.SkyRenderer;
import org.joml.Vector3fc;
import org.joml.Vector4fc;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Our sky dome (OwnSky) takes the place of vanilla's flat sky disc in the sky pass; while it draws, vanilla's sunrise / sunset fan and its sun quad are
 * skipped (the dome has the glow and the disc). The moon and the stars stay vanilla's.
 */
@Mixin(SkyRenderer.class)
abstract class ShadeSkyDomeMixin {
	@Inject(method = "renderSkyDisc", at = @At("HEAD"), cancellable = true)
	private void mcopt$dome(RenderPass renderPass, Vector3fc skyColor, CallbackInfo ci) {
		if (OwnSky.draw()) ci.cancel();
	}

	@Inject(method = "renderSunriseAndSunset", at = @At("HEAD"), cancellable = true)
	private void mcopt$noFan(RenderPass renderPass, PoseStack poseStack, float sunAngle, Vector4fc color, CallbackInfo ci) {
		if (OwnSky.isActive()) ci.cancel();
	}

	@Inject(method = "renderSun", at = @At("HEAD"), cancellable = true)
	private void mcopt$noSun(RenderPass renderPass, float rainBrightness, PoseStack poseStack, CallbackInfo ci) {
		if (OwnSky.isActive()) ci.cancel();
	}
}
