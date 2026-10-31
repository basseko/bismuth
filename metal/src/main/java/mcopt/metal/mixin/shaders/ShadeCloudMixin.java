package mcopt.metal.mixin.shaders;

import com.mojang.renderpearl.api.commands.RenderPass;
import mcopt.metal.own.OwnSky;
import net.minecraft.client.CloudStatus;
import net.minecraft.client.renderer.CloudRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Vanilla's flat-shaded cloud cells are skipped while our dome draws its own soft clouds. */
@Mixin(CloudRenderer.class)
abstract class ShadeCloudMixin {
	@Inject(method = "render", at = @At("HEAD"), cancellable = true)
	private void mcopt$noClouds(CloudStatus status, RenderPass renderPass, CallbackInfo ci) {
		if (OwnSky.ownClouds()) ci.cancel();
	}
}
