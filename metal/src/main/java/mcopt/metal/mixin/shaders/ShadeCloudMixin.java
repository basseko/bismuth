package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnClouds;
import net.minecraft.client.renderer.extract.LevelExtractor;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Vanilla's clouds take the time of day's light colour (data: the cloud colour of the render state; the cloud mesh and shader stay vanilla's). */
@Mixin(LevelExtractor.class)
abstract class ShadeCloudMixin {
	@Shadow
	@Final
	private LevelRenderState levelRenderState;

	@Inject(method = "extract", at = @At("RETURN"))
	private void mcopt$shadeClouds(CallbackInfo ci) {
		OwnClouds.tint(this.levelRenderState);
	}
}
