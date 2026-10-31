package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnLightmap;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.LightmapRenderStateExtractor;
import net.minecraft.client.renderer.state.LightmapRenderState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** The lightmap's inputs coloured by the time of day: the sky light's colour and the block light's tint (data, so every pipeline follows). */
@Mixin(LightmapRenderStateExtractor.class)
abstract class ShadeLightmapMixin {
	@Shadow
	@Final
	private GameRenderer renderer;
	@Shadow
	@Final
	private Minecraft minecraft;

	@Inject(method = "extract", at = @At("RETURN"))
	private void mcopt$shadeLightmap(LightmapRenderState state, float partialTicks, CallbackInfo ci) {
		OwnLightmap.apply(state, this.minecraft, this.renderer, partialTicks);
	}
}
