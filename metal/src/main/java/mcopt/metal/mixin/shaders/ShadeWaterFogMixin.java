package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnWater;
import net.minecraft.client.Camera;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.fog.FogRenderer;
import net.minecraft.world.level.dimension.DimensionType;
import net.minecraft.world.level.material.FogType;
import org.joml.Vector3f;
import org.joml.Vector3fc;
import org.joml.Vector4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * The underwater fog colour (data: it is the clear colour and what every vanilla pipeline's fog blends toward, so the terrain's medium, the sky, entities and
 * particles agree) after vanilla's own processing, which brightens it to full intensity as the player's water vision grows (milestone 4, OwnWater.fogColor).
 */
@Mixin(FogRenderer.class)
abstract class ShadeWaterFogMixin {
	@Shadow
	protected abstract FogType getFogType(Camera camera);

	@Inject(method = "computeFogColor", at = @At("RETURN"))
	private void mcopt$shadeWaterFog(Camera camera, float partialTicks, ClientLevel level, int renderDistance, float darkenWorldAmount, Vector4f dest, CallbackInfo ci) {
		if (level == null || level.dimensionType().skybox() != DimensionType.Skybox.OVERWORLD || !OwnWater.dataOn() || getFogType(camera) != FogType.WATER) return;
		Vector3fc c = OwnWater.fogColor(new Vector3f(dest.x, dest.y, dest.z));
		dest.set(c.x(), c.y(), c.z(), 1.0f);
	}
}
