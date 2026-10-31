package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnWater;
import net.minecraft.client.Camera;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.fog.environment.WaterFogEnvironment;
import net.minecraft.world.level.dimension.DimensionType;
import org.joml.Vector3fc;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * The underwater fog colour (data: it is the clear colour and what every vanilla pipeline's fog blends toward, so the terrain's medium, the sky, entities and
 * particles agree): murkier and darker than vanilla's bright blue (milestone 4, OwnWater.fogColor).
 */
@Mixin(WaterFogEnvironment.class)
abstract class ShadeWaterFogMixin {
	@Inject(method = "getBaseColor", at = @At("RETURN"), cancellable = true)
	private void mcopt$shadeWaterFog(ClientLevel level, Camera camera, int renderDistance, float partialTicks, CallbackInfoReturnable<Vector3fc> cir) {
		if (level == null || level.dimensionType().skybox() != DimensionType.Skybox.OVERWORLD || !OwnWater.dataOn()) return;
		cir.setReturnValue(OwnWater.fogColor(cir.getReturnValue()));
	}
}
