package mcopt.metal.mixin.shaders;

import mcopt.metal.own.OwnLight;
import mcopt.metal.own.OwnProbe;
import net.minecraft.client.Camera;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.fog.environment.AtmosphericFogEnvironment;
import org.joml.Vector3fc;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** The horizon / haze colour by the time of day (data: the fog colour, which is also the horizon of the sky disc): warm at a low sun, lighter at noon. */
@Mixin(AtmosphericFogEnvironment.class)
abstract class ShadeFogMixin {
	@Inject(method = "getBaseColor", at = @At("RETURN"), cancellable = true)
	private void mcopt$shadeFog(ClientLevel level, Camera camera, int renderDistance, float partialTicks, CallbackInfoReturnable<Vector3fc> cir) {
		if (!OwnProbe.bool("shaders", true)) return;
		cir.setReturnValue(OwnLight.tintFog(cir.getReturnValue()));
	}
}
