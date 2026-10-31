package mcopt.metal.mixin.shaders;

import net.minecraft.client.renderer.WeatherEffectRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyVariable;

/** Rain streaks are fainter (the shaded look is an overcast grey-green where vanilla's streaks are saturated blue lines; snow keeps its alpha). */
@Mixin(WeatherEffectRenderer.class)
abstract class ShadeWeatherMixin {
	@ModifyVariable(method = "prepareInstances", at = @At("HEAD"), argsOnly = true, ordinal = 0)
	private float mcopt$fainterRain(float maxAlpha) {
		return maxAlpha >= 0.99f ? maxAlpha * RAIN_ALPHA : maxAlpha;
	}

	private static final float RAIN_ALPHA = Float.parseFloat(System.getProperty("mcopt.shaders.rainAlpha", "0.5"));
}
