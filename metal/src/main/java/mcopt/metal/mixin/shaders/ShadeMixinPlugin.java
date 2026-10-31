package mcopt.metal.mixin.shaders;

import java.util.List;
import java.util.Set;
import org.objectweb.asm.tree.ClassNode;
import org.spongepowered.asm.mixin.extensibility.IMixinConfigPlugin;
import org.spongepowered.asm.mixin.extensibility.IMixinInfo;

/**
 * The shaders' mixins apply only with -Dmcopt.shaders=true (and our own terrain, which the shaders extend): without the switch the game is
 * byte-for-byte what it is without them. Reads the properties itself so nothing of mcopt.metal.own loads before the game does.
 */
public final class ShadeMixinPlugin implements IMixinConfigPlugin {
	static {
		mcopt.metal.Profile.apply(); // before any flag is read
	}

	private static final boolean ENABLED = Boolean.getBoolean("mcopt.shaders");

	@Override
	public void onLoad(String mixinPackage) {
	}

	@Override
	public String getRefMapperConfig() {
		return null;
	}

	@Override
	public boolean shouldApplyMixin(String targetClassName, String mixinClassName) {
		if (mixinClassName.endsWith(".ShadeLightmapMixin")) return ENABLED && !"false".equals(System.getProperty("mcopt.shaders.light"));
		if (mixinClassName.endsWith(".ShadePostMixin")) return ENABLED && !"false".equals(System.getProperty("mcopt.shaders.grade"));
		if (mixinClassName.endsWith(".ShadeWaterFogMixin") || mixinClassName.endsWith(".ShadeWaterOverlayMixin")) return ENABLED && !"false".equals(System.getProperty("mcopt.shaders.water"));
		return ENABLED;
	}

	@Override
	public void acceptTargets(Set<String> myTargets, Set<String> otherTargets) {
	}

	@Override
	public List<String> getMixins() {
		return null;
	}

	@Override
	public void preApply(String targetClassName, ClassNode targetClass, String mixinClassName, IMixinInfo mixinInfo) {
	}

	@Override
	public void postApply(String targetClassName, ClassNode targetClass, String mixinClassName, IMixinInfo mixinInfo) {
	}
}
