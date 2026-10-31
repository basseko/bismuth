package mcopt.metal.mixin.metalfx;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** -Dmcopt.metalfx: the render target's attachments (swapped between the full-size and the reduced set). */
@Mixin(RenderTarget.class)
public interface MetalFxTargetAccess {
	@Accessor("colorTexture")
	void mcopt$colorTexture(GpuTexture texture);

	@Accessor("colorTextureView")
	void mcopt$colorTextureView(GpuTextureView view);

	@Accessor("depthTexture")
	void mcopt$depthTexture(GpuTexture texture);

	@Accessor("depthTextureView")
	void mcopt$depthTextureView(GpuTextureView view);
}
