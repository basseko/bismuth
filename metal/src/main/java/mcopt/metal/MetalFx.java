package mcopt.metal;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandle;
import mcopt.metal.mixin.metalfx.MetalFxTargetAccess;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * MetalFX as an opt-in (-Dmcopt.metalfx=spatial, -Dmcopt.metalfx.scale=0.5..0.9, default 0.67; off by default: with it off none of the
 * metalfx mixins apply, this class is never loaded and the native scaler is never created).
 * <p>
 * Design: the game's main render target is full size and is what the GUI, the screenshot key and the present read. For the world,
 * {@link #begin} swaps the main target's textures (its colour and depth, width and height) for a second, reduced-size set, so everything
 * the world does (the terrain's culling pyramid, the own renderer's passes, post effects, the entity outline blit) sees an ordinary smaller
 * target and needs no knowledge of this. At the point where the shader grade runs (GameRenderer.render before the final depth clear and the
 * GUI), {@link #end} encodes MTLFXSpatialScaler from the reduced colour straight into the full-size colour and puts the full-size textures
 * back; the GUI then draws on the full-size image at native resolution. Projection (aspect only) and mouse picking (camera rays) do not
 * depend on the target size.
 */
public final class MetalFx {
	/** -Dmcopt.metalfx=spatial. */
	public static final boolean SPATIAL = "spatial".equals(System.getProperty("mcopt.metalfx"));
	public static final boolean ON = SPATIAL;
	/** Render scale per axis: the world's size over the window's. */
	public static final double SCALE = Math.max(0.5, Math.min(0.9, Double.parseDouble(System.getProperty("mcopt.metalfx.scale", "0.67"))));

	private static final Linker LINKER = Linker.nativeLinker();
	private static final MethodHandle SUPPORTED = fn("mcfx_supported", JAVA_INT, JAVA_LONG);
	private static final MethodHandle SPATIAL_FN = fn("mcfx_spatial", JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG);

	private static TextureTarget world;
	private static boolean swapped, failed, said, checked;
	private static int fullW, fullH;
	private static GpuTexture savedColor, savedDepth;
	private static GpuTextureView savedColorView, savedDepthView;

	private MetalFx() {
	}

	private static MethodHandle fn(String name, MemoryLayout result, MemoryLayout... args) {
		return LINKER.downcallHandle(MetalBridge.library().find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name)),
			FunctionDescriptor.of(result, args));
	}

	/** Called when the world's rendering starts (GameRenderer.renderLevel): the main target becomes the reduced one. */
	public static void begin(RenderTarget main) {
		if (swapped) end(main, false);  // (a world frame that never reached its end)
		if (failed || main.getColorTexture() == null) return;
		try {
			if (!checked) {
				checked = true;
				long ctx = ((MetalTexture) main.getColorTexture()).encoder.ctx;
				if ((int) SUPPORTED.invokeExact(ctx) == 0) {
					System.out.println("mcopt-metalfx: MetalFX spatial scaling is not supported on this device; off");
					failed = true;
					return;
				}
			}
			int w = main.width, h = main.height;
			int rw = Math.max(64, (int) Math.round(w * SCALE)), rh = Math.max(64, (int) Math.round(h * SCALE));
			if (rw >= w || rh >= h) return;
			if (world == null || world.width != rw || world.height != rh) {
				if (world != null) world.destroyBuffers();
				world = new TextureTarget("mcopt metalfx world", rw, rh, GpuFormat.RGBA8_UNORM, GpuFormat.D32_FLOAT);
			}
			fullW = w;
			fullH = h;
			savedColor = main.getColorTexture();
			savedColorView = main.getColorTextureView();
			savedDepth = main.getDepthTexture();
			savedDepthView = main.getDepthTextureView();
			set(main, world.getColorTexture(), world.getColorTextureView(), world.getDepthTexture(), world.getDepthTextureView(), rw, rh);
			swapped = true;
		} catch (Throwable t) {
			throw rethrow(t);
		}
	}

	/** The world is finished (the shader grade has run, if any): upscale into the full-size target and put it back. */
	public static void end(RenderTarget main) {
		end(main, true);
	}

	private static void end(RenderTarget main, boolean upscale) {
		if (!swapped) return;
		swapped = false;
		set(main, savedColor, savedColorView, savedDepth, savedDepthView, fullW, fullH);
		if (!upscale) return;
		try {
			MetalTexture in = (MetalTexture) world.getColorTexture(), out = (MetalTexture) savedColor;
			int r = (int) SPATIAL_FN.invokeExact(in.encoder.enc, in.handle, out.handle);
			if (r != 0) {
				failed = true;
				System.out.println("mcopt-metalfx: spatial scaling failed (" + r + "); off from the next frame");
				return;
			}
			// the scaler wrote every pixel: a clear still pending on the target (GameRenderer.render's) must not be folded into the GUI's pass
			out.pendingColorClear = null;
			if (!said) {
				said = true;
				System.out.println("mcopt-metalfx: spatial " + world.width + "x" + world.height + " -> " + fullW + "x" + fullH + " (scale " + SCALE + ")");
			}
		} catch (Throwable t) {
			throw rethrow(t);
		}
	}

	private static void set(RenderTarget t, GpuTexture color, GpuTextureView colorView, GpuTexture depth, GpuTextureView depthView, int w, int h) {
		MetalFxTargetAccess a = (MetalFxTargetAccess) (Object) t;
		a.mcopt$colorTexture(color);
		a.mcopt$colorTextureView(colorView);
		a.mcopt$depthTexture(depth);
		a.mcopt$depthTextureView(depthView);
		t.width = w;
		t.height = h;
	}

	private static RuntimeException rethrow(Throwable t) {
		if (t instanceof RuntimeException e) throw e;
		if (t instanceof Error e) throw e;
		return new IllegalStateException(t);
	}
}
