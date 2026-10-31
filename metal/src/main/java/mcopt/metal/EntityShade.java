package mcopt.metal;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.pipeline.BindGroupLayout;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;
import org.joml.Matrix4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Milestone 5 (-Dmcopt.shaders=true; sub-flags {@code mcopt.shaders.entities=false} and {@code mcopt.shaders.entities.cast=false}): entities, block
 * entities, dropped items and armour take the terrain's sun shadows and light balance, and cast shadows into the terrain's shadow map.
 * <p>
 * <b>Receive</b> ({@link ShadePatch}, native mcentity.m {@code mce_bind}): whitelisted pipelines get a patched vertex and fragment stage; each of their draws
 * binds the frame block (the terrain's ShadeFrame plus the entity extras, fragment buffer 29) and this frame's shadow map (texture 24). The block is
 * published by OwnShade each frame the sun applies; in every other case (no sun, the hand, the GUI) it is a zero block and the patched shader returns the
 * vanilla colour untouched.
 * <p>
 * <b>Cast</b>: the draws of the opaque entity pipelines in the world pass are recorded (pipeline, the uniform and vertex/index buffer handles bound at
 * that moment, the draw arguments) and, at the frame's commit, replayed into cascade 0 of this frame's shadow map in the pre command buffer (a depth-only
 * second render pass on that slice with its contents loaded: the terrain's casters are already in it), through a depth-only twin of each pipeline (same
 * vertex function, no fragment stage). The vertex shader computes Projection x ModelView x Position; the replay binds Projection' = LightClip x ModelView^-1
 * so that it yields LightClip x Position (Position is camera-relative world space, as in the terrain). Only cascade 0 (re-rendered every frame) takes
 * entities: the farther cascades are cached, an entity in them would leave a trail.
 */
public final class EntityShade {
	/** -Dmcopt.shaders=true and not -Dmcopt.shaders.entities=false. */
	public static final boolean ON = Boolean.getBoolean("mcopt.shaders") && !"false".equals(System.getProperty("mcopt.shaders.entities"));
	static final boolean RECEIVE = ON;
	/** Entities cast into the shadow map. */
	static final boolean CAST = ON && !"false".equals(System.getProperty("mcopt.shaders.entities.cast"));
	private static final int MAX_DRAWS = Integer.getInteger("mcopt.shaders.entities.maxDraws", 600);
	/** How strongly the sun / shadow term shows on entities (1: as on terrain), the shadow bias scale (entities are thin boxes) and the distance where it fades out. */
	private static final float STRENGTH = Float.parseFloat(System.getProperty("mcopt.shaders.entities.strength", "1.0"));
	private static final float BIAS = Float.parseFloat(System.getProperty("mcopt.shaders.entities.bias", "2.0"));
	private static final float FADE = Float.parseFloat(System.getProperty("mcopt.shaders.entities.fade", "110"));
	private static final int BLOCK = 400, FRAME = 368;

	/** The frame block is published (the sun applies in the world pass). */
	private static boolean active;
	/** Draws of cast pipelines are recorded (cascade 0 is rendered this frame). */
	static boolean castActive;
	private static final Matrix4f clip0 = new Matrix4f();
	private static final Matrix4f view = new Matrix4f();
	private static boolean haveView;
	private static final List<Draw> draws = new ArrayList<>();
	private static boolean failed, said;
	private static long frames, castedFrames, castedDraws, skippedDraws, recordedDraws;
	private static final boolean LOG = Boolean.getBoolean("mcopt.shaders.log");

	private EntityShade() {
	}

	// ---- called by OwnShade / the hand mixin (render thread) ----

	/** Start of OwnShade.frame (drawOpaque, before the frame's terrain draws): nothing applies until the frame publishes. */
	public static void begin() {
		if (active) {
			active = false;
			try {
				Nat.OFF.run();
			} catch (RuntimeException | LinkageError e) {
				fail("off", e);
			}
		}
		castActive = false;
		draws.clear();
		haveView = false;
	}

	/**
	 * End of OwnShade.frame0 when the sun applies: fb: the ShadeFrame block (368 bytes, native), k: the ring slot, cast: cascade 0 was rendered into the
	 * shadow map this frame.
	 */
	public static void publish(long fb, int k, boolean cast) {
		if (failed) return;
		try (MemoryStack stack = MemoryStack.stackPush()) {
			long b = stack.ncalloc(16, 1, BLOCK);
			MemoryUtil.memCopy(fb, b, FRAME);
			MemoryUtil.memPutFloat(b + FRAME, 1.0f);
			MemoryUtil.memPutFloat(b + FRAME + 4, STRENGTH);
			MemoryUtil.memPutFloat(b + FRAME + 8, BIAS);
			MemoryUtil.memPutFloat(b + FRAME + 12, FADE);
			Nat.publish(b, BLOCK);
			clip0.setFromAddress(fb);
		} catch (RuntimeException | LinkageError e) {
			fail("publish", e);
			return;
		}
		active = true;
		castActive = CAST && cast;
		frames++;
		if (!said) {
			said = true;
			System.out.println("mcopt-shaders: entities on (receive" + (CAST ? " + cast" : "") + ")");
		}
	}

	/**
	 * The first-person hand starts (GameRenderer.render3dHud): it still receives the sun and the shadows (its vertex positions are camera-relative and
	 * world-aligned like the world's: the view rotation is in ModelViewMat and the pose stack is un-rotated), but it casts nothing.
	 */
	public static void hand() {
		castActive = false;  // (the world's casters recorded so far are kept for the commit's replay)
	}

	/** The world is done (before the GUI, FogRenderer.endFrame): the GUI's 3D items and entity previews are in their own spaces, so the shading stops. */
	public static void gui() {
		castActive = false;
		if (active) {
			active = false;
			try {
				Nat.OFF.run();
			} catch (RuntimeException | LinkageError e) {
				fail("off", e);
			}
		}
	}

	// ---- called by MetalRenderPass ----

	/** Before a draw of a patched pipeline: its frame block and shadow map. */
	static void bind(long enc) {
		Nat.BIND.run(enc);
	}

	/** Records an indexed draw of a cast pipeline in the world pass (the pass's recorded handles are the ones bound now). */
	static void capture(MetalRenderPass pass, MetalPipeline p, MetalEncoder encoder, int indexCount, int instanceCount, int firstIndex, int vertexOffset, int firstInstance) {
		if (draws.size() >= MAX_DRAWS) return;
		if (!haveView) {
			view.set(RenderSystem.getModelViewMatrixCopy());
			haveView = true;
		}
		Draw d = pass.snapshot(p, encoder, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
		if (d != null) draws.add(d);
	}

	/** One recorded draw: everything the replay binds. */
	static final class Draw {
		MetalPipeline p;
		long[] handle, offset, texture, sampler;
		MetalBuffer[] buf;
		MetalBuffer vbuf, ibuf;
		long vertex, vertexOffset, index;
		int indexInt, count, instances, first, base, firstInstance;
	}

	/** Before the frame's commit: replays the recorded casters into cascade 0 of this frame's shadow map (pre command buffer). */
	static void flush(long enc, long submit) {
		if (draws.isEmpty()) return;
		try {
			// The replay runs in the pre command buffer, before the main one. A buffer a GPU copy (blit, encoded in the main stream) fills in this submit
			// still holds the previous contents at that point (stale vertices, indices or uniforms), so the draws reading one are dropped: a missing
			// shadow for a frame is better than a lagging or garbage one. Buffers the CPU writes (maps, memcpy, renamed memory) are already final.
			recordedDraws += draws.size();
			int before = draws.size();
			draws.removeIf(d -> stale(d, submit));
			skippedDraws += before - draws.size();
			if (LOG && frames % 300 == 0) System.out.println("mcopt-shaders: " + stats());
			if (draws.isEmpty()) return;
			replay(enc);
		} catch (RuntimeException e) {
			fail("cast", e);
		} finally {
			draws.clear();
			castActive = false;
		}
	}

	/** The draw reads a buffer that a blit of this submit writes. */
	private static boolean stale(Draw d, long submit) {
		if (d.vbuf.blitSubmit == submit || d.ibuf != null && d.ibuf.blitSubmit == submit) return true;
		MetalBuffer[] b = d.buf;
		for (int i = 0; i < b.length; i++) {
			if (b[i] != null && b[i].blitSubmit == submit && d.handle[i] != 0) return true;
		}
		return false;
	}

	private static void replay(long enc) {
		long fake = Nat.begin(enc, 0);
		if (fake == 0) return;
		try (MemoryStack stack = MemoryStack.stackPush()) {
			// Projection' = LightClip x ModelView^-1: the vertex shaders' Projection x ModelView x Position becomes LightClip x Position
			Matrix4f proj = new Matrix4f(clip0).mul(new Matrix4f(view).invert());
			long pm = stack.ncalloc(16, 1, 64);
			proj.getToAddress(pm);
			MetalPipeline last = null;
			long lastVertex = -1, lastVertexOffset = -1, lastIndex = -1;
			int n = 0;
			long[] lh = new long[32], lo = new long[32], lt = new long[32], ls = new long[32];
			java.util.Arrays.fill(lh, -1);
			java.util.Arrays.fill(lo, -1);
			java.util.Arrays.fill(lt, -1);
			java.util.Arrays.fill(ls, -1);
			for (Draw d : draws) {
				MetalPipeline p = d.p;
				if (p.shadowPso == 0) continue;
				if (p != last) {
					Native.pipeline(fake, p.shadowPso, 0, 0, 0, 0f, 0f, p.primitive);
					last = p;
					java.util.Arrays.fill(lh, -1);  // (the shadow pipeline's argument tables are the same, but the Projection slot differs per pipeline)
					java.util.Arrays.fill(lt, -1);
				}
				List<BindGroupLayout.UniformDescription> u = p.uniforms;
				for (int i = 0; i < u.size() && i < d.handle.length; i++) {
					switch (u.get(i).type()) {
						case UNIFORM_BUFFER -> {
							if (u.get(i).name().equals("Projection")) {
								if (lh[i] != -2) {
									Native.bytes(fake, i, pm, 64);
									lh[i] = -2;
								}
							} else if (d.handle[i] != 0 && (lh[i] != d.handle[i] || lo[i] != d.offset[i])) {
								Native.buffer(fake, i, d.handle[i], d.offset[i]);
								lh[i] = d.handle[i];
								lo[i] = d.offset[i];
							}
						}
						case COMBINED_IMAGE_SAMPLER -> {
							if (d.texture[i] != 0 && (lt[i] != d.texture[i] || ls[i] != d.sampler[i])) {
								Native.texture(fake, i, d.texture[i], d.sampler[i]);
								lt[i] = d.texture[i];
								ls[i] = d.sampler[i];
							}
						}
						default -> {
						}
					}
				}
				if (d.vertex != lastVertex || d.vertexOffset != lastVertexOffset) {
					Native.vertexBuffer(fake, MetalConst.VERTEX_BUFFER_BASE, d.vertex, d.vertexOffset);
					lastVertex = d.vertex;
					lastVertexOffset = d.vertexOffset;
				}
				if (d.index != lastIndex) {
					Native.index(fake, d.index, d.indexInt);
					lastIndex = d.index;
				}
				Native.drawIndexed(fake, d.count, d.instances, d.first, d.base, d.firstInstance);
				n++;
			}
			castedDraws += n;
			castedFrames++;
		} finally {
			Nat.end(fake);
		}
	}

	private static void fail(String what, Throwable e) {
		if (!failed) {
			System.out.println("mcopt-shaders: entities " + what + " disabled: " + e);
			e.printStackTrace(System.out);
		}
		failed = true;
		castActive = false;
	}

	/** The native side (mcentity.m), linked on first use. */
	private static final class Nat {
		private static final Linker LINKER = Linker.nativeLinker();
		private static final MethodHandle PUBLISH = fn("mce_publish", JAVA_LONG, JAVA_INT);
		private static final MethodHandle OFF_H = fn("mce_off");
		private static final MethodHandle BIND_H = fn("mce_bind", JAVA_LONG);
		private static final MethodHandle BEGIN_H = fnr("mce_begin", JAVA_LONG, JAVA_LONG, JAVA_INT);
		private static final MethodHandle END_H = fn("mce_end", JAVA_LONG);
		static final Runnable OFF = () -> {
			try {
				OFF_H.invokeExact();
			} catch (Throwable t) {
				throw rethrow(t);
			}
		};
		static final LongRunnable BIND = enc -> {
			try {
				BIND_H.invokeExact(enc);
			} catch (Throwable t) {
				throw rethrow(t);
			}
		};

		interface LongRunnable {
			void run(long v);
		}

		static void publish(long bytes, int length) {
			try {
				PUBLISH.invokeExact(bytes, length);
			} catch (Throwable t) {
				throw rethrow(t);
			}
		}

		static long begin(long enc, int slice) {
			try {
				return (long) BEGIN_H.invokeExact(enc, slice);
			} catch (Throwable t) {
				throw rethrow(t);
			}
		}

		static void end(long fake) {
			try {
				END_H.invokeExact(fake);
			} catch (Throwable t) {
				throw rethrow(t);
			}
		}

		private static MethodHandle fn(String name, MemoryLayout... args) {
			var symbol = Native.lookup().find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
			return LINKER.downcallHandle(symbol, FunctionDescriptor.ofVoid(args));
		}

		private static MethodHandle fnr(String name, MemoryLayout result, MemoryLayout... args) {
			var symbol = Native.lookup().find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
			return LINKER.downcallHandle(symbol, FunctionDescriptor.of(result, args));
		}

		private static RuntimeException rethrow(Throwable t) {
			if (t instanceof RuntimeException r) return r;
			if (t instanceof Error e) throw e;
			return new IllegalStateException(t);
		}
	}

	/** Statistics line for -Dmcopt.shaders.log. */
	public static String stats() {
		return "entity shadows: " + castedFrames + " frames, " + castedDraws + " draws replayed, " + skippedDraws + " of " + recordedDraws + " recorded skipped (blit hazard)";
	}
}
