package mcopt.metal.own;

/**
 * -Dmcopt.own.frag.uoccAuto=true: whether uocc pays is measured on this machine, in this scene, instead of guessed from the
 * framebuffer size and GPU core count (-Dmcopt.own.frag.uoccMaxMp). Exact either way (a frame drawn without uocc draws every unit
 * the cull passes), so switching is invisible; only speed differs.
 * <p>
 * Exploring: blocks of BLOCK presented frames alternate with and without uocc (interleaved, so a camera moving through the scene loads
 * both alike); the first SKIP frames of each block are left out (the switch's transient and the frames still in flight from the other
 * mode). Each measured frame's interval (commit to commit of presenting submits: what the player sees, whatever limits it, the GPU,
 * the render thread or the far LOD) goes to its mode. A frame's GPU time alone misleads: uocc's frames spread their work over command
 * buffers that wait on each other, and at long render distances the GPU isn't what limits the frame. With SAMPLES of each, the faster
 * median wins if it is MARGIN faster; otherwise uocc stays on (the configured default; under VSync both modes tie). The choice is held
 * -Dmcopt.own.frag.uoccAutoHoldMs (default 20000), then measured again.
 * <p>
 * Render thread only: submitted() at each commit.
 */
public final class OwnUoccAuto {
	public static final boolean ON = Boolean.getBoolean("mcopt.own.frag.uoccAuto");
	private static final int BLOCK = 12, SKIP = 4, SAMPLES = 96;
	private static final double MARGIN = 0.03;
	/** -Dmcopt.own.frag.uoccAutoLog: every decision, not only changes. */
	private static final boolean LOG = Boolean.getBoolean("mcopt.own.frag.uoccAutoLog");
	private static final long HOLD_NS = Long.getLong("mcopt.own.frag.uoccAutoHoldMs", 20000) * 1_000_000L;
	private static final double[] withUocc = new double[SAMPLES], withoutUocc = new double[SAMPLES];
	private static int nWith, nWithout, inBlock;
	private static boolean exploring = true, uocc = true;
	private static long holdUntil, lastPresent;

	private OwnUoccAuto() {
	}

	/** This frame draws with uocc. */
	static boolean uocc() {
		return uocc;
	}

	/** A submit was committed (presenting: it ends a frame), drawn in the current mode; the next frame's mode is set here. */
	public static void submitted(boolean presenting) {
		if (!presenting) return;
		long now = System.nanoTime(), interval = now - lastPresent;
		lastPresent = now;
		if (!exploring) {
			if (now >= holdUntil) {
				exploring = true;
				nWith = nWithout = inBlock = 0;
				uocc = true;
			}
			return;
		}
		if (inBlock >= SKIP && interval > 0) {
			if (uocc && nWith < SAMPLES) withUocc[nWith++] = interval / 1000.0;
			else if (!uocc && nWithout < SAMPLES) withoutUocc[nWithout++] = interval / 1000.0;
		}
		if (nWith >= SAMPLES && nWithout >= SAMPLES) {
			decide(now);
			return;
		}
		if (++inBlock == BLOCK) {
			inBlock = 0;
			// alternate, but stop feeding a mode that already has its samples
			boolean next = !uocc;
			if (next && nWith >= SAMPLES) next = false;
			if (!next && nWithout >= SAMPLES) next = true;
			uocc = next;
		}
	}

	private static void decide(long now) {
		double on = median(withUocc), off = median(withoutUocc);
		boolean choice = !(off < on * (1 - MARGIN));
		if (choice != uoccHeld || LOG) {
			System.out.printf("mcopt-own-frag: uoccAuto: uocc %s (median frame %.0f us with, %.0f us without)%n", choice ? "on" : "off", on, off);
		}
		uoccHeld = choice;
		uocc = choice;
		exploring = false;
		holdUntil = now + HOLD_NS;
	}

	private static boolean uoccHeld = true;

	private static double median(double[] a) {
		double[] s = a.clone();
		java.util.Arrays.sort(s);
		return (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
	}
}
