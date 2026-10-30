package mcopt.metal.own;

/**
 * -Dmcopt.own.frag.uoccAuto=true: whether uocc pays is measured on this machine, in this scene, instead of guessed from the
 * framebuffer size and GPU core count (-Dmcopt.own.frag.uoccMaxMp). Exact either way (a frame drawn without uocc draws every unit
 * the cull passes), so switching is invisible; only GPU time differs.
 * <p>
 * Exploring: blocks of BLOCK submits alternate with and without uocc (interleaved, so a camera moving through the scene loads both
 * alike); the first SKIP submits of each block are left out (the switch's own transient). Each measured submit's GPU time (command
 * buffer start to end) goes to its mode. With SAMPLES of each, the faster median wins if it is MARGIN faster; otherwise uocc stays on
 * (the configured default). The choice is held -Dmcopt.own.frag.uoccAutoHoldMs (default 20000), then measured again.
 * <p>
 * Render thread only: submitted() at commit, retired() when the command buffer has completed.
 */
public final class OwnUoccAuto {
	public static final boolean ON = Boolean.getBoolean("mcopt.own.frag.uoccAuto");
	private static final int BLOCK = 8, SKIP = 2, SAMPLES = 96;
	private static final double MARGIN = 0.03;
	private static final long HOLD_NS = Long.getLong("mcopt.own.frag.uoccAutoHoldMs", 20000) * 1_000_000L;
	/** Per submit (index & 255): 0 not measured, 1 with uocc, 2 without. */
	private static final byte[] modeOf = new byte[256];
	private static final double[] withUocc = new double[SAMPLES], withoutUocc = new double[SAMPLES];
	private static int nWith, nWithout, inBlock;
	private static boolean exploring = true, uocc = true;
	private static long holdUntil;

	private OwnUoccAuto() {
	}

	/** This frame draws with uocc. */
	static boolean uocc() {
		return uocc;
	}

	/** Submit `index` was committed, drawn in the current mode; the next frame's mode is set here. */
	public static void submitted(long index) {
		if (!exploring) {
			modeOf[(int) (index & 255)] = 0;
			if (System.nanoTime() >= holdUntil) {
				exploring = true;
				nWith = nWithout = inBlock = 0;
				uocc = true;
			}
			return;
		}
		modeOf[(int) (index & 255)] = (byte) (inBlock < SKIP ? 0 : uocc ? 1 : 2);
		if (++inBlock == BLOCK) {
			inBlock = 0;
			// alternate, but stop feeding a mode that already has its samples
			boolean next = !uocc;
			if (next && nWith >= SAMPLES) next = false;
			if (!next && nWithout >= SAMPLES) next = true;
			uocc = next;
		}
	}

	/** Submit `index` completed after `gpuMicros` of GPU time. */
	public static void retired(long index, double gpuMicros) {
		if (!exploring || gpuMicros <= 0) return;
		int mode = modeOf[(int) (index & 255)];
		if (mode == 1 && nWith < SAMPLES) withUocc[nWith++] = gpuMicros;
		else if (mode == 2 && nWithout < SAMPLES) withoutUocc[nWithout++] = gpuMicros;
		if (nWith < SAMPLES || nWithout < SAMPLES) return;
		double on = median(withUocc), off = median(withoutUocc);
		boolean choice = !(off < on * (1 - MARGIN));
		if (choice != uoccHeld) {
			System.out.printf("mcopt-own-frag: uoccAuto: uocc %s (median GPU %.0f us with, %.0f us without)%n", choice ? "on" : "off", on, off);
		}
		uoccHeld = choice;
		uocc = choice;
		exploring = false;
		holdUntil = System.nanoTime() + HOLD_NS;
	}

	private static boolean uoccHeld = true;

	private static double median(double[] a) {
		double[] s = a.clone();
		java.util.Arrays.sort(s);
		return (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
	}
}
