package mcopt.metal;

import org.lwjgl.system.MemoryUtil;

/** The -Dmcopt.batch.* switches: fewer and cheaper draws for what comes through the frontend's render pass (entities, GUI, vanilla terrain...). */
final class BatchDraws {
	/** -Dmcopt.batch.direct: indirect draw lists the CPU can read are issued as direct draws. */
	static final boolean DIRECT = Boolean.getBoolean("mcopt.batch.direct");
	private static long scratch;
	private static int capacity;

	private BatchDraws() {
	}

	/** Room for n {firstIndex, indexCount, baseVertex} triples (render thread only). */
	static long scratch(int n) {
		if (n > capacity) {
			scratch = MemoryUtil.nmemRealloc(scratch, 12L * (capacity = Math.max(n, 2048)));
		}
		return scratch;
	}
}
