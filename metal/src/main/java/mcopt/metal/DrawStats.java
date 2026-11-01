package mcopt.metal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * -Dmcopt.drawstats[=frames]: measurement only. Per render pass label and pipeline, what the frontend sends through
 * MetalRenderPass per frame (averaged over a window): draw commands, native draws, and how many pipeline / uniform /
 * vertex buffer / index buffer / scissor / push constant calls each needed, and how many of those repeated what was already
 * bound. "chain" counts draws that followed another draw of the same pass with no state change in between (merge candidates).
 */
final class DrawStats {
	static final boolean ON = System.getProperty("mcopt.drawstats") != null;
	private static final int WINDOW = Math.max(1, Integer.getInteger("mcopt.drawstats", 300));
	static final int CMDS = 0, DRAWS = 1, PIPE = 2, PIPE_SAME = 3, UBIND = 4, USKIP = 5, VB = 6, VB_SAME = 7, IB = 8, IB_SAME = 9, SCI = 10, SCI_SAME = 11, PC = 12, PC_SAME = 13,
		CHAIN = 14, MERGE_ORDER = 15, MERGE_SORTED = 16, PASSES = 17, N = 18;
	private static final String[] COLS = {"cmds", "draws", "pipe", "=pipe", "ubind", "uskip", "vb", "=vb", "ib", "=ib", "sci", "=sci", "pc", "=pc", "chain", "mrgOrd", "mrgSrt", "passes"};
	private static final Map<String, long[]> byKey = new HashMap<>();
	private static final Map<String, long[]> byUniform = new HashMap<>();
	private static final Map<String, Long> passCount = new HashMap<>();
	private static int frames;
	static String nextLabel = "";

	private DrawStats() {
	}

	static long[] row(String pass, String pipeline) {
		return byKey.computeIfAbsent(pass + "\t" + pipeline, k -> new long[N]);
	}

	static void uniform(String pipeline, String name, boolean bound) {
		long[] r = byUniform.computeIfAbsent(name, k -> new long[2]);
		r[bound ? 0 : 1]++;
	}

	static void pass(String label) {
		passCount.merge(label, 1L, Long::sum);
	}

	static void frame() {
		if (++frames < WINDOW) return;
		print();
		byKey.clear();
		byUniform.clear();
		passCount.clear();
		frames = 0;
	}

	private static void print() {
		Map<String, long[]> perPass = new HashMap<>(), perPipe = new HashMap<>();
		long[] total = new long[N];
		for (var e : byKey.entrySet()) {
			String[] k = e.getKey().split("\t");
			add(perPass.computeIfAbsent(k[0], x -> new long[N]), e.getValue());
			add(perPipe.computeIfAbsent(k[1], x -> new long[N]), e.getValue());
			add(total, e.getValue());
		}
		StringBuilder sb = new StringBuilder("mcopt-drawstats: per frame over " + frames + " frames (pass label / pipeline: " + String.join(" ", COLS) + ")\n");
		sb.append(line("TOTAL", total, 0)).append("passes/frame ").append(String.format("%.1f", passCount.values().stream().mapToLong(Long::longValue).sum() / (double) frames)).append('\n');
		sb.append("-- by pass\n");
		table(sb, perPass, passCount);
		sb.append("-- by pipeline\n");
		table(sb, perPipe, null);
		sb.append("-- by pass+pipeline (top 40)\n");
		List<Map.Entry<String, long[]>> all = new ArrayList<>(byKey.entrySet());
		all.sort((a, b) -> Long.compare(b.getValue()[CMDS], a.getValue()[CMDS]));
		for (int i = 0; i < Math.min(40, all.size()); i++) sb.append(line(all.get(i).getKey().replace("\t", " / "), all.get(i).getValue(), 0));
		sb.append("-- uniform binds (bound, skipped as unchanged) per frame\n");
		byUniform.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue()[0], a.getValue()[0])).forEach(e ->
			sb.append(String.format("  %-28s bound %7.1f skipped %7.1f%n", e.getKey(), e.getValue()[0] / (double) frames, e.getValue()[1] / (double) frames)));
		System.out.print(sb);
	}

	private static void table(StringBuilder sb, Map<String, long[]> m, Map<String, Long> passes) {
		List<Map.Entry<String, long[]>> l = new ArrayList<>(m.entrySet());
		l.sort((a, b) -> Long.compare(b.getValue()[CMDS], a.getValue()[CMDS]));
		for (int i = 0; i < Math.min(30, l.size()); i++) {
			long p = passes == null ? 0 : passes.getOrDefault(l.get(i).getKey(), 0L);
			sb.append(line(l.get(i).getKey(), l.get(i).getValue(), p));
		}
	}

	private static void add(long[] a, long[] b) {
		for (int i = 0; i < N; i++) a[i] += b[i];
	}

	private static String line(String name, long[] r, long passes) {
		StringBuilder sb = new StringBuilder(String.format("  %-52s", name.length() > 52 ? name.substring(0, 52) : name));
		for (int i = 0; i < N - 1; i++) sb.append(String.format(" %7.1f", r[i] / (double) frames));
		if (passes > 0) sb.append(String.format(" [%.1f passes]", passes / (double) frames));
		return sb.append('\n').toString();
	}
}
