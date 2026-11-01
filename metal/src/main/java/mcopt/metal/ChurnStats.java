package mcopt.metal;

import java.util.HashMap;
import java.util.Map;

/**
 * -Dmcopt.churn.stats=true (measurement only): once per 5 s prints, per presented frame, how many submits (command buffer pairs), render passes, texture views
 * and uploads the game asked for, and which callers made the submits that were not the frame's present.
 */
final class ChurnStats {
	static final boolean ON = Boolean.getBoolean("mcopt.churn.stats");
	private static long last, submits, presents, views, uploads, passes;
	private static final Map<String, Integer> callers = new HashMap<>();

	private ChurnStats() {
	}

	static synchronized void count(String what) {
		switch (what) {
			case "view" -> views++;
			case "upload" -> uploads++;
			case "pass" -> passes++;
			default -> {
			}
		}
	}

	static synchronized void submit(boolean presented) {
		submits++;
		if (presented) presents++;
		else {
			String key = StackWalker.getInstance().walk(s -> s.filter(f -> !f.getClassName().equals(ChurnStats.class.getName()) && !f.getClassName().equals(MetalEncoder.class.getName()))
				.limit(4).map(f -> f.getClassName().substring(f.getClassName().lastIndexOf('.') + 1) + "." + f.getMethodName()).reduce((a, b) -> a + " < " + b).orElse("?"));
			callers.merge(key, 1, Integer::sum);
		}
		long now = System.nanoTime();
		if (last == 0) last = now;
		if (now - last < 5_000_000_000L || presents == 0) return;
		double f = presents;
		System.out.printf("mcopt-metal churn: per presented frame: submits %.2f, passes %.2f, texture views %.2f, uploads %.2f (presents %d)%n", submits / f, passes / f, views / f, uploads / f, presents);
		callers.entrySet().stream().sorted((a, b) -> b.getValue() - a.getValue()).limit(6).forEach(e -> System.out.println("  non-present submit x" + e.getValue() + ": " + e.getKey()));
		callers.clear();
		submits = presents = views = uploads = passes = 0;
		last = now;
	}
}
