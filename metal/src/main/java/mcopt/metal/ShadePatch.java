package mcopt.metal;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Milestone 5 (-Dmcopt.shaders=true): sun shadows and the terrain's light balance on the vanilla pipelines' geometry (entities, block entities,
 * dropped items, armour). A text patch of the SPIRV-Cross output of a whitelisted pipeline's vertex and fragment stages (MetalPipeline.compile), so
 * vanilla's shaders stay the source of truth and a change that breaks the recipe leaves the pipeline untouched (logged, never fatal):
 * <ul>
 * <li>vertex: one more varying {@code mcoSh} = (camera-relative position, sky light 0..1), written at the top of the entry point;</li>
 * <li>fragment: {@code entity_shade.metal} is inserted after {@code using namespace metal;}; the entry point gets the frame block (buffer 29), the shadow
 * map (texture 24) and the pixel position as extra arguments; the flat normal is taken at the top of the body (uniform control flow) and the finished
 * {@code fragColor.rgb} is re-lit just before the final {@code return out;}.</li>
 * </ul>
 * The vertex shader's {@code Position} is camera-relative world space for these pipelines (the view rotation lives in ModelViewMat), the same space as
 * the terrain's fragment position, so the cascade matrices of the terrain's ShadeFrame apply unchanged.
 */
final class ShadePatch {
	static final boolean ON = EntityShade.RECEIVE;
	private static final boolean LOG = Boolean.getBoolean("mcopt.shaders.log");

	/** Pipelines whose fragment colour is vertex colour x texture x lightmap (non-OIT). */
	private static final Set<String> RECEIVE = Set.of(
		"minecraft:pipeline/entity_solid", "minecraft:pipeline/entity_solid_offset_forward", "minecraft:pipeline/entity_cutout_cull",
		"minecraft:pipeline/entity_cutout", "minecraft:pipeline/entity_cutout_z_offset", "minecraft:pipeline/entity_translucent",
		"minecraft:pipeline/entity_translucent_cull", "minecraft:pipeline/armor_cutout_no_cull", "minecraft:pipeline/armor_decal_cutout_no_cull",
		"minecraft:pipeline/armor_translucent", "minecraft:pipeline/end_crystal_beam", "minecraft:pipeline/banner_pattern", "minecraft:pipeline/item_cutout",
		"minecraft:pipeline/item_translucent");
	/** The subset that casts into the shadow map (opaque and cutout geometry; translucent layers and decals do not). */
	private static final Set<String> CAST = Set.of(
		"minecraft:pipeline/entity_solid", "minecraft:pipeline/entity_solid_offset_forward", "minecraft:pipeline/entity_cutout_cull",
		"minecraft:pipeline/entity_cutout", "minecraft:pipeline/entity_cutout_z_offset", "minecraft:pipeline/armor_cutout_no_cull", "minecraft:pipeline/item_cutout");

	private static String header;

	private ShadePatch() {
	}

	static boolean receives(String pipeline) {
		return RECEIVE.contains(pipeline);
	}

	static boolean casts(String pipeline) {
		return CAST.contains(pipeline);
	}

	private static final Pattern FRAG_SIG = Pattern.compile("^(fragment \\w+ main0\\()([^\\n]*)(\\))[ \\t]*\\r?\\n\\{", Pattern.MULTILINE);
	private static final Pattern VERT_SIG = Pattern.compile("^vertex \\w+ main0\\(", Pattern.MULTILINE);

	/**
	 * The patched {vertex, fragment} source of pipeline {@code name}, or null when it is not a whitelisted pipeline or its translation does not have the
	 * expected shape (logged).
	 */
	static String[] patch(String name, String vertex, String fragment) {
		if (!ON || !RECEIVE.contains(name)) return null;
		try {
			return patch0(vertex, fragment);
		} catch (IllegalStateException e) {
			System.out.println("mcopt-shaders: shade-patch: skipped " + name + ": " + e.getMessage());
			return null;
		}
	}

	private static String[] patch0(String v, String f) {
		// ---- vertex ----
		if (!v.contains("struct main0_out\n{") || !v.contains("struct main0_in\n{") || !v.contains("float3 Position [[attribute(")) throw new IllegalStateException("vertex shape");
		Matcher vm = VERT_SIG.matcher(v);
		if (!vm.find()) throw new IllegalStateException("vertex entry");
		int bodyStart = v.indexOf("main0_out out = {};", vm.end());
		if (bodyStart < 0) throw new IllegalStateException("vertex out");
		int lineEnd = v.indexOf('\n', bodyStart);
		boolean hasLight = v.contains("int2 UV2 [[attribute(");
		String sky = hasLight ? "clamp(float(in.UV2.y) * (1.0 / 240.0), 0.0, 1.0)" : "1.0";
		String pv = v.substring(0, lineEnd + 1) + "    out.mcoSh = float4(in.Position, " + sky + ");\n" + v.substring(lineEnd + 1);
		pv = pv.replaceFirst("struct main0_out\n\\{\n", "struct main0_out\n{\n    float4 mcoSh [[user(locn14)]];\n");
		// ---- fragment ----
		if (!f.contains("struct main0_in\n{") || !f.contains("float4 lightMapColor [[user(") || !f.contains("float4 fragColor [[color(0)]]")) throw new IllegalStateException("fragment shape");
		if (f.contains("[[position]]") || f.contains("mcoF")) throw new IllegalStateException("fragment has a position argument");
		Matcher fm = FRAG_SIG.matcher(f);
		if (!fm.find()) throw new IllegalStateException("fragment entry");
		int open = f.indexOf("main0_out out = {};", fm.end());
		int ret = f.lastIndexOf("return out;");
		if (open < 0 || ret < open) throw new IllegalStateException("fragment body");
		int openEnd = f.indexOf('\n', open);
		String sigNew = fm.group(1) + fm.group(2) + ", constant MCOFrame& mcoF [[buffer(29)]], depth2d_array<float> mcoMap [[texture(24)]], float4 mcoFC [[position]]" + fm.group(3) + "\n{";
		StringBuilder sb = new StringBuilder(f.length() + 6000);
		sb.append(f, 0, fm.start()).append(sigNew).append(f, fm.end(), openEnd + 1);
		sb.append("    float3 mcoN = mcoNormal(in.mcoSh.xyz);\n");
		sb.append(f, openEnd + 1, ret);
		sb.append("    out.fragColor.xyz = mcoLit(out.fragColor.xyz, in.lightMapColor.xyz, in.mcoSh.xyz, in.mcoSh.w, mcoN, mcoFC.xy, mcoF, mcoMap);\n    ");
		sb.append(f, ret, f.length());
		String pf = sb.toString();
		pf = pf.replaceFirst("struct main0_in\n\\{\n", "struct main0_in\n{\n    float4 mcoSh [[user(locn14)]];\n");
		int ns = pf.indexOf("using namespace metal;");
		if (ns < 0) throw new IllegalStateException("namespace");
		ns = pf.indexOf('\n', ns) + 1;
		pf = pf.substring(0, ns) + "\n" + header() + "\n" + pf.substring(ns);
		return new String[] {pv, pf};
	}

	private static synchronized String header() {
		if (header == null) {
			String path = System.getProperty("mcopt.shaders.entityShadeFile");  // (offline checks)
			try {
				if (path != null) header = java.nio.file.Files.readString(java.nio.file.Path.of(path));
				else try (InputStream in = ShadePatch.class.getResourceAsStream("/mcopt/shade/entity_shade.metal")) {
					header = new String(in.readAllBytes(), StandardCharsets.UTF_8);
				}
			} catch (IOException | RuntimeException e) {
				throw new IllegalStateException("entity_shade.metal: " + e);
			}
		}
		return header;
	}

	static void log(String s) {
		if (LOG) System.out.println("mcopt-shaders: " + s);
	}
}
