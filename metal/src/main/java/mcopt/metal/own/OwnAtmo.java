package mcopt.metal.own;

/**
 * The CPU twin of the sky table's atmosphere (terrain.metal sky_lut): single scattering of the sun's light by Rayleigh and Mie particles and
 * the ozone layer's absorption in a spherical atmosphere with exponential densities (Nishita 1993, Bruneton and Neyret 2008, Hillaire 2020: public).
 * It gives the sun's colour at the camera (a transmittance), the colour of the sky toward a few directions (the horizon haze, the zenith) and the
 * ambient light colours, so the lighting of the terrain agrees with the sky the dome draws. Units: kilometres.
 */
public final class OwnAtmo {
	static final double RG = 6360.0, RT = 6420.0;
	static final double[] BR = {5.802e-3, 13.558e-3, 33.1e-3};
	static final double BMS = 7.0e-3, BME = 7.8e-3;
	static final double[] OZ = {0.650e-3, 1.881e-3, 0.085e-3};
	/** Camera altitude (km) and the sun's intensity, as the table's kernel gets them. */
	static final float CAM_KM = 0.12f, SUN_I = 20f, MS = 1.0f;

	private OwnAtmo() {
	}

	/** Nearest positive distance of the ray (o, d) to the sphere of radius r about the origin, or -1. */
	private static double hit(double[] o, double[] d, double r) {
		double b = o[0] * d[0] + o[1] * d[1] + o[2] * d[2], c = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - r * r, disc = b * b - c;
		if (disc < 0) return -1;
		double q = Math.sqrt(disc), t = -b - q;
		return t < 0 ? -b + q : t;
	}

	private static double ext(double h, double[] out) {
		h = Math.max(h, 0);
		double dr = Math.exp(-h * 0.125), dm = Math.exp(-h / 1.2), doz = Math.max(0, 1 - Math.abs(h - 25) / 15);
		for (int i = 0; i < 3; i++) out[i] = BR[i] * dr + BME * dm + OZ[i] * doz;
		out[3] = dr;
		out[4] = dm;
		return h;
	}

	private static void optical(double[] p, double[] d, double len, int n, double[] tau) {
		double[] e = new double[5];
		double dt = len / n;
		tau[0] = tau[1] = tau[2] = 0;
		for (int i = 0; i < n; i++) {
			double s = (i + 0.5) * dt, x = p[0] + d[0] * s, y = p[1] + d[1] * s, z = p[2] + d[2] * s;
			ext(Math.sqrt(x * x + y * y + z * z) - RG, e);
			for (int k = 0; k < 3; k++) tau[k] += e[k] * dt;
		}
	}

	/** The sun's light as it arrives at the camera (transmittance 0..1 per channel) for the sun's elevation (sine). */
	public static float[] sunTransmittance(double sinEl) {
		double cos = Math.sqrt(Math.max(0, 1 - sinEl * sinEl));
		double[] o = {0, RG + CAM_KM, 0}, s = {cos, sinEl, 0};
		if (hit(o, s, RG) > 0) return new float[] {0, 0, 0};
		double[] tau = new double[3];
		optical(o, s, hit(o, s, RT), 24, tau);
		return new float[] {(float) Math.exp(-tau[0]), (float) Math.exp(-tau[1]), (float) Math.exp(-tau[2])};
	}

	/**
	 * The radiance (the table's units, before the exposure) toward the view elevation e (radians) at azimuth phi (radians from the sun's) for the sun's
	 * elevation (sine). The same maths as sky_lut.
	 */
	public static void radiance(double e, double phi, double sinEl, float[] out) {
		double cosEl = Math.sqrt(Math.max(0, 1 - sinEl * sinEl));
		double[] V = {Math.cos(e) * Math.cos(phi), Math.sin(e), Math.cos(e) * Math.sin(phi)}, S = {cosEl, sinEl, 0}, o = {0, RG + CAM_KM, 0};
		double cosT = V[0] * S[0] + V[1] * S[1] + V[2] * S[2];
		double tAtm = hit(o, V, RT), tG = hit(o, V, RG);
		double tMax = (V[1] < 0 && tG > 0) ? tG : tAtm;
		double phR = 0.05968310 * (1 + cosT * cosT), g = 0.8;
		double phM = (1 - g * g) / (12.5663706 * Math.pow(1 + g * g - 2 * g * cosT, 1.5));
		int n = 24;
		double[] tauV = new double[3], e5 = new double[5], tauS = new double[3], L = new double[3], Lms = new double[3], p = new double[3];
		double tPrev = 0;
		for (int i = 0; i < n; i++) {
			double x1 = (i + 1.0) / n, t1 = tMax * x1 * x1, dt = t1 - tPrev, tm = tPrev + 0.5 * dt;
			tPrev = t1;
			for (int k = 0; k < 3; k++) p[k] = o[k] + V[k] * tm;
			ext(Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - RG, e5);
			double[] Tv = new double[3];
			for (int k = 0; k < 3; k++) {
				Tv[k] = Math.exp(-(tauV[k] + 0.5 * e5[k] * dt));
				tauV[k] += e5[k] * dt;
			}
			optical(p, S, hit(p, S, RT), 6, tauS);
			double lit = hit(p, S, RG) > 0 ? 0 : 1, dr = e5[3], dm = e5[4];
			for (int k = 0; k < 3; k++) {
				double Ts = Math.exp(-tauS[k]);
				L[k] += Tv[k] * (BR[k] * dr * phR + BMS * dm * phM) * Ts * lit * dt;
				Lms[k] += Tv[k] * (BR[k] * dr + BMS * dm) * Ts * dt;
			}
		}
		for (int k = 0; k < 3; k++) out[k] = (float) ((L[k] + Lms[k] * 0.0795775 * MS) * SUN_I);
	}
}
