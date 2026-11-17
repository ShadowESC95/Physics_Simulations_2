#!/usr/bin/env python3
"""
Wall-network estimate: how much flux does a photon beam lose after crossing
many photon-mass walls at random angles?

    python3 wall_network_estimate.py                     default FRB-like scenario
    python3 wall_network_estimate.py --help              all options
    python3 wall_network_estimate.py --monte-carlo       also run the amplitude-level check
    python3 wall_network_estimate.py --plot loss.png     save a figure

Builds on proca_wall_solver.py (same folder).  No files are read; nothing is
written unless --plot or --csv is given.  Needs numpy; matplotlib only for --plot.

Physics (c = hbar = 1 internally)
---------------------------------
Space is filled with domains in which the photon mass takes one of two values,
m_a and m_b, separated by walls that are thick compared with the wavelength.
A sightline of length D crosses N = D / spacing walls with random orientation.

Per wall, for a photon of frequency omega arriving at angle theta to the wall
normal (see proca_wall_solver.py for the derivation):

  * TE-polarised light passes unchanged.
  * TM-polarised light is rotated into the longitudinal polarisation by the angle
        d_alpha = alpha_b - alpha_a,     tan(alpha) = k m / (omega q),
    i.e. a fraction sin^2(d_alpha) ~ tan^2(theta) (m_b - m_a)^2 / omega^2 converts.
  * If the mass rises and the angle is close enough to grazing that
        q_b^2 = q_a^2 - (m_b^2 - m_a^2) < 0,
    the photon is totally reflected and leaves the beam.

Walls crossed by a straight line are met with cos(theta) = u distributed as
p(u) = 2u on (0, 1].  Averaging over u and over the random angle between the
polarisation and the plane of incidence (factor 1/2 for the TM share) gives the
mean loss per wall.  For random wall orientations successive conversions add as
a random walk, so the total loss after N walls is N times the mean loss per wall
while it is small; at large N the longitudinal share saturates at one third.  --monte-carlo checks that statement by propagating
polarisation amplitudes, coherently, through randomly oriented walls.

Stated assumptions
------------------
1. Walls are thick (adiabatic).  For thin walls the conversion is larger by a
   factor of order one; --thin prints that case for mass steps.
2. Wall orientations are independent.  Perfectly aligned walls would not
   accumulate at all (the conversion is undone at the next wall).
3. The observed frequency is used along the whole path.  The emitted frequency
   is higher, which only lowers the loss, so the result is an upper estimate.
4. Longitudinal photons count as lost (they couple to matter more weakly by
   m / omega).
5. Most of the mean conversion comes from near-grazing incidence (the angle
   average grows like ln(omega / m)), which is where the thick-wall formula is
   least secure.  Counting only angles below 80 degrees would lower the loss
   about 15-fold in the default scenario.
"""
import argparse
import sys
import numpy as np

HBAR_EV_S = 6.582119569e-16        # eV s
C_M_S = 299792458.0                # m / s
PARSEC_M = 3.0856775814913673e16   # m


def _trapezoid(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# --------------------------------------------------------------------------
# One wall, thick-wall limit.  Everything in units of omega (pass m / omega).
# --------------------------------------------------------------------------
def wall_rotation(u, ra, rb):
    """
    Crossing from a domain with m_a/omega = ra into m_b/omega = rb at
    cos(theta) = u.  Masses are signed (a kink wall is ra = -rb).

    Returns (sin_dalpha, cos_dalpha, reflected, qb_over_omega) where `reflected`
    is True for total reflection.  The (transverse-TM, longitudinal) amplitudes
    transform as  [[cos, -sin], [sin, cos]].
    """
    u = np.asarray(u, float)
    K2 = 1.0 - ra * ra
    k2 = K2 * (1.0 - u * u)
    qa = np.sqrt(K2) * u
    qb2 = qa * qa + ra * ra - rb * rb
    reflected = qb2 <= 0.0
    qb = np.sqrt(np.where(reflected, 1.0, qb2))
    k = np.sqrt(k2)
    C = qa * qb + k2 * ra * rb
    S = k * (rb * qa - ra * qb)
    norm = np.sqrt(C * C + S * S)
    return S / norm, C / norm, reflected, qb


def thin_wall_conversion(u, ra, rb):
    """Thin-wall limit of the transmitted TM -> longitudinal power, and the reflected power."""
    u = np.asarray(u, float)
    K2 = 1.0 - ra * ra
    k = np.sqrt(K2 * (1.0 - u * u))
    qa = np.sqrt(K2) * u
    qb2 = qa * qa + ra * ra - rb * rb
    reflected = qb2 <= 0.0
    qb = np.sqrt(np.where(reflected, 1.0, qb2))
    tE = 2 * qa / (qa + qb)
    rE = (qa - qb) / (qa + qb)
    pa, pb = qa / ra ** 2, qb / rb ** 2
    tPi = 2 * pa / (pa + pb)
    rPi = (pa - pb) / (pa + pb)
    Kb = np.sqrt(1.0 - rb * rb)
    Db = qb * qb + rb * rb * k * k
    Da = qa * qa + ra * ra * k * k
    Ka = np.sqrt(K2)
    aL = (Kb * k / (rb * Db * Ka)) * (rb ** 2 * qa * tE - ra ** 2 * qb * tPi)
    conv = aL ** 2 * qb / qa
    refl = ((rPi * ra ** 2 * k * k - rE * qa * qa) ** 2 + (qa * ra * k * (rE + rPi)) ** 2) / Da ** 2
    return np.where(reflected, 0.0, conv), np.where(reflected, 1.0, refl)


def mean_loss_per_wall(ra, rb, thin=False, n=6000):
    """
    Mean fractional flux loss per wall for unpolarised light, averaged over
    incidence angle (p(u) = 2u) and over both crossing directions.

    Returns dict(conversion, reflection, total): conversion already includes the
    factor 1/2 for the TM share; reflection applies to both polarisations in the
    total-reflection window and to TM only outside it (thin walls).
    """
    conv_sum = refl_sum = 0.0
    for a, b in ((ra, rb), (rb, ra)):
        scale = max(abs(a), abs(b), 1e-300)
        x = np.linspace(np.log(scale * 1e-4), 0.0, n)     # x = ln u, resolves grazing incidence
        u = np.exp(x)
        if thin:
            conv, refl = thin_wall_conversion(u, a, b)
            total_refl = np.zeros_like(u, bool) | (refl >= 1.0)
            conv_y = 0.5 * conv
            refl_y = np.where(total_refl, 1.0, 0.5 * refl)
        else:
            s, _, total_refl, _ = wall_rotation(u, a, b)
            conv_y = np.where(total_refl, 0.0, 0.5 * s * s)
            refl_y = total_refl.astype(float)
        w = 2.0 * u * u                                    # p(u) du = 2 u^2 dx
        conv_sum += _trapezoid(conv_y * w, x)
        refl_sum += _trapezoid(refl_y * w, x)
    conv, refl = 0.5 * conv_sum, 0.5 * refl_sum
    return dict(conversion=conv, reflection=refl, total=conv + refl)


# --------------------------------------------------------------------------
# Amplitude-level Monte Carlo through randomly oriented thick walls
# --------------------------------------------------------------------------
def monte_carlo(ra, rb, n_walls, n_rays=20000, seed=0):
    """
    Propagate n_rays unpolarised photons through n_walls walls.  Each ray keeps
    its direction d, a complex transverse polarisation vector and a longitudinal
    amplitude; every wall refracts the ray and rotates (TM, longitudinal)
    coherently.  Totally reflected rays are removed.

    Returns dict of arrays over wall number 1..n_walls:
        longitudinal : mean |a_L|^2 over all launched rays
        reflected    : fraction of rays removed so far
    """
    rng = np.random.default_rng(seed)
    d = np.tile([0.0, 0.0, 1.0], (n_rays, 1))
    phi = rng.uniform(0, 2 * np.pi, n_rays)
    pol = np.stack([np.cos(phi), np.sin(phi), np.zeros(n_rays)], axis=1).astype(complex)
    aL = np.zeros(n_rays, complex)
    alive = np.ones(n_rays, bool)
    masses = (ra, rb)
    out_L, out_R = np.empty(n_walls), np.empty(n_walls)

    for i in range(n_walls):
        ma, mb = masses[i % 2], masses[(i + 1) % 2]
        # random wall normal with cos(theta) ~ p(u) = 2u about the ray direction
        u = np.sqrt(1.0 - rng.uniform(0, 1, n_rays))          # in (0, 1]
        az = rng.uniform(0, 2 * np.pi, n_rays)
        helper = np.where(np.abs(d[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        e1 = np.cross(d, helper)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(d, e1)
        sin_t = np.sqrt(1.0 - u * u)
        tang = np.cos(az)[:, None] * e1 + np.sin(az)[:, None] * e2      # unit, perpendicular to d
        nrm = u[:, None] * d + sin_t[:, None] * tang                    # wall normal, n.d = u > 0
        t_hat = d - u[:, None] * nrm                                    # tangential direction of d
        t_norm = np.linalg.norm(t_hat, axis=1, keepdims=True)
        t_hat = t_hat / np.where(t_norm > 0, t_norm, 1.0)

        sin_a, cos_a, refl, qb = wall_rotation(u, ma, mb)
        Ka, Kb = np.sqrt(1 - ma * ma), np.sqrt(1 - mb * mb)
        k = Ka * sin_t
        qa = Ka * u
        s_dir = np.cross(nrm, d)                                        # TE direction
        s_norm = np.linalg.norm(s_dir, axis=1, keepdims=True)
        s_dir = s_dir / np.where(s_norm > 0, s_norm, 1.0)
        p_in = (qa[:, None] * t_hat - k[:, None] * nrm) / Ka
        p_out = (qb[:, None] * t_hat - k[:, None] * nrm) / Kb
        a_s = np.einsum("ij,ij->i", pol, s_dir)
        a_p = np.einsum("ij,ij->i", pol, p_in)
        new_p = cos_a * a_p - sin_a * aL
        new_L = sin_a * a_p + cos_a * aL

        go = alive & ~refl
        alive &= ~refl
        pol = np.where(go[:, None], a_s[:, None] * s_dir + new_p[:, None] * p_out, 0.0)
        aL = np.where(go, new_L, 0.0)
        d_new = (k[:, None] * t_hat + qb[:, None] * nrm) / Kb
        d = np.where(go[:, None], d_new, d)

        out_L[i] = np.mean(np.abs(aL) ** 2)
        out_R[i] = 1.0 - alive.mean()
    return dict(longitudinal=out_L, reflected=out_R)


def network_loss(n_walls, per_wall):
    """
    Fractional loss of transverse (detectable) flux after n_walls walls.

    With c = conversion and r = reflection per wall:  reflection removes photons,
    exp(-N r) survive.  Conversion is reversible (a longitudinal photon converts
    back at rate 2c), so the longitudinal share relaxes to one third of the
    surviving flux:  P_L = (1 - exp(-3 N c)) / 3.   For small N this is N (c + r).
    """
    n = np.asarray(n_walls, float)
    p_long = -np.expm1(-3.0 * n * per_wall["conversion"]) / 3.0
    return 1.0 - np.exp(-n * per_wall["reflection"]) * (1.0 - p_long)


# --------------------------------------------------------------------------
# Scenario in physical units
# --------------------------------------------------------------------------
def scenario(mass_ev, contrast, freq_ghz, distance_gpc, spacing_pc, kink=False, thin=False):
    """
    Loss for a sightline of distance_gpc through domains with mean wall spacing
    spacing_pc.  Domain masses are mass_ev and mass_ev * (1 + contrast), or
    +mass_ev / -mass_ev for kink walls.
    """
    omega = 2 * np.pi * freq_ghz * 1e9                     # rad / s
    ra = mass_ev / HBAR_EV_S / omega
    rb = -ra if kink else ra * (1.0 + contrast)
    if max(abs(ra), abs(rb)) >= 1.0:
        raise ValueError("photon mass exceeds the photon energy: the wave does not propagate")
    per_wall = mean_loss_per_wall(ra, rb, thin=thin and not kink)
    n_walls = distance_gpc * 1e9 / np.asarray(spacing_pc, float)
    return dict(m_over_omega=(ra, rb), per_wall=per_wall, n_walls=n_walls,
                loss=network_loss(n_walls, per_wall),
                wavelength_m=C_M_S / (freq_ghz * 1e9))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mass-ev", type=float, default=1e-14, help="photon mass in one domain [eV]")
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="fractional mass change between domains: m_b = m_a (1 + contrast)")
    ap.add_argument("--kink", action="store_true", help="kink walls: the mass changes sign instead")
    ap.add_argument("--thin", action="store_true", help="use the thin-wall limit (mass steps only)")
    ap.add_argument("--freq-ghz", type=float, default=1.0, help="observed frequency [GHz]")
    ap.add_argument("--distance-gpc", type=float, default=1.0, help="path length [Gpc]")
    ap.add_argument("--spacing-pc", type=float, nargs="+",
                    default=[1e6, 1e3, 1.0, 1e-3, 1e-6, 1e-9],
                    help="mean wall spacing(s) [pc]")
    ap.add_argument("--target", type=float, default=1e-7, help="fractional loss taken as detectable")
    ap.add_argument("--monte-carlo", action="store_true",
                    help="check the N-wall scaling with an amplitude-level simulation")
    ap.add_argument("--plot", metavar="FILE", help="save a figure of loss against wall spacing")
    ap.add_argument("--csv", metavar="FILE", help="save the table as CSV")
    a = ap.parse_args(argv)

    res = scenario(a.mass_ev, a.contrast, a.freq_ghz, a.distance_gpc, a.spacing_pc, a.kink, a.thin)
    pw = res["per_wall"]
    ra, rb = res["m_over_omega"]
    print("Wall-network estimate")
    print(f"  photon mass        {a.mass_ev:.3g} eV   ->  {'-' if a.kink else ''}"
          f"{a.mass_ev * (1 if a.kink else 1 + a.contrast):.3g} eV  "
          f"({'kink' if a.kink else 'step'} walls, {'thin' if a.thin and not a.kink else 'thick'})")
    print(f"  frequency          {a.freq_ghz:g} GHz   (wavelength {res['wavelength_m']:.3g} m)")
    print(f"  m c^2 / hbar omega {abs(ra):.3e}, {abs(rb):.3e}")
    print(f"  path length        {a.distance_gpc:g} Gpc")
    print()
    print("Mean loss per wall (unpolarised light, angle-averaged)")
    print(f"  conversion to longitudinal   {pw['conversion']:.3e}")
    print(f"  total reflection             {pw['reflection']:.3e}")
    print(f"  total                        {pw['total']:.3e}")
    print()
    print(f"  {'wall spacing':>14s}  {'walls crossed':>14s}  {'fractional loss':>16s}")
    rows = []
    for sp, n, loss in zip(a.spacing_pc, np.atleast_1d(res["n_walls"]), np.atleast_1d(res["loss"])):
        metres = sp * PARSEC_M
        label = f"{sp:g} pc" if sp >= 1e-3 else f"{metres:.3g} m"
        note = "   (below the wavelength: thick-wall limit not valid)" if metres < res["wavelength_m"] else ""
        print(f"  {label:>14s}  {n:14.3e}  {loss:16.3e}{note}")
        rows.append((sp, metres, n, loss))
    n_need = a.target / pw["total"]                        # small-loss regime
    sp_need = a.distance_gpc * 1e9 * PARSEC_M / n_need
    print()
    print(f"To lose a fraction {a.target:g} needs {n_need:.2e} walls,")
    print(f"i.e. a wall every {sp_need:.3g} m ({sp_need / res['wavelength_m']:.3g} wavelengths).")
    if sp_need < res["wavelength_m"]:
        print("That is closer together than one wavelength, so no thick-wall network can reach it.")

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("spacing_pc,spacing_m,walls_crossed,fractional_loss\n")
            f.writelines(f"{s:.6g},{m:.6g},{n:.6g},{l:.6g}\n" for s, m, n, l in rows)
        print("wrote", a.csv)

    if a.monte_carlo:
        # Run at mass ratios large enough to simulate; the per-wall law is the same.
        mc_ra = 0.005
        mc_rb = -mc_ra if a.kink else mc_ra * (1.0 + a.contrast)
        n_w = 200
        mc = monte_carlo(mc_ra, mc_rb, n_w, n_rays=40000, seed=1)
        pred = mean_loss_per_wall(mc_ra, mc_rb)
        print()
        print(f"Monte Carlo check at m/omega = {mc_ra:g}, {abs(mc_rb):g}  ({n_w} walls, 40000 rays)")
        print(f"  {'walls':>6s}  {'longitudinal (MC)':>18s}  {'N x conversion':>15s}  "
              f"{'reflected (MC)':>15s}  {'N x reflection':>15s}")
        for n in (10, 50, 100, 200):
            print(f"  {n:6d}  {mc['longitudinal'][n - 1]:18.3e}  {n * pred['conversion']:15.3e}  "
                  f"{mc['reflected'][n - 1]:15.3e}  {n * pred['reflection']:15.3e}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sp = np.geomspace(res["wavelength_m"] / PARSEC_M, 1e6, 200)
        r = scenario(a.mass_ev, a.contrast, a.freq_ghz, a.distance_gpc, sp, a.kink, a.thin)
        fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
        ax.loglog(sp * PARSEC_M, r["loss"], color="#2a78d6", lw=2, label="fractional flux loss")
        ax.axhline(a.target, color="#7a7972", ls="--", lw=1, label=f"target {a.target:g}")
        ax.set_xlabel("mean wall spacing [m]  (left edge = one wavelength)")
        ax.set_ylabel("fractional flux loss")
        ax.set_title(f"{a.distance_gpc:g} Gpc sightline, {a.freq_ghz:g} GHz, m = {a.mass_ev:g} eV", loc="left")
        ax.grid(True, which="major", color="#e1e0d9")
        ax.legend(frameon=False)
        fig.savefig(a.plot, dpi=160)
        print("wrote", a.plot)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
