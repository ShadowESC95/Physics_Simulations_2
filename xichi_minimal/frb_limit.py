"""
Step 5: confront the model with data. Photon-mass limit from localized FRBs.

In the vacuum Xi = v the model's photon has mass m = g v, so radio waves obey
w^2 = k^2 c^2 + m^2 c^4 / hbar^2. Lower frequencies arrive later, with the
same 1/nu^2 law as dispersion by free electrons. A burst from redshift z
therefore acquires an apparent extra dispersion measure
(Bonetti et al. 2016, arXiv:1602.09135, Eqs. 3, 6, 7)

    DM_gamma(z) = [4 pi^2 eps0 m_e c^5 / (h^2 e^2)] * m^2 * H_gamma(z) / H0,
    H_gamma(z)  = int_0^z dz' / [(1+z')^2 E(z')],
    E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda),

while the electrons of the intergalactic medium give
    DM_IGM(z) = [3 c H0 Omega_b f_IGM chi_e / (8 pi G m_p)] * H_e(z),
    H_e(z) = int_0^z (1+z') dz' / E(z').
H_gamma flattens with redshift and H_e keeps growing, so a sample of bursts
with known redshift can tell the two apart.

Data: data/localized_frbs.csv, taken from the public FRBs/FRB repository
(see data/README.md). No simulated bursts are used anywhere in this file.

Two analyses
------------
1. Conservative, model-free: for each burst attribute *all* dispersion beyond
   the Milky Way disc (NE2001) to the photon mass. Every burst gives an upper
   limit on m; none can give a detection.
2. Joint fit: DM_obs - DM_MW = DM_IGM(z) + DM_host/(1+z) + DM_gamma(z; m),
   Gaussian scatter, profiled over the host term and f_IGM. Gives an upper
   limit on m and a chi-square verdict on any chosen mass.

The Gaussian scatter model in (2) is crude: the true distributions are skewed
and the scatter values are assumptions, listed in `FitConfig`. The result
should be read to within a factor of about two. Analysis (1) has no such
assumptions. frb_likelihood.py repeats the joint fit with the skewed
distributions used in FRB cosmology and gives a limit about twice as tight.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

# CODATA 2018 (SI)
C = 299792458.0
H = 6.62607015e-34
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
EPS0 = 8.8541878128e-12
G_NEWTON = 6.67430e-11
PARSEC = 3.0856775814913673e16
EV = E_CHARGE
PC_CM3 = PARSEC * 1e6                      # 1 pc cm^-3 in m^-2

# Planck 2018 flat LambdaCDM
H0 = 67.4e3 / (1e6 * PARSEC)               # s^-1
OMEGA_M = 0.315
OMEGA_L = 1.0 - OMEGA_M
OMEGA_B = 0.0224 / 0.674**2
CHI_E = 0.875                              # electrons per baryon, H + He fully ionised

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "localized_frbs.csv")


def e_of_z(z):
    return np.sqrt(OMEGA_M * (1 + z) ** 3 + OMEGA_L)


def h_gamma(z):
    return quad(lambda u: 1.0 / ((1 + u) ** 2 * e_of_z(u)), 0.0, z)[0]


def h_e(z):
    return quad(lambda u: (1 + u) / e_of_z(u), 0.0, z)[0]


def dispersion_constant():
    """K in  dt = K * DM * nu^-2  (SI: s Hz^2 m^2)."""
    return E_CHARGE**2 / (8 * np.pi**2 * EPS0 * M_E * C)


def mass_kg(m_ev):
    return m_ev * EV / C**2


def dm_gamma(z, m_ev):
    """Apparent dispersion measure from a photon mass, in pc cm^-3."""
    coeff = 4 * np.pi**2 * EPS0 * M_E * C**5 / (H**2 * E_CHARGE**2)
    return coeff * mass_kg(m_ev) ** 2 * h_gamma(z) / H0 / PC_CM3


def photon_delay(z, m_ev, nu_hz):
    """Arrival delay relative to infinite frequency, from first principles (s)."""
    return (mass_kg(m_ev) * C**2 / H) ** 2 / (2 * nu_hz**2) * h_gamma(z) / H0


def dm_igm(z, f_igm=0.84):
    coeff = 3 * C * H0 * OMEGA_B * CHI_E / (8 * np.pi * G_NEWTON * M_P)
    return coeff * f_igm * h_e(z) / PC_CM3


def mass_for_dm(z, dm):
    """Photon mass (eV) whose DM_gamma at redshift z equals dm (pc cm^-3)."""
    return np.sqrt(dm / dm_gamma(z, 1.0))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(path=DATA, z_min=0.0):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            z = float(r["z"])
            if z <= z_min:
                continue
            rows.append((r["frb"], float(r["dm_obs_pc_cm3"]),
                         float(r["dm_mw_ism_ne2001_pc_cm3"]), z))
    return rows


# --------------------------------------------------------------------------
# Analysis 1: conservative per-burst limits
# --------------------------------------------------------------------------
def conservative_limits(rows=None):
    """[(frb, z, excess DM, m_max in eV)], sorted by m_max.

    Only the Milky Way disc is subtracted. The halo, the intergalactic medium
    and the host all add dispersion, so the true bound is tighter.
    """
    rows = load() if rows is None else rows
    out = []
    for name, dm, ism, z in rows:
        excess = dm - ism
        if excess > 0:
            out.append((name, z, excess, mass_for_dm(z, excess)))
    return sorted(out, key=lambda r: r[3])


# --------------------------------------------------------------------------
# Analysis 2: joint fit
# --------------------------------------------------------------------------
@dataclass
class FitConfig:
    z_min: float = 0.01          # below this, peculiar velocities dominate z
    dm_halo: float = 50.0        # Milky Way halo, pc cm^-3 (assumed)
    sigma_mw: float = 30.0       # uncertainty of disc + halo (assumed)
    sigma_host: float = 100.0    # rest-frame host scatter (assumed)
    sigma_igm_coeff: float = 173.8   # sigma_IGM = coeff * z^0.4 (assumed)
    sigma_scale: float = 1.0     # multiplies every scatter, for sensitivity checks
    f_igm_range: tuple = (0.5, 1.0)
    host_range: tuple = (0.0, 400.0)


class JointFit:
    def __init__(self, rows=None, config: FitConfig | None = None):
        self.cfg = cfg = config or FitConfig()
        rows = load(z_min=cfg.z_min) if rows is None else [r for r in rows if r[3] > cfg.z_min]
        self.names = [r[0] for r in rows]
        self.z = np.array([r[3] for r in rows])
        self.excess = np.array([r[1] - r[2] for r in rows]) - cfg.dm_halo
        self.he = np.array([h_e(z) for z in self.z])
        self.igm_unit = np.array([dm_igm(z, 1.0) for z in self.z])
        self.gamma_unit = np.array([dm_gamma(z, 1.0) for z in self.z])   # per eV^2
        self.sigma = cfg.sigma_scale * np.sqrt(
            cfg.sigma_mw**2 + (cfg.sigma_igm_coeff * self.z**0.4) ** 2
            + (cfg.sigma_host / (1 + self.z)) ** 2)

    def chi2(self, m_ev, f_igm, host):
        model = f_igm * self.igm_unit + host / (1 + self.z) + m_ev**2 * self.gamma_unit
        return float(np.sum(((self.excess - model) / self.sigma) ** 2))

    def profile(self, m_ev):
        """chi-square at fixed mass, minimised over f_IGM and the host term."""
        cfg = self.cfg
        best = None
        for f0 in (0.55, 0.75, 0.95):
            for h0 in (20.0, 150.0, 300.0):
                r = minimize(lambda p: self.chi2(m_ev, p[0], p[1]), [f0, h0],
                             bounds=[cfg.f_igm_range, cfg.host_range], method="L-BFGS-B")
                if best is None or r.fun < best.fun:
                    best = r
        return best.fun, best.x

    def best_fit(self):
        if getattr(self, "_best", None) is None:
            self._best = self._scan()
        return self._best

    def _scan(self):
        grid = np.concatenate([[0.0], np.logspace(-16.5, -13, 60)])
        vals = [self.profile(m)[0] for m in grid]
        i = int(np.argmin(vals))
        return grid[i], vals[i]

    def upper_limit(self, delta_chi2=2.71):
        """One-sided limit: mass where the profile rises by delta_chi2 (2.71 = 95%)."""
        m_best, c_best = self.best_fit()
        f = lambda logm: self.profile(10.0**logm)[0] - c_best - delta_chi2
        lo = np.log10(max(m_best, 1e-17))
        return 10.0 ** brentq(f, lo, -12.0, xtol=1e-4)

    def unit_chi2_scale(self):
        """Factor by which the scatters must grow for chi2/dof = 1 at m = 0."""
        return float(np.sqrt(self.profile(0.0)[0] / (len(self.z) - 2)))

    def verdict(self, m_ev):
        """Delta chi-square of a proposed mass relative to the best fit."""
        return self.profile(m_ev)[0] - self.best_fit()[1]


def rescaled_fit(**overrides):
    """Joint fit with the scatters enlarged until chi2/dof = 1 at m = 0.

    The assumed scatters underestimate the real spread (a few bursts have very
    large host contributions), so the unscaled chi-square is too large to be
    read as a probability. Enlarging the errors is the conservative repair.
    """
    scale = JointFit(config=FitConfig(**overrides)).unit_chi2_scale()
    return JointFit(config=FitConfig(sigma_scale=max(scale, 1.0), **overrides))


def report():
    lims = conservative_limits()
    raw = JointFit()
    fit = rescaled_fit()
    fixed = rescaled_fit(f_igm_range=(0.84, 0.84))
    m_best, c_best = fit.best_fit()
    lines = [
        f"Localized FRBs in the table: {len(load())}; used in the joint fit (z > {fit.cfg.z_min}): {len(fit.z)}",
        "",
        "Analysis 1 - conservative per-burst limits (all extragalactic DM given to the photon mass):",
    ]
    for name, z, exc, m in lims[:5]:
        lines.append(f"  {name}  z = {z:.4f}  DM excess = {exc:7.1f}  ->  m < {m:.2e} eV")
    lines += [
        f"  best single-burst limit: m < {lims[0][3]:.2e} eV  ({mass_kg(lims[0][3]):.2e} kg)",
        "",
        "Analysis 2 - joint fit:",
        f"  chi2 at m = 0 with the assumed scatters: {raw.profile(0.0)[0]:.1f} for {len(raw.z)} bursts"
        f"  -> scatters enlarged x{fit.cfg.sigma_scale:.2f}",
        f"  best-fit mass: {m_best:.2e} eV; chi2 improvement over m = 0: {fit.profile(0.0)[0] - c_best:.2f}"
        "  (below 1: no evidence for a mass)",
        f"  95% upper limit, f_IGM free in {fit.cfg.f_igm_range}: m < {fit.upper_limit():.2e} eV",
        f"  95% upper limit, f_IGM fixed at 0.84:      m < {fixed.upper_limit():.2e} eV",
        "  verdict on proposed masses (delta chi2 > 3.84 means excluded at 95%):",
    ]
    for m in (1e-15, 1e-14, 2e-14, 3e-14):
        lines.append(f"    m = {m:.0e} eV: delta chi2 = {fit.verdict(m):.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
