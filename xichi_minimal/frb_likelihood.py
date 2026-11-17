"""
Photon-mass limit from localized FRBs with skewed dispersion distributions.

frb_limit.JointFit treats the scatter of the intergalactic and host dispersion
as Gaussian. Neither is: both are bounded below and have long upper tails.
This module replaces the Gaussians with the distributions used in FRB
cosmology (Macquart et al. 2020, Nature 581, 391, arXiv:2005.13161).

Budget for one burst
--------------------
    DM_obs - DM_disc - DM_halo = DM_cosmic + DM_host / (1 + z) + DM_gamma(z; m) + e

* DM_cosmic = Delta * <DM_cosmic>(z), with <DM_cosmic> = frb_limit.dm_igm and

      p(Delta) = A Delta^-beta exp( -(Delta^-alpha - C0)^2 / (2 alpha^2 s^2) ),
      alpha = beta = 3,   s = F / sqrt(z),

  where C0 is fixed by <Delta> = 1 and A by normalisation. F measures how
  strongly galactic feedback smooths the intergalactic gas.
* DM_host is log-normal in the host frame: ln DM_host ~ N(mu, sigma_host^2).
* DM_gamma is the photon-mass term of frb_limit.dm_gamma. It is not random.
* e is the error of the Milky Way subtraction (disc model plus halo),
  Gaussian with width sigma_mw. Without it a burst whose dispersion is below
  the Milky Way model (FRB 20220319D) would have zero likelihood.

The likelihood of a burst is the convolution of the three random terms. It is
computed on a uniform DM grid: each term is turned into exact cell
probabilities (from its cumulative distribution), the three are convolved by
FFT, and the result is read off at the burst's DM. Nuisance parameters F, mu,
sigma_host and f_IGM are profiled out. A photon mass can only add dispersion,
so the lower edges of the distributions are what constrain it; a Gaussian has
no such edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import erf

import frb_limit as fl

ALPHA = 3.0   # p(Delta) uses alpha = beta = 3 throughout (Macquart et al. 2020)
FLOOR = 1e-14  # smallest density kept (per pc cm^-3): below it the FFT is round-off


# --------------------------------------------------------------------------
# Intergalactic term
# --------------------------------------------------------------------------
def _moments(c0, s):
    """Integrals of Delta^-3 exp(...) and Delta^-2 exp(...) over Delta > 0.

    With w = 1 / Delta they become  int w g dw  and  int g dw,
    g = exp(-(w^3 - c0)^2 / (18 s^2)), which are smooth.
    """
    g = lambda w: np.exp(-((w**3 - c0) ** 2) / (18.0 * s * s))
    top = (max(c0, 0.0) + 40.0 * s) ** (1.0 / 3.0)
    peak = max(c0, 0.0) ** (1.0 / 3.0)
    pts = [peak] if 0.0 < peak < top else None
    norm = quad(lambda w: w * g(w), 0.0, top, points=pts, limit=200)[0]
    mean = quad(g, 0.0, top, points=pts, limit=200)[0]
    return norm, mean


def cosmic_constants(s):
    """(C0, A) for one value of s: mean of Delta equal to 1, unit integral."""
    f = lambda c0: _moments(c0, s)[1] / _moments(c0, s)[0] - 1.0
    c0 = brentq(f, -30.0 * s, 5.0 + 5.0 * s, xtol=1e-10)
    return c0, 1.0 / _moments(c0, s)[0]


@lru_cache(maxsize=1)
def _table():
    ln_s = np.linspace(np.log(0.004), np.log(8.0), 160)
    vals = np.array([cosmic_constants(np.exp(v)) for v in ln_s])
    return ln_s, vals[:, 0], np.log(vals[:, 1])


def p_cosmic(delta, s):
    """Probability density of Delta = DM_cosmic / <DM_cosmic>; zero for Delta <= 0."""
    delta = np.asarray(delta, float)
    ln_s, c0_tab, ln_a_tab = _table()
    c0 = np.interp(np.log(s), ln_s, c0_tab)
    ln_a = np.interp(np.log(s), ln_s, ln_a_tab)
    ok = delta > 0
    d = np.where(ok, delta, 1.0)
    expo = ln_a - ALPHA * np.log(d) - (d**-ALPHA - c0) ** 2 / (2.0 * ALPHA**2 * s * s)
    return np.where(ok, np.exp(expo), 0.0)


def p_host(dm_host, mu, sigma):
    """Log-normal density of the host-frame DM (pc cm^-3)."""
    dm_host = np.asarray(dm_host, float)
    ok = dm_host > 0
    h = np.where(ok, dm_host, 1.0)
    return np.where(ok, np.exp(-((np.log(h) - mu) ** 2) / (2 * sigma * sigma))
                    / (h * sigma * np.sqrt(2 * np.pi)), 0.0)


# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------
@dataclass
class SkewConfig:
    z_min: float = 0.01          # below this, peculiar velocities dominate z
    dm_halo: float = 50.0        # Milky Way halo, pc cm^-3 (assumed, as Macquart et al.)
    sigma_mw: float = 30.0       # error of disc + halo subtraction (assumed)
    f_range: tuple = (0.011, 0.5)        # F: prior range of Macquart et al.
    mu_range: tuple = (np.log(20.0), np.log(300.0))   # median host DM 20-300
    sigma_host_range: tuple = (0.2, 2.0)
    f_igm_range: tuple = (0.5, 1.0)
    step: float = 1.0            # DM grid spacing, pc cm^-3
    n_grid: int | None = None    # FFT length; None picks the smallest power of two that fits
    n_w: int = 1500              # nodes for the cumulative intergalactic distribution


# starting points (F, mu, sigma_host, f_IGM) for the first minimisation
STARTS = [(0.3, np.log(100.0), 0.8, 0.84), (0.15, np.log(60.0), 1.2, 0.7),
          (0.45, np.log(150.0), 0.5, 0.95)]


class SkewFit:
    """Profile likelihood for the photon mass with skewed distributions."""

    def __init__(self, rows=None, config: SkewConfig | None = None):
        self.cfg = cfg = config or SkewConfig()
        rows = fl.load(z_min=cfg.z_min) if rows is None else [r for r in rows if r[3] > cfg.z_min]
        self.frbs = [r[0] for r in rows]
        self.z = np.array([r[3] for r in rows])
        self.excess = np.array([r[1] - r[2] for r in rows]) - cfg.dm_halo
        self.igm_unit = np.array([fl.dm_igm(z, 1.0) for z in self.z])
        self.gamma_unit = np.array([fl.dm_gamma(z, 1.0) for z in self.z])   # per eV^2
        d = cfg.step
        need = 2 * (self.excess.max() / d + int(8 * cfg.sigma_mw / d) + 200)
        n = self._n = cfg.n_grid or int(2 ** np.ceil(np.log2(need)))
        # Cells kept below zero for the Milky Way error. The two positive terms
        # are kept on `m` cells each, so their sum plus the error never wraps
        # round the periodic FFT grid into the range that is read out.
        self._neg = int(8 * cfg.sigma_mw / d) + 2
        self._m = m = n // 2 - self._neg
        if self.excess.max() >= m * d:
            raise ValueError("n_grid * step is too small for the largest DM in the sample")
        self._edges = d * np.arange(m + 1)                 # cell edges of the positive terms
        k = np.fft.fftfreq(n, 1.0 / n)                     # 0, 1, ..., -2, -1 (wrapped)
        cdf = lambda x: 0.5 * (1.0 + erf(x / (np.sqrt(2.0) * cfg.sigma_mw)))
        self._mw_hat = np.fft.rfft(cdf((k + 0.5) * d) - cdf((k - 0.5) * d))
        self._axis = d * (np.arange(-self._neg, m) + 1.0)  # DM at the centre of each sum cell
        self._bounds = [cfg.f_range, cfg.mu_range, cfg.sigma_host_range, cfg.f_igm_range]
        self._cache = {}

    # ---- one burst --------------------------------------------------------
    def _host_cells(self, mu, sigma_host):
        """P(host DM / (1+z) in each grid cell), exact from the log-normal CDF."""
        e = self._edges[None, 1:] * (1 + self.z[:, None])
        c = 0.5 * (1.0 + erf((np.log(e) - mu) / (np.sqrt(2.0) * sigma_host)))
        return np.diff(np.concatenate([np.zeros((len(self.z), 1)), c], axis=1), axis=1)

    def _cosmic_cells(self, F, f_igm):
        """P(DM_cosmic in each grid cell), from the cumulative of p(Delta).

        In w = 1 / Delta the density is proportional to w exp(-(w^3 - C0)^2 /
        (18 s^2)), which is smooth, so its running integral on a uniform w
        grid gives P(Delta > 1/w) accurately.
        """
        s = F / np.sqrt(self.z)
        ln_s, c0_tab, _ = _table()
        c0 = np.interp(np.log(s), ln_s, c0_tab)
        top = (np.maximum(c0, 0.0) + 40.0 * s) ** (1.0 / 3.0)
        w = top[:, None] * np.linspace(0.0, 1.0, self.cfg.n_w)[None, :]
        dens = w * np.exp(-((w**3 - c0[:, None]) ** 2) / (18.0 * s[:, None] ** 2))
        # integrate downwards from the top, so that the small probabilities
        # of the lower tail of Delta are sums, not differences of numbers near 1
        slab = 0.5 * (dens[:, 1:] + dens[:, :-1])
        above = np.concatenate([np.cumsum(slab[:, ::-1], axis=1)[:, ::-1],
                                np.zeros((len(s), 1))], axis=1)
        # ln P(w' > w) = ln P(Delta < 1/w); interpolated in the logarithm
        # because the lower tail of Delta falls faster than exponentially
        ln_frac = np.log(np.maximum(above / above[:, :1], 1e-300))
        mean = f_igm * self.igm_unit
        out = np.empty((len(s), len(self._edges) - 1))
        for i in range(len(s)):
            w_edge = mean[i] / self._edges[1:]             # decreasing in DM
            c = np.exp(np.interp(w_edge, w[i], ln_frac[i], right=-np.inf))
            out[i] = np.diff(np.concatenate([[0.0], c]))
        return out

    def extragalactic_density(self, F, mu, sigma_host, f_igm):
        """Density of DM_cosmic + DM_host/(1+z) + e on self._axis, one row per burst."""
        n = self._n
        hat = (np.fft.rfft(self._host_cells(mu, sigma_host), n=n, axis=1)
               * np.fft.rfft(self._cosmic_cells(F, f_igm), n=n, axis=1) * self._mw_hat[None, :])
        pmf = np.fft.irfft(hat, n=n, axis=1)
        pmf = np.concatenate([pmf[:, n - self._neg:], pmf[:, :self._m]], axis=1)
        return np.maximum(pmf / self.cfg.step, FLOOR)

    def burst_likelihoods(self, m_ev, F, mu, sigma_host, f_igm):
        """Probability density of each burst's observed excess DM (per pc cm^-3)."""
        dens = self.extragalactic_density(F, mu, sigma_host, f_igm)
        x = self.excess - m_ev**2 * self.gamma_unit
        # interpolate ln(density): it is close to quadratic in the tails
        ln_d = np.log(dens)
        return np.exp([np.interp(x[i], self._axis, ln_d[i]) for i in range(len(x))])

    def nll(self, m_ev, F, mu, sigma_host, f_igm):
        """-2 ln L, the analogue of chi-square."""
        like = self.burst_likelihoods(m_ev, F, mu, sigma_host, f_igm)
        return float(-2.0 * np.sum(np.log(like)))

    # ---- profiling --------------------------------------------------------
    def profile(self, m_ev):
        """-2 ln L at fixed mass, minimised over the nuisance parameters.

        The first call tries several starting points. Later calls start from
        the solution at the nearest mass already solved, which is close to
        the new optimum because the profile is continuous in the mass.
        """
        key = float(m_ev)
        if key not in self._cache:
            if self._cache:
                near = min(self._cache, key=lambda k: abs(k - key))
                starts = [tuple(self._cache[near][1])]
            else:
                starts = STARTS
            best = None
            for p0 in starts:
                r = minimize(lambda p: self.nll(m_ev, *p), p0, bounds=self._bounds,
                             method="L-BFGS-B")
                if best is None or r.fun < best.fun:
                    best = r
            self._cache[key] = (float(best.fun), best.x)
        return self._cache[key]

    def curve(self, masses):
        """Profile along increasing masses. Returns (-2 ln L values, parameters)."""
        out = [self.profile(m) for m in masses]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    def best_fit(self):
        """(mass, -2 ln L) at the minimum of the profile on a grid from zero."""
        if getattr(self, "_best", None) is None:
            grid = np.linspace(0.0, 3e-14, 16)
            vals, _ = self.curve(grid)
            i = int(np.argmin(vals))
            self._best = (grid[i], vals[i])
        return self._best

    def upper_limit(self, delta=2.71):
        """Mass where -2 ln L exceeds its minimum by delta.

        delta = 2.71 is the one-sided 95% limit and delta = 1.0 the one-sided
        84% limit (the "1 sigma" convention).
        """
        m_best, c_best = self.best_fit()
        f = lambda m: self.profile(m)[0] - c_best - delta
        return brentq(f, max(m_best, 1e-16), 1e-13, xtol=1e-17)

    def credible_limit(self, level=0.95):
        """Upper limit with a prior flat in the mass, from the profile likelihood.

        The profile stands in for the marginal likelihood, so this is an
        approximation to a full Bayesian limit. It is given because published
        FRB limits are of this kind.
        """
        self.best_fit()
        grid = np.array(sorted(k for k in self._cache if k <= 3e-14))
        vals, _ = self.curve(grid)
        like = np.exp(-0.5 * (vals - vals.min()))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (like[1:] + like[:-1]) * np.diff(grid))])
        return float(np.interp(level, cdf / cdf[-1], grid))

    def verdict(self, m_ev):
        """-2 ln L of a proposed mass relative to the best fit."""
        return self.profile(m_ev)[0] - self.best_fit()[1]


def report():
    fit = SkewFit()
    fixed = SkewFit(config=SkewConfig(f_igm_range=(0.84, 0.84)))
    gauss = fl.rescaled_fit()
    m_best, c_best = fit.best_fit()
    c0, p0 = fit.profile(0.0)
    lines = [
        f"Bursts used (z > {fit.cfg.z_min}): {len(fit.z)}",
        f"Best fit at m = 0:  F = {p0[0]:.2f}, median host DM = {np.exp(p0[1]):.0f} pc cm^-3, "
        f"sigma_host = {p0[2]:.2f}, f_IGM = {p0[3]:.2f}",
        f"Best-fit mass: {m_best:.1e} eV; -2 ln L improvement over m = 0: {c0 - c_best:.2f}"
        "  (below 1: no evidence for a mass)",
        "",
        "Upper limits on the photon mass (eV):               84%        95%",
        f"  skewed distributions, f_IGM free              {fit.upper_limit(1.0):.2e}   {fit.upper_limit():.2e}",
        f"  skewed distributions, f_IGM = 0.84            {fixed.upper_limit(1.0):.2e}   {fixed.upper_limit():.2e}",
        f"  Gaussian scatter (frb_limit), f_IGM free      {gauss.upper_limit(1.0):.2e}   {gauss.upper_limit():.2e}",
        f"  flat prior in m, skewed, f_IGM free: 68% {fit.credible_limit(0.68):.2e}, 95% {fit.credible_limit(0.95):.2e}",
        "",
        "Verdict on proposed masses (delta > 3.84 means excluded at 95%):",
    ]
    for m in (1e-15, 5e-15, 1e-14, 2e-14):
        lines.append(f"  m = {m:.0e} eV: delta(-2 ln L) = {fit.verdict(m):.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
