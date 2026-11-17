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
* e is the error of the Milky Way subtraction, Gaussian, with a width set for
  each sightline: sigma^2 = sigma_halo^2 + (disc_fraction * DM_disc)^2. The
  disc model is uncertain in proportion to what it predicts, so a burst seen
  through the Galactic plane is known less well than one seen at high
  latitude. Without e a burst whose dispersion is below the Milky Way model
  (FRB 20220319D) would have zero likelihood.

* The host may be the wrong galaxy. With probability e_i the burst's redshift
  is not the one in the table, and its dispersion is then drawn from the
  model at the redshift of a random burst of the sample:

      L_i = (1 - e_i) L(DM_i | z_i) + e_i * mean_j L(DM_i | z_j).

  e_i is 1 minus the published host-association probability where one was
  looked up (HOST_PROBABILITY), and `wrong_host` otherwise. This stops a
  single low-dispersion burst from carrying the limit on its own. The
  wrong-host term uses the Milky Way error of sightline j, an approximation.

The likelihood of a burst is the convolution of the three random terms. It is
computed on a uniform DM grid: each term is turned into exact cell
probabilities (from its cumulative distribution), the three are convolved by
FFT, and the result is read off at the burst's DM. A photon mass can only add
dispersion, so the lower edges of the distributions are what constrain it; a
Gaussian has no such edge.

Two ways of removing the nuisance parameters F, mu, sigma_host and f_IGM:
`SkewFit` profiles them out (maximum likelihood at each mass), and `sample`
integrates over them with Metropolis chains (flat priors).

    python3 frb_likelihood.py             profile limits, about 5 min
    python3 frb_likelihood.py --sample    also the marginalised limit, about 15 min more
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import erf

import frb_limit as fl

# Published probability that the listed galaxy is the host (PATH posterior),
# for the bursts checked against their papers; see data/README.md.
# FRB 20220610A: 0.87 + 0.11 for two members of the same galaxy group, so the
# redshift is the group's either way.
HOST_PROBABILITY = {
    "FRB20240209A": 0.99, "FRB20231201A": 0.911, "FRB20221029A": 0.92,
    "FRB20220610A": 0.98, "FRB20210807D": 1.0, "FRB20190611B": 0.98,
    "FRB20220319D": 0.99,
}

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
    sigma_halo: float = 15.0     # error of the halo value (assumed; covers 20-80)
    disc_fraction: float = 0.2   # fractional error of the NE2001 disc DM (assumed)
    sigma_mw: float | None = None    # if set: one error for every sightline instead
    wrong_host: float = 0.05     # probability of a wrong host where none was looked up (assumed)
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
        self.eps = np.array([1.0 - HOST_PROBABILITY[n] if n in HOST_PROBABILITY and cfg.wrong_host > 0
                             else cfg.wrong_host for n in self.frbs])
        disc = np.array([r[2] for r in rows])
        self.sigma_mw = (np.full(len(rows), float(cfg.sigma_mw)) if cfg.sigma_mw is not None
                         else np.sqrt(cfg.sigma_halo**2 + (cfg.disc_fraction * disc) ** 2))
        self.igm_unit = np.array([fl.dm_igm(z, 1.0) for z in self.z])
        self.gamma_unit = np.array([fl.dm_gamma(z, 1.0) for z in self.z])   # per eV^2
        d = cfg.step
        self._neg = int(8 * self.sigma_mw.max() / d) + 2
        need = 2 * (self.excess.max() / d + self._neg + 200)
        n = self._n = cfg.n_grid or int(2 ** np.ceil(np.log2(need)))
        # Cells kept below zero for the Milky Way error. The two positive terms
        # are kept on `m` cells each, so their sum plus the error never wraps
        # round the periodic FFT grid into the range that is read out.
        self._m = m = n // 2 - self._neg
        if self.excess.max() >= m * d:
            raise ValueError("n_grid * step is too small for the largest DM in the sample")
        self._edges = d * np.arange(m + 1)                 # cell edges of the positive terms
        k = np.fft.fftfreq(n, 1.0 / n)                     # 0, 1, ..., -2, -1 (wrapped)
        cdf = lambda x: 0.5 * (1.0 + erf(x / (np.sqrt(2.0) * self.sigma_mw[:, None])))
        self._mw_hat = np.fft.rfft(cdf((k[None, :] + 0.5) * d) - cdf((k[None, :] - 0.5) * d), axis=1)
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
               * np.fft.rfft(self._cosmic_cells(F, f_igm), n=n, axis=1) * self._mw_hat)
        pmf = np.fft.irfft(hat, n=n, axis=1)
        pmf = np.concatenate([pmf[:, n - self._neg:], pmf[:, :self._m]], axis=1)
        return np.maximum(pmf / self.cfg.step, FLOOR)

    def likelihood_grid(self, masses, F, mu, sigma_host, f_igm):
        """Likelihood of each burst at each mass: array (bursts, masses).

        The mass only shifts the DM at which the density is read, so one
        density evaluation serves every mass.
        """
        ln_d = np.log(self.extragalactic_density(F, mu, sigma_host, f_igm))
        shift = np.atleast_1d(np.asarray(masses, float)) ** 2
        # interpolate ln(density): it is close to quadratic in the tails
        x = self.excess[:, None] - shift[None, :] * self.gamma_unit[:, None]
        like = np.exp([np.interp(x[i], self._axis, ln_d[i]) for i in range(len(x))])
        if np.any(self.eps > 0):
            wrong = np.zeros_like(like)
            for j in range(len(ln_d)):                 # the burst placed at redshift z_j
                xj = self.excess[:, None] - shift[None, :] * self.gamma_unit[j]
                wrong += np.exp(np.interp(xj, self._axis, ln_d[j]))
            like = (1.0 - self.eps)[:, None] * like + self.eps[:, None] * wrong / len(ln_d)
        return like

    def burst_likelihoods(self, m_ev, F, mu, sigma_host, f_igm):
        """Probability density of each burst's observed excess DM (per pc cm^-3)."""
        return self.likelihood_grid(m_ev, F, mu, sigma_host, f_igm)[:, 0]

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


# --------------------------------------------------------------------------
# Marginalised limit: sample the nuisance parameters
# --------------------------------------------------------------------------
def _mass_loglike(fit, theta, masses):
    """ln L at every mass in `masses` for one set of nuisance parameters."""
    return np.sum(np.log(fit.likelihood_grid(masses, *theta)), axis=0)


def _chain(args):
    """One Metropolis chain over (F, mu, sigma_host, f_IGM), mass integrated out.

    Priors are flat inside the configured ranges and flat in the mass on
    [0, m_max]. The first `n_burn` steps use fixed step sizes and are used to
    measure the covariance of the posterior; the remaining steps propose from
    that covariance and are the ones returned.
    """
    rows, config, start, n_burn, n_keep, m_max, seed = args
    fit = SkewFit(rows=rows, config=config)
    rng = np.random.default_rng(seed)
    masses = np.linspace(0.0, m_max, 61)
    lo, hi = np.array(fit._bounds).T

    def target(theta):
        """ln of the mass-integrated likelihood, and p(m | theta) on the grid."""
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf, None
        ll = _mass_loglike(fit, theta, masses)
        w = np.exp(ll - ll.max())
        norm = np.sum(0.5 * (w[1:] + w[:-1])) * (masses[1] - masses[0])
        return ll.max() + np.log(norm), w / norm

    theta = np.clip(np.array(start, float), lo, hi)
    lp, pm = target(theta)
    step = np.diag(np.array([0.04, 0.10, 0.08, 0.05]) ** 2)
    burn, kept, post, accepted = [], [], np.zeros_like(masses), 0
    for it in range(n_burn + n_keep):
        if it == n_burn:
            step = 2.38**2 / 4.0 * np.cov(np.array(burn[n_burn // 4:]).T)
        trial = rng.multivariate_normal(theta, step)
        lp_new, pm_new = target(trial)
        if np.log(rng.uniform()) < lp_new - lp:
            theta, lp, pm = trial, lp_new, pm_new
            accepted += it >= n_burn
        if it < n_burn:
            burn.append(theta.copy())
        else:
            kept.append(theta.copy())
            post += pm
    return np.array(kept), post / n_keep, accepted / n_keep


def sample(rows=None, config=None, n_chains=4, n_burn=1500, n_keep=4000, m_max=3e-14,
           seed=0, workers=None):
    """Posterior of the photon mass, marginalised over the nuisance parameters.

    Runs independent chains from scattered starting points. Returns a dict:
        masses, posterior   p(m | data) on a grid, flat prior in m
        limit68, limit95    upper credible limits (eV)
        mean, std           posterior mean and width of F, mu, sigma_host, f_IGM
        r_hat               Gelman-Rubin statistic of each parameter (near 1 if
                            the chains agree)
        acceptance          mean acceptance rate
    """
    from concurrent.futures import ProcessPoolExecutor
    rows = fl.load() if rows is None else rows
    config = config or SkewConfig()
    centre = SkewFit(rows=rows, config=config).profile(0.0)[1]
    rng = np.random.default_rng(seed)
    spread = np.array([0.05, 0.15, 0.10, 0.05])
    jobs = [(rows, config, centre + spread * rng.standard_normal(4), n_burn, n_keep,
             m_max, seed + 1 + c) for c in range(n_chains)]
    with ProcessPoolExecutor(max_workers=workers or n_chains) as pool:
        out = list(pool.map(_chain, jobs))
    chains = np.array([o[0] for o in out])                 # (chain, step, parameter)
    post = np.mean([o[1] for o in out], axis=0)
    masses = np.linspace(0.0, m_max, len(post))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(masses))])
    cdf /= cdf[-1]
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = n_keep * chains.mean(axis=1).var(axis=0, ddof=1)
    r_hat = np.sqrt(((n_keep - 1) / n_keep * within + between / n_keep) / within)
    flat = chains.reshape(-1, 4)
    return dict(masses=masses, posterior=post,
                limit68=float(np.interp(0.68, cdf, masses)),
                limit95=float(np.interp(0.95, cdf, masses)),
                mean=flat.mean(axis=0), std=flat.std(axis=0), r_hat=r_hat,
                acceptance=float(np.mean([o[2] for o in out])),
                chain_limits95=[float(np.interp(0.95, np.cumsum(o[1]) / np.sum(o[1]), masses))
                                for o in out])


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
        "",
        "Verdict on proposed masses (delta > 3.84 means excluded at 95%):",
    ]
    for m in (1e-15, 5e-15, 1e-14, 2e-14):
        lines.append(f"  m = {m:.0e} eV: delta(-2 ln L) = {fit.verdict(m):.1f}")
    return "\n".join(lines)


def sample_report(**kw):
    out = sample(**kw)
    F, mu, sh, fi = out["mean"]
    dF, dmu, dsh, dfi = out["std"]
    return "\n".join([
        "Marginalised over the nuisance parameters (flat priors, flat in m):",
        f"  m < {out['limit68']:.2e} eV (68%),  m < {out['limit95']:.2e} eV (95%)",
        f"  F = {F:.2f} +- {dF:.2f}, median host DM = {np.exp(mu):.0f} pc cm^-3 "
        f"(ln: +- {dmu:.2f}), sigma_host = {sh:.2f} +- {dsh:.2f}, f_IGM = {fi:.2f} +- {dfi:.2f}",
        "  95% limit of each chain: " + ", ".join(f"{v:.2e}" for v in out["chain_limits95"]),
        f"  Gelman-Rubin R: {np.max(out['r_hat']):.3f} (worst parameter); "
        f"acceptance {out['acceptance']:.2f}",
    ])


if __name__ == "__main__":
    import sys
    print(report())
    if "--sample" in sys.argv:
        print()
        print(sample_report())
