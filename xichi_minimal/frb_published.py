"""
Like-for-like check against a published FRB photon-mass limit.

Wei & Wu (arXiv:2006.09680) obtain m < 4.0e-15 eV (68%) and 7.3e-15 eV (95%)
from nine localized bursts. This file repeats their analysis as the paper
describes it, with the values of their Table 1, so that the difference from
frb_likelihood.py can be traced to the method or to the data.

Their model (Eqs. 6-13 of the paper):

    DM_obs = DM_MW + DM_halo + DM_IGM(z) + DM_host(z) / (1 + z) + DM_gamma(z; m)
    DM_halo = 50,  f_IGM = 0.83,  Planck 2018 cosmology
    DM_host(z) = DM_host,0 * sqrt(SFR(z) / SFR(0))
    Gaussian likelihood with
        sigma^2 = sigma_obs^2 + sigma_MW^2 + sigma_halo^2 + sigma_int^2,
        sigma_MW = 10,  sigma_halo = 30   (pc cm^-3)

Free parameters: the photon mass, DM_host,0 and the intrinsic scatter
sigma_int. The paper samples them with emcee and does not state its priors;
here the posterior is integrated on a grid with flat priors on
m in [0, 3e-14] eV, DM_host,0 in [0, 300] and sigma_int in [0, 400].

The same three-parameter Gaussian model can be run on any burst list, which
shows what the method gives on the larger modern sample.
"""
from __future__ import annotations

import numpy as np

import frb_limit as fl

# Table 1 of arXiv:2006.09680: name, z, DM_obs, its error, DM_MW (pc cm^-3)
WEI_WU_TABLE = [
    ("FRB121102", 0.19273, 558.1, 3.3, 188.0),
    ("FRB180916", 0.0337, 348.76, 0.1, 199.0),
    ("FRB180924", 0.3214, 361.42, 0.06, 40.5),
    ("FRB181112", 0.4755, 589.27, 0.03, 102.0),
    ("FRB190102", 0.291, 363.6, 0.3, 57.3),
    ("FRB190523", 0.66, 760.8, 0.6, 37.0),
    ("FRB190608", 0.1178, 338.7, 0.5, 37.2),
    ("FRB190611", 0.378, 321.4, 0.2, 57.83),
    ("FRB190711", 0.522, 593.1, 0.4, 56.4),
]
PUBLISHED_68, PUBLISHED_95 = 4.0e-15, 7.3e-15      # eV, their Eqs. 14 and 15


def sfr(z):
    """Star-formation history used for the host term (their Section 2)."""
    return (0.0157 + 0.118 * z) / (1.0 + (z / 3.23) ** 4.66)


def gaussian_posterior(bursts, f_igm=0.83, dm_halo=50.0, sigma_mw=10.0, sigma_halo=30.0,
                       sfr_scaling=True, m_max=3e-14, n=(121, 121, 161)):
    """Posterior of the mass for the three-parameter Gaussian model.

    bursts: list of (name, z, DM_obs, sigma_obs, DM_MW). Returns a dict with
    the mass grid, its marginal posterior, and the 68% and 95% upper limits.
    """
    z = np.array([b[1] for b in bursts])
    resid0 = (np.array([b[2] - b[4] for b in bursts]) - dm_halo
              - np.array([fl.dm_igm(v, f_igm) for v in z]))
    sig_obs2 = np.array([b[3] for b in bursts]) ** 2 + sigma_mw**2 + sigma_halo**2
    gamma_unit = np.array([fl.dm_gamma(v, 1.0) for v in z])
    host_unit = (np.sqrt(sfr(z) / sfr(0.0)) if sfr_scaling else 1.0) / (1.0 + z)

    m = np.linspace(0.0, m_max, n[0])
    host = np.linspace(0.0, 300.0, n[1])
    s_int = np.linspace(0.0, 400.0, n[2])
    # residual for every (mass, host) pair and burst
    r = (resid0[None, None, :] - m[:, None, None] ** 2 * gamma_unit[None, None, :]
         - host[None, :, None] * host_unit[None, None, :])
    ll = np.empty((len(s_int), len(m), len(host)))
    for k, s in enumerate(s_int):
        var = sig_obs2 + s * s
        ll[k] = -0.5 * np.sum(r * r / var + np.log(var), axis=2)
    post = np.exp(ll - ll.max()).sum(axis=(0, 2))      # integrate sigma_int and host
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]))])
    cdf /= cdf[-1]
    return dict(masses=m, posterior=post / post.max(),
                limit68=float(np.interp(0.68, cdf, m)), limit95=float(np.interp(0.95, cdf, m)))


def table_from_repository(names=None, z_min=0.01):
    """Bursts from data/localized_frbs.csv in the format used here.

    The repository gives no error for some DMs; 1 pc cm^-3 is used, which is
    negligible beside the other terms.
    """
    out = []
    for name, dm, ism, z in fl.load(z_min=z_min):
        if names is None or any(name.startswith("FRB20" + k[3:]) for k in names):
            out.append((name, z, dm, 1.0, ism))
    return out


def report():
    nine = [b[0] for b in WEI_WU_TABLE]
    rows = [
        ("Published (arXiv:2006.09680)", PUBLISHED_68, PUBLISHED_95),
        ("Their method, their table", *_lims(gaussian_posterior(WEI_WU_TABLE))),
        ("Their method, no star-formation scaling", *_lims(gaussian_posterior(WEI_WU_TABLE, sfr_scaling=False))),
        ("Their method, same nine bursts from this table", *_lims(gaussian_posterior(table_from_repository(nine)))),
        ("Their method, all 98 bursts of this table", *_lims(gaussian_posterior(table_from_repository()))),
    ]
    lines = ["Photon-mass upper limits (eV)                           68%        95%"]
    lines += [f"  {name:<50s} {a:.1e}   {b:.1e}" for name, a, b in rows]
    return "\n".join(lines)


def _lims(out):
    return out["limit68"], out["limit95"]


if __name__ == "__main__":
    print(report())
