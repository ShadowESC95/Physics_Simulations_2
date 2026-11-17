"""The skewed-distribution likelihood: its pieces, its numerics, and its result."""
import numpy as np
import pytest
from scipy.integrate import quad

import frb_likelihood as lk
import frb_limit as fl


def _integral(f, s):
    """Integrate over Delta > 0, telling quad where the peak is."""
    c0, _ = lk.cosmic_constants(s)
    peak = max(c0, 1e-3) ** (-1.0 / 3.0)
    pts = sorted({0.5 * peak, peak, 2 * peak})
    return (quad(f, 0, 4 * peak, points=pts, limit=400)[0]
            + quad(f, 4 * peak, np.inf, limit=400)[0])


@pytest.mark.parametrize("s", [0.05, 0.3, 1.0, 3.0])
def test_cosmic_density_is_normalised_with_unit_mean(s):
    assert _integral(lambda d: lk.p_cosmic(d, s), s) == pytest.approx(1.0, abs=2e-3)
    assert _integral(lambda d: d * lk.p_cosmic(d, s), s) == pytest.approx(1.0, abs=5e-3)


def test_cosmic_density_is_skewed_and_narrows_with_redshift():
    """Long upper tail (median below the mean of 1), and less scatter for
    smaller s = F / sqrt(z), i.e. for more distant bursts."""
    d = np.linspace(1e-3, 6, 60001)
    for s in (0.3, 1.0):
        cdf = np.cumsum(lk.p_cosmic(d, s)) * (d[1] - d[0])
        assert d[np.searchsorted(cdf, 0.5)] < 1.0
    width = lambda s: np.sqrt(np.sum((d - 1) ** 2 * lk.p_cosmic(d, s)) * (d[1] - d[0]))
    assert width(0.1) < width(0.3) < width(0.6)


def test_host_density_is_lognormal():
    mu, sig = np.log(80.0), 0.9
    assert quad(lambda h: lk.p_host(h, mu, sig), 0, np.inf)[0] == pytest.approx(1.0, abs=1e-6)
    assert quad(lambda h: lk.p_host(h, mu, sig), 0, 80.0)[0] == pytest.approx(0.5, abs=1e-6)


@pytest.fixture(scope="module")
def fit():
    return lk.SkewFit()


PARS = (0.3, np.log(100.0), 0.8, 0.84)


def test_cell_probabilities_match_the_densities(fit):
    i = int(np.argmin(np.abs(fit.z - 0.3)))
    z, mean = fit.z[i], 0.84 * fit.igm_unit[i]
    s = 0.3 / np.sqrt(z)
    cosmic = fit._cosmic_cells(0.3, 0.84)[i]
    host = fit._host_cells(PARS[1], PARS[2])[i]
    for lo, hi in ((100, 140), (250, 300), (600, 800)):
        exact = quad(lambda c: lk.p_cosmic(c / mean, s) / mean, lo, hi)[0]
        assert cosmic[lo:hi].sum() == pytest.approx(exact, rel=2e-3)
        exact = quad(lambda h: lk.p_host(h, PARS[1], PARS[2]), lo * (1 + z), hi * (1 + z))[0]
        assert host[lo:hi].sum() == pytest.approx(exact, rel=1e-6)


def test_convolution_matches_a_direct_monte_carlo(fit):
    """Draw cosmic + host/(1+z) + error directly and compare with the density."""
    i = int(np.argmin(np.abs(fit.z - 0.3)))
    z, mean = fit.z[i], 0.84 * fit.igm_unit[i]
    rng = np.random.default_rng(3)
    n = 400000
    d = np.linspace(1e-3, 40, 400001)
    cdf = np.cumsum(lk.p_cosmic(d, 0.3 / np.sqrt(z)))
    cosmic = mean * np.interp(rng.uniform(0, 1, n), cdf / cdf[-1], d)
    host = np.exp(PARS[1] + PARS[2] * rng.standard_normal(n)) / (1 + z)
    total = cosmic + host + fit.sigma_mw[i] * rng.standard_normal(n)
    dens = fit.extragalactic_density(*PARS)[i]
    for lo, hi in ((150, 250), (250, 350), (350, 500), (500, 900)):
        sel = (fit._axis >= lo) & (fit._axis < hi)
        expected = dens[sel].sum() * fit.cfg.step
        got = np.mean((total >= lo) & (total < hi))
        assert got == pytest.approx(expected, abs=4 * np.sqrt(expected / n) + 1e-3)


def test_density_is_normalised_and_grid_converged(fit):
    dens = fit.extragalactic_density(*PARS)
    total = dens.sum(axis=1) * fit.cfg.step
    assert np.all((total > 0.9) & (total < 1.0 + 1e-6))       # the rest lies beyond the grid
    fine = lk.SkewFit(config=lk.SkewConfig(step=0.5, n_grid=8192, n_w=3000))
    for m in (0.0, 1e-14):
        assert fit.nll(m, *PARS) == pytest.approx(fine.nll(m, *PARS), abs=0.05)


def test_photon_mass_only_removes_dispersion(fit):
    """A mass m acts exactly like observing every burst with DM_gamma less."""
    m = 8e-15
    lowered = [(n, dm - fl.dm_gamma(z, m), ism, z) for n, dm, ism, z in fl.load()]
    assert fit.nll(m, *PARS) == pytest.approx(lk.SkewFit(rows=lowered).nll(0.0, *PARS), rel=1e-9)


def test_no_evidence_for_a_mass_and_the_limit_is_near_1e_minus_14(fit):
    at_zero = fit.profile(0.0)[0]
    assert fit.profile(5e-15)[0] - at_zero > -0.5          # no better than m = 0
    assert fit.profile(5e-15)[0] - at_zero < 2.71          # 5e-15 eV allowed
    assert fit.profile(1.5e-14)[0] - at_zero > 3.84        # 1.5e-14 eV excluded


def test_limit_is_tighter_than_the_gaussian_fit(fit):
    """The Gaussian fit of frb_limit allows 1.5e-14 eV; this one does not."""
    assert fl.rescaled_fit().verdict(1.5e-14) < 2.71
    assert fit.profile(1.5e-14)[0] - fit.profile(0.0)[0] > 3.84


def test_fitted_nuisance_parameters_are_physical(fit):
    F, mu, sigma_host, f_igm = fit.profile(0.0)[1]
    assert 0.1 < F <= 0.5
    assert 40 < np.exp(mu) < 200            # median host DM, pc cm^-3
    assert 0.5 < sigma_host < 1.5
    assert 0.7 < f_igm <= 1.0


def test_pipeline_detects_an_injected_mass():
    m_true = 3e-14
    boosted = [(n, dm + fl.dm_gamma(z, m_true), ism, z) for n, dm, ism, z in fl.load()]
    f = lk.SkewFit(rows=boosted)
    at_true = f.profile(m_true)[0]
    assert f.profile(0.0)[0] - at_true > 25                # m = 0 rejected
    assert f.profile(2.4e-14)[0] > at_true < f.profile(3.6e-14)[0]


def test_milky_way_error_grows_with_the_disc_contribution(fit):
    """A sightline through more of the Galactic disc is known less well."""
    disc = np.array([r[2] for r in fl.load(z_min=fit.cfg.z_min)])
    assert np.all(np.diff(fit.sigma_mw[np.argsort(disc)]) >= 0)
    assert fit.sigma_mw.min() == pytest.approx(
        np.hypot(fit.cfg.sigma_halo, fit.cfg.disc_fraction * disc.min()))
    flat = lk.SkewFit(config=lk.SkewConfig(sigma_mw=30.0))
    assert np.all(flat.sigma_mw == 30.0)


def test_mass_grid_likelihood_equals_the_direct_one(fit):
    masses = np.array([0.0, 4e-15, 1.2e-14])
    ll = lk._mass_loglike(fit, PARS, masses)
    for m, v in zip(masses, ll):
        assert -2.0 * v == pytest.approx(fit.nll(m, *PARS), rel=1e-10)


def test_sampler_gives_a_limit_consistent_with_the_profile(fit):
    """Short chains: enough to check the machinery, not for quoting numbers."""
    out = lk.sample(n_chains=2, n_burn=250, n_keep=400, seed=5)
    post, m = out["posterior"], out["masses"]
    assert np.sum(0.5 * (post[1:] + post[:-1]) * np.diff(m)) == pytest.approx(1.0, rel=1e-6)
    assert post[0] > post[-1] * 1e3                      # mass zero favoured, 3e-14 excluded
    assert 3e-15 < out["limit68"] < out["limit95"] < 1.5e-14
    assert 0.05 < out["acceptance"] < 0.8
