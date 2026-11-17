"""Step 5: the comparison with real data, and checks that it could fail."""
import numpy as np
import pytest

import frb_limit as fl


def test_dispersion_constant_matches_the_standard_value():
    k = fl.dispersion_constant() * fl.PC_CM3 / 1e12        # MHz^2 pc^-1 cm^3 s
    assert k == pytest.approx(4148.808, abs=2e-3)


def test_photon_mass_dm_reproduces_the_first_principles_delay():
    """DM_gamma was defined so that K * DM_gamma / nu^2 equals the delay
    computed directly from the massive-photon group velocity."""
    for z, m, nu in [(0.1, 1e-14, 1.4e9), (1.0, 3e-15, 6e8)]:
        via_dm = fl.dispersion_constant() * fl.dm_gamma(z, m) * fl.PC_CM3 / nu**2
        assert via_dm == pytest.approx(fl.photon_delay(z, m, nu), rel=1e-12)


def test_redshift_functions():
    assert fl.h_gamma(1e-4) == pytest.approx(1e-4, rel=1e-3)
    assert fl.h_e(1e-4) == pytest.approx(1e-4, rel=1e-3)
    assert fl.h_gamma(1.0) < 1.0 < fl.h_e(1.0)            # the two differ in shape
    assert 800 < fl.dm_igm(1.0) < 1000                     # pc cm^-3 at z = 1


def test_agrees_with_published_single_burst_limits():
    """FRB 121102, z = 0.19273: Bonetti et al. (arXiv:1701.03097) quote
    m < 2.2e-14 eV. Their exact DM budget is not in the abstract, so this
    only checks agreement to 25% when ~340 pc cm^-3 is given to the photon.
    FRB 150418 at the originally claimed z = 0.492: published limits are
    1.8e-14 eV (arXiv:1602.09135) and 5.2e-50 kg = 2.9e-14 eV (arXiv:1602.07835);
    the same formula must land between them."""
    assert fl.mass_for_dm(0.19273, 340.0) == pytest.approx(2.2e-14, rel=0.25)
    assert 1.8e-14 < fl.mass_for_dm(0.492, 776.2 - 188.5) < 2.9e-14
    assert fl.mass_kg(2.2e-14) == pytest.approx(3.9e-50, rel=0.02)


def test_data_table_is_real_and_sane():
    rows = fl.load()
    assert len(rows) >= 90
    names = [r[0] for r in rows]
    assert "FRB20121102A" in names and len(set(names)) == len(names)
    row = rows[names.index("FRB20121102A")]
    assert row[1] == pytest.approx(558.1) and row[3] == pytest.approx(0.19273)
    assert all(0 < r[3] < 2 and r[1] > 0 for r in rows)


def test_conservative_limit():
    best = fl.conservative_limits()[0]
    assert 1e-14 < best[3] < 3e-14


@pytest.fixture(scope="module")
def fit():
    return fl.rescaled_fit()


def test_no_evidence_for_a_mass_and_limit_is_near_2e_minus_14(fit):
    m_best, c_best = fit.best_fit()
    assert fit.profile(0.0)[0] - c_best < 1.0
    assert 5e-15 < fit.upper_limit() < 3e-14


def test_the_data_can_exclude_a_mass(fit):
    assert fit.verdict(3e-14) > 3.84          # excluded
    assert fit.verdict(1e-15) < 1.0           # allowed


def test_pipeline_detects_an_injected_mass():
    """Add the dispersion of a 5e-14 eV photon to the real bursts. The fit must
    recover it and reject m = 0, i.e. the analysis is able to see a signal."""
    rows = fl.load()
    m_true = 5e-14
    boosted = [(n, dm + fl.dm_gamma(z, m_true), ism, z) for n, dm, ism, z in rows]
    fit = fl.JointFit(rows=boosted, config=fl.FitConfig(sigma_scale=1.5))
    m_best, c_best = fit.best_fit()
    assert m_best == pytest.approx(m_true, rel=0.15)
    assert fit.profile(0.0)[0] - c_best > 25
