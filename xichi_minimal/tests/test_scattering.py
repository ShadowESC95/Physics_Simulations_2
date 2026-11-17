"""Step 4: simulate, and compare with the exact Poeschl-Teller transmission."""
import numpy as np
import pytest

import scattering as sc
from solver import Params, Solver, add_wave_packet, kink


@pytest.mark.parametrize("s", [0.2, 0.5, 1.3, 3.7])
@pytest.mark.parametrize("k", [0.05, 0.3, 1.0])
def test_mode_equation_matches_exact_transmission(s, k):
    T, R = sc.transmission_numeric(k, 1.0, s)
    assert T + R == pytest.approx(1.0, abs=1e-9)
    assert T == pytest.approx(float(sc.transmission_exact(k, 1.0, s)), abs=1e-8)


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_wall_is_reflectionless_when_s_is_l_times_l_plus_one(ell):
    for k in (0.05, 0.4, 2.0):
        T, R = sc.transmission_numeric(k, 1.0, ell * (ell + 1))
        assert R < 1e-9 and T == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("g", [0.5, 1.0, 2.2])
def test_trapped_photon_frequencies(g):
    exact = sc.bound_states_exact(g, 2.0, 1.0)
    numeric = sc.bound_states_numeric(g, 2.0, 1.0)
    assert len(exact) == len(numeric) >= 1
    np.testing.assert_allclose(numeric, exact, atol=2e-5)


def _packet_transmission(g, k0, amplitude=1e-4):
    """Fraction of the packet's energy that ends up beyond the wall."""
    p = Params(g=g, lam=2.0, v=1.0)
    x = np.linspace(-320, 320, 6401)
    sol = Solver(p, x, dt=0.05)
    s = kink(x, p)
    k, weight = add_wave_packet(s, amplitude, -130.0, k0, 20.0, mass=p.photon_mass)
    e_in = sol.energy(s, "a")
    vg = k0 / np.hypot(k0, p.photon_mass)
    sol.step(s, int(260.0 / vg / sol.dt))
    right, left = s.x > 30, s.x < -30
    measured = sol.energy(s, "a", right) / e_in
    reflected = sol.energy(s, "a", left) / e_in
    expected = np.sum(weight * sc.transmission_exact(k, p.delta, sc.strength(g, p.lam))) / np.sum(weight)
    return measured, reflected, expected


def test_time_domain_packet_matches_exact_transmission():
    measured, reflected, expected = _packet_transmission(np.sqrt(0.5), 0.3)
    assert 0.4 < expected < 0.75                    # a regime where the test has teeth
    assert measured == pytest.approx(expected, abs=0.01)
    assert measured + reflected == pytest.approx(1.0, abs=0.01)


def test_time_domain_reflectionless_wall():
    measured, reflected, expected = _packet_transmission(np.sqrt(2.0), 0.3)   # s = 2
    assert expected == pytest.approx(1.0, abs=1e-12)
    assert reflected < 2e-3
