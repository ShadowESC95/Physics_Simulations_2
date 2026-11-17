#!/usr/bin/env python3
"""
Validation of wall_network_estimate.py.   Run:  python3 -m pytest -q   (or run directly)
"""
import numpy as np
from proca_wall_solver import scan_fast, sharp_wall_conversion
from wall_network_estimate import (wall_rotation, thin_wall_conversion, mean_loss_per_wall,
                                   monte_carlo, network_loss, scenario)


def test_thick_wall_formula_matches_solver():
    om, mL, mR = 20.0, 0.5, 1.0
    u = np.linspace(0.12, 0.999, 15)
    num = scan_fast("step", mL, mR, 4.0, om, np.arccos(u), ppw=60)
    s, c, refl, _ = wall_rotation(u, mL / om, mR / om)
    assert not refl.any()
    assert np.allclose(num["T"][1, 0], s * s, atol=2e-5)
    assert np.allclose(s * s + c * c, 1.0)


def test_thin_wall_formula_matches_solver_assembly():
    om, mL, mR = 20.0, 0.5, 1.0
    u = np.linspace(0.12, 0.999, 15)
    conv, _ = thin_wall_conversion(u, mL / om, mR / om)
    ref = [sharp_wall_conversion("step", mL, mR, om, t) for t in np.arccos(u)]
    assert np.allclose(conv, ref, atol=1e-12)


def test_total_reflection_window():
    # rising mass: reflected weight is the probability that q_b^2 < 0 under p(u) = 2u
    ra, rb = 0.01, 0.02
    up = (rb ** 2 - ra ** 2) / (1 - ra ** 2)
    assert abs(mean_loss_per_wall(ra, rb)["reflection"] - 0.5 * up) < 1e-3 * up


def test_identical_domains_lose_nothing():
    out = mean_loss_per_wall(0.01, 0.01)
    assert out["total"] < 1e-20


def test_monte_carlo_matches_per_wall_law():
    for ra, rb in ((0.005, 0.01), (0.005, -0.005)):
        mc = monte_carlo(ra, rb, 60, n_rays=40000, seed=3)
        pw = mean_loss_per_wall(ra, rb)
        pred_L = (1 - np.exp(-3 * 60 * pw["conversion"])) / 3
        assert abs(mc["longitudinal"][-1] / pred_L - 1) < 0.15, (ra, rb)
        if pw["reflection"] > 0:
            assert abs(mc["reflected"][-1] / (60 * pw["reflection"]) - 1) < 0.25


def test_network_loss_limits():
    pw = dict(conversion=1e-6, reflection=0.0, total=1e-6)
    assert abs(network_loss(100, pw) / 1e-4 - 1) < 1e-3          # linear for few walls
    assert abs(network_loss(1e12, pw) - 1 / 3) < 1e-9            # saturates at one third


def test_scenario_scales_as_inverse_frequency_squared():
    a = scenario(1e-14, 1.0, 1.0, 1.0, 1.0)["per_wall"]["total"]
    b = scenario(1e-14, 1.0, 2.0, 1.0, 1.0)["per_wall"]["total"]
    assert 3.5 < a / b < 4.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
