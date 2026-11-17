"""Step 3: known limits. Each test compares the solver with an exact result."""
import numpy as np
import pytest

import model
import sympy as sp
from solver import Params, Solver, add_wave_packet, kink, vacuum


def test_zero_coupling_is_the_free_wave_equation():
    """Symbolic: at g = 0 the A equation is exactly Maxwell's wave equation."""
    eq = model.reduced_equations()[model.a_].subs(model.g, 0)
    a, t, x = model.a_, model.t, model.x
    assert sp.expand(eq + sp.diff(a, t, 2) - sp.diff(a, x, 2)) == 0


def _centroid(solver, s):
    e = solver.energy_density(s, "a")
    xm = 0.5 * (s.x[1:] + s.x[:-1])
    return np.sum(xm * e) / np.sum(e)


def test_zero_coupling_pulse_travels_at_light_speed():
    p = Params(g=0.0)
    x = np.linspace(-100, 100, 4001)
    sol = Solver(p, x, dt=0.02)
    s = vacuum(x, p)
    add_wave_packet(s, 1e-3, -40.0, 2.0, 5.0, mass=0.0)
    x0 = _centroid(sol, s)
    sol.step(s, 3000)                        # t = 60
    speed = (_centroid(sol, s) - x0) / s.time
    assert speed == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("k0", [0.5, 1.5])
def test_uniform_vacuum_gives_proca_dispersion(k0):
    """Group velocity k/sqrt(k^2 + g^2 v^2) in the uniform vacuum Xi = v."""
    p = Params(g=1.0, lam=2.0, v=1.0)
    x = np.linspace(-150, 150, 6001)
    sol = Solver(p, x, dt=0.02)
    s = vacuum(x, p)
    add_wave_packet(s, 1e-4, -60.0, k0, 8.0, mass=p.photon_mass)
    x0 = _centroid(sol, s)
    sol.step(s, 4000)                        # t = 80
    speed = (_centroid(sol, s) - x0) / s.time
    assert speed == pytest.approx(k0 / np.hypot(k0, p.photon_mass), abs=4e-3)


def test_static_kink_stays_put_and_energy_is_the_bogomolny_mass():
    p = Params(g=1.0, lam=2.0, v=1.0)
    x = np.linspace(-30, 30, 3001)
    sol = Solver(p, x, dt=0.01)
    s = kink(x, p)
    e0 = sol.energy(s)
    # exact kink mass: (2 sqrt(2)/3) sqrt(lambda) v^3
    assert e0 == pytest.approx(2 * np.sqrt(2) / 3 * np.sqrt(p.lam) * p.v**3, rel=1e-4)
    start = s.xi.copy()
    sol.step(s, 2000)
    assert np.max(np.abs(s.xi - start)) < 1e-3
    assert sol.energy(s) == pytest.approx(e0, rel=1e-6)


def _energy_drift(dt):
    p = Params(g=1.0, lam=2.0, v=1.0)
    x = np.linspace(-60, 60, 2401)
    sol = Solver(p, x, dt=dt)
    s = kink(x, p)
    add_wave_packet(s, 0.3, -25.0, 1.0, 4.0, mass=p.photon_mass)   # nonlinear amplitude
    e0, worst = sol.energy(s), 0.0
    for _ in range(20):
        sol.step(s, int(round(1.0 / dt)))
        worst = max(worst, abs(sol.energy(s) - e0) / e0)
    return worst


def test_energy_is_conserved_at_second_order_in_dt():
    coarse, fine = _energy_drift(0.04), _energy_drift(0.02)
    assert fine < 2e-4
    assert coarse / fine == pytest.approx(4.0, rel=0.25)
