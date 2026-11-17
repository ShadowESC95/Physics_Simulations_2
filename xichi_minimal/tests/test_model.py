"""Steps 1-2: one Lagrangian, consistent units, equations derived from it."""
import sympy as sp

import model


def test_every_term_has_mass_dimension_four():
    assert set(model.check_dimensions().values()) == {4}


def test_lagrangian_is_gauge_invariant():
    assert model.gauge_variation() == 0


def test_reduction_is_a_consistent_truncation():
    assert all(r == 0 for r in model.reduction_is_consistent().values())


def test_reduced_equations_match_the_expected_form():
    t, x, g, lam, v = model.t, model.x, model.g, model.lam, model.v
    xi, a = model.xi_, model.a_
    red = model.reduced_equations()
    box = lambda f: sp.diff(f, t, 2) - sp.diff(f, x, 2)
    assert sp.expand(red[xi] + box(xi) + lam * xi * (xi**2 - v**2) + g**2 * xi * a**2) == 0
    assert sp.expand(red[a] + box(a) + g**2 * xi**2 * a) == 0


def test_energy_density_is_positive_definite_form():
    H = model.energy_density()
    xi, a, t, x = model.xi_, model.a_, model.t, model.x
    expected = (sp.Rational(1, 2) * (sp.diff(xi, t)**2 + sp.diff(xi, x)**2
                                     + sp.diff(a, t)**2 + sp.diff(a, x)**2)
                + model.potential(xi) + sp.Rational(1, 2) * model.g**2 * xi**2 * a**2)
    assert sp.expand(H - expected) == 0


def test_static_kink_solves_the_field_equation():
    assert model.kink_residual() == 0


def test_masses():
    assert sp.simplify(model.scalar_mass() - sp.sqrt(2 * model.lam) * model.v) == 0
    assert model.photon_mass() == model.g * model.v
