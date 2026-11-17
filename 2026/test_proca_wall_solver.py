#!/usr/bin/env python3
"""
Validation of proca_wall_solver.py.   Run:  python3 -m pytest -q   (or run directly)

1. Uniform medium: no reflection, no conversion.
2. Flux closure for step, kink and bump walls, both incident polarisations.
3. Sharp-step limit against the closed-form Fresnel-type coefficients.
4. Independent cross-check: brute-force integration of the full coupled
   4x4 unitary-gauge Proca system, with no use of the channel decoupling.
5. Thick-wall limit against the adiabatic polarisation-rotation formula.
6. Normal incidence: no T <-> L conversion.
"""
import numpy as np
from scipy.integrate import solve_ivp
from proca_wall_solver import (Step, Kink, Bump, scatter, channel_coefficients,
                               sharp_step_channels, adiabatic_conversion, _csqrt,
                               scatter_fast)


def brute_force(wall, omega, theta, Z=None):
    """
    Solve the coupled Proca TM system directly in unitary gauge.
    State y = (A_x, B_y, phi, Pi),  Pi = m^2 A_z,   y' = M(z) y.
    Only valid when m(z) never vanishes.
    """
    Z = Z or 14.0 * wall.delta
    mL, mR = wall.m_left, wall.m_right
    k = np.sqrt(omega ** 2 - mL ** 2) * np.sin(theta)
    qL = _csqrt(omega ** 2 - k ** 2 - mL ** 2)
    qR = _csqrt(omega ** 2 - k ** 2 - mR ** 2)

    def rhs(z, y):
        Ax, By, phi, Pi = y
        m2 = wall.m(z) ** 2
        return [By + 1j * k * Pi / m2,
                -(omega ** 2 - m2) * Ax + omega * k * phi,
                (k / omega) * By + 1j * (omega ** 2 - m2) / (omega * m2) * Pi,
                1j * m2 * (omega * phi - k * Ax)]

    def modes(m, kappa):
        K = np.sqrt(k * k + kappa * kappa)
        T = np.array([kappa / K, 1j * K, 0.0, -m * m * k / K], dtype=complex)
        L = np.array([omega * k / (m * K), 0.0, K / m, m * omega * kappa / K], dtype=complex)
        return T, L

    TR, LR = modes(mR, qR)
    sols = []
    for v in (TR, LR):
        s = solve_ivp(rhs, (Z, -Z), v * np.exp(1j * qR * Z), method="DOP853",
                      rtol=1e-11, atol=1e-13)
        sols.append(s.y[:, -1])
    Tp, Lp = modes(mL, qL)
    Tm, Lm = modes(mL, -qL)
    basis = np.column_stack([Tp * np.exp(-1j * qL * Z), Lp * np.exp(-1j * qL * Z),
                             Tm * np.exp(1j * qL * Z), Lm * np.exp(1j * qL * Z)])
    C = np.linalg.solve(basis, np.column_stack(sols))   # rows: T+,L+,T-,L- ; cols: outgoing T,L
    inc, ref = C[:2], C[2:]
    tran = np.linalg.inv(inc)                           # incident unit -> transmitted amps
    refl = ref @ tran
    return np.abs(refl) ** 2, np.abs(tran) ** 2 * (qR.real / qL.real)


def test_uniform_medium():
    out = scatter(Step(0.7, 0.7, 1.0), omega=2.0, theta=0.6)
    assert np.allclose(out["R"], 0, atol=1e-9)
    assert np.allclose(out["T"], np.eye(2), atol=1e-8)


def test_flux_closure():
    walls = [Step(0.5, 1.0, 1.0), Step(1.2, 0.3, 0.4), Kink(1.0, 1.0), Kink(0.8, 0.3),
             Bump(0.6, 0.9, 0.7), Bump(1.0, -0.6, 1.5)]
    for w in walls:
        for omega in (1.6, 3.0, 8.0):
            for th in (0.1, 0.7, 1.3):
                if omega ** 2 <= w.m_left ** 2:
                    continue
                assert scatter(w, omega, th)["closure"] < 1e-6, (w, omega, th)


def test_sharp_step_limit():
    omega, th, mL, mR = 3.0, 0.9, 0.5, 1.4
    k = np.sqrt(omega ** 2 - mL ** 2) * np.sin(th)
    num = channel_coefficients(Step(mL, mR, 2e-3), omega, k, Z=0.2)
    ref = sharp_step_channels(omega, k, mL, mR)
    for key in ("rE", "rPi"):
        assert abs(abs(num[key]) - abs(ref[key])) < 2e-3, key
    for key in ("tE", "tPi"):
        assert abs(abs(num[key]) - abs(ref[key])) < 2e-3, key


def test_against_brute_force_coupled_system():
    for w, omega, th in [(Step(0.5, 1.0, 1.0), 3.0, 0.9), (Step(1.2, 0.4, 0.3), 2.0, 0.5),
                         (Bump(0.6, 0.9, 0.7), 2.5, 1.1)]:
        out = scatter(w, omega, th)
        Rb, Tb = brute_force(w, omega, th)
        assert np.allclose(out["R"], Rb, atol=1e-6), (w, out["R"], Rb)
        assert np.allclose(out["T"], Tb, atol=1e-6), (w, out["T"], Tb)


def test_adiabatic_limit():
    omega, th, mL, mR = 6.0, 1.0, 0.5, 1.5
    out = scatter(Step(mL, mR, 12.0), omega, th)
    assert abs(out["T"][1, 0] - adiabatic_conversion(omega, th, mL, mR)) < 1e-4
    assert out["R"].max() < 1e-8


def test_normal_incidence_no_conversion():
    out = scatter(Step(0.5, 1.0, 0.5), omega=2.0, theta=0.0)
    assert out["R"][1, 0] < 1e-12 and out["T"][1, 0] < 1e-12


def test_kink_against_complex_contour():
    """
    The Pi equation is singular at the kink centre (m = 0) on the real axis.
    Both local solutions are analytic there, so integrating the *raw* Pi equation
    along the shifted line z = x - i*eta must reproduce the regular-gauge result.
    """
    w, omega, th, eta = Kink(1.0, 1.0), 3.0, 0.8, 0.4
    k = np.sqrt(omega ** 2 - 1.0) * np.sin(th)
    q = _csqrt(omega ** 2 - k ** 2 - 1.0)
    Z = 14.0

    def rhs(x, y):                      # y = (Pi, P),  P = Pi'/m^2
        z = x - 1j * eta
        m2 = np.tanh(z) ** 2
        return [m2 * y[1], -(omega ** 2 - k ** 2 - m2) * y[0] / m2]

    zR, zL = Z - 1j * eta, -Z - 1j * eta
    e0 = np.exp(1j * q * zR)
    s = solve_ivp(rhs, (Z, -Z), [e0, 1j * q * e0], method="DOP853", rtol=1e-11, atol=1e-13)
    Pi, dPi = s.y[0, -1], s.y[1, -1]    # m^2 = 1 at both ends
    a = 0.5 * (Pi + dPi / (1j * q)) * np.exp(-1j * q * zL)
    b = 0.5 * (Pi - dPi / (1j * q)) * np.exp(1j * q * zL)
    c = channel_coefficients(w, omega, k)
    assert abs(b / a - c["rPi"]) < 1e-7
    assert abs(1.0 / a - c["tPi"]) < 1e-7


def test_fast_solver_matches_reference():
    cases = [("step", 0.5, 1.0, 1.0, 3.0, 0.87, Step(0.5, 1.0, 1.0)),
             ("kink", 1.0, 0.0, 1.0, 3.0, 0.79, Kink(1.0, 1.0)),
             ("bump", 0.6, 0.9, 0.7, 2.5, 1.10, Bump(0.6, 0.9, 0.7)),
             ("step", 0.5, 3.0, 0.5, 2.5, 0.60, Step(0.5, 3.0, 0.5))]   # total reflection
    for kind, p1, p2, d, om, th, wall in cases:
        ref = scatter(wall, om, th)
        fast = scatter_fast(kind, p1, p2, d, om, th)
        assert np.allclose(fast["R"][..., 0], ref["R"], atol=2e-5), kind
        assert np.allclose(fast["T"][..., 0], ref["T"], atol=2e-5), kind
    # a scan evaluated in one call agrees with point-by-point evaluation
    th = np.linspace(0.05, 1.4, 7)
    scan = scatter_fast("step", 0.5, 1.0, 1.0, 3.0, th)
    for i, t_ in enumerate(th):
        assert np.allclose(scan["T"][..., i], scatter(Step(0.5, 1.0, 1.0), 3.0, t_)["T"], atol=2e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
