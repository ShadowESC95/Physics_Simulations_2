"""
Linear scattering of the photon off the Xi kink, and the exact answer.

On the static kink Xi = v tanh(x/delta), a mode a = Re[f(x) exp(-i w t)] of the
linearised A equation obeys

    f'' + [k^2 + U0 sech^2(x/delta)] f = 0,
    k^2 = w^2 - g^2 v^2,   U0 = g^2 v^2,   delta = sqrt(2/lambda)/v,

because g^2 v^2 tanh^2 = g^2 v^2 - g^2 v^2 sech^2. This is the Poeschl-Teller
problem (Landau & Lifshitz, Quantum Mechanics, section 25, problem 4), whose
transmission probability is known in closed form:

    T(k) = sinh^2(pi k delta) / [sinh^2(pi k delta) + cos^2(pi/2 sqrt(1 + 4 s))],
    s = U0 delta^2 = 2 g^2 / lambda.

Consequences that the numerics can be tested against:
  * the wall is exactly reflectionless at every frequency when s = l(l+1),
    l = 1, 2, ...;
  * photons can be trapped on the wall, with frequencies below g v:
    kappa_n delta = (sqrt(1+4s) - 1)/2 - n > 0,  w_n^2 = g^2 v^2 - kappa_n^2.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal


def strength(g, lam):
    """s = U0 delta^2 = 2 g^2 / lambda (dimensionless)."""
    return 2.0 * g**2 / lam


def transmission_exact(k, delta, s):
    k = np.asarray(k, dtype=float)
    sh = np.sinh(np.pi * k * delta) ** 2
    c = np.cos(0.5 * np.pi * np.sqrt(1.0 + 4.0 * s)) ** 2
    return sh / (sh + c)


def transmission_numeric(k, delta, s, span=30.0, rtol=1e-11, atol=1e-13):
    """Integrate the mode equation with outgoing-wave conditions.

    Start from a pure transmitted wave exp(i k x) on the right, integrate to
    the left, and split the result into incident and reflected parts.
    Returns (T, R).
    """
    X = span * delta
    U0 = s / delta**2

    def rhs(xx, yv):
        f, fp = yv
        return [fp, -(k**2 + U0 / np.cosh(xx / delta) ** 2) * f]

    y0 = [np.exp(1j * k * X), 1j * k * np.exp(1j * k * X)]
    sol = solve_ivp(rhs, (X, -X), np.array(y0, dtype=complex),
                    method="DOP853", rtol=rtol, atol=atol)
    f, fp = sol.y[0, -1], sol.y[1, -1]
    inc = 0.5 * (f + fp / (1j * k)) * np.exp(1j * k * X)      # coefficient of exp(ikx)
    ref = 0.5 * (f - fp / (1j * k)) * np.exp(-1j * k * X)     # coefficient of exp(-ikx)
    return 1.0 / abs(inc) ** 2, abs(ref) ** 2 / abs(inc) ** 2


def bound_states_exact(g, lam, v):
    """Frequencies of photon modes trapped on the wall."""
    s = strength(g, lam)
    delta = np.sqrt(2.0 / lam) / v
    top = 0.5 * (np.sqrt(1.0 + 4.0 * s) - 1.0)
    n = np.arange(0, int(np.ceil(top)))
    kappa = (top - n) / delta
    kappa = kappa[kappa > 1e-12]
    return np.sqrt((g * v) ** 2 - kappa**2)


def bound_states_numeric(g, lam, v, half_width=40.0, n=8001):
    """Same frequencies from a finite-difference eigenvalue problem."""
    delta = np.sqrt(2.0 / lam) / v
    xx = np.linspace(-half_width * delta, half_width * delta, n)
    h = xx[1] - xx[0]
    m2 = (g * v * np.tanh(xx / delta)) ** 2
    w2 = eigh_tridiagonal(2.0 / h**2 + m2, -np.ones(n - 1) / h**2,
                          select="v", select_range=(-1.0, (g * v) ** 2 * (1 - 1e-9)))[0]
    return np.sqrt(w2)


if __name__ == "__main__":
    g, lam, v = np.sqrt(0.5), 2.0, 1.0
    delta, s = np.sqrt(2 / lam) / v, strength(g, lam)
    print(f"s = {s:.3f}, delta = {delta:.3f}")
    for k in (0.1, 0.3, 0.6, 1.0):
        T, R = transmission_numeric(k, delta, s)
        print(f"k = {k:4.1f}: T numeric = {T:.10f}, exact = {transmission_exact(k, delta, s):.10f}, R + T = {R + T:.12f}")
    print("trapped modes:", bound_states_exact(g, lam, v), bound_states_numeric(g, lam, v))
