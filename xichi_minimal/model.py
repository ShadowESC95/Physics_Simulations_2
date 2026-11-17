"""
The Xi-chi model: one Lagrangian, its units, and its equations of motion.

Everything downstream (solver.py, scattering.py, the tests) takes its equations
from this file. Nothing is typed in by hand a second time.

Model (natural units hbar = c = 1, metric +---)
-----------------------------------------------
A complex scalar Phi = Xi * exp(i chi) / sqrt(2) coupled to a U(1) gauge field:

    L = -1/4 F_mu_nu F^mu_nu
        + 1/2 (d_mu Xi)(d^mu Xi)
        + 1/2 Xi^2 (d_mu chi - g A_mu)(d^mu chi - g A^mu)
        - lambda/4 (Xi^2 - v^2)^2

Xi is the modulus, chi the phase. This is the Abelian Higgs model written in
polar field variables. It is standard physics, chosen because it is the
consistent version of "a scalar field gives the photon a mass":

  * the photon mass is  m_gamma = g * Xi  (a Lorentz scalar), so in vacuum
    m_gamma = g v;
  * chi is the Stueckelberg field: the gauge symmetry
        A_mu -> A_mu + d_mu Lambda / g,   chi -> chi + Lambda
    is exact, so the mass does not break gauge invariance;
  * with Xi frozen it reduces to the Stueckelberg Lagrangian used in
    ../2026/proca_wall_solver.py, with m(z) = g Xi(z) and theta = chi / g.

Mass dimensions in 3+1 dimensions: [A] = [Xi] = [v] = 1, [chi] = [g] =
[lambda] = 0, and every term of L has dimension 4. `check_dimensions` verifies
this term by term.
"""
from __future__ import annotations

import sympy as sp
from sympy.calculus.euler import euler_equations

t, x, y, z = COORDS = sp.symbols("t x y z", real=True)
g, lam, v = sp.symbols("g lambda v", positive=True)
ETA = sp.diag(1, -1, -1, -1)

A = [sp.Function(f"A{mu}")(*COORDS) for mu in range(4)]   # A_mu, lower index
Xi = sp.Function("Xi")(*COORDS)
chi = sp.Function("chi")(*COORDS)
FIELDS = A + [Xi, chi]


def _d(f, mu):
    return sp.diff(f, COORDS[mu])


def potential(xi):
    return lam / 4 * (xi**2 - v**2) ** 2


def lagrangian_terms(a=A, xi=Xi, ch=chi):
    """The four terms of L as a dict, for any field configuration."""
    F = sp.Matrix(4, 4, lambda m, n: _d(a[n], m) - _d(a[m], n))
    maxwell = -sp.Rational(1, 4) * sum(
        ETA[m, m] * ETA[n, n] * F[m, n] ** 2 for m in range(4) for n in range(4))
    kin_xi = sp.Rational(1, 2) * sum(ETA[m, m] * _d(xi, m) ** 2 for m in range(4))
    stueckelberg = sp.Rational(1, 2) * xi**2 * sum(
        ETA[m, m] * (_d(ch, m) - g * a[m]) ** 2 for m in range(4))
    return {"maxwell": maxwell, "kinetic_Xi": kin_xi,
            "stueckelberg": stueckelberg, "potential": -potential(xi)}


def lagrangian(a=A, xi=Xi, ch=chi):
    return sum(lagrangian_terms(a, xi, ch).values())


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
def check_dimensions():
    """Mass dimension of each term of L; all must equal 4.

    Rescale lengths by 1/s, so every derivative gains a factor s, and every
    field or constant by s**(its mass dimension). A term of dimension n is
    then multiplied by s**n.
    """
    s = sp.symbols("s", positive=True)
    X = sp.symbols("X0:4", real=True)
    a_s = [s * sp.Function(f"a{m}")(*X) for m in range(4)]
    xi_s = s * sp.Function("xi")(*X)
    ch_s = sp.Function("ch")(*X)
    d = lambda f, m: s * sp.diff(f, X[m])
    F = sp.Matrix(4, 4, lambda m, n: d(a_s[n], m) - d(a_s[m], n))
    terms = {
        "maxwell": -sp.Rational(1, 4) * sum(
            ETA[m, m] * ETA[n, n] * F[m, n] ** 2 for m in range(4) for n in range(4)),
        "kinetic_Xi": sp.Rational(1, 2) * sum(ETA[m, m] * d(xi_s, m) ** 2 for m in range(4)),
        "stueckelberg": sp.Rational(1, 2) * xi_s**2 * sum(
            ETA[m, m] * (d(ch_s, m) - g * a_s[m]) ** 2 for m in range(4)),
        "potential": -lam / 4 * (xi_s**2 - (s * v) ** 2) ** 2,
    }
    out = {}
    for name, term in terms.items():
        ratio = sp.simplify(term / term.subs(s, 1))
        out[name] = sp.degree(ratio, s)
        assert sp.simplify(ratio - s ** out[name]) == 0, name
    return out


# --------------------------------------------------------------------------
# Symmetry and equations of motion
# --------------------------------------------------------------------------
def gauge_variation():
    """L after a gauge transformation minus L; must simplify to zero."""
    Lam = sp.Function("Lambda")(*COORDS)
    a_new = [A[m] + _d(Lam, m) / g for m in range(4)]
    return sp.simplify(lagrangian(a_new, Xi, chi + Lam) - lagrangian())


def equations_of_motion():
    """Euler-Lagrange equations of the full 3+1 dimensional model.

    Returns a dict {field: expression}, each expression = 0 on shell.
    """
    eqs = euler_equations(lagrangian(), FIELDS, COORDS)
    return {f: sp.expand(e.lhs - e.rhs) for f, e in zip(FIELDS, eqs)}


# --------------------------------------------------------------------------
# 1+1 dimensional reduction used by the solver
# --------------------------------------------------------------------------
# Unitary gauge chi = 0, fields depending on (t, x) only, and a single
# transverse polarisation A_y = a(t, x).
a_ = sp.Function("a")(t, x)
xi_ = sp.Function("xi")(t, x)
REDUCTION = {A[0]: 0, A[1]: 0, A[2]: a_, A[3]: 0, Xi: xi_, chi: 0}


def reduced_lagrangian():
    L = lagrangian([0, 0, a_, 0], xi_, sp.Integer(0))
    return sp.expand(L)


def reduced_equations():
    """{xi: ..., a: ...} = 0, derived from the reduced Lagrangian."""
    eqs = euler_equations(reduced_lagrangian(), [xi_, a_], [t, x])
    return {f: sp.expand(e.lhs - e.rhs) for f, e in zip([xi_, a_], eqs)}


def reduction_is_consistent():
    """Insert the ansatz in the full equations of motion.

    Returns the six residuals. A consistent truncation needs the A_0, A_x,
    A_z and chi equations to vanish identically and the Xi and A_y equations
    to coincide with `reduced_equations`.
    """
    full = equations_of_motion()
    red = reduced_equations()
    res = {}
    for f, e in full.items():
        res[f] = sp.simplify(e.subs(REDUCTION).doit())
    res[Xi] = sp.simplify(res[Xi] - red[xi_])
    res[A[2]] = sp.simplify(res[A[2]] - red[a_])
    return res


def accelerations():
    """Solve the reduced equations for the second time derivatives.

    Returns (expr_xi_tt, expr_a_tt, symbols) with the fields and their second
    space derivatives replaced by plain symbols (xi, a, xi_xx, a_xx).
    """
    red = reduced_equations()
    s_xi, s_a, s_xixx, s_axx = sp.symbols("xi a xi_xx a_xx", real=True)
    out = []
    for f in (xi_, a_):
        tt = sp.Derivative(f, (t, 2))
        sol = sp.solve(red[f], tt)
        assert len(sol) == 1
        expr = sol[0].subs({sp.Derivative(xi_, (x, 2)): s_xixx,
                            sp.Derivative(a_, (x, 2)): s_axx}).subs({xi_: s_xi, a_: s_a})
        out.append(sp.expand(expr))
    return out[0], out[1], (s_xi, s_a, s_xixx, s_axx)


def energy_density():
    """Hamiltonian density of the reduced model, by Legendre transform."""
    L = reduced_lagrangian()
    H = sum(sp.diff(L, sp.Derivative(f, t)) * sp.Derivative(f, t) for f in (xi_, a_)) - L
    return sp.expand(H)


def numeric_rhs(g_val, lam_val, v_val):
    """NumPy functions (acc_xi, acc_a) generated from the symbolic equations."""
    e_xi, e_a, syms = accelerations()
    sub = {g: g_val, lam: lam_val, v: v_val}
    return (sp.lambdify(syms, e_xi.subs(sub), "numpy"),
            sp.lambdify(syms, e_a.subs(sub), "numpy"))


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------
def photon_mass():
    """Vacuum photon mass m_gamma = g v."""
    return g * v


def scalar_mass():
    """Mass of the Xi excitation about the vacuum, sqrt(V''(v))."""
    xi = sp.symbols("xi")
    return sp.sqrt(sp.simplify(sp.diff(potential(xi), xi, 2).subs(xi, v)))


def kink_width():
    """delta in Xi = v tanh(x / delta)."""
    return sp.sqrt(2 / lam) / v


def kink(xpos=x):
    return v * sp.tanh(xpos / kink_width())


def kink_residual():
    """The static kink with a = 0 inserted in the Xi equation; must be zero."""
    red = reduced_equations()[xi_]
    return sp.simplify(red.subs({a_: 0, xi_: kink()}).doit())


if __name__ == "__main__":
    print("L (1+1 reduction) =", reduced_lagrangian())
    for f, e in reduced_equations().items():
        print(f"EOM[{f}]: {e} = 0")
    print("dimensions:", check_dimensions())
    print("gauge variation:", gauge_variation())
    print("m_gamma =", photon_mass(), "  m_Xi =", scalar_mass(), "  delta =", kink_width())
