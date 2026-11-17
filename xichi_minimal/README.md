# Xi-chi minimal model

One Lagrangian, its equations derived by computer algebra, a solver that takes
its equations from that derivation, tests against exact results, and a
comparison with real data.

```
python3 -m pip install -r requirements.txt
python3 model.py          # Lagrangian, equations of motion, unit check
python3 scattering.py     # wall transmission: numerical against exact
python3 frb_limit.py      # photon-mass limit from 101 localized FRBs
python3 -m pytest -q      # 42 tests, about 20 s
```

## What this model is

Natural units (hbar = c = 1), metric (+,-,-,-):

    L = -1/4 F_mu_nu F^mu_nu + 1/2 (d Xi)^2
        + 1/2 Xi^2 (d_mu chi - g A_mu)^2 - lambda/4 (Xi^2 - v^2)^2

Xi and chi are the modulus and phase of one complex field,
Phi = Xi exp(i chi)/sqrt(2). **This is the Abelian Higgs model in polar
variables. It is textbook physics, not a new theory.** It is used here because
it is the consistent form of the idea "a scalar field gives the photon a mass":

| Earlier Xi-chi documents | This model |
| --- | --- |
| Photon mass from kappa (grad Xi)^2, which is not Lorentz invariant | m_gamma = g Xi, a Lorentz scalar; in vacuum m_gamma = g v |
| Gauge invariance "softly broken" | Gauge invariance exact; chi is the Stueckelberg field |
| Several incompatible definitions of Xi | Xi is one field with mass dimension 1 |
| Equations typed separately in each script | One Lagrangian; every equation derived from it in `model.py` |

With Xi frozen to a profile it is the model of `../2026/proca_wall_solver.py`,
with m(z) = g Xi(z).

## The five steps and what each one found

**1. One Lagrangian with units.** [A] = [Xi] = [v] = 1, [chi] = [g] =
[lambda] = 0. `model.check_dimensions()` confirms every term has mass
dimension 4, and `model.gauge_variation()` confirms the gauge symmetry.

**2. Equations of motion derived symbolically.** `model.equations_of_motion()`
applies the Euler-Lagrange equations in 3+1 dimensions. The solver uses the
reduction to one transverse polarisation a(t, x) and xi(t, x):

    xi_tt - xi_xx + lambda xi (xi^2 - v^2) + g^2 xi a^2 = 0
    a_tt  - a_xx  + g^2 xi^2 a = 0

`model.reduction_is_consistent()` inserts this ansatz in the full equations and
shows the remaining four vanish identically. The solver's right-hand sides are
generated from these expressions (`model.numeric_rhs`), so code and equations
cannot drift apart.

**3. Known limits** (`tests/test_limits.py`).

- g = 0 gives Maxwell's wave equation exactly; a pulse travels at speed 1.000.
- Uniform vacuum gives Proca dispersion: measured group velocity equals
  k / sqrt(k^2 + g^2 v^2).
- The kink Xi = v tanh(x/delta), delta = sqrt(2/lambda)/v, solves the field
  equation exactly, stays static on the grid, and has the exact mass
  (2 sqrt(2)/3) sqrt(lambda) v^3.
- Energy error falls by 4.00 when the time step is halved (second order).

**4. Simulation with an exact benchmark** (`scattering.py`,
`tests/test_scattering.py`). A photon crossing the kink sees the
Poeschl-Teller potential, whose transmission is known in closed form:

    T(k) = sinh^2(pi k delta) / [sinh^2(pi k delta) + cos^2(pi/2 sqrt(1 + 4s))],
    s = 2 g^2 / lambda

- Mode-equation solver against the formula: agreement to 1e-8.
- Full nonlinear time-domain run of a wave packet: transmitted energy
  fraction 0.5852, formula 0.5854.
- Prediction checked: the wall is reflectionless at all frequencies when
  s = l(l+1). Simulated reflection at s = 2 is 1e-6.
- Prediction checked: photons can be trapped on the wall, at the frequencies
  given by the Poeschl-Teller levels.

**5. Real data** (`frb_limit.py`, `tests/test_frb.py`). A photon mass delays
low radio frequencies like extra dispersion, with a different redshift
dependence from intergalactic electrons. Using 101 localized fast radio
bursts (see `data/README.md`):

| Analysis | Result |
| --- | --- |
| Conservative, per burst (all extragalactic dispersion given to the photon) | m_gamma < 1.8e-14 eV |
| Joint fit, baryon fraction free | m_gamma < 1.8e-14 eV (95%) |
| Joint fit, baryon fraction fixed at 0.84 | m_gamma < 1.2e-14 eV (95%) |
| Evidence for a non-zero mass | None: chi-square improves by 0.00 |

So the data constrain the product g v < about 2e-14 eV and show no sign of a
photon mass. A mass of 3e-14 eV is excluded (delta chi2 = 230); 1e-14 eV, the
default of `../2026/wall_network_estimate.py`, is not excluded by this
analysis. An injection test confirms the fit would detect a 5e-14 eV mass.

## Limits of what is shown

- Nothing here is evidence *for* the model. Steps 3 and 4 show the code solves
  the stated equations; step 5 is an upper limit.
- The joint fit uses Gaussian scatter with assumed widths. With those widths
  chi-square is 205 for 98 bursts, driven by a few bursts with very large host
  dispersion, so the scatters are enlarged by 1.46 before quoting limits.
  Read the joint limits to within a factor of about two.
- Published FRB analyses quote limits between 1.8e-14 and 2.9e-14 eV for
  single bursts, which this code matches, and 4.0e-15 eV (68%) from nine
  bursts (Wei & Wu, arXiv:2006.09680), which it does not reach; that analysis
  makes stronger assumptions about the intergalactic and host terms.
- Laboratory and solar-system limits on a constant photon mass are far
  stronger than any FRB limit (the Particle Data Group lists about 1e-18 eV).
  They are not re-derived here.
- The simulation is 1+1 dimensional with one polarisation. Oblique incidence
  and the longitudinal mode are treated in `../2026/proca_wall_solver.py`.
- The earlier Xi-chi documents (not kept in this repository) describe additional
  effects (varying constants, entropy laws, birefringence). None of them
  follows from this Lagrangian, and none is tested here.

## References

- L. D. Landau, E. M. Lifshitz, *Quantum Mechanics*, section 25, problem 4
  (transmission through a 1/cosh^2 potential).
- L. Bonetti et al., arXiv:1602.09135 and arXiv:1701.03097 (photon mass from
  FRB dispersion; formulas for H_gamma and H_e).
- X.-F. Wu et al., arXiv:1602.07835; J.-J. Wei, X.-F. Wu, arXiv:2006.09680.
- FRB data: <https://github.com/FRBs/FRB>.
