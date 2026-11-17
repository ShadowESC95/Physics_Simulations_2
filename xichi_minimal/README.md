# Xi-chi minimal model

One Lagrangian, its equations derived by computer algebra, a solver that takes
its equations from that derivation, tests against exact results, and a
comparison with real data.

```
python3 -m pip install -r requirements.txt
python3 model.py          # Lagrangian, equations of motion, unit check
python3 scattering.py     # wall transmission: numerical against exact
python3 frb_limit.py      # photon-mass limit from 101 localized FRBs
python3 frb_likelihood.py # the same with skewed distributions, about 4 min
python3 -m pytest -q      # 56 tests, about 90 s
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

**5. Real data** (`frb_limit.py`, `frb_likelihood.py`, `tests/test_frb.py`,
`tests/test_frb_likelihood.py`). A photon mass delays low radio frequencies
like extra dispersion, with a different redshift dependence from intergalactic
electrons. Using 101 localized fast radio bursts (see `data/README.md`), 98 of
them above z = 0.01 in the fits:

| Analysis | 95% upper limit on m_gamma |
| --- | --- |
| Conservative, per burst (all extragalactic dispersion given to the photon) | 1.8e-14 eV |
| Joint fit, Gaussian scatter (`frb_limit.py`) | 1.8e-14 eV |
| Joint fit, skewed distributions (`frb_likelihood.py`) | **9.5e-15 eV** |

No fit shows evidence for a non-zero mass: the best-fit mass is zero and the
likelihood does not improve when a mass is allowed.

The skewed fit uses the distributions of FRB cosmology (Macquart et al. 2020):
the intergalactic dispersion follows a long-tailed law whose width shrinks
with redshift, and the host dispersion is log-normal. Both are bounded below,
and a photon mass can only add dispersion, so the lower edges are what
constrain it. A Gaussian has no edge, which is why that fit is weaker. Four
nuisance parameters are fitted with the mass: F = 0.26, median host
dispersion 86 pc cm^-3, log-normal width 1.06, and f_IGM = 1.0 (at the edge
of its range).

How much the skewed limit moves with the assumptions:

| Change | 95% limit |
| --- | --- |
| Baseline (halo 50 pc cm^-3, Milky Way error 30 pc cm^-3) | 9.5e-15 eV |
| Milky Way halo 30 / 80 pc cm^-3 | 1.1e-14 / 7.1e-15 eV |
| Milky Way error 15 / 50 pc cm^-3 | 6.2e-15 / 1.3e-14 eV |
| f_IGM fixed at 0.84 | 1.1e-14 eV |
| F fixed at 0.32 | 9.5e-15 eV |
| Only bursts above z = 0.05 (88 bursts) | 8.9e-15 eV |
| Without the three most constraining bursts | 1.3e-14 eV |
| Prior flat in the mass instead of the profile likelihood | 9.9e-15 eV (68%: 5.5e-15 eV) |

So the data constrain the product g v < about 1e-14 eV, to within a factor of
about 1.5 either way set by how well the Milky Way contribution is known. A
mass of 2e-14 eV is excluded (delta of -2 ln L = 299). The value 1e-14 eV,
the default of `../2026/wall_network_estimate.py`, sits at the edge of the
95% limit. Injection tests confirm that both fits detect a mass added to the
real bursts.

## Limits of what is shown

- Nothing here is evidence *for* the model. Steps 3 and 4 show the code solves
  the stated equations; step 5 is an upper limit.
- The Gaussian fit needs its assumed scatters enlarged by 1.46 to describe
  the data (chi-square 205 for 98 bursts), because a few bursts have very
  large host dispersion. The skewed fit has no such rescaling; its long tails
  absorb those bursts.
- The skewed limit depends on the Milky Way subtraction: the halo value and
  the size of its error are assumed, not fitted, and they move the limit
  between 6e-15 and 1.3e-14 eV. The limit also rests on a handful of bursts
  with low dispersion for their redshift.
- The limits use a profile likelihood. The flat-prior number in the table
  uses the profile in place of a full marginalisation over the nuisance
  parameters.
- Published FRB analyses quote limits between 1.8e-14 and 2.9e-14 eV for
  single bursts, which this code matches, and 4.0e-15 eV (68%) from nine
  bursts (Wei & Wu, arXiv:2006.09680). The comparable number here is
  5.5e-15 eV (68%), close to it but not below it despite ten times as many
  bursts; this analysis lets the Milky Way error and the host distribution
  take more of the dispersion.
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
- J.-P. Macquart et al., Nature 581, 391 (2020), arXiv:2005.13161
  (distributions of the intergalactic and host dispersion).
- FRB data: <https://github.com/FRBs/FRB>.
