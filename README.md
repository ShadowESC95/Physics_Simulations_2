# Physics Simulations 2

Two small projects on a photon mass that comes from a scalar field. Both are
standard physics (the Abelian Higgs / Stueckelberg model), solved numerically
and checked against exact results and public data.

| Folder | What it does |
| --- | --- |
| [`xichi_minimal/`](xichi_minimal/README.md) | One Lagrangian, equations of motion derived by computer algebra, a 1+1 dimensional solver, and a photon-mass limit from 101 localized fast radio bursts |
| [`2026/`](2026/README.md) | Scattering of a massive photon off a planar wall where the mass changes, an interactive desktop app, and an estimate of the flux lost crossing a network of such walls |

## Quick start

Python 3.9 or newer.

```
cd xichi_minimal
python3 -m pip install -r requirements.txt
python3 -m pytest -q          # 56 tests
python3 frb_limit.py          # photon-mass limit from the FRB table

cd ../2026
python3 -m pip install -r requirements.txt pytest
python3 -m pytest -q          # 15 tests
python3 wall_network_estimate.py
python3 proca_wall_app.py     # needs tkinter
```

## Main results

- A photon crossing the scalar-field kink sees a Poeschl-Teller potential; the
  simulated transmission matches the closed-form result.
- Fast radio burst dispersion gives m_gamma < about 1e-14 eV (95%) and no
  evidence for a non-zero mass. Laboratory limits are far stronger.
- A thick wall converts a fraction of about tan^2(theta) (dm/omega)^2 of
  transverse photons to longitudinal ones. For a 1e-14 eV mass change at 1 GHz
  that is about 1e-16 per wall.

Nothing here is evidence for the model. The tests show that the code solves the
stated equations, and the data analysis is an upper limit.

## Licence

Code: MIT, see [`LICENSE`](LICENSE). The burst table in `xichi_minimal/data/`
comes from the [FRBs/FRB](https://github.com/FRBs/FRB) project under the BSD
3-Clause licence, see `xichi_minimal/data/LICENSE-FRBs`.
