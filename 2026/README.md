# Photon-mass wall scattering (2026)

Scattering of a massive (Stueckelberg / Proca) photon off a planar wall where the
photon mass changes, and what a network of such walls does to a beam.

| File | What it is |
| --- | --- |
| `proca_wall_solver.py` | The physics: derivation in the docstring, reference solver, fast solver for scans |
| `proca_wall_app.py` | Interactive desktop window on top of the solver |
| `wall_network_estimate.py` | Flux loss after many walls at random angles, in physical units |
| `test_proca_wall_solver.py`, `test_wall_network.py` | Validation tests |

## Archive

`archive/` holds the earlier scripts these files replace: `S8_7_TM_Scattering_Solver.py`,
`S10_1_Parameter_Scan_TM_Longitudinal.py` and the `FRB_*` scripts. They are kept for
reference only and are not tested. Do not use their numbers:

- `FRB_PhotonMass_Propagation.py`, `FRB_PhotonMass_vs_Plasma_WithNoise.py`,
  `FRB_Polarization_CMB_Extended.py` and `FRB_Professional_Analysis_WallNetwork.py`
  divide a mass in eV by a frequency in rad/s, so their residuals are not in
  consistent units.
- `FRB_Polarization_EndToEnd_Pipeline.py` builds a residual that is zero by
  construction, and the polarization scripts need an input table that is not
  included.

## Setup

Python 3.9 or newer. From this folder:

```
python3 -m pip install -r requirements.txt
```

Everything runs offline. No paths are hard-coded: run the scripts from any
directory, and files are written only where you ask (`--plot`, `--csv`, or the
app's save dialogs).

## Run

```
python3 proca_wall_app.py                 # interactive solver
python3 wall_network_estimate.py          # default scenario: 3e-15 eV, 1 GHz, 1 Gpc
python3 wall_network_estimate.py --help   # all options
python3 -m pytest -q                      # tests (or run each test file directly)
```

## Result in one paragraph

With `q^2 = omega^2 - k^2 - m(z)^2`, the TM problem separates exactly into two
scalar equations, `E_x'' + q^2 E_x = 0` and `m^2 (Pi'/m^2)' + q^2 Pi = 0` with
`Pi = m^2 A_z`. Transverse and longitudinal photons are mixtures of these two
channels with mixing angle `tan(alpha) = k m / (omega q)`. A thick wall converts
a fraction `sin^2(alpha_R - alpha_L) ~ tan^2(theta) (dm/omega)^2`, independent of
its thickness. Averaged over random wall orientations this is about `1e-17` per
wall for a `3e-15 eV` mass change at 1 GHz, so a fractional loss of `1e-7` over
1 Gpc needs roughly ten walls per parsec.

The default masses (3e-15 and 6e-15 eV in alternating domains) are chosen to
lie inside the limit from fast radio burst dispersion, about 9e-15 eV at 95%
(`../xichi_minimal/frb_likelihood.py`). The script prints the rms mass along
the path and says so when a chosen scenario is already excluded; the earlier
default of 1e-14 eV was.
