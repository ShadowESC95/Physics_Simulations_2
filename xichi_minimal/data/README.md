# Data

`localized_frbs.csv` lists fast radio bursts that have a measured host redshift.

- **Source:** the per-burst JSON files in `frb/data/FRBs/` of the public
  repository <https://github.com/FRBs/FRB> (BSD 3-Clause, see `LICENSE-FRBs`),
  at commit `20e1aa5fc590738e6a51263306853e1e324f7ebd`, downloaded 17 Nov 2026.
- **Selection:** every burst whose file has a redshift `z`, a dispersion
  measure `DM` and a Milky Way disc estimate `DMISM`. Nothing else was
  filtered. 101 bursts.
- **Columns:** burst name, observed DM and its error (pc cm^-3), the NE2001
  Milky Way disc DM along that sightline (pc cm^-3), host redshift, repeater
  flag, and the reference keys given in the source file.

The values are observational measurements compiled by that project. Nothing in
this folder is simulated.
