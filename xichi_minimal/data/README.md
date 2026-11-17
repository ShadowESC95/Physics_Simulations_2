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

## Checks against the source papers

The bursts that matter most for the photon-mass limit were compared with
their primary papers on 17 Nov 2026. Dispersion measures and redshifts in the
table agree with the papers except where noted. All redshifts are
spectroscopic.

| Burst | Source | Finding |
| --- | --- | --- |
| FRB 20240209A | arXiv:2410.23374, 2410.23336 | Values agree. Host probability 0.99. The burst lies 40 kpc from a quiescent elliptical, probably in a globular cluster, so almost no host dispersion is expected. |
| FRB 20231201A | arXiv:2502.11217 | Values agree. Host probability 0.91, the weakest of this group; the localization is a 2 by 20 arcsecond ellipse. The paper does not tabulate the Milky Way value, so the 67.46 here is unverified. |
| FRB 20221029A | arXiv:2308.06813, 2409.16964 | Values agree. Host probability 0.92. The sightline passes the outskirts of a galaxy cluster. |
| FRB 20220610A | arXiv:2210.04680, 2311.10815 | Values agree. The host is a compact group of galaxies; the group redshift is secure. The dispersion is about 650 pc cm^-3 above the mean for its redshift. |
| FRB 20210807D | arXiv:2408.02083, 2208.00819 | The papers give DM 251.9, the table 251.3. Host probability 1.00. The Milky Way term is about half the total. Using 251.9 leaves the limit unchanged. |
| FRB 20190611B | arXiv:2005.13162, 2005.13161 | DM is 332.63 in one paper (as here) and 321.4 to 322.2 in two others. The host association has been questioned in print. Using 322.2 or dropping the burst moves the limit by under 5%. |
| FRB 20220319D | arXiv:2301.01000 | The paper quotes a Milky Way disc value of 132.9, the table 126.77. Both exceed the observed DM of 110.95, so the excess is negative. Dropping the burst changes the limit from 7.6e-15 to 7.2e-15 eV. |

The table is left as downloaded; none of these values was edited.
