# cyclotron-spectrum-analyzer

Code and data behind *"Magnetic-field constraints for four magnetic
cataclysmic variables from simultaneous continuum and cyclotron modelling"*
(Lin et al.). Everything needed to regenerate the paper's figures, tables and
magnetic fields is bundled; re-fitting from the raw survey spectra is also
supported.

Targets: `J0005` = DESI J000558.72+294103.8, `J0022` = DESI J002253.23+134040.7,
`J0749` = DESI J074917.11+365427.9, `J0035` = LAMOST J003553.36+433341.4.

## Method

Each spectrum is fitted with one forward model,

```
F_λ = s_wd · Koester(T_wd, log g) + s_spot · Planck(T_spot)
                                  + A_cyc · Cyc(B, kT, θ, Λ)
```

on the de-reddened, line-masked, 25-Å binned spectrum with inverse-variance
weights. The three amplitudes are solved by bounded linear least squares at
every trial of the non-linear parameters; the global search is differential
evolution; the harmonic-number ambiguity is mapped by profiling χ² over `B`;
uncertainties come from an MCMC over all ten parameters. The quoted field error
is `σ_B = sqrt(σ_formal² + (0.026·B)²)`, the 2.6 % floor calibrated on EQ Cet,
BS Tri, MQ Dra and PZ Vir.

## Quick start

```bash
pip install -r requirements.txt
./reproduce.sh                 # all figures + the Table 4 numbers
CYC_ROOT=$PWD python code/reproduce_numbers.py    # numbers only
```

No archive downloads needed: the Koester DA grid ships resampled as
`data/koester_cache.npz` (optical) and `data/koester_cache_wide.npz`
(1150–60000 Å, needed for the SED figure).

## Which script makes which figure

Paths are relative to the repository root; `{src}` ∈ `{J0005, J0022, J0749, J0035}`.

| paper item | script |
|---|---|
| Table 4 fields (B, kT, θ, logΛ, σ_B) | `code/reproduce_numbers.py` |
| Fig. 1 benchmark validation | `code/plot_profile_curves.py` |
| Fig. 2 light curves | `code/lightcurve_panels.py` |
| Fig. 3 joint fits | `validation_candidates/full_refit/code/regen_joint_fullrefit.py` |
| Fig. 4 profiled χ²(B) | `validation_candidates/full_refit/code/plot_full_refit_profiles.py` |
| Fig. 5 emission lines | `code/emission_lines.py` |
| Fig. 6 SEDs + donor limits | `code/sed_figure.py` |
| Table 3 continuum parameters, Table 6 photometry | `code/make_referee_tables.py` |
| Fig. A.1/A.2 corner plots | `code/make_corner.py`, `code/plot_corner_full.py` |
| App. B SDSS validation | `validation_candidates/full_refit/code/plot_full_refit_profiles.py` |
| App. C competing branches | `validation_candidates/full_refit/code/plot_branch_decomposition.py` |
| App. D constrained-WD fits | `code/wd_prior_refit.py`, `code/plot_wd_prior.py` |
| Field-error budget | `validation_candidates/full_refit/code/branch_errors.py` |

## SEDs and donor limits

```bash
CYC_ROOT=$PWD python code/fetch_sed_photometry.py   # GALEX/SDSS/PS1/2MASS/WISE
CYC_ROOT=$PWD python code/donor_limits.py           # spectral-type limits
CYC_ROOT=$PWD python code/sed_figure.py             # Fig. 6
```

`code/sed_state_test.py` quantifies how far the archival epochs disagree
(they do, by up to 1.9 mag), which is why the infrared gives only an upper
limit on the donor. `code/fetch_spherex.py` and `code/spherex_bin.py` retrieve
and bin the SPHEREx forced photometry; the service is an on-demand job of
order 20 min per target and needs `astro_toolbox` on `PYTHONPATH`.

## Constrained white dwarf

In the published fits `T_wd`, `log g` and `s_wd = (R_wd/d)²` are free and
independent, so their combination need not describe a real star. Table 3 gives
those values and, beside them, a refit with the photosphere parametrised by
mass alone (Nauenberg mass–radius relation, `M_wd` ∈ 0.6–1.0 M☉, `T_wd` ∈
8000–20000 K, Gaia distance):

```bash
CYC_ROOT=$PWD python code/wd_prior_refit.py --mode branches
CYC_ROOT=$PWD python code/plot_wd_prior.py
CYC_ROOT=$PWD python code/wd_diagnostics.py     # why the free fit is unphysical
```

The field is unchanged: below 0.01 MG at the adopted branch of each DESI
target, 1.6 MG for the low-S/N LAMOST spectrum (movement along its shallow
51–54 MG ridge).

## Re-fitting from scratch

Needs the **full** Koester grid, not the cache:

```bash
export KOESTER_DIR=/path/to/koester2      # SVO; see DATA_SOURCES.md
rm -f data/koester_cache.npz
CYC_ROOT=$PWD python validation_candidates/full_refit/code/full_refit_scan.py J0749
```

Re-fits are stochastic (differential evolution, MCMC) and take minutes per
source. The profiled χ²(B) field, which sets the adopted `B`, is the
deterministic part and recovers the published value.

## Layout

```
code/
  cyclotron_m2.py         numba cyclotron kernel (exponentially scaled,
                          stable down to kT ~ 0.1 keV)
  joint_pipeline.py       joint fit: Koester loader, DE search, profiled
                          χ²(B), branch polishing, MCMC
  benchmark_fit.py        EQ Cet / BS Tri validation fits
  chi2_maps.py            profiled χ² on a (B, kT) grid
  regen_bstri_map.py      converged BS Tri envelope for Fig. 1
  wd_prior_refit.py       fits with the WD tied to a mass–radius relation
  wd_diagnostics.py       physicality check on the free-fit WD
  sed_figure.py           Fig. 6; donor_limits.py, sed_state_test.py
  fetch_sed_photometry.py, fetch_spherex.py, spherex_bin.py
  emission_lines.py, emission_profiles.py
  lightcurve_analysis.py, lightcurve_panels.py
  plot_profile_curves.py, make_corner.py, plot_corner_full.py, plot_wd_prior.py
  make_referee_tables.py  Tables 3 and 6
  reproduce_numbers.py, pubstyle.py
validation_candidates/full_refit/
  code/, data/            the adopted fits, kt families, error budget
data/
  raw/                    DESI/LAMOST FITS, ZTF CSVs, benchmark spectra
  koester_cache.npz       resampled Koester DA cube (optical)
  koester_cache_wide.npz  same, 1150-60000 A, for the SEDs
  {src}_binned_spectrum.txt, {src}_fit_components.txt
  {src}_joint_results.json, {src}_mcmc_chain.npy, {src}_BT_map.npz
  {src}_lc_results.json, *_sdss/, emission_lines.{json,csv}
  sed_photometry.json     compiled archival photometry
  donor_limits.json       donor spectral-type limits
  eem_dwarf_sequence.txt  Pecaut & Mamajek dwarf sequence (2022.04.16)
  spherex/                SPHEREx forced photometry, raw and binned
  wd_prior/               constrained-WD refits
figures/                  PDF versions of the paper figures
DATA_SOURCES.md           provenance and archive links
reproduce.sh              regenerates all figures + numbers
```

## Notes

- χ² is inverse-variance weighted; posterior errors are rescaled by
  `sqrt(χ²_min/dof)`.
- The harmonic ambiguity is reported, not suppressed: alternative acceptable
  `(B, n)` solutions are kept in each `*_full_refit.json`.
- White-dwarf and hot-spot temperatures and flux ratios are nuisance
  parameters. The free decomposition is phenomenological; use the constrained
  refit if you need a physical photosphere.
- Donor limits are approximate: the distance uncertainty alone is 0.3–0.7 mag
  in absolute magnitude, comparable to half a spectral subtype.
- The kernel uses exponentially scaled forms; unscaled `exp(-x)` underflows
  below kT ~ 0.7 keV and silently flattens the model to Rayleigh–Jeans.

## License

Code under the MIT License (`LICENSE`). Bundled survey data remain subject to
the DESI, LAMOST, ZTF, SDSS and SVO archive policies (`DATA_SOURCES.md`);
please cite those sources and this paper.
