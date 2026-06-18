# cyclotron-spectrum-analyzer

Code and data behind *"Magnetic-field constraints for four magnetic
cataclysmic variables from simultaneous continuum and cyclotron modelling"*
(Lin et al.).  Everything needed to regenerate the
figures and the tabulated magnetic-field results from the bundled fit products
is included; re-fitting from the raw survey spectra is also supported.

The four targets are
`J0005` = DESI J000558.72+294103.8,
`J0022` = DESI J002253.23+134040.7,
`J0749` = DESI J074917.11+365427.9, and
`J0035` = LAMOST J003553.36+433341.4
(`J0035b` is the independent 2016-epoch LAMOST spectrum of J0035).

## What the code does

Each spectrum is fitted with a single forward model that adds a Koester
white-dwarf photosphere, an accretion-spot blackbody, and an isothermal
constant-Λ cyclotron component,

```
F_λ = s_wd · Koester(T_wd, log g) + s_spot · Planck(T_spot)
                                  + A_cyc · Cyc(B, kT, θ, Λ)
```

fitted directly to the de-reddened, line-masked, 25-Å binned spectrum with
inverse-variance weights.  The three amplitudes are solved by bounded linear
least squares at every trial of the non-linear parameters; the global search
is a differential-evolution optimisation; the harmonic-number ambiguity is
mapped by profiling χ² over the field strength `B`; and uncertainties come from
an MCMC over all ten parameters.  The quoted field error is
`σ_B = sqrt(σ_formal² + (0.026·B)²)`, the 2.6 % floor being calibrated on
benchmark polars (EQ Cet, BS Tri, MQ Dra, PZ Vir).

## Quick start

```bash
pip install -r requirements.txt
./reproduce.sh                      # or:  PYTHON=/path/to/python ./reproduce.sh
```

This prints the Table 3 numbers and writes every paper figure to `figures/`.
**No archive downloads are needed** — the Koester DA grid is bundled in compact
resampled form as `data/koester_cache.npz`, so the white-dwarf component is
rebuilt without the full 140 MB model grid.

To check just the headline numbers:

```bash
CYC_ROOT=$PWD python code/reproduce_numbers.py
```

## How each figure / number is produced

All paths below are relative to the repository root.  `{src}` ∈
`{J0005, J0022, J0749, J0035}`.

| paper item | script | key inputs |
|------------|--------|------------|
| Headline adopted branch values (B, kT, θ, logΛ, σ_B) | `code/reproduce_numbers.py` | `validation_candidates/full_refit/data/{src}_full_refit.json`, `…/branch_error_summary.json` |
| Joint continuum–cyclotron fits | `validation_candidates/full_refit/code/regen_joint_fullrefit.py` | `…/{src}_full_refit.json`, `data/raw/`, Koester cache |
| Profiled χ²(B), fixed-kT families | `validation_candidates/full_refit/code/plot_full_refit_profiles.py` | `…/{src}_full_refit.json`, `…/{src}_kt_family.npz` |
| Competing harmonic branches (App. C) | `validation_candidates/full_refit/code/plot_branch_decomposition.py` | `…/{src}_full_refit.json` |
| Benchmark validation | `code/plot_profile_curves.py` | `data/bench_*_spectrum.txt`, `data/*_BT_map.npz` |
| Long-term + folded light curves | `code/lightcurve_panels.py` | `data/raw/ZTF_*.csv`, `data/{src}_lc_results.json` |
| Emission-line diagnostics | `code/emission_lines.py` | `data/emission_lines.{json,csv}`, `data/raw/` |
| J0005 cyclotron corner | `code/make_corner.py` | `data/J0005_mcmc_chain.npy` |
| SDSS validation panel (App. B) | `validation_candidates/full_refit/code/plot_full_refit_profiles.py` | `data/*_sdss/`, `…/{MQDra,PZVir,J1344}_full_refit.json` |
| Field-error budget | `validation_candidates/full_refit/code/branch_errors.py` | `…/{src}_full_refit.json` |

## Re-fitting from scratch (slow path)

The bundled `*_full_refit.json` solutions can be regenerated from the raw
spectra.  This needs the **full** Koester grid (not the cache):

```bash
# 1. download the Koester (2010) DA grid from SVO (see DATA_SOURCES.md)
export KOESTER_DIR=/path/to/koester2
rm -f data/koester_cache.npz             # force a rebuild from the raw grid
# 2. profiled χ²(B) scan + branch polishing for one source
CYC_ROOT=$PWD python validation_candidates/full_refit/code/full_refit_scan.py J0749
```

Because the search uses differential evolution and MCMC, re-fits are stochastic
and take minutes per source; the profiled χ²(B) field, which sets the adopted
`B`, is the deterministic part and recovers the published value.

## Layout

```
code/
  cyclotron_m2.py          numba cyclotron kernel (harmonic emissivities,
                           constant-Λ transfer; exponentially scaled K2 and
                           Boltzmann factor, stable down to kT ~ 0.1 keV)
  joint_pipeline.py        joint continuum+cyclotron fit: Koester grid loader,
                           DE global search, profiled χ²(B), branch polishing,
                           MCMC
  chi2_maps.py             profiled χ² on a (B, kT) grid
  benchmark_fit.py         EQ Cet / BS Tri validation fits
  emission_lines.py        Balmer / He I / He II 4686 line measurements
  emission_profiles.py     Hα/Hβ velocity-profile fits
  lightcurve_analysis.py   ZTF detrending, Lomb–Scargle, alias control, folding
  lightcurve_panels.py     long-term + folded light-curve figure
  plot_profile_curves.py   χ²(B) benchmark-validation figure
  make_corner.py           cyclotron-parameter corner plot
  reproduce_numbers.py     prints the Table 3 magnetic-field results
  pubstyle.py              shared matplotlib style
validation_candidates/full_refit/
  code/                    the adopted (full-refit) fits and their figures
  data/                    {src}_full_refit.json (adopted + alternative
                           branches, posteriors), kt_family.npz,
                           branch_error_summary.json
data/
  raw/                     DESI/LAMOST FITS, ZTF CSVs, benchmark spectra
  koester_cache.npz        resampled Koester DA cube (replaces the 140 MB grid)
  {src}_binned_spectrum.txt   de-reddened, line-masked, 25-Å binned spectrum
  {src}_fit_components.txt    wave, F_obs, err, total, WD, spot, cyclotron
  {src}_joint_results.json    per-source fit record (profiled χ², branches)
  {src}_mcmc_chain.npy        thinned joint-MCMC chain (10 parameters)
  {src}_BT_map.npz            profiled χ² on the (B, kT) grid
  {src}_lc_results.json       periods, alias powers, FAP levels
  *_sdss/                     SDSS validation-polar spectra
  emission_lines.{json,csv}, emission_profiles.json
figures/                   PDF versions of the paper figures
DATA_SOURCES.md            provenance and archive links for every input
reproduce.sh               regenerates all figures + numbers
```

## Notes

- χ² is inverse-variance weighted; posterior errors are rescaled by
  `sqrt(χ²_min/dof)`.
- The harmonic ambiguity is reported, not suppressed: alternative acceptable
  `(B, n)` solutions are kept in each `*_full_refit.json` and shown in the
  branch-decomposition figures.
- White-dwarf and hot-spot temperatures and flux ratios are nuisance parameters
  and are not quoted.
- The kernel uses exponentially scaled forms; the unscaled `exp(-x)` underflows
  for kT below ~0.7 keV and silently flattens the model to Rayleigh–Jeans.

## License

Code under the MIT License (see `LICENSE`).  Bundled raw survey data remain
subject to the policies of the DESI, LAMOST, ZTF, SDSS, and SVO archives
(see `DATA_SOURCES.md`); please cite those sources and this paper if you use
them.
