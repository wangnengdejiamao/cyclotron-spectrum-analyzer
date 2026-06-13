# cyclotron-spectrum-analyzer

Code and data behind *"Magnetic fields and orbital periods of four
magnetic cataclysmic variables"* (Lin et al.). The package reproduces
the fits, figures, and tabulated numbers of the paper from the public
survey spectra and light curves.

The four targets are
`J0005` = DESI J000558.72+294103.8,
`J0022` = DESI J002253.23+134040.7,
`J0749` = DESI J074917.11+365427.9, and
`J0035` = LAMOST J003553.36+433341.4
(`J0035b` is the independent 2016-epoch LAMOST spectrum of J0035).

## What the code does

The spectrum of a polar is fitted with one forward model that adds a
Koester white-dwarf photosphere, an accretion-spot blackbody, and an
isothermal constant-Λ cyclotron component. All components are varied
together in an inverse-variance-weighted χ², so the cyclotron
parameters do not inherit the bias of a separately subtracted
continuum. The harmonic-number ambiguity is mapped by profiling χ²
over the field strength B; uncertainties come from an MCMC over all ten
parameters, with the error bars rescaled by sqrt(χ²_min/dof).

## Install

```bash
pip install -r requirements.txt
```

## Files

```
code/
  cyclotron_m2.py          numba cyclotron kernel (harmonic emissivities
                           + constant-Λ transfer; K2 and the Boltzmann
                           factor in exponentially scaled form, stable to
                           kT ~ 0.1 keV)
  joint_pipeline.py        joint continuum+cyclotron fit: DE global
                           search, profiled χ²(B), branch polishing, MCMC
  chi2_maps.py             profiled χ² on a (B, kT) grid
  benchmark_fit.py         EQ Cet and BS Tri validation fits
  lightcurve_analysis.py   ZTF detrending, Lomb-Scargle, alias control,
                           folded light curves
  lightcurve_panels.py     long-term + folded light-curve figure
  plot_profile_curves.py   χ²(B) profile figures (sources + benchmarks)
  replot_all.py            spectral decomposition figures
  polarcat_compare_v2.py   places the targets on the PolarCat B-vs-P_orb
                           plane (the population figure)
  polarcat_gaia_correlate.py  population B-vs-period analysis (exploratory)
  update_macros.py         regenerates the LaTeX number macros
  pubstyle.py              shared matplotlib style

data/fits/
  {src}_binned_spectrum.txt   dereddened, line-masked, 25-Å binned spectrum
  {src}_fit_components.txt     wave, F_obs, err, total, WD, spot, cyclotron
  {src}_joint_results.json     adopted + alternative solutions, posteriors
  {src}_mcmc_chain.npy         thinned MCMC chains

data/maps/{src}_BT_map.npz      profiled χ² on the (B, kT) grid
data/lightcurves/{src}_lc_results.json   periods, alias powers, FAP levels

benchmarks/
  EQCet_residual_spectrum.txt          full-resolution phase-differenced
  BSTri_residual_spectrum.txt          cyclotron residual spectra
                                       (wavelength [m], relative flux)
  bench_EQCet.json, bench_BSTri.json   validation fit results
  BSTri_BT_map.npz                     BS Tri (B, kT) grid

figures/   PDF versions of the paper figures
```

## Reproducing a fit

```bash
# set the input paths at the top of code/joint_pipeline.py first
python code/joint_pipeline.py --source J0005
```

The input spectra are not redistributed here: the DESI DR1 coadds,
LAMOST DR11 spectra, and ZTF light curves come from the respective
public archives, and the Koester (2010) DA model grid from the SVO
theory server.

## Notes

- χ² is inverse-variance weighted; posterior errors are rescaled by
  sqrt(χ²_min/dof) to absorb the observed scatter.
- The harmonic ambiguity is reported, not suppressed: alternative
  acceptable (B, n) solutions appear in the profiled χ²(B) curves.
- The kernel uses the exponentially scaled forms exp(-x(γ-1)) with
  x = m_e c²/kT and the scaled K2. The unscaled exp(-x) underflows for
  kT below ~0.7 keV, which silently turns the model into a featureless
  Rayleigh-Jeans spectrum; the scaled forms avoid this.
