# Data provenance

Every input that the analysis depends on is listed here, with its public
origin.  The derived products under `data/` and
`validation_candidates/full_refit/data/` are our own and are released with
this package; the raw survey spectra and light curves are redistributed here
for convenience and remain subject to the original archives' usage policies.

## Science targets (raw spectra — `data/raw/`)

| target | file | archive |
|--------|------|---------|
| DESI J000558.72+294103.8 (J0005) | `39628472875746272.fits` | DESI DR1 coadd |
| DESI J002253.23+134040.7 (J0022) | `39628114199839787.fits` | DESI DR1 coadd |
| DESI J074917.11+365427.9 (J0749) | `39633019182516048.fits` | DESI DR1 coadd |
| LAMOST J003553.36+433341.4 (J0035)        | `DR11LRS_256702174.fits` | LAMOST DR11 low-resolution |
| LAMOST J003553.36+433341.4 (J0035, 2016)  | `DR11LRS_475312249.fits` | LAMOST DR11 low-resolution |

DESI: <https://data.desi.lbl.gov/> (DR1).  LAMOST: <https://www.lamost.org/dr11/>.

## Light curves (`data/raw/ZTF_*.csv`)

ZTF DR photometry, retrieved from the IRSA/ZTF light-curve service
(<https://irsa.ipac.caltech.edu/Missions/ztf.html>), one CSV per target,
named by the query coordinates.

## White-dwarf model atmospheres

Koester (2010) DA LTE model grid, from the Spanish Virtual Observatory
Theoretical Spectra service:
<http://svo2.cab.inta-csic.es/theory/newov2/index.php> (Koester DA models).

The full grid (~140 MB of `da*.dk.dat.txt` files, T_eff 5000-80000 K,
log g 6.5-9.5) is **not** redistributed here.  Instead we ship the resampled
interpolation cube actually used by the fits,
`data/koester_cache.npz` (~23 MB), which `joint_pipeline.KoesterGrid` loads
directly — so reproducing the figures and numbers needs no grid download.
To re-fit truly from scratch, download the grid from SVO, point `KOESTER_DIR`
at it, and delete the cache so it is rebuilt.

The same grid resampled to 1150-60000 A is bundled as
`data/koester_cache_wide.npz` for the ultraviolet-to-infrared SED figure.

## Benchmark systems (`data/raw/benchmarks/`)

Phase-differenced cyclotron residual spectra of two well-studied polars,
used to validate the field recovery:

| system | file | field reference |
|--------|------|-----------------|
| EQ Cet | `EQCet_processed_spectrum.txt` | Schwope et al. (2008) |
| BS Tri | `BSTri_processed_spectrum.txt` | Kolbin et al. (2022) |

The SDSS validation polars (MQ Dra, PZ Vir, SDSS J1344+2044) under
`data/*_sdss/` are donor-subtracted / extracted SDSS spectra from the SDSS
public data releases (<https://www.sdss.org/>).

## Archival photometry and dwarf sequence

Compiled by `code/fetch_sed_photometry.py` (VizieR) into
`data/sed_photometry.json`:

| catalogue | VizieR | bands |
|---|---|---|
| GALEX GR6+7 AIS | II/335 | FUV, NUV |
| SDSS DR16 | V/154 | u g r i z |
| Pan-STARRS 1 | II/349 | g r i z y |
| 2MASS PSC | II/246 | J H Ks (no detections) |
| UKIRT Hemisphere Survey DR11 | II/384 | J K (no detections) |
| AllWISE | II/328 | W1–W4 |
| CatWISE2020 | II/365 | W1, W2 |
| Spitzer SEIP | II/368 | IRAC/MIPS (no detections) |

SPHEREx forced photometry (`data/spherex/`) comes from the IRSA
spectrophotometry service, which runs Tractor photometry on every Level-2
image covering a position; `code/fetch_spherex.py` submits the job and
`code/spherex_bin.py` bins the result. Only J0005+2941 was retrieved; at these
magnitudes the per-channel S/N is below 2.

`data/eem_dwarf_sequence.txt` is the online dwarf sequence maintained by
E. Mamajek (version 2022.04.16), which extends Pecaut & Mamajek (2013) with
absolute magnitudes and WISE colours:
<https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt>

Infrared extinction ratios A_λ/A_V for the WISE bands are from
Wang & Chen (2019); at E(B−V) < 0.07 these corrections are below 0.01 mag.

## Distances and reddening

Gaia DR3 parallaxes (distances in `code/joint_pipeline.py:SOURCES`) and
`E(B-V)` from the Schlegel/Schlafly & Finkbeiner dust maps.
