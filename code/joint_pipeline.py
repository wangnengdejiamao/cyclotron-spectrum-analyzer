#!/usr/bin/env python3
"""
joint_pipeline.py — Simultaneous continuum + cyclotron forward modelling.

Replaces the two-step "subtract continuum, then fit residual" approach of
extract_cyclotron_koester.py, whose problems were:

  1. Lower-envelope bias: the negativity penalty forced the continuum to lie
     below the data everywhere, so the "cyclotron residual" absorbed every
     imperfection of the continuum model.
  2. Fixed continuum windows (~4950/6100/7400 A) presuppose where the cyclotron
     humps are NOT — circular, since hump positions depend on the B we measure.
  3. Soft-clipping of negative residuals introduced an asymmetric bias.
  4. Emission lines were replaced by interpolation (invents correlated data)
     instead of being masked out of the likelihood.
  5. The cyclotron fit used an UNWEIGHTED chi^2, so the chi^2 scale was
     meaningless (reduced chi2 ~ 0.01) and curvature errors were nonsense.
  6. Continuum systematics entered only as per-bin Gaussian errors; the
     covariance between continuum and cyclotron parameters was lost.

New approach (this file):
  F_model(lam) = s_wd * Koester(lam; T_wd, logg)
               + s_spot * Planck(lam; T_spot)
               + A_cyc * Cyc(lam; B, kT, theta, Lambda)        [normalised]

  * fitted directly to the de-reddened, line-masked, 25-A binned spectrum
    with inverse-variance weights;
  * amplitudes (s_wd, s_spot, A_cyc) solved by bounded weighted linear
    least squares at every trial of the non-linear parameters, with the
    s_wd bound set by the Gaia distance and a physical WD radius range;
  * global search by differential evolution over the non-linear parameters;
  * harmonic-number ambiguity made explicit by a profiled chi^2(B) scan;
  * uncertainties from a full MCMC over ALL parameters (continuum and
    cyclotron jointly), so continuum uncertainty propagates into B, kT,
    theta, Lambda automatically.

Usage:
    python3 joint_pipeline.py --source J0005 [--quick]
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import differential_evolution, lsq_linear, minimize
import emcee
import corner as corner_mod

warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/ljm/Desktop/cyc/优化算法/m2_optimized')
from cyclotron_m2 import cal_cy_spec  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
KEV_J = 1.602176634e-16          # keV -> J
MG_T = 100.0                     # MG -> Tesla
R_SUN_CM = 6.957e10
PC_CM = 3.0857e18

KOESTER_DIR = '/Users/ljm/Desktop/cyc/long expose/koester2'
OUT_BASE = '/Users/ljm/Desktop/cyc/paper_v2'

SOURCES = {
    'J0005': dict(
        name='DESI J000558.72+294103.8',
        fits='/Users/ljm/Desktop/csst/long expose/desi/39628472875746272.fits',
        kind='desi', flux_scale=1e-17,
        dist_pc=604.0, dist_err=98.0, ebv=0.0416,
        seeds=[[6250, 7.25, 25500, 56.8, 0.723, 57.8, 5.43],
               [7000, 8.0, 35000, 58.0, 1.0, 68.8, 7.0],
               [6500, 7.25, 40600, 57.2, 0.72, 65.9, 8.99]]),
    'J0022': dict(
        name='DESI J002253.23+134040.7',
        fits='/Users/ljm/Desktop/csst/long expose/desi/39628114199839787.fits',
        kind='desi', flux_scale=1e-17,
        dist_pc=583.0, dist_err=38.0, ebv=0.0677,
        seeds=[[10000, 8.0, 30000, 44.6, 12.3, 71.3, 6.5],
               [10000, 8.0, 30000, 60.0, 5.0, 60.0, 5.0]]),
    'J0749': dict(
        name='DESI J074917.11+365427.9',
        fits='/Users/ljm/Desktop/csst/long expose/desi/39633019182516048.fits',
        kind='desi', flux_scale=1e-17,
        dist_pc=946.0, dist_err=135.0, ebv=0.0544,
        seeds=[[17500, 9.0, 79000, 48.0, 10.0, 50.0, 5.0],
               [17500, 9.0, 79000, 77.2, 25.4, 35.8, 4.01],
               [12000, 8.5, 50000, 40.2, 12.2, 72.9, 3.4]]),
    'J0035': dict(
        name='LAMOST J003553.36+433341.4',
        fits='/Users/ljm/Desktop/csst/long expose/lamost/DR11LRS_256702174.fits',
        fits_alt='/Users/ljm/Desktop/csst/long expose/lamost/DR11LRS_475312249.fits',
        kind='lamost', flux_scale=1.0,     # relative flux; no absolute anchor
        dist_pc=None, dist_err=None, ebv=0.0582,
        seeds=[[10000, 8.0, 30000, 50.0, 5.0, 60.0, 5.0],
               [10000, 8.0, 30000, 30.0, 5.0, 60.0, 5.0]]),
}

LINE_MASKS = [
    # (centre, half-width)  Balmer (incl. high-order lines near the jump)
    (6563, 50), (4861, 45), (4340, 38), (4102, 32), (3970, 25), (3889, 25),
    (3835, 20), (3798, 18),
    # He I
    (4471, 22), (5876, 25), (6678, 22), (7065, 22), (4026, 20), (4922, 18),
    (5015, 18), (4713, 15), (7281, 18),
    # He II
    (4686, 28), (5411, 18), (4541, 15), (4200, 15),
    # Na D / Ca II / O I
    (5893, 18), (8498, 15), (8542, 15), (8662, 15), (7774, 15), (8446, 15),
    # telluric / sky
    (7620, 45), (6870, 35), (7180, 30), (5577, 12), (6300, 12), (6360, 10),
    (8950, 60), (9300, 80), (8230, 40),
]

# ---------------------------------------------------------------------------
# Koester grid (trimmed copy of the loader from extract_cyclotron_koester.py)
# ---------------------------------------------------------------------------
class KoesterGrid:
    def __init__(self, grid_dir, wave_grid=None):
        cache = os.path.join(OUT_BASE, 'data', 'koester_cache.npz')
        if wave_grid is None and os.path.exists(cache):
            z = np.load(cache)
            self.teffs, self.loggs = z['teffs'], z['loggs']
            self.wave = z['wave']
            self.interp = RegularGridInterpolator(
                (self.teffs, self.loggs, self.wave), z['cube'],
                method='linear', bounds_error=False, fill_value=0.0)
            print(f'  Koester grid: cached ({len(self.teffs)} Teff x '
                  f'{len(self.loggs)} logg)')
            return
        files = sorted(glob.glob(os.path.join(grid_dir, 'da*.dk.dat.txt')))
        params, waves, fluxes = [], [], []
        for f in files:
            m = re.match(r'da(\d{5})_(\d{3})\.dk\.dat\.txt', os.path.basename(f))
            if not m:
                continue
            try:
                data = np.loadtxt(f, comments='#')
            except Exception:
                continue
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            params.append((float(m.group(1)), float(m.group(2)) / 100.0))
            waves.append(data[:, 0])
            fluxes.append(data[:, 1])
        teffs = sorted({p[0] for p in params})
        loggs = sorted({p[1] for p in params})
        if wave_grid is None:
            wave_grid = np.arange(3500.0, 10001.0, 1.0)
        cube = np.full((len(teffs), len(loggs), len(wave_grid)), np.nan)
        idx = {(t, g): i for i, (t, g) in enumerate(params)}
        for (t, g), i in idx.items():
            w, f = waves[i], fluxes[i]
            o = np.argsort(w)
            w, f = w[o], f[o]
            u = np.concatenate([[True], np.diff(w) > 0])
            itp = interp1d(w[u], f[u], bounds_error=False, fill_value=0.0)
            cube[teffs.index(t), loggs.index(g)] = itp(wave_grid)
        # nearest-neighbour fill for missing grid nodes
        for it in range(len(teffs)):
            for ig in range(len(loggs)):
                if np.any(np.isnan(cube[it, ig])):
                    best, bd = None, 1e30
                    for (t, g) in params:
                        d = (it - teffs.index(t))**2 + (ig - loggs.index(g))**2
                        if 0 < d < bd:
                            bd, best = d, (teffs.index(t), loggs.index(g))
                    if best is not None:
                        cube[it, ig] = cube[best]
        self.teffs = np.array(teffs)
        self.loggs = np.array(loggs)
        self.wave = wave_grid
        cube = np.nan_to_num(cube)
        self.interp = RegularGridInterpolator(
            (self.teffs, self.loggs, wave_grid), cube,
            method='linear', bounds_error=False, fill_value=0.0)
        print(f'  Koester grid: {len(params)} models, '
              f'T {self.teffs.min():.0f}-{self.teffs.max():.0f} K, '
              f'logg {self.loggs.min():.2f}-{self.loggs.max():.2f}')
        try:
            os.makedirs(os.path.join(OUT_BASE, 'data'), exist_ok=True)
            np.savez_compressed(cache, teffs=self.teffs, loggs=self.loggs,
                                wave=self.wave, cube=cube)
            print('  Koester grid cached for future runs')
        except Exception as exc:
            print(f'  (cache write failed: {exc})')

    def shape(self, wave_AA, teff, logg):
        pts = np.column_stack([np.full_like(wave_AA, teff),
                               np.full_like(wave_AA, logg), wave_AA])
        return self.interp(pts)


def planck(wave_AA, T):
    h, c, kb = 6.626e-27, 2.998e10, 1.381e-16
    lam = np.asarray(wave_AA) * 1e-8
    x = np.clip(h * c / (lam * kb * T), None, 500)
    return 2 * h * c**2 / lam**5 / (np.exp(x) - 1.0)


def ccm89_alam_av(wave_AA, rv=3.1):
    """CCM89 A(lambda)/A(V) for optical/NIR wavelengths."""
    x = 1e4 / np.asarray(wave_AA, dtype=float)   # 1/micron
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    # infrared 0.3-1.1
    m = (x >= 0.3) & (x < 1.1)
    a[m] = 0.574 * x[m]**1.61
    b[m] = -0.527 * x[m]**1.61
    # optical 1.1-3.3
    m = (x >= 1.1) & (x <= 3.3)
    y = x[m] - 1.82
    a[m] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
            + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b[m] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
            - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    return a + b / rv


# ---------------------------------------------------------------------------
# Data loading / preparation
# ---------------------------------------------------------------------------
def load_spectrum(meta):
    with fits.open(meta['fits']) as hdul:
        if meta['kind'] == 'desi':
            d = hdul[1].data
            w = np.asarray(d['WAVELENGTH'], float).ravel()
            f = np.asarray(d['FLUX'], float).ravel()
            iv = np.asarray(d['IVAR'], float).ravel()
        else:  # lamost COADD
            d = hdul['COADD'].data
            w = np.asarray(d['WAVELENGTH'], float).ravel()
            f = np.asarray(d['FLUX'], float).ravel()
            iv = np.asarray(d['IVAR'], float).ravel()
    o = np.argsort(w)
    w, f, iv = w[o], f[o], iv[o]
    good = (iv > 0) & np.isfinite(f) & (f != 0)
    w, f, iv = w[good], f[good], iv[good]
    e = 1.0 / np.sqrt(iv)
    s = meta['flux_scale']
    f, e = f * s, e * s
    # de-redden
    if meta['ebv'] > 0:
        alam = ccm89_alam_av(w) * 3.1 * meta['ebv']
        corr = 10 ** (0.4 * alam)
        f, e = f * corr, e * corr
    return w, f, e


def mask_lines(w):
    m = np.ones_like(w, dtype=bool)
    for c, hw in LINE_MASKS:
        m &= ~((w > c - hw) & (w < c + hw))
    return m


def bin_spectrum(w, f, e, bw=25.0, wmin=3950.0, wmax=9300.0, min_frac=0.3):
    # wmin = 3950: excludes the crowded Balmer-jump region (H8-H11 + Ca II H/K
    # blend below ~3950 A) where line masking is incomplete and survey flux
    # calibration is least reliable; residuals there previously produced
    # spurious negative (data - continuum) excursions.
    """IVAR-weighted binning. Returns wave, flux, err (with 3% flux floor)."""
    edges = np.arange(wmin, wmax + bw, bw)
    bw_, bf_, be_ = [], [], []
    npix_full = np.median(np.diff(w))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (w >= lo) & (w < hi)
        n = m.sum()
        if n < min_frac * (hi - lo) / npix_full:
            continue
        wgt = 1.0 / e[m]**2
        mu = np.sum(f[m] * wgt) / np.sum(wgt)
        sig_formal = np.sqrt(1.0 / np.sum(wgt))
        # robust scatter contribution (handles unmodelled pixel-to-pixel noise)
        sig_scat = 1.4826 * np.median(np.abs(f[m] - mu)) / np.sqrt(n)
        sig = max(sig_formal, sig_scat)
        bw_.append(0.5 * (lo + hi))
        bf_.append(mu)
        be_.append(sig)
    bwv = np.array(bw_)
    bfv = np.array(bf_)
    bev = np.array(be_)
    # 3% calibration floor relative to a smooth version of the flux
    floor = 0.03 * np.abs(bfv)
    bev = np.sqrt(bev**2 + floor**2)
    return bwv, bfv, bev


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
class JointModel:
    """Continuum (Koester WD + spot BB) + cyclotron, amplitudes via
    bounded weighted linear least squares."""

    def __init__(self, koester, wave_AA, flux, err, s_wd_bounds):
        self.k = koester
        self.w = wave_AA
        self.wm = wave_AA * 1e-10          # m, for cyclotron model
        self.f = flux
        self.e = err
        self.s_wd_bounds = s_wd_bounds      # (lo, hi) for s_wd
        self._cyc_cache = {}

    def cyc_shape(self, B_mg, kT_kev, theta_deg, log_lam):
        key = (round(B_mg, 3), round(kT_kev, 3), round(theta_deg, 2),
               round(log_lam, 3))
        if key in self._cyc_cache:
            return self._cyc_cache[key]
        spec = cal_cy_spec(self.wm, kT_kev * KEV_J, B_mg * MG_T,
                           np.deg2rad(theta_deg), 10.0 ** log_lam)
        mx = spec.max()
        shape = spec / mx if mx > 0 else spec
        if len(self._cyc_cache) > 4000:
            self._cyc_cache.clear()
        self._cyc_cache[key] = shape
        return shape

    def components(self, p):
        """p = [T_wd, logg, T_spot, B, kT, theta, loglam]"""
        wd = self.k.shape(self.w, p[0], p[1])
        sp = planck(self.w, p[2])
        sp = sp / sp.max()
        cy = self.cyc_shape(p[3], p[4], p[5], p[6])
        return wd, sp, cy

    def solve_amplitudes(self, p):
        wd, sp, cy = self.components(p)
        wd = np.nan_to_num(wd, nan=0.0, posinf=0.0, neginf=0.0)
        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
        cy = np.nan_to_num(cy, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.column_stack([wd, sp, cy]) / self.e[:, None]
        y = self.f / self.e
        # column normalisation: amplitudes span ~25 dex, which breaks LAPACK
        norms = np.sqrt(np.sum(A ** 2, axis=0))
        norms[norms <= 0] = 1.0
        An = A / norms[None, :]
        lo = np.array([self.s_wd_bounds[0] * norms[0], 0.0, 0.0])
        hi = np.array([self.s_wd_bounds[1] * norms[0], np.inf, np.inf])
        try:
            res = lsq_linear(An, y, bounds=(lo, hi), method='bvls',
                             max_iter=200)
            amps = res.x / norms
        except Exception:
            return np.zeros(3), np.zeros_like(self.f), 1e12
        model = wd * amps[0] + sp * amps[1] + cy * amps[2]
        chi2 = float(np.sum(((self.f - model) / self.e) ** 2))
        if not np.isfinite(chi2):
            return amps, model, 1e12
        return amps, model, chi2

    def chi2(self, p):
        if p[2] <= p[0]:          # T_spot must exceed T_WD
            return 1e12
        _, _, c2 = self.solve_amplitudes(p)
        return c2

    def chi2_full(self, q):
        """q = [T_wd, logg, T_spot, B, kT, theta, loglam,
                log_s_wd, log_s_spot, log_A_cyc] — for MCMC."""
        p = q[:7]
        if p[2] <= p[0]:
            return 1e12
        wd, sp, cy = self.components(p)
        wd = np.nan_to_num(wd, nan=0.0, posinf=0.0, neginf=0.0)
        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
        cy = np.nan_to_num(cy, nan=0.0, posinf=0.0, neginf=0.0)
        model = (10**q[7]) * wd + (10**q[8]) * sp + (10**q[9]) * cy
        c2 = float(np.sum(((self.f - model) / self.e) ** 2))
        return c2 if np.isfinite(c2) else 1e12


NONLIN_BOUNDS = [
    (6000.0, 40000.0),    # T_wd
    (7.0, 9.49),          # logg
    (8000.0, 100000.0),   # T_spot
    (10.0, 100.0),        # B [MG]
    (0.3, 30.0),          # kT [keV]
    (10.0, 89.0),         # theta [deg]
    (0.0, 9.0),           # log10 Lambda
]
PNAMES = ['T_wd', 'logg', 'T_spot', 'B_MG', 'kT_keV', 'theta_deg', 'log_Lambda']


def run_global_fit(model, maxiter=250, popsize=15, seed=7, seeds=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    ndim = len(NONLIN_BOUNDS)
    n_pop = max(popsize * ndim, 30)
    lo = np.array([b[0] for b in NONLIN_BOUNDS])
    hi = np.array([b[1] for b in NONLIN_BOUNDS])
    init = lo + (hi - lo) * rng.random((n_pop, ndim))
    if seeds:
        for i, s in enumerate(seeds[:max(1, n_pop // 4)]):
            base = np.clip(np.asarray(s, float), lo + 1e-9, hi - 1e-9)
            init[i] = base
            # a few jittered copies of each seed
            for j in range(3):
                row = base * (1 + 0.03 * rng.standard_normal(ndim))
                init[(i + 1) * 4 + j] = np.clip(row, lo + 1e-9, hi - 1e-9)
    res = differential_evolution(
        model.chi2, NONLIN_BOUNDS, maxiter=maxiter, popsize=popsize,
        tol=1e-8, seed=seed, polish=True, init=init,
        mutation=(0.4, 1.2), recombination=0.8)
    amps, mdl, chi2 = model.solve_amplitudes(res.x)
    print(f'  global DE: chi2={chi2:.1f} in {time.time()-t0:.0f}s -> '
          + ', '.join(f'{n}={v:.3g}' for n, v in zip(PNAMES, res.x)))
    return res.x, amps, chi2


def profile_B(model, best_p, B_grid, maxiter=40, popsize=8, seed=11):
    """Profiled chi2 over B with warm-started local optimization.

    Two sweeps (up then down) over the grid; at each B the sub-space
    (kT, theta, loglam) is optimized with Nelder-Mead started from the
    neighbouring grid solution, plus a small DE refresh on a subset of
    points to escape local traps. Continuum shape parameters are held at
    the reference solution; amplitudes are re-solved at every trial.
    """
    sub_bounds = [NONLIN_BOUNDS[4], NONLIN_BOUNDS[5], NONLIN_BOUNDS[6]]
    s_lo = np.array([b[0] for b in sub_bounds])
    s_hi = np.array([b[1] for b in sub_bounds])
    n = len(B_grid)
    chis = np.full(n, np.inf)
    sols = [None] * n

    def obj_factory(B):
        def obj(s):
            s = np.clip(s, s_lo, s_hi)
            p = np.array([best_p[0], best_p[1], best_p[2], B,
                          s[0], s[1], s[2]])
            return model.chi2(p)
        return obj

    start = np.array([best_p[4], best_p[5], best_p[6]])
    for sweep, order in enumerate([range(n), range(n - 1, -1, -1)]):
        prev = start.copy()
        for i in order:
            obj = obj_factory(B_grid[i])
            r = minimize(obj, prev, method='Nelder-Mead',
                         options=dict(maxfev=220, fatol=0.05, xatol=1e-3))
            if r.fun < chis[i]:
                chis[i] = r.fun
                sols[i] = np.clip(r.x, s_lo, s_hi)
            # occasional DE refresh on first sweep
            if sweep == 0 and i % 4 == 0:
                rde = differential_evolution(obj, sub_bounds,
                                             maxiter=max(10, maxiter // 3),
                                             popsize=max(5, popsize - 2),
                                             tol=1e-5, seed=seed,
                                             polish=False)
                if rde.fun < chis[i]:
                    chis[i] = rde.fun
                    sols[i] = np.clip(rde.x, s_lo, s_hi)
            prev = sols[i].copy() if sols[i] is not None else prev
            if sweep == 0 and i % 8 == 0:
                print(f'    B={B_grid[i]:5.1f} MG  chi2={chis[i]:9.1f}',
                      flush=True)
    return chis, np.array([s if s is not None else start for s in sols])


def fine_profile(model, ref_p, B_center, sub_seed, half_width=3.0,
                 step=0.25):
    """Fine-grained profiled chi2 around a minimum (warm-started NM),
    used for a smooth profile curve and a profile-based 1-sigma interval
    on B that is independent of the MCMC."""
    sub_bounds = [NONLIN_BOUNDS[4], NONLIN_BOUNDS[5], NONLIN_BOUNDS[6]]
    s_lo = np.array([b[0] for b in sub_bounds])
    s_hi = np.array([b[1] for b in sub_bounds])
    Bg = np.arange(max(NONLIN_BOUNDS[3][0], B_center - half_width),
                   min(NONLIN_BOUNDS[3][1], B_center + half_width) + step,
                   step)
    chis = np.empty_like(Bg)
    prev = np.clip(np.asarray(sub_seed, float), s_lo, s_hi)
    order = np.argsort(np.abs(Bg - B_center))   # outward from the centre
    tmp = {}
    for i in order:
        def obj(s):
            s = np.clip(s, s_lo, s_hi)
            p = np.array([ref_p[0], ref_p[1], ref_p[2], Bg[i],
                          s[0], s[1], s[2]])
            return model.chi2(p)
        r = minimize(obj, prev, method='Nelder-Mead',
                     options=dict(maxfev=260, fatol=0.02, xatol=1e-3))
        tmp[i] = (r.fun, np.clip(r.x, s_lo, s_hi))
        prev = tmp[i][1]
    for i, (c, _s) in tmp.items():
        chis[i] = c
    return Bg, chis


def profile_interval(Bg, chis, chi2_red):
    """1-sigma interval on B from the rescaled fine profile."""
    scale = max(chi2_red, 1.0)
    d = (chis - chis.min()) / scale
    ok = Bg[d < 1.0]
    if len(ok) == 0:
        return None
    return [float(ok.min()), float(ok.max())]


def find_branches(B_grid, chis, dchi2_max=25.0):
    """Local minima of the profiled curve within dchi2_max of the global
    min. Grid endpoints are included: a minimum running into the edge
    usually signals the high-B/high-kT pseudo-continuum degeneracy and
    must be polished and inspected, not silently dropped."""
    c0 = chis.min()
    branches = []
    for i in range(1, len(B_grid) - 1):
        if chis[i] <= chis[i-1] and chis[i] <= chis[i+1] \
                and chis[i] - c0 < dchi2_max:
            branches.append((float(B_grid[i]), float(chis[i] - c0)))
    if chis[0] < chis[1] and chis[0] - c0 < dchi2_max:
        branches.append((float(B_grid[0]), float(chis[0] - c0)))
    if chis[-1] < chis[-2] and chis[-1] - c0 < dchi2_max:
        branches.append((float(B_grid[-1]), float(chis[-1] - c0)))
    # merge near-duplicates (within 3 MG)
    merged = []
    for b, d in sorted(branches, key=lambda t: t[1]):
        if all(abs(b - mb) > 3.0 for mb, _ in merged):
            merged.append((b, d))
    return merged


def cyc_contrast(model, p, wave_lo=4500.0, wave_hi=9000.0):
    """Hump contrast of the cyclotron component: (max-min)/max within the
    well-observed window. Featureless pseudo-continuum solutions have low
    contrast and do not explain discrete humps."""
    cy = model.cyc_shape(p[3], p[4], p[5], p[6])
    m = (model.w >= wave_lo) & (model.w <= wave_hi)
    seg = cy[m]
    if seg.max() <= 0:
        return 0.0
    return float((seg.max() - seg.min()) / seg.max())


def polish_branch(model, best_p, B_center, seed=23, sub_seed=None):
    bounds = list(NONLIN_BOUNDS)
    bounds[3] = (max(10.0, B_center - 4.0), min(100.0, B_center + 4.0))
    rng = np.random.default_rng(seed)
    ndim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    n_pop = 8 * ndim
    init = lo + (hi - lo) * rng.random((n_pop, ndim))
    base = np.array([best_p[0], best_p[1], best_p[2], B_center,
                     best_p[4], best_p[5], best_p[6]])
    if sub_seed is not None:
        base[4:7] = sub_seed
    init[0] = np.clip(base, lo + 1e-9, hi - 1e-9)
    for j in range(1, 6):
        init[j] = np.clip(base * (1 + 0.04 * rng.standard_normal(ndim)),
                          lo + 1e-9, hi - 1e-9)
    res = differential_evolution(model.chi2, bounds, maxiter=70, popsize=8,
                                 tol=1e-8, seed=seed, polish=True,
                                 init=init)
    amps, mdl, chi2 = model.solve_amplitudes(res.x)
    return res.x, amps, chi2


def run_mcmc(model, best_p, best_amps, n_walkers=40, n_burn=800,
             n_steps=2200, seed=42, sigma_scale=1.0):
    """sigma_scale: error-bar inflation sqrt(chi2_min/ndof) so the posterior
    widths absorb the observed scatter about the best model (conservative
    treatment of unmodelled systematics)."""
    rng = np.random.default_rng(seed)
    q0 = np.concatenate([best_p, np.log10(np.maximum(best_amps, 1e-30))])
    ndim = len(q0)
    s2 = max(sigma_scale, 1.0) ** 2

    lo = np.array([b[0] for b in NONLIN_BOUNDS]
                  + [np.log10(max(model.s_wd_bounds[0], 1e-30)), -30, -30])
    hi = np.array([b[1] for b in NONLIN_BOUNDS]
                  + [np.log10(model.s_wd_bounds[1]), 0, 0])
    # amplitude upper bounds: generous (log10 space)
    hi[8] = q0[8] + 4 if np.isfinite(q0[8]) else 0
    hi[9] = q0[9] + 4 if np.isfinite(q0[9]) else 0
    lo[8] = q0[8] - 6 if np.isfinite(q0[8]) else -30
    lo[9] = q0[9] - 6 if np.isfinite(q0[9]) else -30

    def log_prob(q):
        if np.any(q < lo) or np.any(q > hi):
            return -np.inf
        c2 = model.chi2_full(q)
        if not np.isfinite(c2) or c2 >= 1e12:
            return -np.inf
        return -0.5 * c2 / s2

    scale = np.array([200, 0.05, 2000, 0.3, 0.3, 1.0, 0.1, 0.05, 0.05, 0.05])
    pos = q0[None, :] + scale[None, :] * rng.standard_normal((n_walkers, ndim))
    pos = np.clip(pos, lo + 1e-9, hi - 1e-9)
    # ensure T_spot > T_wd at start
    bad = pos[:, 2] <= pos[:, 0]
    pos[bad, 2] = pos[bad, 0] + 2000.0

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    t0 = time.time()
    state = sampler.run_mcmc(pos, n_burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)
    print(f'  MCMC: {n_walkers}x{n_steps} in {time.time()-t0:.0f}s, '
          f'acc={np.mean(sampler.acceptance_fraction):.2f}')
    return sampler.get_chain(flat=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_decomposition(src, model, p, amps, outdir, tag='', raw=None):
    wd, sp, cy = model.components(p)
    total = amps[0]*wd + amps[1]*sp + amps[2]*cy
    cont = amps[0]*wd + amps[1]*sp
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1.4],
                                              hspace=0.05))
    ax = axes[0]
    if raw is not None:
        w_raw, f_raw = raw
        sel = (w_raw >= model.w.min() - 150) & (w_raw <= model.w.max() + 150)
        ax.plot(w_raw[sel], f_raw[sel], color='0.55', lw=0.4, alpha=0.8,
                label='observed spectrum', zorder=1)
        ax.set_xlim(model.w.min() - 120, model.w.max() + 120)
        ymax = np.percentile(f_raw[sel], 99.5)
        ax.set_ylim(min(0, np.percentile(f_raw[sel], 0.5)), 1.25 * ymax)
    else:
        ax.errorbar(model.w, model.f, yerr=model.e, fmt='o', ms=2.5, lw=0.7,
                    color='0.25', ecolor='0.65', label='binned spectrum',
                    zorder=2)
    ax.plot(model.w, total, color='crimson', lw=1.6, label='total model',
            zorder=4)
    ax.plot(model.w, amps[0]*wd, '--', color='royalblue', lw=1.1,
            label=f'WD  (T={p[0]:.0f} K, logg={p[1]:.2f})')
    ax.plot(model.w, amps[1]*sp, '--', color='darkorange', lw=1.1,
            label=f'spot (T={p[2]:.0f} K)')
    ax.plot(model.w, amps[2]*cy, '-', color='seagreen', lw=1.3,
            label=(f'cyclotron (B={p[3]:.1f} MG, kT={p[4]:.1f} keV, '
                   f'θ={p[5]:.0f}°, logΛ={p[6]:.1f})'))
    ax.set_ylabel(r'$F_\lambda$  [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'{SOURCES[src]["name"]}  — joint fit {tag}', fontsize=11)

    ax = axes[1]
    ax.errorbar(model.w, model.f - cont, yerr=model.e, fmt='o', ms=2.5,
                lw=0.7, color='navy', ecolor='0.7',
                label='data − continuum')
    ax.plot(model.w, amps[2]*cy, color='seagreen', lw=1.6,
            label='cyclotron component')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'Wavelength  [$\mathrm{\AA}$]')
    ax.set_ylabel(r'$F_\lambda - F_{\rm cont}$')
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(outdir, f'{src}_joint_fit{tag}.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def plot_profile(src, B_grid, chis, branches, outdir, fine=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    c0 = chis.min()
    if fine is not None:
        c0 = min(c0, fine[1].min())
    d = chis - c0
    ax.plot(B_grid, d, '-', color='k', lw=1.4)
    if fine is not None:
        ax.plot(fine[0], fine[1] - c0, '-', color='crimson', lw=1.1,
                label='fine scan (0.25 MG)')
        ax.legend(fontsize=8, loc='upper left')
    for thr, lab in [(1, r'1$\sigma$'), (4, r'2$\sigma$'), (9, r'3$\sigma$')]:
        ax.axhline(thr, ls=':', lw=0.8, color='gray')
        ax.text(B_grid[-1], thr, ' ' + lab, va='center', fontsize=8,
                color='gray')
    for b, dd in branches:
        ax.plot(b, dd, 'v', color='crimson', ms=8)
    ax.set_xlabel('B  [MG]')
    ax.set_ylabel(r'$\Delta\chi^2$ (profiled over all other parameters)')
    ax.set_ylim(-1, min(60, max(12, d.max() * 1.05)))
    ax.set_title(SOURCES[src]['name'], fontsize=11)
    fig.savefig(os.path.join(outdir, f'{src}_B_profile.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def plot_corner(src, chain, outdir):
    labels = [r'$T_{\rm WD}$', r'$\log g$', r'$T_{\rm spot}$', r'$B$ [MG]',
              r'$kT$ [keV]', r'$\theta$ [deg]', r'$\log\Lambda$',
              r'$\log s_{\rm WD}$', r'$\log s_{\rm spot}$',
              r'$\log A_{\rm cyc}$']
    fig = corner_mod.corner(chain, labels=labels,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            title_kwargs={'fontsize': 8},
                            label_kwargs={'fontsize': 9})
    fig.savefig(os.path.join(outdir, f'{src}_corner.pdf'),
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def refine(src):
    """Re-profile and re-sample around the best polished solution from an
    existing results JSON (use when the polish stage found a deeper minimum
    than the global DE, so the saved profile/posterior used a stale
    continuum shape)."""
    meta = SOURCES[src]
    outdir = os.path.join(OUT_BASE, 'figures')
    datadir = os.path.join(OUT_BASE, 'data')
    with open(os.path.join(datadir, f'{src}_joint_results.json')) as fh:
        prev = json.load(fh)

    print(f'=== {src} (refine): {meta["name"]} ===')
    w, f, e = load_spectrum(meta)
    keep = mask_lines(w)
    wb, fb, eb = bin_spectrum(w[keep], f[keep], e[keep])
    s_lo, s_hi = prev['s_wd_bounds']
    koester = KoesterGrid(KOESTER_DIR)
    model = JointModel(koester, wb, fb, eb, (s_lo, s_hi))
    _ = model.cyc_shape(40.0, 5.0, 70.0, 6.0)

    sols = sorted(prev['solutions'], key=lambda s: s['chi2'])
    best_p = np.array(sols[0]['p'])
    # polish once more from the stored best
    pb, ab, cb = polish_branch(model, best_p, best_p[3], seed=31)
    if cb > sols[0]['chi2']:
        pb = best_p
        ab, _, cb = model.solve_amplitudes(pb)
    print(f'  refined best: chi2={cb:.1f}, B={pb[3]:.2f}, kT={pb[4]:.2f}, '
          f'th={pb[5]:.1f}, logL={pb[6]:.2f}')

    B_grid = np.arange(12.0, 96.0, 1.5)
    chis, _ = profile_B(model, pb, B_grid, maxiter=30, popsize=7)
    branches = find_branches(B_grid, chis)
    print(f'  branches: {branches}')
    solutions = [dict(p=pb.tolist(), amps=ab.tolist(), chi2=cb,
                      contrast=cyc_contrast(model, pb))]
    for bB, _d in branches[:4]:
        if abs(bB - pb[3]) < 3.0:
            continue
        p2, a2, c2 = polish_branch(model, pb, bB)
        solutions.append(dict(p=p2.tolist(), amps=a2.tolist(), chi2=c2,
                              contrast=cyc_contrast(model, p2)))
        print(f'    branch B~{bB:.0f}: chi2={c2:.1f}, B={p2[3]:.2f}, '
              f'contrast={solutions[-1]["contrast"]:.2f}')
    solutions.sort(key=lambda s: s['chi2'])
    # adopt the best STRUCTURED solution (hump contrast > 0.15); a lower
    # chi2 from a featureless pseudo-continuum corner is reported but not
    # adopted, since it does not explain the observed humps
    structured = [s for s in solutions if s['contrast'] > 0.15]
    adopted = structured[0] if structured else solutions[0]
    pb = np.array(adopted['p'])
    ab = np.array(adopted['amps'])
    cb = adopted['chi2']
    for s in solutions:
        s['adopted'] = (s is adopted)

    ndof = len(wb) - 10
    chi2_red = cb / ndof
    out = dict(prev)
    out.update(best_chi2=cb, chi2_red=chi2_red,
               sigma_rescale=float(np.sqrt(max(chi2_red, 1.0))),
               solutions=solutions, branches=branches,
               B_grid=B_grid.tolist(), profile_chi2=chis.tolist(),
               refined=True)
    with open(os.path.join(datadir, f'{src}_joint_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    plot_decomposition(src, model, pb, ab, outdir)
    if len(solutions) > 1:
        plot_decomposition(src, model, np.array(solutions[1]['p']),
                           np.array(solutions[1]['amps']), outdir,
                           tag='_alt')
    plot_profile(src, B_grid, chis, branches, outdir)

    chain = run_mcmc(model, pb, ab, n_walkers=32, n_burn=700, n_steps=1800,
                     sigma_scale=np.sqrt(max(chi2_red, 1.0)))
    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    post = {}
    names = PNAMES + ['log_s_wd', 'log_s_spot', 'log_A_cyc']
    print('  posterior (16/50/84):')
    for i, n in enumerate(names):
        post[n] = dict(q16=float(q16[i]), q50=float(q50[i]),
                       q84=float(q84[i]))
        print(f'    {n:>10s} = {q50[i]:9.3f}  (-{q50[i]-q16[i]:.3f} '
              f'+{q84[i]-q50[i]:.3f})')
    out['posterior'] = post
    with open(os.path.join(datadir, f'{src}_joint_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    np.save(os.path.join(datadir, f'{src}_mcmc_chain.npy'), chain[::4])
    plot_corner(src, chain, outdir)
    print(f'  refined results saved for {src}\n')
    return out


def process(src, quick=False):
    meta = SOURCES[src]
    outdir = os.path.join(OUT_BASE, 'figures')
    datadir = os.path.join(OUT_BASE, 'data')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    print(f'=== {src}: {meta["name"]} ===')
    w, f, e = load_spectrum(meta)
    keep = mask_lines(w)
    wb, fb, eb = bin_spectrum(w[keep], f[keep], e[keep])
    print(f'  {len(w)} pixels -> {keep.sum()} unmasked -> {len(wb)} bins; '
          f'median S/N per bin = {np.median(fb/eb):.1f}')

    # s_wd bounds from Gaia distance (DESI absolute fluxes); free for LAMOST
    if meta['dist_pc'] is not None:
        d_lo = (meta['dist_pc'] - 2 * meta['dist_err']) * PC_CM
        d_hi = (meta['dist_pc'] + 2 * meta['dist_err']) * PC_CM
        s_lo = (0.003 * R_SUN_CM / d_hi) ** 2
        s_hi = (0.030 * R_SUN_CM / d_lo) ** 2
    else:
        s_lo, s_hi = 1e-30, 1e-5
    print(f'  s_wd bounds: [{s_lo:.2e}, {s_hi:.2e}]')

    koester = KoesterGrid(KOESTER_DIR)
    model = JointModel(koester, wb, fb, eb, (s_lo, s_hi))

    # warm-up compile
    _ = model.cyc_shape(40.0, 5.0, 70.0, 6.0)

    mi, ps = (60, 8) if quick else (120, 10)
    best_p, best_amps, best_chi2 = run_global_fit(model, maxiter=mi,
                                                  popsize=ps,
                                                  seeds=meta.get('seeds'))
    # checkpoint after global fit
    with open(os.path.join(datadir, f'{src}_checkpoint.json'), 'w') as fh:
        json.dump(dict(stage='global', p=best_p.tolist(),
                       amps=best_amps.tolist(), chi2=best_chi2), fh)

    B_grid = np.arange(12.0, 96.0, 3.0 if quick else 1.5)
    prof_mi, prof_ps = (15, 6) if quick else (30, 7)
    chis, prof_sols = profile_B(model, best_p, B_grid, maxiter=prof_mi,
                                popsize=prof_ps)
    branches = find_branches(B_grid, chis)
    print(f'  branches (B, dchi2): {branches}')
    # checkpoint after profile
    with open(os.path.join(datadir, f'{src}_checkpoint.json'), 'w') as fh:
        json.dump(dict(stage='profile', p=best_p.tolist(),
                       chi2=best_chi2, B_grid=B_grid.tolist(),
                       profile_chi2=chis.tolist(),
                       branches=branches), fh)

    # polish every branch; keep global best & alternates
    solutions = []
    for bB, _d in branches[:3]:
        iB = int(np.argmin(np.abs(B_grid - bB)))
        pb, ab, cb = polish_branch(model, best_p, bB,
                                   sub_seed=prof_sols[iB])
        solutions.append(dict(p=pb.tolist(), amps=ab.tolist(), chi2=cb))
        print(f'    polished branch B~{bB:.0f}: chi2={cb:.1f}, '
              f'B={pb[3]:.2f}, kT={pb[4]:.2f}, th={pb[5]:.1f}, '
              f'logL={pb[6]:.2f}')
    if not solutions:
        solutions = [dict(p=best_p.tolist(), amps=best_amps.tolist(),
                          chi2=best_chi2)]
    solutions.sort(key=lambda s: s['chi2'])
    if solutions[0]['chi2'] > best_chi2:
        solutions.insert(0, dict(p=best_p.tolist(),
                                 amps=best_amps.tolist(), chi2=best_chi2))

    # adopt the best STRUCTURED solution (hump contrast > 0.15); a lower
    # chi2 from a featureless pseudo-continuum corner is reported but not
    # adopted (see paper, systematics section)
    for s in solutions:
        s['contrast'] = cyc_contrast(model, np.array(s['p']))
    structured = [s for s in solutions if s['contrast'] > 0.15]
    adopted = structured[0] if structured else solutions[0]
    for s in solutions:
        s['adopted'] = (s is adopted)
    p1 = np.array(adopted['p'])
    a1 = np.array(adopted['amps'])

    ndof = len(wb) - 10
    chi2_red = adopted['chi2'] / ndof

    # fine-grained profile around the adopted minimum (0.25 MG steps)
    iB = int(np.argmin(np.abs(B_grid - p1[3])))
    Bg_fine, chis_fine = fine_profile(model, p1, p1[3],
                                      prof_sols[iB])
    B_1sig = profile_interval(Bg_fine, chis_fine, chi2_red)
    print(f'  fine profile: B 1-sigma (rescaled) = {B_1sig}')

    out = dict(
        source=src, name=meta['name'], fits=meta['fits'],
        n_bins=len(wb), ndof=ndof,
        best_chi2=adopted['chi2'],
        chi2_red=chi2_red,
        sigma_rescale=float(np.sqrt(max(chi2_red, 1.0))),
        solutions=solutions,
        branches=branches,
        B_grid=B_grid.tolist(), profile_chi2=chis.tolist(),
        B_grid_fine=Bg_fine.tolist(), profile_chi2_fine=chis_fine.tolist(),
        B_1sigma_profile=B_1sig,
        s_wd_bounds=[s_lo, s_hi],
        quick=quick,
    )
    with open(os.path.join(datadir, f'{src}_joint_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    np.savetxt(os.path.join(datadir, f'{src}_binned_spectrum.txt'),
               np.column_stack([wb, fb, eb]),
               header='wave_AA  flux_cgs  err_cgs (dereddened, line-masked)')
    # model components on the binned grid, for publication/reproduction
    wd_c, sp_c, cy_c = model.components(p1)
    np.savetxt(os.path.join(datadir, f'{src}_fit_components.txt'),
               np.column_stack([wb, fb, eb,
                                a1[0]*wd_c + a1[1]*sp_c + a1[2]*cy_c,
                                a1[0]*wd_c, a1[1]*sp_c, a1[2]*cy_c]),
               fmt='%.6e',
               header=('wave_AA  F_obs  F_err  F_model_total  F_WD  '
                       'F_spot  F_cyclotron   (dereddened, cgs per AA; '
                       f'adopted solution B={p1[3]:.2f} MG)'))
    plot_decomposition(src, model, p1, a1, outdir, raw=(w, f))
    if len(solutions) > 1:
        p2 = np.array(solutions[1]['p'])
        a2 = np.array(solutions[1]['amps'])
        plot_decomposition(src, model, p2, a2, outdir, tag='_alt',
                           raw=(w, f))
    plot_profile(src, B_grid, chis, branches, outdir,
                 fine=(Bg_fine, chis_fine))

    # MCMC around primary solution, with error-bar inflation
    nw, nb, ns = (24, 200, 400) if quick else (32, 700, 1800)
    chain = run_mcmc(model, p1, a1, n_walkers=nw, n_burn=nb, n_steps=ns,
                     sigma_scale=np.sqrt(max(chi2_red, 1.0)))
    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    post = {}
    names = PNAMES + ['log_s_wd', 'log_s_spot', 'log_A_cyc']
    print('  posterior (16/50/84, sigma rescaled by '
          f'{np.sqrt(max(chi2_red,1.0)):.2f}):')
    for i, n in enumerate(names):
        post[n] = dict(q16=float(q16[i]), q50=float(q50[i]),
                       q84=float(q84[i]))
        print(f'    {n:>10s} = {q50[i]:9.3f}  (-{q50[i]-q16[i]:.3f} '
              f'+{q84[i]-q50[i]:.3f})')

    out['posterior'] = post
    with open(os.path.join(datadir, f'{src}_joint_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    np.save(os.path.join(datadir, f'{src}_mcmc_chain.npy'), chain[::4])
    plot_corner(src, chain, outdir)
    print(f'  saved results for {src}\n')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                    choices=list(SOURCES) + ['all'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--refine', action='store_true')
    args = ap.parse_args()
    targets = list(SOURCES) if args.source == 'all' else [args.source]
    for s in targets:
        if args.refine:
            refine(s)
        else:
            process(s, quick=args.quick)
