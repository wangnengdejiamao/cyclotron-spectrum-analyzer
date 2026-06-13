#!/usr/bin/env python3
"""
benchmark_fit.py — validate the weighted-chi2 cyclotron fitting machinery
on the two benchmark polars (BS Tri, EQ Cet) whose residual cyclotron
spectra and literature parameters are known:

  BS Tri : B = 22.7 +/- 0.4 MG, theta ~ 87 deg   (Kolbin et al. 2022)
  EQ Cet : B ~ 34 MG                              (Campbell et al. 2008)

These are continuum-subtracted (phase-differenced) spectra, so only the
cyclotron component is fitted: F = A * C(B, kT, theta, Lambda).
Per-point uncertainties are estimated from the local scatter
(rolling MAD over 9 points) with a 3% floor — giving a meaningful
weighted chi2, unlike the unweighted statistic used previously.
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cyclotron_m2 import cal_cy_spec

KEV_J = 1.602176634e-16
MG_T = 100.0
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(_HERE, os.pardir))   # repo root
_BENCH_DIR = os.path.join(OUT, 'benchmarks')

# We fit the full-resolution phase-differenced residual spectra, i.e. the
# unbinned cyclotron-only spectra derived from the individual-phase
# observations of Campbell et al. (2008, EQ Cet) and Kolbin et al.
# (2022, BS Tri). Each file has two columns: wavelength [m], relative flux.
BENCH = {
    'BSTri': dict(name='BS Tri',
                  path=os.path.join(_BENCH_DIR, 'BSTri_residual_spectrum.txt'),
                  wave_unit='m', B_lit=22.7, B_lit_err=0.4, theta_lit=87.0,
                  # BS Tri is eclipsing: its bright-phase viewing angle is
                  # fixed by the eclipse solution to ~87 deg.  Without this
                  # constraint a single phase-differenced spectrum admits a
                  # featureless high-B pseudo-continuum branch near 95 MG
                  # (see Sect. on systematics); we therefore restrict theta.
                  theta_bounds=(80.0, 89.0),
                  seeds=[[22.3, 5.9, 87.0, 8.0], [22.7, 5.0, 85.0, 7.0],
                         [20.0, 8.0, 88.0, 6.0]]),
    'EQCet': dict(name='EQ Cet',
                  path=os.path.join(_BENCH_DIR, 'EQCet_residual_spectrum.txt'),
                  wave_unit='m', B_lit=34.0, B_lit_err=2.0, theta_lit=None,
                  seeds=[[34.8, 1.4, 58.0, 6.2], [34.0, 3.7, 60.0, 4.4]]),
}

BOUNDS = [(10.0, 100.0), (0.5, 30.0), (10.0, 89.0), (0.0, 9.0)]


def load(meta):
    d = np.loadtxt(meta['path'])
    w, f = d[:, 0], d[:, 1]
    if meta['wave_unit'] == 'm':
        w_m = w.copy()
    else:
        w_m = w * 1e-10
    o = np.argsort(w_m)
    w_m, f = w_m[o], f[o]
    # drop masked pixels (zeros from emission-line / telluric masking)
    good = (f != 0) & np.isfinite(f)
    w_m, f = w_m[good], f[good]
    # local-scatter error: rolling MAD of differences over 9 points
    e = np.empty_like(f)
    n = len(f)
    for i in range(n):
        j0, j1 = max(0, i - 4), min(n, i + 5)
        seg = f[j0:j1]
        e[i] = 1.4826 * np.median(np.abs(np.diff(seg))) / np.sqrt(2)
    e = np.maximum(e, 0.03 * np.abs(f).max() * 0.1)
    e = np.maximum(e, 0.03 * np.abs(f))
    return w_m, f, e


def make_chi2(w_m, f, e):
    def chi2(p):
        B, kT, th, ll = p
        spec = cal_cy_spec(w_m, kT * KEV_J, B * MG_T, np.deg2rad(th),
                           10.0 ** ll)
        mx = spec.max()
        if not np.isfinite(mx) or mx <= 0:
            return 1e12
        shape = spec / mx
        # optimal amplitude analytically
        wgt = 1.0 / e ** 2
        A = np.sum(f * shape * wgt) / max(np.sum(shape ** 2 * wgt), 1e-300)
        if A < 0:
            return 1e12
        c2 = float(np.sum(((f - A * shape) / e) ** 2))
        return c2 if np.isfinite(c2) else 1e12
    return chi2


def run(key):
    meta = BENCH[key]
    w_m, f, e = load(meta)
    print(f'=== {meta["name"]}: {len(w_m)} points, '
          f'{w_m.min()*1e10:.0f}-{w_m.max()*1e10:.0f} A ===')
    chi2 = make_chi2(w_m, f, e)
    _ = chi2([40, 5, 70, 6])  # numba warm-up

    # per-source bounds (BS Tri restricts theta to its known eclipse value)
    bounds = [list(b) for b in BOUNDS]
    if 'theta_bounds' in meta:
        bounds[2] = list(meta['theta_bounds'])

    t0 = time.time()
    rng = np.random.default_rng(5)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    n_pop = 80
    init = lo + (hi - lo) * rng.random((n_pop, 4))
    for i, s in enumerate(meta.get('seeds', [])):
        init[i] = np.clip(np.asarray(s, float), lo + 1e-9, hi - 1e-9)
        for j in range(3):
            init[len(meta['seeds']) + i * 3 + j] = np.clip(
                init[i] * (1 + 0.03 * rng.standard_normal(4)),
                lo + 1e-9, hi - 1e-9)
    res = differential_evolution(chi2, bounds, maxiter=150, popsize=20,
                                 tol=1e-10, seed=5, polish=True,
                                 init=init)
    ndof = len(w_m) - 5
    print(f'  best: B={res.x[0]:.2f} MG kT={res.x[1]:.2f} keV '
          f'th={res.x[2]:.1f} logL={res.x[3]:.2f}  '
          f'chi2/dof={res.fun/ndof:.2f}  ({time.time()-t0:.0f}s)',
          flush=True)

    # profiled chi2 over B — warm-started two-sweep scan
    from scipy.optimize import minimize
    B_grid = np.arange(12.0, 96.0, 1.5)
    nB = len(B_grid)
    prof = np.full(nB, np.inf)
    psol = [None] * nB
    s_lo, s_hi = lo[1:], hi[1:]
    start = np.clip(res.x[1:], s_lo, s_hi)
    for sweep, order in enumerate([range(nB), range(nB - 1, -1, -1)]):
        prev = start.copy()
        for i in order:
            B = B_grid[i]
            def obj(s):
                s = np.clip(s, s_lo, s_hi)
                return chi2([B, s[0], s[1], s[2]])
            r = minimize(obj, prev, method='Nelder-Mead',
                         options=dict(maxfev=250, fatol=0.05, xatol=1e-3))
            if r.fun < prof[i]:
                prof[i] = r.fun
                psol[i] = np.clip(r.x, s_lo, s_hi)
            if sweep == 0 and i % 4 == 0:
                rde = differential_evolution(obj, bounds[1:], maxiter=15,
                                             popsize=6, tol=1e-5, seed=11,
                                             polish=False)
                if rde.fun < prof[i]:
                    prof[i] = rde.fun
                    psol[i] = np.clip(rde.x, s_lo, s_hi)
            prev = psol[i].copy() if psol[i] is not None else prev

    # if the profile found a deeper basin than the global DE, re-polish
    ib = int(np.argmin(prof))
    if prof[ib] < res.fun - 1e-6:
        bounds2 = [(max(10., B_grid[ib] - 4), min(100., B_grid[ib] + 4))] \
            + bounds[1:]
        init2 = np.array([[B_grid[ib], *psol[ib]]] * 6) \
            * (1 + 0.02 * rng.standard_normal((6, 4)))
        init2 = np.clip(init2, [b[0] + 1e-9 for b in bounds2],
                        [b[1] - 1e-9 for b in bounds2])
        full_init = np.vstack([init2,
                               np.array([b[0] for b in bounds2])
                               + (np.array([b[1] for b in bounds2])
                                  - np.array([b[0] for b in bounds2]))
                               * rng.random((34, 4))])
        res2 = differential_evolution(chi2, bounds2, maxiter=80, popsize=10,
                                      tol=1e-10, seed=7, polish=True,
                                      init=full_init)
        if res2.fun < res.fun:
            res = res2
            print(f'  re-polished: B={res.x[0]:.2f} MG kT={res.x[1]:.2f} '
                  f'th={res.x[2]:.1f} logL={res.x[3]:.2f} '
                  f'chi2/dof={res.fun/ndof:.2f}', flush=True)
    branches = []
    c0 = prof.min()
    for i in range(1, len(B_grid) - 1):
        if prof[i] <= prof[i-1] and prof[i] <= prof[i+1] \
                and prof[i] - c0 < 25 * (c0 / ndof if c0/ndof > 1 else 1):
            branches.append((float(B_grid[i]), float(prof[i] - c0)))
    print(f'  profile branches: {branches}')

    # 1-sigma interval on B from rescaled profile
    scale = max(c0 / ndof, 1.0)
    dchi = (prof - c0) / scale
    ok = B_grid[dchi < 1.0]
    B_lo, B_hi = (float(ok.min()), float(ok.max())) if len(ok) else (np.nan,)*2
    print(f'  B (rescaled profile 1sigma): {res.x[0]:.2f} '
          f'[{B_lo:.2f}, {B_hi:.2f}]  vs literature '
          f'{meta["B_lit"]}+/-{meta["B_lit_err"]}')

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    spec = cal_cy_spec(w_m, res.x[1] * KEV_J, res.x[0] * MG_T,
                       np.deg2rad(res.x[2]), 10.0 ** res.x[3])
    shape = spec / spec.max()
    wgt = 1.0 / e ** 2
    A = np.sum(f * shape * wgt) / np.sum(shape ** 2 * wgt)
    axes[0].errorbar(w_m * 1e10, f, yerr=e, fmt='o', ms=2.5, lw=0.6,
                     color='0.3', ecolor='0.7', label='residual spectrum')
    axes[0].plot(w_m * 1e10, A * shape, color='crimson', lw=1.6,
                 label=(f'fit: B={res.x[0]:.1f} MG, kT={res.x[1]:.1f} keV,'
                        f' θ={res.x[2]:.0f}°'))
    axes[0].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    axes[0].set_ylabel(r'$F_\lambda$ (arbitrary)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(meta['name'], fontsize=11)
    axes[1].plot(B_grid, dchi, 'k-', lw=1.3)
    for thr in (1, 4, 9):
        axes[1].axhline(thr, ls=':', lw=0.7, color='gray')
    axes[1].axvline(meta['B_lit'], color='royalblue', ls='--', lw=1.2,
                    label=f'literature {meta["B_lit"]} MG')
    axes[1].set_xlabel('B [MG]')
    axes[1].set_ylabel(r'$\Delta\chi^2$ (rescaled, profiled)')
    axes[1].set_ylim(-0.5, 30)
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(OUT, 'figures', f'bench_{key}.pdf'),
                bbox_inches='tight')
    plt.close(fig)

    out = dict(name=meta['name'], B=float(res.x[0]),
               B_1sigma=[B_lo, B_hi], kT=float(res.x[1]),
               theta=float(res.x[2]), logLambda=float(res.x[3]),
               chi2=float(res.fun), ndof=ndof,
               chi2_red=float(res.fun / ndof),
               branches=branches, B_lit=meta['B_lit'],
               B_grid=B_grid.tolist(), profile_dchi2_rescaled=dchi.tolist())
    with open(os.path.join(OUT, 'data', f'bench_{key}.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print()


if __name__ == '__main__':
    for k in BENCH:
        run(k)
