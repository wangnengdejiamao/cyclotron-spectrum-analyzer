#!/usr/bin/env python3
"""
chi2_maps.py — profiled chi^2 maps in the (B, kT) plane.

At every grid point (B, kT) the remaining cyclotron parameters
(theta, log Lambda) are re-optimized with warm-started Nelder-Mead and
the component amplitudes are re-solved by bounded weighted linear least
squares; the continuum shape parameters are held at the adopted
solution, as in the 1-D profile.  Output: {src}_BT_map.npz.

Usage:  python3 chi2_maps.py J0005 J0022 ...     (or 'EQCet')
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pipeline as jp
from cyclotron_m2 import cal_cy_spec

OUT = '/Users/ljm/Desktop/cyc/paper_v2'
KEV_J = 1.602176634e-16

B_GRID = np.arange(12.0, 95.01, 2.0)
KT_GRID = np.geomspace(0.5, 30.0, 14)


def snake(n_i, n_j):
    for i in range(n_i):
        rng = range(n_j) if i % 2 == 0 else range(n_j - 1, -1, -1)
        for j in rng:
            yield i, j


_W = {}


def _init_worker(src):
    import os as _os
    _os.environ['NUMBA_NUM_THREADS'] = '1'
    meta = jp.SOURCES[src]
    r = json.load(open(f'{OUT}/data/{src}_joint_results.json'))
    adopted = next(s for s in r['solutions'] if s.get('adopted'))
    p0 = np.array(adopted['p'])
    w, f, e = jp.load_spectrum(meta)
    keep = jp.mask_lines(w)
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep])
    if meta['dist_pc'] is not None:
        d_lo = (meta['dist_pc'] - 2 * meta['dist_err']) * jp.PC_CM
        d_hi = (meta['dist_pc'] + 2 * meta['dist_err']) * jp.PC_CM
        s_b = ((0.003 * jp.R_SUN_CM / d_hi) ** 2,
               (0.030 * jp.R_SUN_CM / d_lo) ** 2)
    else:
        s_b = (1e-30, 1e-5)
    koester = jp.KoesterGrid(jp.KOESTER_DIR)
    model = jp.JointModel(koester, wb, fb, eb, s_b)
    _ = model.cyc_shape(40.0, 5.0, 70.0, 6.0)
    _W['model'] = model
    _W['p0'] = p0


def _row(args):
    i, kT = args
    model, p0 = _W['model'], _W['p0']
    lo = np.array([20.0, 0.0])
    hi = np.array([89.0, 9.0])
    prev = np.clip(np.array([p0[5], p0[6]]), lo, hi)
    out = np.empty(len(B_GRID))
    order = np.argsort(np.abs(B_GRID - p0[3]))
    tmp = {}
    for j in order:
        B = B_GRID[j]

        def obj(s):
            s = np.clip(s, lo, hi)
            p = np.array([p0[0], p0[1], p0[2], B, kT, s[0], s[1]])
            return model.chi2(p)
        res = minimize(obj, prev, method='Nelder-Mead',
                       options=dict(maxfev=140, fatol=0.05, xatol=2e-3))
        tmp[j] = res.fun
        prev = np.clip(res.x, lo, hi)
    for j, v in tmp.items():
        out[j] = v
    print(f'  row {i+1}/{len(KT_GRID)} (kT={kT:.2f}) done', flush=True)
    return i, out


def map_source(src):
    from multiprocessing import get_context
    r = json.load(open(f'{OUT}/data/{src}_joint_results.json'))
    adopted = next(s for s in r['solutions'] if s.get('adopted'))
    p0 = np.array(adopted['p'])
    chi = np.full((len(KT_GRID), len(B_GRID)), np.inf)
    ctx = get_context('fork')
    with ctx.Pool(6, initializer=_init_worker, initargs=(src,)) as pool:
        for i, row in pool.imap_unordered(_row, list(enumerate(KT_GRID))):
            chi[i] = row
    np.savez(f'{OUT}/data/{src}_BT_map.npz', B=B_GRID, kT=KT_GRID, chi2=chi,
             best=p0, chi2_red=r['chi2_red'], best_chi2=adopted['chi2'])
    print(f'{src} map saved', flush=True)


def map_eqcet():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_fit import load, make_chi2, BENCH
    w_m, fl, e = load(BENCH['EQCet'])
    chi2fn = make_chi2(w_m, fl, e)
    _ = chi2fn([34, 2, 60, 6])
    rr = json.load(open(f'{OUT}/data/bench_EQCet.json'))
    chi = np.full((len(KT_GRID), len(B_GRID)), np.inf)
    lo = np.array([10.0, 0.0])
    hi = np.array([89.0, 9.0])
    prev = np.array([rr['theta'], rr['logLambda']])
    for i, j in snake(len(KT_GRID), len(B_GRID)):
        kT, B = KT_GRID[i], B_GRID[j]

        def obj(s):
            s = np.clip(s, lo, hi)
            return chi2fn([B, kT, s[0], s[1]])
        res = minimize(obj, prev, method='Nelder-Mead',
                       options=dict(maxfev=160, fatol=0.05, xatol=2e-3))
        chi[i, j] = res.fun
        prev = np.clip(res.x, lo, hi)
        if j == len(B_GRID) - 1:
            print(f'  EQCet kT={kT:5.2f} done (row {i+1}/{len(KT_GRID)})',
                  flush=True)
    np.savez(f'{OUT}/data/EQCet_BT_map.npz', B=B_GRID, kT=KT_GRID, chi2=chi,
             best=np.array([rr['B'], rr['kT'], rr['theta'],
                            rr['logLambda']]),
             chi2_red=rr['chi2_red'], best_chi2=rr['chi2'])
    print('EQCet map saved', flush=True)


def map_bstri():
    """BS Tri (B, kT) map with theta constrained to its known eclipse
    geometry (80-89 deg), as in benchmark_fit.py."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_fit import load, make_chi2, BENCH
    w_m, fl, e = load(BENCH['BSTri'])
    chi2fn = make_chi2(w_m, fl, e)
    _ = chi2fn([22, 5, 87, 8])
    rr = json.load(open(f'{OUT}/data/bench_BSTri.json'))
    chi = np.full((len(KT_GRID), len(B_GRID)), np.inf)
    lo = np.array([80.0, 0.0])      # theta restricted to eclipse value
    hi = np.array([89.0, 9.0])
    prev = np.array([min(max(rr['theta'], 80.0), 89.0), rr['logLambda']])
    for i, j in snake(len(KT_GRID), len(B_GRID)):
        kT, B = KT_GRID[i], B_GRID[j]

        def obj(s):
            s = np.clip(s, lo, hi)
            return chi2fn([B, kT, s[0], s[1]])
        res = minimize(obj, prev, method='Nelder-Mead',
                       options=dict(maxfev=160, fatol=0.05, xatol=2e-3))
        chi[i, j] = res.fun
        prev = np.clip(res.x, lo, hi)
        if j == len(B_GRID) - 1:
            print(f'  BSTri kT={kT:5.2f} done (row {i+1}/{len(KT_GRID)})',
                  flush=True)
    np.savez(f'{OUT}/data/BSTri_BT_map.npz', B=B_GRID, kT=KT_GRID, chi2=chi,
             best=np.array([rr['B'], rr['kT'], rr['theta'],
                            rr['logLambda']]),
             chi2_red=rr['chi2_red'], best_chi2=rr['chi2'])
    print('BSTri map saved', flush=True)


if __name__ == '__main__':
    for t in sys.argv[1:]:
        if t == 'EQCet':
            map_eqcet()
        elif t == 'BSTri':
            map_bstri()
        else:
            map_source(t)
