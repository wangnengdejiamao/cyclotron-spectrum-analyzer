#!/usr/bin/env python3
"""wd_prior_refit.py — joint fits with a physically constrained white dwarf.

Referee request (Sect. 4.1): the unconstrained fit can place the WD
photosphere at a (T_wd, log g, R_wd) combination that no real white dwarf
occupies, because log g and the solid angle s_wd = (R_wd/d)^2 are free and
independent.  Here the WD component is re-parametrised by its MASS:

    R_wd  = R(M_wd)      Nauenberg (1972) mass-radius relation
    log g = log10(G M_wd / R_wd^2)
    s_wd  = (R_wd / d)^2 , d within +-2 sigma of the Gaia distance

with M_wd in [0.6, 1.0] Msun (the CV range around the 0.8 Msun mean of
Pala et al. 2022) and T_wd in [8000, 20000] K (the CV WD temperature
range).  The hot spot keeps its 8000-50000 K blackbody bounds.  Everything
else -- the likelihood, the binning, the masking, the cyclotron model --
is identical to joint_pipeline.py, so the fields are directly comparable.

Two modes:
  --branches   constrained refit of every published branch (fast, minutes)
  --profile    constrained full-refit profile chi2(B) over 12-96 MG
               (slow, ~15 min/source; verifies the branch structure)

Outputs data/wd_prior/{src}_wdprior_branches.json  and  ..._profile.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, lsq_linear, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pipeline as jp

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FR = os.path.join(ROOT, 'validation_candidates', 'full_refit', 'data')
OUTD = os.path.join(ROOT, 'data', 'wd_prior')
os.makedirs(OUTD, exist_ok=True)

G_CGS = 6.674e-8
MSUN = 1.989e33

# published branches to re-fit (source -> list of B in MG)
BRANCHES = {'J0005': [56.77], 'J0022': [64.35, 77.70],
            'J0749': [29.26, 96.39], 'J0035': [51.13]}

# constrained non-linear bounds: [T_wd, M_wd, T_spot, B, kT, theta, logLam]
BOUNDS = [(8000.0, 20000.0),     # T_wd  : CV white-dwarf range
          (0.60, 1.00),          # M_wd  : CV white-dwarf mass range
          (8000.0, 50000.0),     # T_spot
          (10.0, 100.0),         # B [MG]
          (0.3, 30.0),           # kT [keV]
          (10.0, 89.0),          # theta [deg]
          (0.0, 9.0)]            # log10 Lambda
PN = ['T_wd', 'M_wd', 'T_spot', 'B_MG', 'kT_keV', 'theta_deg', 'log_Lambda']
FREE = [0, 1, 2, 4, 5, 6]        # re-optimised at fixed B


def r_of_m(m_msun):
    """Nauenberg (1972) zero-temperature WD mass-radius relation [cm]."""
    m_ch = 5.816 / 2.0 ** 2
    x = (np.asarray(m_msun, float) / m_ch) ** (2.0 / 3.0)
    return 7.83e8 * np.sqrt(np.clip(1.0 - x, 0.0, None)) / \
        (np.asarray(m_msun, float) / m_ch) ** (1.0 / 3.0)


def logg_of_m(m_msun):
    return np.log10(G_CGS * np.asarray(m_msun, float) * MSUN /
                    r_of_m(m_msun) ** 2)


class PriorModel(jp.JointModel):
    """JointModel with the WD solid angle tied to the fitted WD mass."""

    def __init__(self, koester, wave, flux, err, d_lo_cm, d_hi_cm,
                 free_swd=False):
        super().__init__(koester, wave, flux, err, (0.0, np.inf))
        self.d_lo, self.d_hi = d_lo_cm, d_hi_cm
        self.free_swd = free_swd      # LAMOST: relative flux, no scale

    def _translate(self, q):
        """(T_wd, M_wd, ...) -> joint_pipeline p vector + s_wd bounds."""
        m = float(np.clip(q[1], 0.2, 1.35))
        lg = float(np.clip(logg_of_m(m), self.k.loggs.min(),
                           self.k.loggs.max()))
        p = np.array([q[0], lg, q[2], q[3], q[4], q[5], q[6]], float)
        if self.free_swd:
            return p, (0.0, np.inf)
        R = float(r_of_m(m))
        return p, ((R / self.d_hi) ** 2, (R / self.d_lo) ** 2)

    def solve_amplitudes_q(self, q):
        p, (s_lo, s_hi) = self._translate(q)
        wd, sp, cy = self.components(p)
        wd, sp, cy = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                      for x in (wd, sp, cy))
        A = np.column_stack([wd, sp, cy]) / self.e[:, None]
        y = self.f / self.e
        norms = np.sqrt(np.sum(A ** 2, axis=0))
        norms[norms <= 0] = 1.0
        An = A / norms[None, :]
        lo = np.array([s_lo * norms[0], 0.0, 0.0])
        hi = np.array([(s_hi if np.isfinite(s_hi) else 1e30) * norms[0],
                       np.inf, np.inf])
        if hi[0] <= lo[0]:
            hi[0] = lo[0] * (1 + 1e-9)
        try:
            res = lsq_linear(An, y, bounds=(lo, hi), method='bvls',
                             max_iter=200)
            amps = res.x / norms
        except Exception:
            return p, np.zeros(3), np.zeros_like(self.f), 1e12
        model = wd * amps[0] + sp * amps[1] + cy * amps[2]
        c2 = float(np.sum(((self.f - model) / self.e) ** 2))
        return p, amps, model, (c2 if np.isfinite(c2) else 1e12)

    def chi2_q(self, q):
        if q[2] <= q[0]:                      # T_spot must exceed T_wd
            return 1e12
        return self.solve_amplitudes_q(q)[3]


def build(src):
    meta = jp.SOURCES[src]
    w, f, e = jp.load_spectrum(meta)
    keep = jp.mask_lines(w)
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep],
                                 wmin=3950.0, wmax=9300.0)
    if meta['dist_pc'] is not None:
        d_lo = (meta['dist_pc'] - 2 * meta['dist_err']) * jp.PC_CM
        d_hi = (meta['dist_pc'] + 2 * meta['dist_err']) * jp.PC_CM
        free_swd = False
    else:
        d_lo = d_hi = 1.0
        free_swd = True
    model = PriorModel(jp.KoesterGrid(jp.KOESTER_DIR), wb, fb, eb,
                       d_lo, d_hi, free_swd=free_swd)
    _ = model.cyc_shape(50.0, 3.0, 60.0, 5.0)
    return model, wb, meta


def polish_at_B(model, B, seed=3, x0=None, maxiter=60):
    """Constrained optimisation of the six free parameters at fixed B."""
    lo = np.array([BOUNDS[i][0] for i in FREE])
    hi = np.array([BOUNDS[i][1] for i in FREE])

    def obj(x):
        x = np.clip(x, lo, hi)
        q = np.array([x[0], x[1], x[2], B, x[3], x[4], x[5]])
        return model.chi2_q(q)

    best_x, best_f = None, np.inf
    if x0 is not None:
        r = minimize(obj, np.clip(x0, lo, hi), method='Nelder-Mead',
                     options=dict(maxfev=1200, fatol=0.02, xatol=1e-3))
        best_x, best_f = np.clip(r.x, lo, hi), r.fun
    rd = differential_evolution(obj, list(zip(lo, hi)), maxiter=maxiter,
                                popsize=12, tol=1e-8, seed=seed, polish=True,
                                mutation=(0.4, 1.2), recombination=0.85)
    if rd.fun < best_f:
        best_x, best_f = np.clip(rd.x, lo, hi), rd.fun
    r = minimize(obj, best_x, method='Nelder-Mead',
                 options=dict(maxfev=1500, fatol=0.01, xatol=1e-4))
    if r.fun < best_f:
        best_x, best_f = np.clip(r.x, lo, hi), r.fun
    q = np.array([best_x[0], best_x[1], best_x[2], B,
                  best_x[3], best_x[4], best_x[5]])
    return q, best_f


def refine_B(model, q0, half=4.0, seed=5):
    """Free B in a window around the branch, all else re-optimised."""
    lo = np.array([BOUNDS[i][0] for i in FREE] + [q0[3] - half])
    hi = np.array([BOUNDS[i][1] for i in FREE] + [q0[3] + half])

    def obj(x):
        x = np.clip(x, lo, hi)
        q = np.array([x[0], x[1], x[2], x[6], x[3], x[4], x[5]])
        return model.chi2_q(q)

    x0 = np.array([q0[0], q0[1], q0[2], q0[4], q0[5], q0[6], q0[3]])
    r = minimize(obj, x0, method='Nelder-Mead',
                 options=dict(maxfev=3000, fatol=0.01, xatol=1e-4))
    x = np.clip(r.x, lo, hi)
    return np.array([x[0], x[1], x[2], x[6], x[3], x[4], x[5]]), r.fun


def frac(model, q):
    """Component flux fractions (at 5500 A and band-integrated)."""
    p, amps, mdl, c2 = model.solve_amplitudes_q(q)
    wd, sp, cy = model.components(p)
    comp = [amps[0] * wd, amps[1] * sp, amps[2] * cy]
    tot = sum(comp)
    i55 = int(np.argmin(np.abs(model.w - 5500.0)))
    f55 = [float(c[i55] / tot[i55]) if tot[i55] > 0 else 0.0 for c in comp]
    ii = [float(np.trapz(c, model.w)) for c in comp]
    s = sum(ii) or 1.0
    return amps, [x / s for x in ii], f55, c2


def do_branches(srcs):
    out_all = {}
    for src in srcs:
        model, wb, meta = build(src)
        dof = max(len(wb) - 10, 1)
        js = json.load(open(os.path.join(FR, f'{src}_full_refit.json')))
        ref_chi2 = min([s['chi2'] for s in js['solutions'] if 'chi2' in s])
        rows = []
        for B0 in BRANCHES[src]:
            t0 = time.time()
            q, c2 = polish_at_B(model, B0, seed=int(B0) + 7)
            q, c2 = refine_B(model, q, half=4.0)
            amps, ifrac, f55, c2b = frac(model, q)
            m = float(q[1])
            rows.append(dict(
                B_start=B0, B=float(q[3]), chi2=float(c2b),
                chi2_red=float(c2b / dof), T_wd=float(q[0]), M_wd=m,
                logg=float(logg_of_m(m)), R_wd_cm=float(r_of_m(m)),
                R_wd_Rsun=float(r_of_m(m) / jp.R_SUN_CM),
                T_spot=float(q[2]), kT=float(q[4]), theta=float(q[5]),
                logLambda=float(q[6]), amps=[float(a) for a in amps],
                frac_int=dict(wd=ifrac[0], spot=ifrac[1], cyc=ifrac[2]),
                frac_5500=dict(wd=f55[0], spot=f55[1], cyc=f55[2]),
                elapsed_s=time.time() - t0))
            print(f'  {src} branch {B0:.2f} -> B={q[3]:.2f} MG  '
                  f'chi2={c2b:.1f} (unconstrained {ref_chi2:.1f})  '
                  f'T_wd={q[0]:.0f} M={m:.2f} T_spot={q[2]:.0f}  '
                  f'kT={q[4]:.2f} th={q[5]:.1f} logL={q[6]:.2f}  '
                  f'WD/spot/cyc={ifrac[0]:.2f}/{ifrac[1]:.2f}/{ifrac[2]:.2f}',
                  flush=True)
        out = dict(source=src, name=meta['name'], n_bins=len(wb), dof=dof,
                   unconstrained_chi2=ref_chi2,
                   unconstrained_chi2_red=ref_chi2 / dof,
                   unconstrained_params=js['adopted_params'],
                   unconstrained_amps=js['adopted_amps'],
                   bounds=dict(zip(PN, [list(b) for b in BOUNDS])),
                   relative_flux=bool(model.free_swd), branches=rows)
        with open(os.path.join(OUTD, f'{src}_wdprior_branches.json'), 'w') as fh:
            json.dump(out, fh, indent=2)
        out_all[src] = out
    return out_all


def do_profile(srcs, step=2.0):
    for src in srcs:
        model, wb, meta = build(src)
        dof = max(len(wb) - 10, 1)
        Bg = np.arange(12.0, 96.0 + 1e-9, step)
        lo = np.array([BOUNDS[i][0] for i in FREE])
        hi = np.array([BOUNDS[i][1] for i in FREE])
        chis = np.full(len(Bg), np.inf)
        sols = [None] * len(Bg)
        start = np.array([12000.0, 0.8, 25000.0, 4.0, 55.0, 5.0])
        t0 = time.time()
        for sweep, order in enumerate([range(len(Bg)),
                                       range(len(Bg) - 1, -1, -1)]):
            prev = start.copy()
            for i in order:
                B = float(Bg[i])

                def obj(x, B=B):
                    x = np.clip(x, lo, hi)
                    return model.chi2_q(np.array([x[0], x[1], x[2], B,
                                                  x[3], x[4], x[5]]))
                r = minimize(obj, prev, method='Nelder-Mead',
                             options=dict(maxfev=400, fatol=0.05, xatol=1e-3))
                bx, bf = np.clip(r.x, lo, hi), r.fun
                if sweep == 0 and i % 4 == 0:
                    rd = differential_evolution(
                        obj, list(zip(lo, hi)), maxiter=14, popsize=6,
                        tol=1e-7, seed=i + 11, polish=True,
                        mutation=(0.4, 1.2), recombination=0.85)
                    if rd.fun < bf:
                        bx, bf = np.clip(rd.x, lo, hi), rd.fun
                if bf < chis[i]:
                    chis[i], sols[i] = bf, bx
                prev = sols[i] if sols[i] is not None else bx
            print(f'  {src} sweep {sweep}: chi2_min={np.nanmin(chis):.1f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        cmin = float(np.nanmin(chis))
        scale = max(cmin / dof, 1.0)
        br = jp.find_branches(Bg, chis, dchi2_max=30.0)
        out = dict(source=src, name=meta['name'], B_grid=Bg.tolist(),
                   profile_chi2=chis.tolist(),
                   profile_dchi2_rescaled=((chis - cmin) / scale).tolist(),
                   branches_coarse=br, chi2_min=cmin, dof=dof,
                   chi2_red=cmin / dof, elapsed_s=time.time() - t0)
        with open(os.path.join(OUTD, f'{src}_wdprior_profile.json'), 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'  {src} constrained branches (B, dchi2): {br}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['branches', 'profile'],
                    default='branches')
    ap.add_argument('--sources', nargs='*',
                    default=['J0005', 'J0022', 'J0749', 'J0035'])
    ap.add_argument('--step', type=float, default=2.0)
    a = ap.parse_args()
    if a.mode == 'branches':
        do_branches(a.sources)
    else:
        do_profile(a.sources, step=a.step)
