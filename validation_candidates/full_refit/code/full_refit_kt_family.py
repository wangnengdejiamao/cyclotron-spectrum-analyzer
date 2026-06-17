#!/usr/bin/env python3
"""full_refit_kt_family.py — fixed-kT chi^2(B) curve families (full-refit).

For Fig.~4 in the benchmark-figure style: at every (kT, B) cell we FIX the
electron temperature kT AND the field B and re-optimise everything else
(WD T/logg, hot-spot T, viewing angle, log Lambda; amplitudes re-solved),
warm-started across B so each fixed-kT curve is smooth.  This is the
full-refit analogue of the continuum-fixed {src}_BT_map.npz grids used for
EQ Cet / BS Tri, so the science panels can be drawn with the same fixed-kT
coloured curves + kT-free envelope, while staying consistent with the
full-refit (A) branch framing (e.g. J0749 bottoms at 29/96, not 45).

The kT-free envelope itself is NOT taken from min over this discrete kT
grid; the figure overlays the exact 1-D full-refit profile from
{src}_full_refit.json.  Here we only need the fixed-kT family.

Saves (so the figure can be redrawn without re-running):
  {src}_kt_family.npz   B grid, kT grid, chi2[kT, B], chi2_red
Runtime ~15-30 min per source.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize

ROOT = os.environ.get("CYC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "validation_candidates", "branch_scan"))
import joint_pipeline as jp
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SCIENCE = ["J0005", "J0022", "J0749", "J0035"]
# free indices at fixed (B, kT): T_wd, logg, T_spot, theta, logLambda
FREE5 = [0, 1, 2, 5, 6]
# B starts at 20 MG: all acceptable branches are >=29 MG and the very
# low-B cells are expensive (many cyclotron harmonics in the optical band)
BGRID = np.arange(20.0, 98.0 + 1e-9, 3.0)
KTGRID = np.geomspace(0.7, 28.0, 6)


def make_p(x5, B, kT):
    p = np.empty(7)
    p[0], p[1], p[2] = x5[0], x5[1], x5[2]
    p[3] = B
    p[4] = kT
    p[5], p[6] = x5[3], x5[4]
    return p


def build_model(src, js):
    meta = jp.SOURCES[src]
    w, f, e = jp.load_spectrum(meta)
    keep = jp.mask_lines(w)
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep],
                                 wmin=3950.0, wmax=9300.0)
    return jp.JointModel(jp.KoesterGrid(jp.KOESTER_DIR), wb, fb, eb,
                         tuple(js["s_wd_bounds"])), len(wb)


def run_src(src):
    t0 = time.time()
    js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
    model, nb = build_model(src, js)
    dof = max(nb - 10, 1)
    p0 = np.array(js["adopted_params"])
    _ = model.cyc_shape(p0[3], p0[4], p0[5], p0[6])
    lo = np.array([jp.NONLIN_BOUNDS[i][0] for i in FREE5])
    hi = np.array([jp.NONLIN_BOUNDS[i][1] for i in FREE5])
    seed5 = p0[FREE5].copy()

    chi = np.full((len(KTGRID), len(BGRID)), np.inf)
    for j, kT in enumerate(KTGRID):
        # warm start each kT row from the adopted continuum, sweep both ways
        def sweep(order, prev):
            for i in order:
                B = BGRID[i]
                obj = lambda x, B=B, kT=kT: model.chi2(
                    make_p(np.clip(x, lo, hi), B, kT))
                r = minimize(obj, prev, method="Nelder-Mead",
                             options=dict(maxfev=300, fatol=0.3, xatol=1e-2))
                if r.fun < chi[j, i]:
                    chi[j, i] = r.fun
                prev = np.clip(r.x, lo, hi)
            return prev
        c0 = int(np.argmin(np.abs(BGRID - p0[3])))
        sweep(range(c0, len(BGRID)), seed5.copy())
        sweep(range(c0, -1, -1), seed5.copy())
        print(f"  {src} kT={kT:5.2f}: chi2_min={chi[j].min():.1f}", flush=True)

    cmin = float(chi.min())
    np.savez(os.path.join(DATA, f"{src}_kt_family.npz"),
             B=BGRID, kT=KTGRID, chi2=chi, chi2_red=cmin / dof,
             best_chi2=cmin, dof=dof)
    print(f"{src}: saved kt_family ({len(KTGRID)}x{len(BGRID)}); "
          f"chi2_red={cmin/dof:.2f}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or SCIENCE):
        run_src(k)
    print("done -> {src}_kt_family.npz in full_refit/data/")
