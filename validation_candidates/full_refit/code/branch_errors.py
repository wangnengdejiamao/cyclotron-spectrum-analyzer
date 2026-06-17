#!/usr/bin/env python3
"""branch_errors.py — realistic 1-sigma field error for every branch.

For each distinct acceptable branch in {src}_full_refit.json we run a
FINE, warm-started full-refit profile in a +-window MG box around the
branch minimum (continuum and the other cyclotron parameters re-optimised
at every fixed B, started from the polished branch solution so the local
curve is smooth, unlike the coarse 2-MG global grid).

The quoted field error combines, in quadrature,
  (i)  the FORMAL statistical precision = the chi^2_red-rescaled
       Delta chi^2 = 1 half-width of that local curve, and
  (ii) a SYSTEMATIC floor = f_sys * B, where f_sys is the RMS fractional
       deviation of the benchmark recoveries (EQ Cet, BS Tri, MQ Dra,
       PZ Vir) from their published fields.
The formal width alone is sub-MG (~0.1-1 MG), which is unrealistically
precise for a cyclotron field set by hump positions in a single-epoch
survey spectrum; the benchmark floor (~few per cent) is the dominant and
empirically-calibrated term, giving fields good to ~1-3 MG, in line with
published polar field measurements.

All run products are saved so figures can be regenerated WITHOUT re-running
the (slow) profiler:
  * B_refined / B_err_formal / B_err (total) / fine_B / fine_chi2 written
    back into each solution of {src}_full_refit.json
  * {src}_branch_fine.npz          (per-branch fine B grid + chi2)
  * branch_error_summary.json      (f_sys, benchmark table, all errors)

Usage:  python3 branch_errors.py [SRC ...]   (default J0022 J0749 J0035)
Runtime ~3-7 min per source.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize

ROOT = os.environ.get("CYC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "validation_candidates", "branch_scan"))
import joint_pipeline as jp
from branch_scan import make_p, FREE
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SCIENCE = ["J0022", "J0749", "J0035"]
WINDOW = 3.0      # +- MG scanned around each branch
STEP = 0.20       # fine grid step [MG]

# Benchmark recoveries (B_fit, B_lit) used to calibrate the systematic
# floor.  EQ Cet / BS Tri are the main-text benchmarks; MQ Dra / PZ Vir
# are the blind SDSS validations (Appendix).  J1344 is excluded: it lands
# on a lower harmonic branch, i.e. a branch error, not a precision error.
BENCH = [("EQ Cet", 34.8, 34.0), ("BS Tri", 22.4, 22.7),
         ("MQ Dra", 61.6, 59.0), ("PZ Vir", 63.3, 63.0)]


def f_sys_floor():
    frac = np.array([(bf - bl) / bl for _, bf, bl in BENCH])
    return float(np.sqrt(np.mean(frac ** 2)))      # RMS fractional dev.


def build_model(src, js):
    meta = jp.SOURCES[src]
    w, f, e = jp.load_spectrum(meta)
    keep = jp.mask_lines(w)
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep],
                                 wmin=3950.0, wmax=9300.0)
    model = jp.JointModel(jp.KoesterGrid(jp.KOESTER_DIR), wb, fb, eb,
                          tuple(js["s_wd_bounds"]))
    return model, len(wb)


def local_profile(model, p0, Bc):
    """warm-started chi2(B) on a fine grid in [Bc-WINDOW, Bc+WINDOW]."""
    lo = np.array([jp.NONLIN_BOUNDS[i][0] for i in FREE])
    hi = np.array([jp.NONLIN_BOUNDS[i][1] for i in FREE])
    Bgrid = np.arange(Bc - WINDOW, Bc + WINDOW + 1e-9, STEP)
    chis = np.full(len(Bgrid), np.inf)

    def run(order):
        prev = np.clip(np.array(p0)[FREE], lo, hi)
        for i in order:
            B = Bgrid[i]
            obj = lambda x, B=B: model.chi2(make_p(np.clip(x, lo, hi), B))
            r = minimize(obj, prev, method="Nelder-Mead",
                         options=dict(maxfev=600, fatol=0.02, xatol=1e-3))
            f = float(r.fun)
            if f < chis[i]:
                chis[i] = f
            prev = np.clip(r.x, lo, hi)
    c = int(np.argmin(np.abs(Bgrid - Bc)))    # sweep outwards both ways
    run(range(c, len(Bgrid)))
    run(range(c, -1, -1))
    return Bgrid, chis


def half_width(Bgrid, dchi):
    """Delta chi^2 = 1 half-width either side of the minimum -> sigma."""
    imin = int(np.argmin(dchi))
    Bmin = Bgrid[imin]

    def cross(side):
        idx = range(imin, len(Bgrid)) if side > 0 else range(imin, -1, -1)
        prevB, prevd = Bmin, dchi[imin]
        for i in idx:
            if dchi[i] >= 1.0:
                if dchi[i] == prevd:
                    return abs(Bgrid[i] - Bmin)
                frac = (1.0 - prevd) / (dchi[i] - prevd)
                return abs(prevB + frac * (Bgrid[i] - prevB) - Bmin)
            prevB, prevd = Bgrid[i], dchi[i]
        return None
    sides = [x for x in (cross(+1), cross(-1)) if x is not None]
    return Bmin, (float(np.mean(sides)) if sides else None)


def run_src(src, fsys, summary, npz_store):
    t0 = time.time()
    js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
    model, nb = build_model(src, js)
    dof = max(nb - 10, 1)
    cmin = min(s["chi2"] for s in js["solutions"] if "chi2" in s)
    sc = max(cmin / dof, 1.0)
    _ = model.cyc_shape(*np.array(js["adopted_params"])[[3, 4, 5, 6]])

    branches = sorted([s for s in js["solutions"] if "p" in s],
                      key=lambda s: s["dchi2_rescaled"])
    uniq = []
    for s in branches:
        if all(abs(s["p"][3] - u["p"][3]) > 2 for u in uniq):
            uniq.append(s)
    print(f"=== {src}  chi2_red={sc:.2f}  f_sys={fsys*100:.1f}% ===",
          flush=True)
    for k, s in enumerate(uniq[:3]):
        Bc = s["p"][3]
        Bgrid, chis = local_profile(model, s["p"], Bc)
        dchi = (chis - chis.min()) / sc
        Bmin, sig = half_width(Bgrid, dchi)
        sig_f = sig if sig is not None else STEP        # unresolved -> 1 step
        sig_sys = fsys * Bmin
        sig_tot = float(np.hypot(sig_f, sig_sys))
        s["B_refined"] = round(float(Bmin), 2)
        s["B_err_formal"] = round(float(sig_f), 3)
        s["B_err_sys"] = round(float(sig_sys), 2)
        s["B_err"] = round(sig_tot, 2)               # quoted total error
        s["fine_B"] = [round(float(b), 3) for b in Bgrid]
        s["fine_chi2"] = [round(float(c), 3) for c in chis]
        # adopt the fine-grid minimum as the branch field ONLY if it stays in
        # the same basin (within 2 MG of the scan position): this corrects the
        # ~0.4 MG coarseness of the 2-MG scan without letting an aggressive
        # warm start walk across a small barrier and merge two genuinely
        # distinct local minima (e.g. the 51 and 54 MG sub-minima of J0035,
        # which the profiled curve resolves but a downhill refine would fold).
        if abs(Bmin - Bc) <= 2.0:
            s["p"][3] = round(float(Bmin), 2)
            if s.get("adopted"):
                js["adopted_params"][3] = round(float(Bmin), 2)
                js["adopted_B"] = round(float(Bmin), 2)
        npz_store[f"{src}_b{k}_B"] = Bgrid
        npz_store[f"{src}_b{k}_chi2"] = chis
        tag = "ADOPT" if s.get("adopted") else (
            "alt" if s.get("acceptable", True) else "EXCL")
        print(f"  {tag:5s} B={Bmin:6.2f}  formal +-{sig_f:.2f}  "
              f"sys +-{sig_sys:.2f}  ->  total +-{sig_tot:.2f} MG", flush=True)
        summary.append(dict(source=src, branch=tag, B=round(float(Bmin), 2),
                            B_err_formal=round(float(sig_f), 3),
                            B_err_sys=round(float(sig_sys), 2),
                            B_err=round(sig_tot, 2),
                            acceptable=bool(s.get("acceptable", True)),
                            adopted=bool(s.get("adopted", False))))
    json.dump(js, open(os.path.join(DATA, f"{src}_full_refit.json"), "w"),
              indent=1)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    fsys = f_sys_floor()
    print(f"systematic floor f_sys = {fsys*100:.2f}% (RMS of "
          + ", ".join(f"{n} {100*(bf-bl)/bl:+.1f}%" for n, bf, bl in BENCH)
          + ")", flush=True)
    summary, npz_store = [], {}
    for k in (sys.argv[1:] or SCIENCE):
        run_src(k, fsys, summary, npz_store)
    np.savez(os.path.join(DATA, "branch_fine_profiles.npz"), **npz_store)
    json.dump(dict(f_sys=fsys, benchmarks=BENCH, branches=summary),
              open(os.path.join(DATA, "branch_error_summary.json"), "w"),
              indent=1)
    print("done -> errors + fine profiles saved to full_refit/data/")
