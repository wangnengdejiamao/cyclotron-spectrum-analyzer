#!/usr/bin/env python3
"""full_refit_scan.py — reproducible full-refit branch scan.

At every trial field B the WHOLE model is re-optimised (T_WD, logg,
T_hot, kT, theta, logLambda; amplitudes re-solved), so competing harmonic
branches are found fairly (continuum-free), unlike the continuum-fixed
profile of the main text.  Writes full_refit/data/{src}_full_refit.json
and a comparison_summary.json.  This regenerates the inputs to
plot_branch_decomposition.py and the appendix tables/figures.

Usage:  python3 full_refit_scan.py [SRC ...]      (default: all 7)
Runtime ~15-20 min per source (cyclotron model is the bottleneck).
"""
import json, os, sys, time
import numpy as np

ROOT = os.environ.get("CYC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "validation_candidates", "branch_scan"))
import joint_pipeline as jp
from branch_scan import profile_B_full, make_p     # full-refit profiler
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

# science targets (from jp.SOURCES) and validation SDSS spectra
VAL = {  # key: (spectrum, wmax, published_B)
    "MQDra": ("J1553_sdss/J1553_sdss_donor_subtracted.txt", 8000.0, 59.2),
    "PZVir": ("PZVir_sdss/PZVir_sdss_spectrum.txt", 9000.0, 63.0),
    "J1344": ("J1344_sdss/J1344_sdss_donor_subtracted.txt", 9000.0, 56.0),
}
SCIENCE = ["J0005", "J0022", "J0749", "J0035"]
NAME = {"J0005": "DESI J0005+2941", "J0022": "DESI J0022+1340",
        "J0749": "DESI J0749+3654", "J0035": "LAMOST J0035+4333"}


def s_bounds(meta):
    if meta.get("dist_pc") is not None:
        d_lo = (meta["dist_pc"] - 2 * meta["dist_err"]) * jp.PC_CM
        d_hi = (meta["dist_pc"] + 2 * meta["dist_err"]) * jp.PC_CM
        return ((0.003 * jp.R_SUN_CM / d_hi) ** 2,
                (0.030 * jp.R_SUN_CM / d_lo) ** 2)
    return (1e-30, 1e-19)


def load(src):
    if src in VAL:
        d = np.loadtxt(os.path.join(ROOT, "data", VAL[src][0]))
        w, f, e = d[:, 0], d[:, 1] * 1e-17, d[:, 2] * 1e-17
        g = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        o = np.argsort(w[g])
        return w[g][o], f[g][o], e[g][o], (1e-30, 1e-19), VAL[src][1]
    meta = jp.SOURCES[src]
    w, f, e = jp.load_spectrum(meta)
    return w, f, e, s_bounds(meta), 9300.0


def scan(src):
    t0 = time.time()
    w, f, e, swb, wmax = load(src)
    keep = jp.mask_lines(w)
    wlo = 4000.0 if src in VAL else 3950.0
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep],
                                 wmin=wlo, wmax=wmax)
    model = jp.JointModel(jp.KoesterGrid(jp.KOESTER_DIR), wb, fb, eb, swb)
    # reference seed = old adopted joint solution (science) or pub field
    if src in VAL:
        p_ref = np.array([12000., 8.0, 25000., VAL[src][2], 4.0, 55.0, 6.0])
        oldB = None
    else:
        r = json.load(open(f"{ROOT}/data/{src}_joint_results.json"))
        p_ref = np.array(next(s for s in r["solutions"]
                              if s.get("adopted"))["p"])
        oldB = float(p_ref[3])
    _ = model.cyc_shape(p_ref[3], p_ref[4], p_ref[5], p_ref[6])
    dof = max(len(wb) - 10, 1)

    Bg = np.arange(14.0, 96.0 + 1e-9, 2.0)
    chis, sols_grid = profile_B_full(model, p_ref, Bg, seed=hash(src) % 1000)
    branches = jp.find_branches(Bg, chis, dchi2_max=30.0)

    polished = []
    for Bc, _ in branches:
        i = int(np.argmin(np.abs(Bg - Bc)))
        x0 = sols_grid[i] if sols_grid[i] is not None else p_ref[[0,1,2,4,5,6]]
        pb, ab, cb = jp.polish_branch(model, make_p(x0, Bc), Bc,
                                      seed=int(Bc) + 7)
        polished.append(dict(p=pb.tolist(), amps=ab.tolist(), chi2=float(cb),
                             contrast=float(jp.cyc_contrast(model, pb))))
    # add the earlier single-component reference branch for science targets
    if oldB is not None:
        ab, _m, cb = model.solve_amplitudes(p_ref)
        polished.append(dict(p=p_ref.tolist(), amps=ab.tolist(),
                             chi2=float(cb),
                             contrast=float(jp.cyc_contrast(model, p_ref)),
                             reference_seed=True))
    cmin = min(d["chi2"] for d in polished)
    sc = max(cmin / dof, 1.0)
    for d in polished:
        d["dchi2_rescaled"] = (d["chi2"] - cmin) / sc
        d["acceptable"] = bool(d["dchi2_rescaled"] < 9.0 and d["contrast"] > 0.15)
    polished.sort(key=lambda d: d["chi2"])
    polished[0]["adopted"] = True
    adopted = polished[0]

    out = dict(source=src, kind=("validation" if src in VAL else "science"),
               name=NAME.get(src, src),
               published_B=(VAL[src][2] if src in VAL else None),
               old_adopted_B=oldB, chi2_red=cmin / dof,
               B_grid=Bg.tolist(), profile_chi2=chis.tolist(),
               profile_dchi2_rescaled=((chis - cmin) / sc).tolist(),
               solutions=polished, adopted_B=float(adopted["p"][3]),
               adopted_params=adopted["p"], adopted_amps=adopted["amps"],
               n_bins=int(len(wb)), s_wd_bounds=list(swb),
               elapsed_s=time.time() - t0)
    json.dump(out, open(os.path.join(DATA, f"{src}_full_refit.json"), "w"),
              indent=1)
    acc = [d for d in polished if d["acceptable"]]
    print(f"{src}: adopted {adopted['p'][3]:.1f} MG (old {oldB}); "
          f"acceptable branches "
          + ", ".join(f"{d['p'][3]:.1f}" for d in acc)
          + f"  [{time.time()-t0:.0f}s]", flush=True)
    return out


if __name__ == "__main__":
    keys = sys.argv[1:] or SCIENCE + list(VAL)
    summ = []
    for k in keys:
        o = scan(k)
        summ.append(dict(source=k, kind=o["kind"], old_B=o["old_adopted_B"],
                         full_refit_B=o["adopted_B"],
                         published_B=o["published_B"], chi2_red=o["chi2_red"],
                         solutions=[dict(B=s["p"][3],
                                         dchi2_rescaled=s["dchi2_rescaled"],
                                         contrast=s["contrast"],
                                         acceptable=s["acceptable"])
                                    for s in o["solutions"]]))
    json.dump(summ, open(os.path.join(DATA, "comparison_summary.json"), "w"),
              indent=1)
    print("done -> full_refit/data/")
