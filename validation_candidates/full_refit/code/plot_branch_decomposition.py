#!/usr/bin/env python3
"""plot_branch_decomposition.py — reproducible Appendix-C figures.

For each branch-ambiguous target, draw one column per acceptable harmonic
branch in the SAME decomposition style as Fig.~3 (Koester WD + smooth
hot blackbody + cyclotron, top; data-continuum vs cyclotron, bottom),
using the full-refit branch solutions in
full_refit/data/{src}_full_refit.json.  No refitting: the branch
parameters are read from the JSON and the components rebuilt.

Outputs (ApJ style via pubstyle):
  full_refit/figures/{src}_branch_decomposition_full_refit.pdf
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("CYC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import pubstyle
import joint_pipeline as jp
pubstyle.apply()
C = pubstyle.COLORS
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGD = os.path.join(HERE, "..", "figures")
os.makedirs(FIGD, exist_ok=True)
DONOR = "#7b3f00"

# validation SDSS spectra (science targets come from jp.SOURCES)
VAL_SPEC = {
    "MQDra": ("J1553_sdss/J1553_sdss_donor_subtracted.txt", 8000.0),
    "PZVir": ("PZVir_sdss/PZVir_sdss_spectrum.txt", 9000.0),
    "J1344": ("J1344_sdss/J1344_sdss_donor_subtracted.txt", 9000.0),
}
SHORT = {"J0749": "DESI J0749+3654", "J0022": "DESI J0022+1340",
         "J0035": "LAMOST J0035+4333", "J0005": "DESI J0005+2941",
         "MQDra": "MQ Dra", "PZVir": "PZ Vir", "J1344": "SDSS J1344+2044"}


def load_spectrum(src, meta_kind, wmax=9300.0):
    if src in VAL_SPEC:
        d = np.loadtxt(os.path.join(ROOT, "data", VAL_SPEC[src][0]))
        w, f, e = d[:, 0], d[:, 1] * 1e-17, d[:, 2] * 1e-17
        g = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        o = np.argsort(w[g])
        return w[g][o], f[g][o], e[g][o]
    return jp.load_spectrum(jp.SOURCES[src])


def build_model(src, wb, fb, eb, s_wd_bounds):
    return jp.JointModel(jp.KoesterGrid(jp.KOESTER_DIR), wb, fb, eb,
                         tuple(s_wd_bounds))


def prep(src):
    """Load spectrum, build the model, and return the ordered, de-duplicated
    list of acceptable branches for one source."""
    js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
    sols = [s for s in js["solutions"] if "p" in s]
    # order: adopted first, then by chi2
    sols.sort(key=lambda s: (not s.get("adopted", False),
                             s.get("dchi2_rescaled", 0)))
    # drop near-duplicate branches (within 2 MG of one already kept)
    uniq = []
    for s in sols:
        if all(abs(s["p"][3] - u["p"][3]) > 2.0 for u in uniq):
            uniq.append(s)
    sols = uniq
    wmax = VAL_SPEC[src][1] if src in VAL_SPEC else 9300.0
    w, f, e = load_spectrum(src, js.get("kind"))
    keep = jp.mask_lines(w)
    wlo = 4000.0 if src in VAL_SPEC else 3950.0
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep],
                                 wmin=wlo, wmax=wmax)
    model = build_model(src, wb, fb, eb, js["s_wd_bounds"])
    _ = model.cyc_shape(sols[0]["p"][3], sols[0]["p"][4],
                        sols[0]["p"][5], sols[0]["p"][6])
    sel = (w >= wb.min() - 100) & (w <= wb.max() + 100)
    return dict(sols=sols, w=w, f=f, wb=wb, fb=fb, eb=eb, model=model, sel=sel)


def branch_tags(n):
    """Column labels matching Table 3: global min. / alt. branch(es)."""
    if n == 1:
        return ["global min."]
    return ["global min."] + [("alt. branch" if n == 2 else f"alt. branch {i}")
                              for i in range(1, n)]


def draw_branch(ax, axr, D, solu, tag, col, src, title_size=7.5):
    """Draw one branch (top spectrum + bottom residual) into a given pair of
    axes, in the same style as Fig. 3."""
    if solu.get("reference_seed") and not solu.get("adopted"):
        tag = "earlier branch"
    p = np.array(solu["p"])
    model, wb, fb, eb = D["model"], D["wb"], D["fb"], D["eb"]
    amps, mdl, _c = model.solve_amplitudes(p)
    wd, sp, cy = model.components(p)
    s = 1e-17
    ax.plot(D["w"][D["sel"]], D["f"][D["sel"]] / s, color=C["data"], lw=0.35,
            alpha=0.8, label="observed")
    ax.plot(wb, mdl / s, color=C["model"], lw=1.5, label="total")
    # WD/spot temperatures are nuisance parameters we do not quote
    ax.plot(wb, amps[0] * wd / s, "--", color=C["wd"], lw=1.0, label="WD")
    ax.plot(wb, amps[1] * sp / s, "--", color=C["spot"], lw=1.0,
            label="hot cont.")
    ax.plot(wb, amps[2] * cy / s, "-", color=C["cyc"], lw=1.2, label="cyclotron")
    ax.set_xlim(wb.min() - 100, wb.max() + 100)
    ax.set_ylim(bottom=min(0, np.percentile(fb / s, 1)))
    ax.tick_params(labelbottom=False)
    flag = "" if solu.get("acceptable", True) else " (excl.)"
    ax.set_title(f"{tag}{flag}\n$B$={p[3]:.1f} MG, $kT$={p[4]:.1f} keV, "
                 f"$\\theta$={p[5]:.0f}$^\\circ$, "
                 f"$\\log\\Lambda$={p[6]:.1f}", fontsize=title_size)
    if col == 0:
        ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$]")
        ax.legend(fontsize=6.2, loc="upper right")
        ax.text(0.03, 0.05, SHORT[src], transform=ax.transAxes,
                fontsize=8.5, va="bottom")
    axr.errorbar(wb, (fb - amps[0] * wd - amps[1] * sp) / s, yerr=eb / s,
                 fmt="o", ms=2.0, lw=0.5, color=C["resid"], ecolor="0.75")
    axr.plot(wb, amps[2] * cy / s, color=C["cyc"], lw=1.3)
    axr.axhline(0, color="k", ls=":", lw=0.6)
    axr.set_xlim(wb.min() - 100, wb.max() + 100)
    axr.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    if col == 0:
        axr.set_ylabel(r"$F_\lambda - F_{\rm cont}$")


def figure(src, adopted_only=False):
    D = prep(src)
    if adopted_only:
        # J0035: the 54 MG sub-minimum is the same harmonic at higher kT
        # (a kT-B degeneracy), not a competing branch, so show only the
        # adopted fit (Appendix-C caption says so; cf. Fig. 4 / Table 3).
        D["sols"] = D["sols"][:1]
    sols = D["sols"]
    n = len(sols)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 4.8), squeeze=False,
                             gridspec_kw=dict(height_ratios=[2.1, 1.2],
                                              hspace=0.05))
    for col, (solu, tag) in enumerate(zip(sols, branch_tags(n))):
        draw_branch(axes[0][col], axes[1][col], D, solu, tag, col, src)
    fig.tight_layout()
    out = os.path.join(FIGD, f"{src}_branch_decomposition_full_refit.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  ({n} branches: "
          + ", ".join(f"{s['p'][3]:.1f}" for s in sols) + " MG)")


def combined(srcs=("J0749", "J0022"),
             out_name="branch_decomposition_combined_full_refit.pdf"):
    """One Appendix-C figure: the branch-ambiguous targets stacked, with
    equal panel widths (J0749 = 3 branches on top, J0022 = 2 branches
    centred below).  Saves a journal page versus two separate floats."""
    Ds = [prep(s) for s in srcs]
    ncols = [len(D["sols"]) for D in Ds]
    base = 6                                   # 6-col base: panels span 2 cols
    spans = {2: [(1, 3), (3, 5)],              # 2 branches -> centred
             3: [(0, 2), (2, 4), (4, 6)]}      # 3 branches -> full width
    fig = plt.figure(figsize=(10.2, 9.7))
    gs = fig.add_gridspec(5, base,
                          height_ratios=[2.0, 1.15, 0.42, 2.0, 1.15],
                          hspace=0.12, wspace=0.40,
                          left=0.07, right=0.985, top=0.965, bottom=0.055)
    row_pairs = [(0, 1), (3, 4)]               # (spectrum row, residual row)
    for (rt, rb), D, n, src in zip(row_pairs, Ds, ncols, srcs):
        for col, (a, b) in enumerate(spans[n]):
            ax = fig.add_subplot(gs[rt, a:b])
            axr = fig.add_subplot(gs[rb, a:b], sharex=ax)
            tag = branch_tags(n)[col]
            draw_branch(ax, axr, D, D["sols"][col], tag, col, src)
    out = os.path.join(FIGD, out_name)
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  ("
          + "; ".join(f"{s}:{n}" for s, n in zip(srcs, ncols)) + " branches)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["combined"] or not args:
        combined()
        if not args:                           # default: also (re)make J0035
            figure("J0035", adopted_only=True)
    else:
        for k in args:
            figure(k)
