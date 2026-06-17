#!/usr/bin/env python3
"""plot_full_refit_profiles.py — reproducible Fig.~4 (B_profiles_full_refit)
and the Appendix-B validation figure (validation_sdss_full_refit) from the
full-refit JSONs in full_refit/data/.

  B_profiles_full_refit.pdf : 2x2 profiled dchi2(B) for the 4 science
      targets, adopted branch (star) + distinct alternatives (open circ.).
  validation_sdss_full_refit.pdf : 3 rows (MQ Dra, PZ Vir, J1344); left
      = adopted-branch decomposition, right = full-refit dchi2(B).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

ROOT = os.environ.get("CYC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import pubstyle
import joint_pipeline as jp
pubstyle.apply()
C = pubstyle.COLORS
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUTD = os.path.join(ROOT, "figures")
SHORT = {"J0005": "DESI J0005+2941", "J0022": "DESI J0022+1340",
         "J0749": "DESI J0749+3654", "J0035": "LAMOST J0035+4333",
         "MQDra": "MQ Dra", "PZVir": "PZ Vir", "J1344": "SDSS J1344+2044"}
VAL_SPEC = {"MQDra": ("J1553_sdss/J1553_sdss_donor_subtracted.txt", 8000.0, 59.2),
            "PZVir": ("PZVir_sdss/PZVir_sdss_spectrum.txt", 9000.0, 63.0),
            "J1344": ("J1344_sdss/J1344_sdss_donor_subtracted.txt", 9000.0, 56.0)}
def prof(js):
    B = np.array(js["B_grid"]); d = np.array(js["profile_dchi2_rescaled"])
    if "B_grid_fine" in js:
        B = np.append(B, js["B_grid_fine"])
        d = np.append(d, js["profile_dchi2_fine_rescaled"])
    o = np.argsort(B)
    return B[o], np.maximum(d[o], 0.25)


def red_label_kwargs():
    return dict(fontsize=6.4, color="crimson", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=0.7))


def draw_profile(ax, js, lab, annotate=True):
    B, d = prof(js)
    # collect the distinct branches first so the envelope can be forced
    # through each refined minimum (guarantees the markers sit ON the curve)
    sols = sorted([s for s in js["solutions"] if "p" in s],
                  key=lambda s: s["dchi2_rescaled"])
    uniq = []
    for s in sols:
        if all(abs(s["p"][3] - u["p"][3]) > 2 for u in uniq):
            uniq.append(s)
    uniq = uniq[:3]
    Be = np.append(B, [s["p"][3] for s in uniq])
    de = np.append(d, [max(s["dchi2_rescaled"], 0.25) for s in uniq])
    o = np.argsort(Be)
    ax.plot(Be[o], np.maximum(de[o], 0.25), "-", color="navy", lw=1.2,
            zorder=2)
    for thr in (1, 4, 9):
        ax.axhline(thr, ls=":", lw=0.7, color="0.6")
    for s in uniq:
        b = s["p"][3]; kt = s["p"][4]; dc = max(s["dchi2_rescaled"], 0.25)
        if not (s.get("adopted") or s.get("acceptable", True)):
            continue          # don't mark/label excluded branches (e.g. 45 MG)
        ax.axvline(b, color="0.55", ls="--", lw=0.6, zorder=1)
        if s.get("adopted"):
            ax.plot(b, dc, "*", ms=14, mfc="gold", mec="k", mew=0.6, zorder=6,
                    clip_on=False)
            if annotate:
                ax.annotate(rf"$B$={b:.0f}, $kT$={kt:.1f}",
                            xy=(b, dc), xytext=(0.03, 0.05),
                            textcoords="axes fraction", fontsize=6.4,
                            color="k", va="bottom")
        elif s.get("acceptable", True):
            ax.plot(b, dc, "o", ms=7.5, mfc="none", mec="crimson", mew=1.3,
                    zorder=6, clip_on=False)
            if annotate:
                near_edge = b > min(B) + 0.8 * (max(B) - min(B))
                ax.annotate(rf"$B$={b:.0f}, $kT$={kt:.1f}",
                            xy=(b, dc),
                            xytext=((-34 if near_edge else 12), 14),
                            textcoords="offset points",
                            ha="right" if near_edge else "left",
                            **red_label_kwargs())
    ax.set_yscale("log"); ax.set_ylim(0.22, 4000)
    bmax = max(max(B), max(s["p"][3] for s in uniq))
    ax.set_xlim(min(B), bmax + 3)
    ax.text(0.03, 0.94, lab, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))


def branch_markers(js):
    """distinct, marked branches (adopted + acceptable alternatives)."""
    sols = sorted([s for s in js["solutions"] if "p" in s],
                  key=lambda s: s["dchi2_rescaled"])
    uniq = []
    for s in sols:
        if all(abs(s["p"][3] - u["p"][3]) > 2 for u in uniq):
            uniq.append(s)
    return [s for s in uniq[:3]
            if s.get("adopted") or s.get("acceptable", True)]


def _label_block(ax, lab, marks, corner):
    """Source name + one line per branch, anchored in a bottom corner so the
    labels never sit on the curves.  Each line carries the ACTUAL marker glyph
    (gold star = adopted, crimson open circle = alternative), so the symbol in
    the legend is the same colour as the one on the curve."""
    left = "left" in corner
    xg = 0.05 if left else 0.515         # glyph x (axes fraction)
    xt = 0.085 if left else 0.55         # text x (anchored so long names fit)
    ytop = 0.30
    bb = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.4)
    ax.text(xt - 0.04, ytop, lab, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.3, zorder=9, bbox=bb)
    for i, s in enumerate(marks):
        b = s["p"][3]; kt = s["p"][4]; yy = ytop - 0.085 * (i + 1)
        if s.get("adopted"):
            ax.plot(xg, yy - 0.018, "*", ms=10, mfc="gold", mec="k", mew=0.5,
                    transform=ax.transAxes, clip_on=False, zorder=10)
        else:
            ax.plot(xg, yy - 0.018, "o", ms=5.5, mfc="none", mec="crimson",
                    mew=1.2, transform=ax.transAxes, clip_on=False, zorder=10)
        ax.text(xt, yy, rf"$B$={b:.0f}, $kT$={kt:.1f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7.2, color="k", zorder=9, bbox=bb)


def draw_family(ax, src, js, lab, ktnorm, corner="lower left"):
    """benchmark-style panel: faint fixed-kT chi2(B) family (context) over the
    full-refit kT-free envelope, with branch markers.  The envelope is the
    JSON profile spliced with each branch's fine warm-started profile, so it
    dips to the true minimum at every branch (the coarse 2-MG grid otherwise
    leaves the alternative markers, e.g. J0022 at 78 MG, on a slope)."""
    Be, de = prof(js)
    pchi = np.array(js["profile_chi2"])
    cmin = float(min(pchi.min(),
                     min(s["chi2"] for s in js["solutions"] if "chi2" in s)))
    dof = max(int(js["n_bins"]) - 10, 1)
    scale = max(cmin / dof, 1.0)
    addB, addd = [Be], [de]
    for s in js["solutions"]:
        if "fine_B" in s and "fine_chi2" in s:
            fb = np.asarray(s["fine_B"], float)
            fc = np.asarray(s["fine_chi2"], float)
            addB.append(fb)
            addd.append(np.maximum((fc - cmin) / scale, 0.25))
    Be = np.concatenate(addB); de = np.concatenate(addd)
    o = np.argsort(Be); Be, de = Be[o], de[o]
    z = np.load(os.path.join(DATA, f"{src}_kt_family.npz"))
    B, kT, chi = z["B"], z["kT"], z["chi2"]
    for j in range(len(kT)):
        d = np.maximum((chi[j] - cmin) / scale, 0.25)
        ax.plot(B, d, "-", lw=0.55, alpha=0.40,
                color=cm.viridis(ktnorm(kT[j])), zorder=2)
    ax.plot(Be, de, "-", color="navy", lw=1.7, zorder=4)    # kT-free envelope
    for thr in (1, 4, 9):
        ax.axhline(thr, ls=":", lw=0.7, color="0.55")
    marks = branch_markers(js)
    bmax = max(B.max(), Be.max())
    for s in marks:
        b = s["p"][3]
        dc = float(np.interp(b, Be, de))
        bmax = max(bmax, b)
        ax.axvline(b, color="0.55", ls="--", lw=0.6, zorder=1)
        if s.get("adopted"):
            ax.plot(b, dc, "*", ms=14, mfc="gold", mec="k", mew=0.6,
                    zorder=6, clip_on=False)
        else:
            ax.plot(b, dc, "o", ms=7.5, mfc="none", mec="crimson", mew=1.3,
                    zorder=6, clip_on=False)
    ax.set_yscale("log"); ax.set_ylim(0.22, 4000)
    ax.set_xlim(min(B), bmax + 3)
    _label_block(ax, lab, marks, corner)


def b_profiles():
    srcs = ["J0005", "J0022", "J0749", "J0035"]
    have_family = all(os.path.exists(os.path.join(DATA, f"{s}_kt_family.npz"))
                      for s in srcs)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True,
                             sharey=True)
    if have_family:
        kt0 = np.load(os.path.join(DATA, f"{srcs[0]}_kt_family.npz"))["kT"]
        norm = LogNorm(kt0.min(), kt0.max())
        ktnorm = lambda t: 0.92 * norm(t)
        # top row labels bottom-left, bottom row bottom-right (away from dips)
        corners = {"J0005": "lower left", "J0022": "lower left",
                   "J0749": "lower right", "J0035": "lower right"}
        for ax, src in zip(axes.ravel(), srcs):
            js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
            draw_family(ax, src, js, SHORT[src], ktnorm, corners[src])
        sm = ScalarMappable(norm=norm, cmap=cm.viridis)
        cbar = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02,
                            location="right")
        cbar.set_label(r"fixed $kT$ [keV]", fontsize=8)
        handles = [Line2D([0], [0], color="navy", lw=1.7,
                          label=r"$kT$-free envelope (full refit)")]
        fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
                   bbox_to_anchor=(0.5, 0.985), fontsize=7.5)
    else:
        for ax, src in zip(axes.ravel(), srcs):
            js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
            draw_profile(ax, js, SHORT[src])
        fig.subplots_adjust(wspace=0.06, hspace=0.08)
    for ax in axes[1]:
        ax.set_xlabel(r"$B$ [MG]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"rescaled $\Delta\tilde\chi^2$")
    fig.savefig(os.path.join(OUTD, "B_profiles_full_refit.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("B_profiles_full_refit.pdf saved",
          "(fixed-kT family)" if have_family else "(single profile)")


def validation():
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0]))
    for row, src in enumerate(["MQDra", "PZVir", "J1344"]):
        js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
        infile, wmax, litB = VAL_SPEC[src]
        d = np.loadtxt(os.path.join(ROOT, "data", infile))
        w, f, e = d[:, 0], d[:, 1] * 1e-17, d[:, 2] * 1e-17
        g = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        o = np.argsort(w[g]); w, f, e = w[g][o], f[g][o], e[g][o]
        keep = jp.mask_lines(w)
        wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep], bw=25.0,
                                     wmin=4000.0, wmax=wmax)
        model = jp.JointModel(jp.KoesterGrid(jp.KOESTER_DIR), wb, fb, eb,
                              tuple(js["s_wd_bounds"]))
        p = np.array(js["adopted_params"])
        _ = model.cyc_shape(p[3], p[4], p[5], p[6])
        amps, mdl, _c = model.solve_amplitudes(p)
        wd, sp, cy = model.components(p)
        axL, axR = axes[row]
        s = 1e-17
        rsel = (w >= wb.min() - 100) & (w <= wb.max() + 100)
        axL.plot(w[rsel], f[rsel] / s, color=C["data"], lw=0.4, alpha=0.8,
                 label="observed (raw)")
        axL.plot(wb, mdl / s, color=C["model"], lw=1.5, label=f"fit ({p[3]:.1f} MG)")
        axL.plot(wb, amps[0] * wd / s, "--", color=C["wd"], lw=0.9)
        axL.plot(wb, amps[1] * sp / s, "--", color=C["spot"], lw=0.9)
        axL.plot(wb, amps[2] * cy / s, "-", color=C["cyc"], lw=1.1)
        axL.set_xlim(wb.min(), wb.max())
        # do not draw into negative flux: clip the view at zero
        rtop = max((mdl / s).max(), np.percentile(f[rsel] / s, 99.5))
        axL.set_ylim(0.0, 1.12 * rtop)
        axL.set_ylabel(r"$F_\lambda$ [$10^{-17}$]")
        axL.legend(fontsize=6.5, loc="upper right")
        axL.set_title(SHORT[src], loc="left", fontsize=9)
        draw_profile(axR, js, SHORT[src], annotate=False)
        axR.axvline(litB, color="k", ls="-.", lw=0.9)
        axR.text(0.97, 0.94, f"lit. {litB:.0f} MG", transform=axR.transAxes,
                 fontsize=7, ha="right", va="top", color="0.3")
        axR.set_ylabel(r"rescaled $\Delta\tilde\chi^2$")
    axes[2, 0].set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    axes[2, 1].set_xlabel(r"$B$ [MG]")
    fig.subplots_adjust(hspace=0.34, wspace=0.27, left=0.10, right=0.98,
                        top=0.96, bottom=0.08)
    fig.savefig(os.path.join(OUTD, "validation_sdss_full_refit.pdf"))
    plt.close(fig)
    print("validation_sdss_full_refit.pdf saved")


if __name__ == "__main__":
    b_profiles()
    validation()
