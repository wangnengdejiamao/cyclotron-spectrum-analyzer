#!/usr/bin/env python3
"""regen_joint_fullrefit.py — regenerate the main joint-fit panels
(Figure 3, figs/{src}_joint_fit.pdf) from the full-refit adopted
solutions, so they match Table 3 and the (A) branch framing (e.g.
J0749 at its 29.3 MG global-minimum branch, J0035 with kT=2.6 etc.),
not the old continuum-fixed fit that replot_all.py used.

Same two-panel ApJ style as code/replot_all.joint_fig.  For the
branch-ambiguous J0749 the panel shows the global-minimum branch and the
label says so (alternatives are in Appendix C).
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
FIGD = os.path.join(ROOT, "figures")

SHORT = {"J0005": "DESI J0005+2941", "J0022": "DESI J0022+1340",
         "J0749": "DESI J0749+3654", "J0035": "LAMOST J0035+4333"}
AMBIG = {"J0749"}          # label as global-minimum branch


def panel(src, koester):
    js = json.load(open(os.path.join(DATA, f"{src}_full_refit.json")))
    p = np.array(js["adopted_params"])
    meta = jp.SOURCES[src]
    w, f, e = jp.load_spectrum(meta)
    keep = jp.mask_lines(w)
    wb, fb, eb = jp.bin_spectrum(w[keep], f[keep], e[keep],
                                 wmin=3950.0, wmax=9300.0)
    model = jp.JointModel(koester, wb, fb, eb, tuple(js["s_wd_bounds"]))
    _ = model.cyc_shape(p[3], p[4], p[5], p[6])
    amps, tot, _c = model.solve_amplitudes(p)
    wd, sp, cy = model.components(p)
    wdc, spc, cyc = amps[0] * wd, amps[1] * sp, amps[2] * cy

    # Each panel is placed at ~0.49\linewidth (2x2 in the manuscript), i.e.
    # scaled to ~half this canvas, so fonts/lines are set LARGE here to match
    # the on-page size of the other (full-width) figures.
    LBL, TICK, LEG = 17, 15, 14
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.1, 1.3],
                                              hspace=0.05))
    ax = axes[0]
    sel = (w >= wb.min() - 120) & (w <= wb.max() + 120)
    ax.plot(w[sel], f[sel], color=C["data"], lw=0.5, alpha=0.85,
            label="observed", zorder=1)
    ax.plot(wb, tot, color=C["model"], lw=2.0, label="total", zorder=5)
    # WD/spot temperatures are nuisance parameters we do not quote, and the
    # cyclotron B/kT/theta/logLambda are listed in the text block below
    # (keeping the legend short so the fonts can be large)
    ax.plot(wb, wdc, "--", color=C["wd"], lw=1.4, label="WD")
    ax.plot(wb, spc, "--", color=C["spot"], lw=1.4, label="hot spot")
    ax.plot(wb, cyc, "-", color=C["cyc"], lw=1.7, label="cyclotron")
    ymax = np.percentile(f[sel], 99.6)
    ax.set_ylim(0, 1.34 * ymax)
    ax.set_xlim(wb.min() - 120, wb.max() + 120)
    ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
                  fontsize=LBL)
    ax.tick_params(labelsize=TICK)
    ax.legend(loc="upper right", fontsize=LEG, ncol=2, handlelength=1.3,
              columnspacing=1.0, borderpad=0.3)
    ax.text(0.025, 0.96, SHORT[src], transform=ax.transAxes, va="top",
            fontsize=LBL, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    # adopted cyclotron parameters, placed below the legend (top right),
    # low enough that it clears the 3-row legend box
    ax.text(0.975, 0.60,
            rf"$B$={p[3]:.1f} MG, $kT$={p[4]:.1f} keV" + "\n"
            rf"$\theta$={p[5]:.0f}$^\circ$, $\log_{{10}}\Lambda$={p[6]:.1f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=TICK,
            zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

    ax = axes[1]
    resid = fb - wdc - spc
    ax.errorbar(wb, resid, yerr=eb, fmt="o", ms=2.6, lw=0.8,
                color=C["resid"], ecolor="0.7", label="data $-$ continuum")
    ax.plot(wb, cyc, color=C["cyc"], lw=1.7, label="cyclotron")
    ax.axhline(0, color="k", ls=":", lw=0.9)
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize=LBL)
    ax.set_ylabel(r"$F_\lambda - F_{\rm cont}$", fontsize=LBL)
    ax.tick_params(labelsize=TICK)
    # Focus the residual panel on the cyclotron signal; the blue end of the
    # low-field branches scatters well below zero (continuum slightly
    # over-absorbed there) and would otherwise stretch the axis. No legend:
    # the caption identifies the points and the line, and a box here would
    # overlap the data.
    rhi = max(float(np.nanpercentile(resid, 98)), float(cyc.max()))
    ax.set_ylim(-0.75 * rhi, 1.20 * rhi)
    fig.savefig(f"{FIGD}/{src}_joint_fit.pdf")
    plt.close(fig)
    print(f"  {src}_joint_fit.pdf  (B={p[3]:.1f}, kT={p[4]:.1f}, "
          f"th={p[5]:.0f}, logL={p[6]:.1f})")


if __name__ == "__main__":
    koester = jp.KoesterGrid(jp.KOESTER_DIR)
    for s in (sys.argv[1:] or ["J0005", "J0022", "J0749", "J0035"]):
        panel(s, koester)
    print("done -> figures/{src}_joint_fit.pdf")
