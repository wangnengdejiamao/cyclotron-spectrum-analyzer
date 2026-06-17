#!/usr/bin/env python3
"""
plot_profile_curves.py — chi^2(B) profile curves at fixed electron
temperatures (one panel per source), in the style of Lin et al. (2025).

Each curve is the minimum chi^2 as a function of B for a fixed kT, with
the viewing angle, optical-depth parameter, and component amplitudes
re-optimized at every point (rows of the {src}_BT_map.npz grids).

Outputs:
  * B_profiles_kT.pdf   2x2 panels, one per target
  * bench_EQCet.pdf     left: unbinned residual spectrum + model;
                        right: fixed-kT chi^2(B) curves for EQ Cet
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cyclotron_m2 import cal_cy_spec

pubstyle.apply()
OUT = os.environ.get('CYC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEV_J = 1.602176634e-16
SHORT = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
         'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}

def draw_curves(ax, npz, label, best_BkT, legend=True, envelope=None,
                note=True):
    """chi^2(B) at fixed grid temperatures bracketing the best-fit kT,
    plus (optionally) the full kT-free profile envelope. The star marks
    the adopted solution at Delta chi^2 = 0 (plotted at the display
    floor), where the envelope reaches its minimum."""
    z = np.load(npz)
    B, kT, chi = z['B'], z['kT'], z['chi2']
    scale = max(float(z['chi2_red']), 1.0)
    ref = min(chi.min(), float(z['best_chi2']))
    B0, kT0 = best_BkT
    i0 = int(np.argmin(np.abs(np.log(kT) - np.log(kT0))))
    rows = sorted({min(max(i0 + di, 0), len(kT) - 1)
                   for di in (-4, -2, 0, 2, 4)})
    cmap = cm.viridis
    for i in rows:
        d = (chi[i] - ref) / scale
        col = cmap(np.log(kT[i] / kT.min())
                   / np.log(kT.max() / kT.min()) * 0.9)
        ax.plot(B, np.maximum(d, 0.3), '-', color=col, lw=1.0,
                label=f'$kT$={kT[i]:.3g} keV')
    if envelope is not None:
        Be, de = envelope
        o = np.argsort(Be)
        ax.plot(Be[o], np.maximum(de[o], 0.3), '-', color='k', lw=1.6,
                label='envelope ($kT$ free)', zorder=5)
    for thr in (1, 4, 9):
        ax.axhline(thr, ls=':', lw=0.7, color='0.55')
    ax.axvline(B0, color='0.4', ls='--', lw=0.7, zorder=1)
    ax.plot(B0, 0.3, '*', ms=12, mfc='gold', mec='k', mew=0.6, zorder=6,
            clip_on=False)
    ax.set_yscale('log')
    ax.set_ylim(0.22, 4000)
    ax.set_xlim(B.min(), B.max())
    ax.text(0.03, 0.94, label, transform=ax.transAxes, va='top',
            fontsize=8.5,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.82,
                      pad=0.6))
    if legend:
        ax.legend(fontsize=6.0, loc='lower right', ncol=2,
                  handlelength=1.3, columnspacing=0.8)
    elif note:
        ax.text(0.03, 0.055,
                'coloured: fixed $kT$\nblack: $kT$ free',
                transform=ax.transAxes, fontsize=6.2, va='bottom',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.88, pad=0.5),
                zorder=10)


def envelope_from_json(src):
    r = json.load(open(f'{OUT}/data/{src}_joint_results.json'))
    scale = max(r['chi2_red'], 1.0)
    ref = r['best_chi2']
    adopted = next(s for s in r['solutions'] if s.get('adopted'))
    B0 = adopted['p'][3]
    # include the adopted solution itself: by definition the profiled
    # curve passes through (B0, chi2_min), i.e. Delta = 0
    B = np.array(r['B_grid'] + r.get('B_grid_fine', []) + [B0])
    c = np.array(r['profile_chi2'] + r.get('profile_chi2_fine', [])
                 + [ref])
    return B, (c - ref) / scale


def curves_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True,
                             sharey=True)
    # secondary chi2(B) minima (competing harmonic / high-B branches),
    # marked so the multiple solutions are explicit; all are excluded.
    SECONDARY = {'J0022': [(78.0, 23.0)],
                 'J0749': [(77.0, 76.0), (93.0, 13.0)]}
    for k, (ax, src) in enumerate(zip(axes.ravel(), SHORT)):
        z = np.load(f'{OUT}/data/{src}_BT_map.npz')
        best = z['best']
        draw_curves(ax, f'{OUT}/data/{src}_BT_map.npz', SHORT[src],
                    best_BkT=(best[3], best[4]), legend=False,
                    envelope=envelope_from_json(src), note=False)
        for bs, ds in SECONDARY.get(src, []):
            ax.plot(bs, ds, 'v', ms=6.5, mfc='none', mec='crimson',
                    mew=1.1, zorder=8, clip_on=False)
            ax.annotate(f'{bs:.0f} MG\n({ds**0.5:.1f}$\\sigma$)',
                        xy=(bs, ds), xytext=(bs, min(ds * 5, 2500)),
                        color='crimson', fontsize=5.6, ha='center',
                        va='bottom', zorder=8)
    for ax in axes[1]:
        ax.set_xlabel(r'$B$ [MG]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'rescaled $\Delta\tilde\chi^2$')
    handles = [
        Line2D([0], [0], color=cm.viridis(0.55), lw=1.2,
               label=r'fixed $kT$ grid'),
        Line2D([0], [0], color='k', lw=1.6,
               label=r'$kT$ free envelope'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.53, 0.995), fontsize=7.0,
               handlelength=1.7, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.05, hspace=0.07, top=0.925)
    fig.savefig(f'{OUT}/figures/B_profiles_kT.pdf')
    plt.close(fig)
    print('B_profiles_kT.pdf saved')


def _bench_row(axL, axR, key, src_lab, note='', legend=True):
    """One validation source, drawn exactly like the per-source panels of
    Figure~\\ref{fig:B_profiles}: the residual spectrum and best-fit model
    over the fitted wavelength range (left), and the fixed-kT chi^2(B)
    curves with the kT-free envelope from the (B, kT) map (right).
    The plotted data are written to data/bench_{key}_spectrum.txt and the
    right-panel grid is data/{key}_BT_map.npz."""
    from benchmark_fit import load, BENCH
    w_m, fl, e = load(BENCH[key])
    r = json.load(open(f'{OUT}/data/bench_{key}.json'))
    B, kT, th, ll = r['B'], r['kT'], r['theta'], r['logLambda']
    spec = cal_cy_spec(w_m, kT * KEV_J, B * 100.0, np.deg2rad(th), 10.0 ** ll)
    shape = spec / spec.max()
    wgt = 1.0 / e ** 2
    A = np.sum(fl * shape * wgt) / np.sum(shape ** 2 * wgt)
    model = A * shape

    axL.plot(w_m * 1e10, fl, color='0.5', lw=0.45, alpha=0.9,
             label='residual spectrum', zorder=1)
    axL.plot(w_m * 1e10, model, color=pubstyle.COLORS['model'], lw=1.5,
             label=f'fit ($B$={B:.1f} MG)', zorder=3)
    axL.set_xlim(w_m.min() * 1e10, w_m.max() * 1e10)   # data range only
    axL.set_ylabel(r'$F_\lambda$ (arbitrary)')
    axL.legend(fontsize=7.0, loc='upper right')
    axL.set_title(src_lab + (f' ({note})' if note else ''), loc='left',
                  fontsize=9, pad=3)

    # data behind the figure
    np.savetxt(f'{OUT}/data/bench_{key}_spectrum.txt',
               np.column_stack([w_m * 1e10, fl, e, model]),
               header='wavelength_A  residual_flux  error  best_fit_model',
               fmt='%.6e')

    # right: fixed-kT curves + kT-free envelope from the (B, kT) map,
    # identical in style to the source panels of Figure~\ref{fig:B_profiles}
    npz = f'{OUT}/data/{key}_BT_map.npz'
    z = np.load(npz)
    sc = max(float(z['chi2_red']), 1.0)
    refm = min(float(z['chi2'].min()), float(z['best_chi2']))
    env = (z['chi2'].min(axis=0) - refm) / sc
    draw_curves(axR, npz, src_lab, best_BkT=(B, kT), legend=legend,
                envelope=(np.append(z['B'], B), np.append(env, 0.0)))
    axR.axvline(r['B_lit'], color='k', ls='-.', lw=0.9)
    axR.text(0.97, 0.94, f"lit. {r['B_lit']:.1f} MG", transform=axR.transAxes,
             fontsize=7, ha='right', va='top', color='0.3',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.78,
                       pad=0.4))
    axR.set_ylabel(r'rescaled $\Delta\tilde\chi^2$')


def bench_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    _bench_row(axes[0, 0], axes[0, 1], 'EQCet', 'EQ Cet',
               note='blind', legend=True)
    _bench_row(axes[1, 0], axes[1, 1], 'BSTri', 'BS Tri',
               note=r'$\theta$ fixed; hump region', legend=False)
    axes[1, 0].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    axes[1, 1].set_xlabel(r'$B$ [MG]')
    fig.subplots_adjust(hspace=0.32, wspace=0.27, left=0.09, right=0.98,
                        top=0.94, bottom=0.09)
    fig.savefig(f'{OUT}/figures/bench_validation.pdf')
    plt.close(fig)
    print('bench_validation.pdf saved')


if __name__ == '__main__':
    curves_figure()
    bench_figure()
