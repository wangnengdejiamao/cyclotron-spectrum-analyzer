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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle
from cyclotron_m2 import cal_cy_spec

pubstyle.apply()
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   os.pardir))   # repo root
KEV_J = 1.602176634e-16
SHORT = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
         'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}

def draw_curves(ax, npz, label, best_BkT, legend=True, envelope=None):
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
    ax.set_ylim(0.3, 4000)
    ax.set_xlim(B.min(), B.max())
    ax.text(0.03, 0.95, label, transform=ax.transAxes, va='top',
            fontsize=9)
    if legend:
        ax.legend(fontsize=6.0, loc='lower right', ncol=2,
                  handlelength=1.3, columnspacing=0.8)


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
    for k, (ax, src) in enumerate(zip(axes.ravel(), SHORT)):
        z = np.load(f'{OUT}/data/{src}_BT_map.npz')
        best = z['best']
        draw_curves(ax, f'{OUT}/data/{src}_BT_map.npz', SHORT[src],
                    best_BkT=(best[3], best[4]),
                    envelope=envelope_from_json(src))
    for ax in axes[1]:
        ax.set_xlabel(r'$B$ [MG]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'rescaled $\Delta\tilde\chi^2$')
    fig.subplots_adjust(wspace=0.05, hspace=0.07)
    fig.savefig(f'{OUT}/figures/B_profiles_kT.pdf')
    plt.close(fig)
    print('B_profiles_kT.pdf saved')


def _bench_row(axL, axR, key, bench_name, src_lab, note='', spec_xlim=None):
    """One validation source: residual spectrum + best-fit model (left),
    fixed-kT profiled chi^2(B) curves + kT-free envelope (right)."""
    from benchmark_fit import load, BENCH
    w_m, fl, e = load(BENCH[key])
    r = json.load(open(f'{OUT}/data/bench_{key}.json'))
    B, kT, th, ll = r['B'], r['kT'], r['theta'], r['logLambda']
    spec = cal_cy_spec(w_m, kT * KEV_J, B * 100.0, np.deg2rad(th),
                       10.0 ** ll)
    shape = spec / spec.max()
    wgt = 1.0 / e ** 2
    A = np.sum(fl * shape * wgt) / np.sum(shape ** 2 * wgt)

    axL.plot(w_m * 1e10, fl, color='0.5', lw=0.45, alpha=0.9,
             label='observed residual spectrum', zorder=1)
    axL.plot(w_m * 1e10, A * shape, color=pubstyle.COLORS['model'], lw=1.5,
             label=f'fit ($B$={B:.1f} MG)', zorder=3)
    axL.set_ylabel(r'$F_\lambda$ (arbitrary)')
    axL.legend(fontsize=7.0, loc='upper right')
    axL.set_title(src_lab + (f' ({note})' if note else ''), loc='left',
                  fontsize=9, pad=3)
    if spec_xlim is not None:   # display truncation only; fit uses full spectrum
        axL.set_xlim(left=spec_xlim, right=w_m.max() * 1e10 * 1.004)

    # right: 1-D profiled chi^2(B) from the full-resolution fit
    Be = np.append(np.array(r['B_grid']), B)
    de = np.append(np.array(r['profile_dchi2_rescaled']), 0.0)
    o = np.argsort(Be)
    axR.plot(Be[o], np.maximum(de[o], 0.3), '-', color='k', lw=1.6,
             label=r'profiled $\Delta\tilde\chi^2(B)$')
    for thr, lab in [(1, r'1$\sigma$'), (4, r'2$\sigma$'), (9, r'3$\sigma$')]:
        axR.axhline(thr, ls=':', lw=0.7, color='0.55')
        axR.text(94, thr, lab, fontsize=6.5, va='center', ha='right',
                 color='0.5')
    axR.axvline(B, color='0.4', ls='--', lw=0.7, zorder=1)
    axR.plot(B, 0.3, '*', ms=12, mfc='gold', mec='k', mew=0.6,
             clip_on=False, zorder=6)
    axR.set_yscale('log')
    axR.set_ylim(0.3, 4000)
    axR.set_xlim(12, 95)
    axR.legend(fontsize=7, loc='upper left')
    axR.axvline(r['B_lit'], color='k', ls='-.', lw=0.9)
    axR.text(0.97, 0.93, f"lit. {r['B_lit']:.1f} MG", transform=axR.transAxes,
             fontsize=7, ha='right', va='top', color='0.3')
    axR.set_ylabel(r'rescaled $\Delta\tilde\chi^2$')


def bench_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    _bench_row(axes[0, 0], axes[0, 1], 'EQCet', 'EQ Cet', 'EQ Cet',
               note='blind')
    _bench_row(axes[1, 0], axes[1, 1], 'BSTri', 'BS Tri', 'BS Tri',
               note=r'$\theta$ fixed by eclipse', spec_xlim=4500.0)
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
