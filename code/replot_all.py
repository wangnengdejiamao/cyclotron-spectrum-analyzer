#!/usr/bin/env python3
"""
replot_all.py — regenerate the spectral figures in a uniform ApJ-like
style from the saved fit products (no refitting):

  * {src}_joint_fit.pdf    raw spectrum + model components (2 panels)
  * B_profiles_combined.pdf  one panel, rescaled profiled chi2(B) curves
  * bench_EQCet.pdf        restyled benchmark figure
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle
import joint_pipeline as jp
from cyclotron_m2 import cal_cy_spec

pubstyle.apply()
C = pubstyle.COLORS
OUT = '/Users/ljm/Desktop/cyc/paper_v2'
FIGD = os.path.join(OUT, 'figures')

SHORT = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
         'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}


def joint_fig(src, koester):
    meta = jp.SOURCES[src]
    r = json.load(open(f'{OUT}/data/{src}_joint_results.json'))
    adopted = next(s for s in r['solutions'] if s.get('adopted'))
    p = np.array(adopted['p'])
    a = np.array(adopted['amps'])
    w, f, e = jp.load_spectrum(meta)
    comp = np.loadtxt(f'{OUT}/data/{src}_fit_components.txt')
    wb, fb, eb, tot, wdc, spc, cyc = comp.T

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.1, 1.3],
                                              hspace=0.04))
    ax = axes[0]
    sel = (w >= wb.min() - 120) & (w <= wb.max() + 120)
    ax.plot(w[sel], f[sel], color=C['data'], lw=0.35, alpha=0.85,
            label='observed', zorder=1)
    ax.plot(wb, tot, color=C['model'], lw=1.5, label='total model',
            zorder=5)
    ax.plot(wb, wdc, '--', color=C['wd'], lw=1.0,
            label=f'WD ({p[0]/1e3:.1f} kK)')
    ax.plot(wb, spc, '--', color=C['spot'], lw=1.0,
            label=f'spot ({p[2]/1e3:.0f} kK)')
    ax.plot(wb, cyc, '-', color=C['cyc'], lw=1.2,
            label=(f'cyclotron ($B$={p[3]:.1f} MG, $kT$={p[4]:.1f} keV, '
                   f'$\\theta$={p[5]:.0f}$^\\circ$, '
                   f'$\\log\\Lambda$={p[6]:.1f})'))
    ymax = np.percentile(f[sel], 99.6)
    ax.set_ylim(0, 1.28 * ymax)
    ax.set_xlim(wb.min() - 120, wb.max() + 120)
    ax.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
    ax.legend(loc='upper right', fontsize=7.5)
    ax.text(0.015, 0.96, SHORT[src], transform=ax.transAxes, va='top',
            fontsize=10)

    ax = axes[1]
    ax.errorbar(wb, fb - wdc - spc, yerr=eb, fmt='o', ms=2.2, lw=0.6,
                color=C['resid'], ecolor='0.7', label='data $-$ continuum')
    ax.plot(wb, cyc, color=C['cyc'], lw=1.4, label='cyclotron component')
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    ax.set_ylabel(r'$F_\lambda - F_{\rm cont}$')
    ax.legend(loc='upper right', fontsize=7.5)
    fig.savefig(f'{FIGD}/{src}_joint_fit.pdf')
    plt.close(fig)
    print(f'  {src}_joint_fit.pdf')


def profiles_fig():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for src, col in zip(['J0005', 'J0022', 'J0749', 'J0035'], C['srcs']):
        r = json.load(open(f'{OUT}/data/{src}_joint_results.json'))
        Bg = np.array(r['B_grid'])
        ch = np.array(r['profile_chi2'])
        scale = max(r['chi2_red'], 1.0)
        c0 = ch.min()
        if 'profile_chi2_fine' in r:
            chf = np.array(r['profile_chi2_fine'])
            c0 = min(c0, chf.min())
        d = (ch - c0) / scale
        ax.plot(Bg, d, '-', color=col, lw=1.2, label=SHORT[src])
        if 'profile_chi2_fine' in r:
            Bf = np.array(r['B_grid_fine'])
            ax.plot(Bf, (chf - c0) / scale, '-', color=col, lw=1.2)
    for thr, lab in [(1, r'1$\sigma$'), (4, r'2$\sigma$'), (9, r'3$\sigma$')]:
        ax.axhline(thr, ls=':', lw=0.7, color='0.6')
        ax.text(95.5, thr, lab, va='center', fontsize=7, color='0.45')
    ax.set_xlabel(r'$B$ [MG]')
    ax.set_ylabel(r'rescaled $\Delta\tilde\chi^2(B)$')
    ax.set_xlim(12, 95)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    fig.savefig(f'{FIGD}/B_profiles_combined.pdf')
    plt.close(fig)
    print('  B_profiles_combined.pdf')


def bench_fig():
    KEV_J = 1.602176634e-16
    meta_path = '/Users/ljm/Desktop/cyc/new/EQcet/processed_spectrum.txt'
    d = np.loadtxt(meta_path)
    w_m, fl = d[:, 0], d[:, 1]
    o = np.argsort(w_m)
    w_m, fl = w_m[o], fl[o]
    e = np.empty_like(fl)
    for i in range(len(fl)):
        seg = fl[max(0, i-4):i+5]
        e[i] = 1.4826 * np.median(np.abs(np.diff(seg))) / np.sqrt(2)
    e = np.maximum(e, 0.03 * np.abs(fl))
    r = json.load(open(f'{OUT}/data/bench_EQCet.json'))
    B, kT, th, ll = r['B'], r['kT'], r['theta'], r['logLambda']
    spec = cal_cy_spec(w_m, kT*KEV_J, B*100.0, np.deg2rad(th), 10.0**ll)
    shape = spec / spec.max()
    wgt = 1.0 / e**2
    A = np.sum(fl*shape*wgt) / np.sum(shape**2*wgt)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    ax.errorbar(w_m*1e10, fl, yerr=e, fmt='o', ms=2.0, lw=0.5,
                color='0.35', ecolor='0.7', label='EQ Cet residual')
    ax.plot(w_m*1e10, A*shape, color=C['model'], lw=1.4,
            label=f'fit ($B$={B:.1f} MG)')
    ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    ax.set_ylabel(r'$F_\lambda$ (arbitrary)')
    ax.legend(fontsize=7.5)
    ax = axes[1]
    Bg = np.array(r['B_grid'])
    dchi = np.array(r['profile_dchi2_rescaled'])
    ax.plot(Bg, dchi, '-', color='k', lw=1.2)
    ax.axvline(r['B_lit'], color=C['wd'], ls='--', lw=1.1,
               label=f'published {r["B_lit"]:.0f} MG')
    for thr in (1, 4, 9):
        ax.axhline(thr, ls=':', lw=0.7, color='0.6')
    ax.set_xlabel(r'$B$ [MG]')
    ax.set_ylabel(r'rescaled $\Delta\tilde\chi^2(B)$')
    ax.set_ylim(0, 30)
    ax.legend(fontsize=7.5)
    fig.savefig(f'{FIGD}/bench_EQCet.pdf')
    plt.close(fig)
    print('  bench_EQCet.pdf')


if __name__ == '__main__':
    koester = jp.KoesterGrid(jp.KOESTER_DIR)
    for s in ['J0005', 'J0022', 'J0749', 'J0035']:
        joint_fig(s, koester)
    profiles_fig()
    bench_fig()
    print('done')
