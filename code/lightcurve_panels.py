#!/usr/bin/env python3
"""
lightcurve_panels.py — combined ZTF light-curve figure for the paper.

One row per source:
  left  : long-term light curve (g, r) with 40-day rolling medians,
  right : phase-folded, detrended light curve at the adopted period
          (g, r points) with phase-binned median trend lines.
The source without a significant period (J0022) shows only the long-term
panel; its right panel is left blank with an annotation.
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
pubstyle.apply()
from lightcurve_analysis import load_ztf, rolling_median, SOURCES

OUT = '/Users/ljm/Desktop/cyc/paper_v2'
COL = dict(zg='#1b7837', zr='#c1272d')

# (label, fold period in minutes);  J0749 folded at 2 x P_phot.
# J0022 has no significant period and is omitted from this figure.
ROWS = [
    ('J0005', 'DESI J0005+2941', 105.349, None),
    ('J0749', 'DESI J0749+3654', 80.406, r'$=2P_{\rm phot}$'),
    ('J0035', 'LAMOST J0035+4333', 143.765, '1-d alias unresolved'),
]


def phase_binned(ph, dm, nb=24):
    cent, med, err = [], [], []
    for k in range(nb):
        m = (ph >= k / nb) & (ph < (k + 1) / nb)
        if m.sum() >= 3:
            cent.append((k + 0.5) / nb)
            med.append(np.median(dm[m]))
            err.append(1.4826 * np.median(np.abs(dm[m] - np.median(dm[m])))
                       / np.sqrt(m.sum()))
    return np.array(cent), np.array(med), np.array(err)


def main():
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(7.2, 6.6),
                             gridspec_kw=dict(width_ratios=[1.7, 1.0]))
    last = len(ROWS) - 1
    for row, (src, name, P_min, note) in enumerate(ROWS):
        bands = load_ztf(SOURCES[src]['csv'])
        axL, axR = axes[row]

        # ---- long-term ----
        detr = {}
        for b in ('zg', 'zr'):
            if b not in bands:
                continue
            d = bands[b]
            rm = rolling_median(d["t"], d["m"], 40.0)
            axL.plot(d['t'] - 2458000, d['m'], '.', ms=1.6, color=COL[b],
                     alpha=0.35, rasterized=True)
            axL.plot(d['t'] - 2458000, rm, '-', color=COL[b], lw=0.9,
                     label=b[1])
            detr[b] = (d['t'], d['m'] - rm)
        axL.invert_yaxis()
        axL.set_ylabel('mag')
        axL.text(0.015, 0.93, name, transform=axL.transAxes, va='top',
                 fontsize=8.5)
        if row == 0:
            axL.legend(fontsize=7.5, loc='lower right', ncol=2)
        if row == last:
            axL.set_xlabel('HJD $-$ 2458000 [d]')

        # ---- folded ----
        if P_min is None:
            axR.text(0.5, 0.5, note, transform=axR.transAxes, ha='center',
                     va='center', fontsize=8.5, style='italic',
                     color='0.35')
            axR.set_xticks([]); axR.set_yticks([])
            for s in axR.spines.values():
                s.set_visible(False)
            continue
        P_day = P_min / 1440.0
        for b in ('zg', 'zr'):
            if b not in detr:
                continue
            t, dm = detr[b]
            ph = (t / P_day) % 1.0
            axR.plot(np.concatenate([ph, ph + 1]),
                     np.concatenate([dm, dm]), '.', ms=1.8, color=COL[b],
                     alpha=0.3, rasterized=True)
            c, m, er = phase_binned(ph, dm)
            axR.errorbar(np.concatenate([c, c + 1]),
                         np.concatenate([m, m]),
                         yerr=np.concatenate([er, er]), fmt='o-', ms=3.5,
                         lw=1.3, color=COL[b], mec='k', mew=0.4,
                         ecolor=COL[b], capsize=0, zorder=5)
        axR.invert_yaxis()
        axR.axhline(0, color='0.6', ls=':', lw=0.6)
        axR.set_ylabel(r'$\Delta$mag')
        lab = f'$P$ = {P_min:.3f} min'
        if note:
            lab += f'  {note}'
        axR.text(0.5, 1.02, lab, transform=axR.transAxes, ha='center',
                 va='bottom', fontsize=7.5)
        axR.set_xlim(0, 2)
        if row == last:
            axR.set_xlabel('orbital phase')

    fig.subplots_adjust(hspace=0.18, wspace=0.28, left=0.09, right=0.98,
                        top=0.97, bottom=0.06)
    fig.savefig(f'{OUT}/figures/lightcurve_panels.pdf', dpi=200)
    print('lightcurve_panels.pdf saved')


if __name__ == '__main__':
    main()
