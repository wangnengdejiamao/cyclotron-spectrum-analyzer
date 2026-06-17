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

OUT = os.environ.get('CYC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = dict(zg='#1b7837', zr='#c1272d')

# (label, fold period in minutes);  J0749 folded at 2 x P_phot.
# J0022 has no significant period and is omitted from this figure.
ROWS = [
    ('J0005', 'DESI J0005+2941', 105.34857383655087, None),
    ('J0749', 'DESI J0749+3654', 80.4062, r'$=2P_{\rm phot}$'),
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


def centre_faint_minimum(detr, P_day, target_phase=0.5, nb=40):
    """Return a phase shift that places the deepest binned median at target."""
    all_ph, all_dm = [], []
    for t, dm in detr.values():
        all_ph.append((t / P_day) % 1.0)
        all_dm.append(dm)
    ph = np.concatenate(all_ph)
    dm = np.concatenate(all_dm)
    med = np.full(nb, np.nan)
    for k in range(nb):
        m = (ph >= k / nb) & (ph < (k + 1) / nb)
        if m.sum() >= 3:
            med[k] = np.median(dm[m])
    dip_phase = (np.nanargmax(med) + 0.5) / nb
    return target_phase - dip_phase


def combined_binned(detr, P_day, phase_shift=0.0, nb=40):
    ph_all, dm_all = [], []
    for t, dm in detr.values():
        ph_all.append((t / P_day + phase_shift) % 1.0)
        dm_all.append(dm)
    return phase_binned(np.concatenate(ph_all), np.concatenate(dm_all), nb=nb)


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
        phase_shift = centre_faint_minimum(detr, P_day) if src == 'J0005' else 0.0
        nbins = 40 if src == 'J0005' else 24
        if src == 'J0005':
            axR.axvspan(0.47, 0.53, color='0.87', zorder=0)
            axR.axvspan(1.47, 1.53, color='0.87', zorder=0)
        fold_bands = ('zg', 'zr')
        for b in fold_bands:
            if b not in detr:
                continue
            t, dm = detr[b]
            ph = (t / P_day + phase_shift) % 1.0
            axR.plot(np.concatenate([ph, ph + 1]),
                     np.concatenate([dm, dm]), '.', ms=1.8, color=COL[b],
                     alpha=0.3, rasterized=True)
            c, m, er = phase_binned(ph, dm, nb=nbins)
            axR.errorbar(np.concatenate([c, c + 1]),
                         np.concatenate([m, m]),
                         yerr=np.concatenate([er, er]), fmt='o-', ms=3.5,
                         lw=1.3, color=COL[b], mec='k', mew=0.4,
                         ecolor=COL[b], capsize=0, zorder=5)
        if src == 'J0005':
            c, m, er = combined_binned(detr, P_day, phase_shift, nb=40)
            axR.plot(np.concatenate([c, c + 1]), np.concatenate([m, m]),
                     color='k', lw=1.5, zorder=7)
        axR.invert_yaxis()
        axR.axhline(0, color='0.6', ls=':', lw=0.6)
        axR.set_ylabel(r'$\Delta$mag')
        lab = f'$P$ = {P_min:.3f} min'
        if note:
            lab += f'  {note}'
        if src == 'J0005':
            lab += '  (minimum centred)'
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
