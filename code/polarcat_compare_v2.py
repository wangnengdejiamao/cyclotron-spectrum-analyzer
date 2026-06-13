#!/usr/bin/env python3
"""
polarcat_compare_v2.py — place the paper_v2 measurements in the PolarCat
(Schwope 2025) population. Reads B values from the joint-fit JSONs and
periods from the LC JSONs; writes the (B, P_orb) scatter figure and a
summary CSV with population percentiles.
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io.votable import parse
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)) )
import pubstyle
pubstyle.apply()

OUT = _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),
                                   _o.pardir))   # repo root
# PolarCat VOTable tables (Schwope 2025); obtain from the published
# catalogue and place under data/ (not redistributed here).
VOT_POLARS = _o.path.join(OUT, 'data', '2025a_polars.vot')
VOT_LARPS = _o.path.join(OUT, 'data', '2025a_larps.vot')

LABELS = {
    'J0005': ('DESI J0005+2941', 105.349),
    'J0022': ('DESI J0022+1340', None),
    'J0749': ('DESI J0749+3654', 80.406),
    'J0035': ('LAMOST J0035+4333', 143.765),
}


def main():
    polars = parse(VOT_POLARS).get_first_table().to_table()
    larps = parse(VOT_LARPS).get_first_table().to_table()
    P_pol = np.asarray(polars['P_orb'].filled(np.nan), float)
    B_pol = np.asarray(polars['B1'].filled(np.nan), float)
    P_lp = np.asarray(larps['P_orb'].filled(np.nan), float)
    B_lp = np.asarray(larps['B1'].filled(np.nan), float)
    B_sorted = np.sort(B_pol[np.isfinite(B_pol)])
    P_sorted = np.sort(P_pol[np.isfinite(P_pol)])

    ours = []
    for key, (label, P_min) in LABELS.items():
        jf = os.path.join(OUT, 'data', f'{key}_joint_results.json')
        if not os.path.exists(jf):
            print(f'  {key}: no joint results yet, skipped')
            continue
        with open(jf) as f:
            r = json.load(f)
        post = r.get('posterior', {})
        if 'B_MG' in post:
            B = post['B_MG']['q50']
            Berr = 0.5 * (post['B_MG']['q84'] - post['B_MG']['q16'])
        else:
            B = r['solutions'][0]['p'][3]
            Berr = np.nan
        alts = [s['p'][3] for s in r['solutions'][1:]
                if abs(s['p'][3] - B) > 3.0]
        ours.append(dict(key=key, label=label, B=B, Berr=Berr, P=P_min,
                         alt=alts))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    m = np.isfinite(P_pol) & np.isfinite(B_pol)
    ax.scatter(P_pol[m], B_pol[m], s=14, c='lightcoral',
               edgecolor='firebrick', lw=0.4,
               label=f'Polars (PolarCat, N={m.sum()})')
    m2 = np.isfinite(P_lp) & np.isfinite(B_lp)
    ax.scatter(P_lp[m2], B_lp[m2], s=14, c='skyblue',
               edgecolor='steelblue', lw=0.4, label=f'LARPs (N={m2.sum()})')
    rows = ['key,label,B_MG,B_err,P_min,B_percentile,P_percentile,alt_B']
    for s in ours:
        if s['P'] is None:
            ax.axhline(s['B'], color='goldenrod', ls='--', lw=1.2,
                       zorder=3)
            ax.annotate(s['label'] + ' ($P$ unknown)', (32, s['B']),
                        xytext=(0, 4), textcoords='offset points',
                        fontsize=8, style='italic', color='darkgoldenrod')
            bp = np.searchsorted(B_sorted, s['B']) / len(B_sorted) * 100
            rows.append(f"{s['key']},{s['label']},{s['B']:.2f},"
                        f"{s['Berr']:.2f},,{bp:.0f},,"
                        f"{';'.join(f'{a:.1f}' for a in s['alt'])}")
            print(rows[-1])
            continue
        x = s['P']
        ax.errorbar(x, s['B'], yerr=s['Berr'] if np.isfinite(s['Berr'])
                    else None, fmt='*', ms=17, mfc='gold', mec='k',
                    ecolor='k', lw=1, zorder=5)
        ax.annotate(s['label'], (x, s['B']), xytext=(8, 4),
                    textcoords='offset points', fontsize=8)
        for ab in s['alt']:
            ax.plot(x, ab, '*', ms=11, mfc='none', mec='goldenrod',
                    mew=1.4, zorder=4)
        bp = np.searchsorted(B_sorted, s['B']) / len(B_sorted) * 100
        pp = (np.searchsorted(P_sorted, s['P']) / len(P_sorted) * 100
              if s['P'] else np.nan)
        rows.append(f"{s['key']},{s['label']},{s['B']:.2f},"
                    f"{s['Berr']:.2f},{s['P'] or ''},{bp:.0f},"
                    f"{pp:.0f},{';'.join(f'{a:.1f}' for a in s['alt'])}")
        print(rows[-1])
    ax.set_xscale('log')
    ax.set_xlabel('Orbital period [min]')
    ax.set_ylabel(r'$B$ [MG]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'polarcat_B_vs_Porb.pdf'))
    fig.savefig(os.path.join(OUT, 'manuscript', 'figs',
                             'polarcat_B_vs_Porb.pdf'))
    with open(os.path.join(OUT, 'data', 'polarcat_summary.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('saved polarcat figure + summary')


if __name__ == '__main__':
    main()
