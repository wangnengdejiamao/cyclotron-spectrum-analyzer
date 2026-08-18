#!/usr/bin/env python3
"""plot_wd_prior.py — appendix figure of the physically constrained fits.

Same decomposition style as Fig. 3, but with the white dwarf component
tied to a Nauenberg (1972) mass-radius relation, M_wd in 0.6-1.0 Msun,
T_wd in 8000-20000 K, and the Gaia distance (see wd_prior_refit.py).
Each panel also states the field of the free fit and of this one, which
is the point of the figure: the continuum decomposition changes, the
field does not.
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
from wd_prior_refit import build, PriorModel  # noqa: F401

pubstyle.apply()
C = pubstyle.COLORS
ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
WP = os.path.join(ROOT, 'data', 'wd_prior')
FR = os.path.join(ROOT, 'validation_candidates', 'full_refit', 'data')
ORDER = ['J0005', 'J0022', 'J0749', 'J0035']
SHORT = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
         'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, src in zip(axes.ravel(), ORDER):
        model, wb, meta = build(src)
        wp = json.load(open(os.path.join(WP, f'{src}_wdprior_branches.json')))
        row = wp['branches'][0]
        q = np.array([row['T_wd'], row['M_wd'], row['T_spot'], row['B'],
                      row['kT'], row['theta'], row['logLambda']])
        p, amps, mdl, c2 = model.solve_amplitudes_q(q)
        wd, sp, cy = model.components(p)
        s = 1e-17 if meta['kind'] == 'desi' else 1.0
        unit = r'$F_\lambda$ [$10^{-17}$]' if meta['kind'] == 'desi' \
            else r'$F_\lambda$ (relative)'

        w_s, f_s, _ = jp.load_spectrum(meta)
        sel = (w_s >= wb.min() - 60) & (w_s <= wb.max() + 60)
        ax.plot(w_s[sel], f_s[sel] / s, color=C['data'], lw=0.3, alpha=0.6,
                zorder=1)
        ax.plot(wb, mdl / s, color=C['model'], lw=1.4, zorder=5,
                label='total')
        ax.plot(wb, amps[0] * wd / s, '--', color=C['wd'], lw=1.0,
                label='WD (constrained)')
        ax.plot(wb, amps[1] * sp / s, '--', color=C['spot'], lw=1.0,
                label='hot spot')
        ax.plot(wb, amps[2] * cy / s, '-', color=C['cyc'], lw=1.1,
                label='cyclotron')
        ax.set_xlim(wb.min(), wb.max())
        top = max(float((mdl / s).max()),
                  float(np.percentile(f_s[sel] / s, 99.0)))
        ax.set_ylim(0.0, 1.18 * top)
        ax.set_ylabel(unit)
        ax.set_title(SHORT[src], loc='left', fontsize=9, pad=3)

        js = json.load(open(os.path.join(FR, f'{src}_full_refit.json')))
        bfree = float(js['adopted_params'][3])
        if src == 'J0035':
            wdtxt = rf"$T_{{\rm wd}}$={row['T_wd']:.0f} K"
        else:
            wdtxt = (rf"$M_{{\rm wd}}$={row['M_wd']:.2f} $M_\odot$, "
                     rf"$T_{{\rm wd}}$={row['T_wd']:.0f} K")
        ax.text(0.975, 0.94,
                rf"$B$={row['B']:.2f} MG (free fit {bfree:.2f})" + '\n'
                + wdtxt + '\n'
                + rf"$T_{{\rm spot}}$={row['T_spot']:.0f} K, "
                  rf"$kT$={row['kT']:.2f} keV",
                transform=ax.transAxes, ha='right', va='top', fontsize=6.4,
                linespacing=1.3,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                          pad=1.2))
        if src == 'J0005':
            ax.legend(fontsize=6.2, loc='upper right',
                      bbox_to_anchor=(0.99, 0.62), framealpha=0.85)
    for ax in axes[1]:
        ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    fig.subplots_adjust(hspace=0.33, wspace=0.24, left=0.085, right=0.985,
                        top=0.945, bottom=0.10)
    fig.savefig(f'{ROOT}/figures/wd_prior_fits.pdf')
    plt.close(fig)
    print('wd_prior_fits.pdf saved')


if __name__ == '__main__':
    main()
