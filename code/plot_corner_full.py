#!/usr/bin/env python3
"""plot_corner_full.py — all ten posteriors of the J0005 joint fit.

The referee asked for the posteriors of every fitted parameter, not just
the four cyclotron ones already shown.  This draws the released chain
(data/J0005_mcmc_chain.npy) over all ten parameters of Eq. (2) and
prints the medians quoted in Appendix A, so the figure and the text
cannot drift apart.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle

pubstyle.apply()
ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
G_CGS, MSUN, RSUN, PC = 6.674e-8, 1.989e33, 6.957e10, 3.0857e18
D_J0005 = 604.0

LABELS = [r'$T_{\rm wd}$ [kK]', r'$\log g$', r'$T_{\rm spot}$ [kK]',
          r'$B$ [MG]', r'$kT$ [keV]', r'$\theta$ [deg]',
          r'$\log_{10}\Lambda$', r'$\log_{10} s_{\rm wd}$',
          r'$\log_{10} s_{\rm spot}$', r'$\log_{10} A_{\rm cyc}$']
NAMES = ['T_wd', 'logg', 'T_spot', 'B', 'kT', 'theta', 'logLambda',
         'log_s_wd', 'log_s_spot', 'log_A_cyc']


def main():
    c = np.load(f'{ROOT}/data/J0005_mcmc_chain.npy')
    f = c.reshape(-1, c.shape[-1]) if c.ndim == 3 else c
    show = f.copy()
    show[:, 0] /= 1000.0          # K -> kK, so the labels stay readable
    show[:, 2] /= 1000.0

    fig = corner.corner(
        show, labels=LABELS, quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt='.2f',
        title_kwargs=dict(fontsize=6.5), label_kwargs=dict(fontsize=7.5),
        hist_kwargs=dict(lw=0.8), plot_datapoints=False, fill_contours=True,
        levels=(0.393, 0.865), smooth=0.9,
        contour_kwargs=dict(linewidths=0.6))
    for ax in fig.get_axes():
        ax.tick_params(labelsize=5.5)
    fig.savefig(f'{ROOT}/figures/J0005_corner_full.pdf',
                bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('J0005_corner_full.pdf saved')

    med = np.median(f, axis=0)
    lo, hi = np.percentile(f, [16, 84], axis=0)
    print('\nposterior medians (16th-84th):')
    for n, m, a, b in zip(NAMES, med, lo, hi):
        print(f'  {n:11s} {m:11.4f}  [{a:.4f}, {b:.4f}]')
    s_wd = 10 ** med[NAMES.index('log_s_wd')]
    R = np.sqrt(s_wd) * D_J0005 * PC
    M = 10 ** med[1] * R ** 2 / G_CGS / MSUN
    print(f'\n  B half-width       = {(hi[3]-lo[3])/2:.3f} MG')
    print(f'  R_wd(median s_wd)  = {R/RSUN:.4f} Rsun')
    print(f'  M_wd = gR^2/G      = {M:.2f} Msun')
    print(f'  logg 16-84 width   = {hi[1]-lo[1]:.2f} dex '
          f'(prior width 2.49)')


if __name__ == '__main__':
    main()
