#!/usr/bin/env python3
"""make_corner.py — joint-MCMC corner plot of the four cyclotron parameters.

Reads the thinned joint-MCMC chain stored in data/{src}_mcmc_chain.npy and
draws the (B, kT, theta, log Lambda) corner used in the appendix
(figures/{src}_corner_cyc.pdf).  Default source is J0005, the one shown in
the paper; pass other source keys on the command line to make theirs too.

The chain columns are the full joint parameter vector
[T_wd, logg, T_spot, B_MG, kT_keV, theta_deg, log_Lambda,
 log_s_wd, log_s_spot, log_A_cyc]; columns 3-6 are the cyclotron block.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
import pubstyle
pubstyle.apply()

DATA = os.path.join(ROOT, 'data')
FIGD = os.path.join(ROOT, 'figures')
os.makedirs(FIGD, exist_ok=True)

SCI = [3, 4, 5, 6]   # B, kT, theta, log Lambda
LAB = [r'$B$ [MG]', r'$kT$ [keV]', r'$\theta$ [deg]', r'$\log_{10}\Lambda$']
SRC = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
       'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}


def make(src):
    ch = np.load(os.path.join(DATA, f'{src}_mcmc_chain.npy'))
    d = ch[:, SCI]
    q = np.percentile(d, [16, 50, 84], axis=0)
    fig = corner.corner(
        d, labels=LAB, quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_fmt='.2g', title_kwargs=dict(fontsize=9),
        label_kwargs=dict(fontsize=11), bins=40,
        plot_datapoints=False, fill_contours=True,
        levels=(0.39, 0.86, 0.99), color='#243b6b',
        hist_kwargs=dict(color='#243b6b'))
    out = os.path.join(FIGD, f'{src}_corner_cyc.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  {src}_corner_cyc.pdf  '
          f'B={q[1, 0]:.2f} (+{q[2, 0]-q[1, 0]:.2f}/-{q[1, 0]-q[0, 0]:.2f}), '
          f'kT={q[1, 1]:.2f}, theta={q[1, 2]:.0f}, logL={q[1, 3]:.2f}  '
          f'[{d.shape[0]} samples]')


if __name__ == '__main__':
    for s in (sys.argv[1:] or ['J0005']):
        make(s)
