#!/usr/bin/env python3
"""reproduce_numbers.py — print the headline magnetic-field results.

Reproduces, straight from the stored fit products, the cyclotron parameters
of the adopted branches:

  * adopted B, kT, theta, log10 Lambda
      <- validation_candidates/full_refit/data/{src}_full_refit.json["adopted_params"]
  * the field uncertainty
      sigma_B = sqrt( sigma_formal^2 + (0.026 * B)^2 )
    where sigma_formal is the joint-MCMC width and 0.026*B is the 2.6 %
    systematic floor calibrated on the benchmarks.  For the three
    branch-ambiguous targets the value is taken directly from
    full_refit/data/branch_error_summary.json (computed by branch_errors.py);
    for J0005 (a unique field) it is recomputed here from the MCMC chain.

No fitting is done: this reads the adopted solutions only.  The manuscript
table may differ in the last digit where it quotes posterior medians or
rounded branch-summary values rather than the chi-squared minimum itself.
"""
import json
import os

import numpy as np

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FRD = os.path.join(ROOT, 'validation_candidates', 'full_refit', 'data')
DATA = os.path.join(ROOT, 'data')
SYS_FLOOR = 0.026   # 2.6 % benchmark-calibrated systematic floor

NAME = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
        'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}


def adopted_err():
    """{source: sigma_B} for the branch-ambiguous targets (branch_errors.py)."""
    s = json.load(open(os.path.join(FRD, 'branch_error_summary.json')))
    return {b['source']: b['B_err'] for b in s['branches'] if b.get('adopted')}


def j0005_err(B):
    """Unique-field source: combine MCMC formal width with the 2.6 % floor."""
    ch = np.load(os.path.join(DATA, 'J0005_mcmc_chain.npy'))
    formal = float(np.std(ch[:, 3]))
    return (formal**2 + (SYS_FLOOR * B)**2) ** 0.5


def main():
    errs = adopted_err()
    print(f"{'source':16s} {'B [MG]':>14s} {'kT [keV]':>9s} "
          f"{'theta':>7s} {'log10 Lambda':>13s}")
    print('-' * 64)
    for src in ('J0005', 'J0022', 'J0749', 'J0035'):
        p = json.load(open(os.path.join(
            FRD, f'{src}_full_refit.json')))['adopted_params']
        B, kT, th, lam = p[3], p[4], p[5], p[6]
        sB = errs.get(src) or j0005_err(B)
        print(f"{NAME[src]:16s} {B:7.1f} +/- {sB:4.1f} {kT:9.1f} "
              f"{th:6.0f} {lam:13.1f}")
    print('-' * 64)

    bench = json.load(open(os.path.join(
        FRD, 'branch_error_summary.json')))['benchmarks']
    print('\nbenchmark recovery (this code  vs  literature):')
    for name, got, lit in bench:
        print(f"  {name:8s} {got:5.1f} MG  vs  {lit:5.1f} MG")
    print(f"\n(field error = sqrt(sigma_formal^2 + ({SYS_FLOOR}*B)^2); "
          'benchmark scatter sets the 2.6% floor)')


if __name__ == '__main__':
    main()
