#!/usr/bin/env python3
"""spherex_bin.py — bin the SPHEREx forced photometry into usable points.

IRSA returns one flux per (channel, exposure), each of low signal-to-noise
for targets this faint.  We apply the standard quality cuts and then
inverse-variance average into logarithmic wavelength bins, so the result
is a handful of points with meaningful errors between 0.75 and 5 um.

Writes data/spherex/spherex_binned.json.
"""
import glob
import json
import os

import numpy as np

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SPX = os.path.join(ROOT, 'data', 'spherex')
NBIN = 8
FIT_QL_RANGE = (0.5, 2.0)      # forced-photometry fit quality
SIG_CLIP = 4.0


def arr(v):
    """The archive JSON stores numpy arrays as their repr; parse either."""
    if isinstance(v, str):
        return np.array([float(x) for x in
                         v.strip('[] \n').replace('\n', ' ').split()])
    return np.asarray(v, float)


def bin_one(path, nbin=NBIN):
    d = json.load(open(path))
    lam = arr(d['lambda_um'])
    f = arr(d['flux_uJy'])
    e = arr(d['flux_err_uJy'])
    flags = arr(d['flags']) if 'flags' in d else np.zeros_like(f)
    ql = arr(d['fit_ql']) if 'fit_ql' in d else np.ones_like(f)

    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    good &= (ql > FIT_QL_RANGE[0]) & (ql < FIT_QL_RANGE[1])
    n_all = good.sum()
    # SPHEREx flag bits: keep the cleanest half by flag value, but never
    # cut so hard that a bin empties -- record what each cut costs.
    strict = good & (flags == 0)
    used_strict = strict.sum() >= 0.15 * n_all
    sel = strict if used_strict else good

    edges = np.geomspace(lam[sel].min() * 0.999, lam[sel].max() * 1.001,
                         nbin + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = sel & (lam >= lo) & (lam < hi)
        if m.sum() < 3:
            continue
        w = 1.0 / e[m] ** 2
        mu = np.sum(f[m] * w) / np.sum(w)
        # sigma clip once
        keep = np.abs(f[m] - mu) < SIG_CLIP * e[m]
        if keep.sum() >= 3:
            w = w[keep]
            mu = np.sum(f[m][keep] * w) / np.sum(w)
        sig = 1.0 / np.sqrt(np.sum(w))
        scat = (np.std(f[m][keep] if keep.sum() >= 3 else f[m], ddof=1)
                / np.sqrt(max(keep.sum() if keep.sum() >= 3 else m.sum(), 1)))
        rows.append(dict(lam_um=float(np.average(lam[m], weights=1/e[m]**2)),
                         lam_lo=float(lo), lam_hi=float(hi),
                         n=int(keep.sum() if keep.sum() >= 3 else m.sum()),
                         f_uJy=float(mu), e_formal=float(sig),
                         e_scatter=float(scat),
                         e_uJy=float(max(sig, scat)),
                         snr=float(mu / max(sig, scat))))
    return dict(n_measurements=int(len(lam)), n_good=int(n_all),
                n_used=int(sel.sum()), strict_flags=bool(used_strict),
                bins=rows)


def main():
    out = {}
    for path in sorted(glob.glob(os.path.join(SPX, '*_spherex.json'))):
        src = os.path.basename(path).split('_')[0]
        r = bin_one(path)
        out[src] = r
        print(f'=== {src}: {r["n_measurements"]} measurements, '
              f'{r["n_used"]} used '
              f'({"flags==0" if r["strict_flags"] else "quality cut only"})')
        for b in r['bins']:
            print(f'   {b["lam_um"]:5.2f} um  '
                  f'{b["f_uJy"]:8.2f} +- {b["e_uJy"]:6.2f} uJy  '
                  f'(N={b["n"]:3d}, S/N={b["snr"]:5.1f})')
    with open(os.path.join(SPX, 'spherex_binned.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwritten {SPX}/spherex_binned.json')


if __name__ == '__main__':
    main()
