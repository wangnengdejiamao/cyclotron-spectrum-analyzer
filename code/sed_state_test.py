#!/usr/bin/env python3
"""sed_state_test.py — how much does the accretion state corrupt the SED?

The archival UV/optical/IR photometry is NOT simultaneous with the DESI or
LAMOST spectra, and all four systems change accretion state by 1.5-2.5 mag
in the ZTF bands.  Before the archival SED can be used for anything, this
script measures the size of that effect:

  1. synthetic photometry of the fitted spectrum in the Pan-STARRS/SDSS
     bands, compared with the catalogue values (two independent epochs);
  2. the SDSS - Pan-STARRS difference itself (two epochs, same band);
  3. the AllWISE - CatWISE difference (two epoch ranges, W1/W2);
  4. the ZTF g/r magnitude at the spectrum epoch relative to the full
     bright/faint range of the light curve.

Conclusion drawn in the paper: the UV cannot be used to constrain the
white-dwarf temperature, and the IR can only give an UPPER limit on the
donor (the observed flux is an upper bound on the donor in any state).
"""
import csv as _csv
import json
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pipeline as jp

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get('CYC_RAW', os.path.join(ROOT, 'data', 'raw'))
PHOT = json.load(open(f'{ROOT}/data/sed_photometry.json'))

ZTF = {'J0005': f'{BASE}/ZTF_RA1.49466_DEC29.68439.csv',
       'J0022': f'{BASE}/ZTF_RA5.72177_DEC13.67799.csv',
       'J0749': f'{BASE}/ZTF_RA117.32130_DEC36.90776.csv',
       'J0035': (f'{BASE}/每个源分析/LAMOST_J003553.36+433341.4/'
                 'ZTF_RA8.9724_DEC43.5615.csv')}

# effective wavelength [AA] and AB zero point for synthetic photometry
BANDS = {'gP1': 4849.0, 'rP1': 6201.0, 'iP1': 7535.0, 'zP1': 8674.0,
         'gSDSS': 4702.0, 'rSDSS': 6175.0, 'iSDSS': 7490.0, 'zSDSS': 8946.0}


def load_ztf(path):
    rows = {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            try:
                if int(r['catflags'], 0) != 0:
                    continue
                m, e, t = float(r['mag']), float(r['magerr']), float(r['hjd'])
            except (ValueError, KeyError):
                continue
            if e <= 0 or e > 0.30:
                continue
            rows.setdefault(r['filtercode'], []).append((t, m))
    return {b: np.array(sorted(v)) for b, v in rows.items() if len(v) > 20}


def spec_mjd(src):
    meta = jp.SOURCES[src]
    with fits.open(meta['fits']) as h:
        for hdu in h:
            for k in ('MJD', 'MJD-OBS', 'MJDMID', 'MJDLIST', 'DATE-OBS',
                      'MJDBEG', 'TAI-BEG'):
                if k in hdu.header:
                    return k, hdu.header[k]
    return None, None


def synth_mag(w, f, lam0, width=300.0):
    """Crude AB magnitude: mean F_lambda in a +-width/2 box at lam0."""
    m = (w > lam0 - width / 2) & (w < lam0 + width / 2)
    if m.sum() < 5:
        return None
    flam = float(np.nanmedian(f[m]))
    if not np.isfinite(flam) or flam <= 0:
        return None
    fnu = flam * lam0 ** 2 / 2.998e18
    return -2.5 * np.log10(fnu / 3631e-23)


def main():
    print('=' * 74)
    print('1) Two-epoch catalogue differences (same or near-same band)')
    print('=' * 74)
    for src in PHOT:
        b = PHOT[src]['bands']
        line = [f'{src}:']
        if 'gP1' in b and 'gSDSS' in b:
            line.append(f"g(PS1)-g(SDSS) = {b['gP1']['mag']-b['gSDSS']['mag']:+.2f}")
        if 'rP1' in b and 'rSDSS' in b:
            line.append(f"r(PS1)-r(SDSS) = {b['rP1']['mag']-b['rSDSS']['mag']:+.2f}")
        if 'W1' in b and 'W1cw' in b:
            line.append(f"W1(AllWISE)-W1(CatWISE) = {b['W1']['mag']-b['W1cw']['mag']:+.2f}")
        if 'W2' in b and 'W2cw' in b:
            line.append(f"W2(AllWISE)-W2(CatWISE) = {b['W2']['mag']-b['W2cw']['mag']:+.2f}")
        print('  ' + '  '.join(line))

    print()
    print('=' * 74)
    print('2) ZTF full range and the level at the spectrum epoch')
    print('=' * 74)
    for src, path in ZTF.items():
        if not os.path.exists(path):
            print(f'  {src}: ZTF csv missing')
            continue
        lc = load_ztf(path)
        k, mjd = spec_mjd(src)
        print(f'  {src}: spectrum epoch header {k}={mjd}')
        for band, a in lc.items():
            t, m = a[:, 0], a[:, 1]
            lo, hi = np.percentile(m, [2, 98])
            print(f'     {band}: N={len(m)}  median={np.median(m):.2f}  '
                  f'2-98% range = {hi-lo:.2f} mag  '
                  f'[{lo:.2f} .. {hi:.2f}]')

    print()
    print('=' * 74)
    print('3) Spectrum vs catalogue photometry (state at the spectrum epoch)')
    print('=' * 74)
    for src in PHOT:
        meta = jp.SOURCES[src]
        if meta['kind'] != 'desi':
            print(f'  {src}: relative flux, skipped')
            continue
        w, f, e = jp.load_spectrum(meta)          # dereddened
        # undo dereddening for a fair comparison with catalogue magnitudes
        alam = jp.ccm89_alam_av(w) * 3.1 * meta['ebv']
        fobs = f / 10 ** (0.4 * alam)
        b = PHOT[src]['bands']
        out = []
        for band, lam in BANDS.items():
            if band not in b:
                continue
            sm = synth_mag(w, fobs, lam)
            if sm is None:
                continue
            out.append(f'{band}: spec {sm:.2f} vs cat {b[band]["mag"]:.2f} '
                       f'({sm-b[band]["mag"]:+.2f})')
        print(f'  {src}: ' + ' | '.join(out))


if __name__ == '__main__':
    main()
