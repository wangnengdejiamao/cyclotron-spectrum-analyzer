#!/usr/bin/env python3
"""fetch_sed_photometry.py — archival UV-to-mid-IR photometry for the SED.

Referee (general comment): "it would be beneficial to plot the SED of all 4
systems, in particular showing any available IR photometry (WISE, Spitzer,
SPHEREx etc), and placing upper limits on the donor type."

This collects, per target and from the public archives, everything between
the far-UV and 5 um:

  GALEX GR6+7 (II/335)        FUV, NUV
  Pan-STARRS DR1 (II/349)     g r i z y
  SDSS DR16 (V/154)           u g r i z          (where covered)
  2MASS PSC (II/246)          J H Ks
  UKIDSS/UHS DR11 (II/384)    J K                (where covered)
  AllWISE (II/328)            W1 W2 W3 W4
  CatWISE2020 (II/365)        W1 W2              (deeper than AllWISE)
  Spitzer SEIP (II/368)       IRAC/MIPS          (where covered)

Written to data/sed_photometry.json with the full provenance so the SED
figure and the donor limits are reproducible.
"""
import json
import os
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

warnings.filterwarnings('ignore')

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'sed_photometry.json')

TARGETS = {
    'J0005': (1.49466, 29.68439),
    'J0022': (5.72177, 13.67799),
    'J0749': (117.32130, 36.90776),
    'J0035': (8.97235, 43.56152),
}

# catalogue: (VizieR id, radius", {band: (mag col, err col)}, system)
CATS = [
    ('GALEX', 'II/335/galex_ais', 5.0,
     {'FUV': ('FUVmag', 'e_FUVmag'), 'NUV': ('NUVmag', 'e_NUVmag')}, 'AB'),
    ('PanSTARRS', 'II/349/ps1', 2.0,
     {'gP1': ('gmag', 'e_gmag'), 'rP1': ('rmag', 'e_rmag'),
      'iP1': ('imag', 'e_imag'), 'zP1': ('zmag', 'e_zmag'),
      'yP1': ('ymag', 'e_ymag')}, 'AB'),
    ('SDSS16', 'V/154/sdss16', 2.0,
     {'uSDSS': ('umag', 'e_umag'), 'gSDSS': ('gmag', 'e_gmag'),
      'rSDSS': ('rmag', 'e_rmag'), 'iSDSS': ('imag', 'e_imag'),
      'zSDSS': ('zmag', 'e_zmag')}, 'AB'),
    ('2MASS', 'II/246/out', 3.0,
     {'J': ('Jmag', 'e_Jmag'), 'H': ('Hmag', 'e_Hmag'),
      'Ks': ('Kmag', 'e_Kmag')}, 'Vega'),
    ('UHS', 'II/384/uhsdr11', 2.0,
     {'JUH': ('Jmag', 'e_Jmag'), 'KUH': ('Kmag', 'e_Kmag')}, 'Vega'),
    ('AllWISE', 'II/328/allwise', 3.0,
     {'W1': ('W1mag', 'e_W1mag'), 'W2': ('W2mag', 'e_W2mag'),
      'W3': ('W3mag', 'e_W3mag'), 'W4': ('W4mag', 'e_W4mag')}, 'Vega'),
    ('CatWISE', 'II/365/catwise', 3.0,
     {'W1cw': ('W1mproPM', 'e_W1mproPM'),
      'W2cw': ('W2mproPM', 'e_W2mproPM')}, 'Vega'),
    ('SEIP', 'II/368/sstsl2', 3.0,
     {'I1': ('IRAC1', 'e_IRAC1'), 'I2': ('IRAC2', 'e_IRAC2'),
      'I3': ('IRAC3', 'e_IRAC3'), 'I4': ('IRAC4', 'e_IRAC4'),
      'M1': ('MIPS1', 'e_MIPS1')}, 'Vega'),
]


def query(cat_id, coord, radius, cols):
    v = Vizier(columns=['**', '+_r'], row_limit=20)
    try:
        res = v.query_region(coord, radius=radius * u.arcsec,
                             catalog=cat_id)
    except Exception as exc:
        return None, f'query error: {exc}'
    if not res or len(res) == 0:
        return None, 'no match'
    t = res[0]
    if '_r' in t.colnames:
        t = t[np.argsort(t['_r'])]
    return t, None


def main():
    out = {}
    for key, (ra, dec) in TARGETS.items():
        c = SkyCoord(ra, dec, unit='deg')
        entry = {'ra': ra, 'dec': dec, 'bands': {}, 'notes': {}}
        for name, cid, rad, cols, system in CATS:
            t, err = query(cid, c, rad, cols)
            if t is None:
                entry['notes'][name] = err
                print(f'{key:6s} {name:10s} -- {err}')
                continue
            row = t[0]
            sep = float(row['_r']) if '_r' in t.colnames else np.nan
            got = []
            for band, (mc, ec) in cols.items():
                if mc not in t.colnames:
                    continue
                m = row[mc]
                if m is None or (hasattr(m, 'mask') and m is np.ma.masked) \
                        or not np.isfinite(float(m if m is not None else np.nan)):
                    continue
                e = row[ec] if ec in t.colnames else None
                try:
                    e = float(e)
                    if not np.isfinite(e):
                        e = None
                except Exception:
                    e = None
                entry['bands'][band] = dict(mag=float(m), err=e,
                                            system=system, catalog=name,
                                            sep_arcsec=sep,
                                            upper_limit=(e is None))
                got.append(f'{band}={float(m):.2f}'
                           + (f'+-{e:.2f}' if e is not None else ' (UL)'))
            print(f'{key:6s} {name:10s} r={sep:.2f}"  ' + ', '.join(got))
        out[key] = entry
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwritten {OUT}')


if __name__ == '__main__':
    main()
