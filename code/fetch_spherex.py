#!/usr/bin/env python3
"""fetch_spherex.py — SPHEREx forced-photometry spectra for the four targets.

The referee asks for any available IR photometry including SPHEREx.  IRSA
only serves SPHEREx spectrophotometry as an on-demand forced-photometry
job (minutes per target); this script runs one per source and writes the
resulting table (or the failure reason) to data/spherex/.
"""
import json
import os
import sys
import traceback

# requires astro_toolbox on PYTHONPATH (see README)

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'spherex')
os.makedirs(OUT, exist_ok=True)

TARGETS = {
    'J0005': (1.49466, 29.68439),
    'J0022': (5.72177, 13.67799),
    'J0749': (117.32130, 36.90776),
    'J0035': (8.97235, 43.56152),
}

if __name__ == '__main__':
    from astro_toolbox.spherex import query_spectrum
    for key, (ra, dec) in TARGETS.items():
        path = os.path.join(OUT, f'{key}_spherex.json')
        if os.path.exists(path):
            print(f'{key}: already done', flush=True)
            continue
        print(f'=== {key} ({ra}, {dec}) ===', flush=True)
        try:
            spec = query_spectrum(ra, dec)
            payload = spec
            if hasattr(spec, 'to_dict'):
                payload = spec.to_dict()
            with open(path, 'w') as fh:
                json.dump(payload, fh, default=str, indent=1)
            print(f'{key}: written {path}', flush=True)
        except Exception:
            with open(os.path.join(OUT, f'{key}_spherex_FAILED.txt'), 'w') as fh:
                fh.write(traceback.format_exc())
            print(f'{key}: FAILED\n{traceback.format_exc()}', flush=True)
