#!/usr/bin/env python3
"""donor_limits.py — upper limits on the donor star from the archival IR.

Referee (general comment and Sect. 5): "place upper limits on the donor
type ... Can you exclude M-dwarf donors in any of these systems?"

Two independent constraints are combined.

(1) DYNAMICAL.  A Roche-lobe-filling donor has a mean density fixed by the
    orbital period alone,  rho_2 = 107 / P_hr^2 g cm^-3, i.e.

        R_2 = 0.234 (M_2/Msun)^(1/3) (P_orb/hr)^(2/3) Rsun .

    A donor of spectral type SpT has a main-sequence mass and radius; it
    can only fill its Roche lobe at period P if R_2(M_SpT, P) >= R_MS(SpT)
    (CV donors are inflated by ~0-30 per cent relative to the main
    sequence, never compressed).  Earlier types are excluded by the period
    alone, with no photometry involved.

(2) PHOTOMETRIC.  None of the four targets is detected by 2MASS or UKIDSS,
    so the reddest archival measurements are WISE W1 (3.35 um) and W2
    (4.60 um).  The observed mid-IR flux is the SUM of the donor, the white
    dwarf, the accretion-heated pole cap and the low cyclotron harmonics,
    and the systems change accretion state by 1.5-2.5 mag
    (sed_state_test.py), so the only statement valid in every state is

        F_donor(W1) <= min over epochs of F_observed(W1) .

    We adopt the FAINTEST of the AllWISE and CatWISE2020 W1 measurements,
    deredden it and convert it to an absolute magnitude with the Gaia
    distance.  A donor of type SpT filling its Roche lobe at P_orb has

        M_W1(donor) = M_W1(SpT) - 5 log10( R_2 / R_MS(SpT) ) ,

    i.e. the main-sequence surface brightness of that type over the larger
    Roche radius.  Types predicted brighter than the observed limit are
    excluded.

Spectral types, Teff, masses, radii and M_Ks, Ks-W1 come from the
Pecaut & Mamajek (2013) dwarf sequence (2022.04.16 revision);
M_W1 = M_Ks - (Ks-W1).  Writes data/donor_limits.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pipeline as jp

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
EEM = os.path.join(ROOT, 'data', 'eem_dwarf_sequence.txt')
PHOT = json.load(open(f'{ROOT}/data/sed_photometry.json'))

W1_LAM, W2_LAM = 33526.0, 46028.0        # A
# A_lambda / A_V in the WISE bands (Wang & Chen 2019); CCM89 is not defined
# this far into the IR.  With E(B-V) < 0.07 these corrections are < 0.01 mag.
A_RATIO = {'W1': 0.039, 'W2': 0.026}

PORB = {'J0005': 105.349, 'J0022': None, 'J0749': 80.4062, 'J0035': 143.765}
DIST = {'J0005': (604.0, 98.0), 'J0022': (583.0, 38.0),
        'J0749': (946.0, 135.0), 'J0035': (1000.0, 400.0)}
SHORT = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
         'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}
EBV = {s: jp.SOURCES[s]['ebv'] for s in SHORT}
# maximum radius inflation of a CV donor over the main sequence
MAX_INFLATION = 1.30


def load_eem():
    rows = []
    for line in open(EEM):
        if line.startswith('#') or not line.strip():
            continue
        c = line.split()
        if len(c) < 31 or not c[0].endswith('V'):
            continue
        if c[0][0] not in 'KML':
            continue

        def g(i):
            try:
                return float(c[i])
            except ValueError:
                return np.nan
        spt, teff, rad = c[0], g(1), g(6)
        m_ks, ks_w1, msun = g(22), g(23), g(30)
        if not (np.isfinite(m_ks) and np.isfinite(rad) and np.isfinite(msun)):
            continue
        rows.append(dict(spt=spt, teff=teff, R=rad, M=msun, M_Ks=m_ks,
                         M_W1=m_ks - (ks_w1 if np.isfinite(ks_w1) else 0.0)))
    return rows


def roche_radius(m2, p_hr):
    """Radius of a Roche-lobe-filling donor [Rsun] at period p_hr."""
    return 0.234 * np.asarray(m2, float) ** (1.0 / 3.0) * p_hr ** (2.0 / 3.0)


def donor_mass_from_period(seq, p_hr):
    """Latest main-sequence type that still fits inside its Roche lobe,
    i.e. the standard 'main-sequence donor' at this period."""
    best = None
    for r in seq:
        if roche_radius(r['M'], p_hr) >= r['R']:
            if best is None or r['M'] > best['M']:
                best = r
    return best


def main():
    seq = load_eem()
    out = {}
    for src in SHORT:
        b = PHOT[src]['bands']
        d, derr = DIST[src]
        p_hr = PORB[src] / 60.0 if PORB[src] else None
        res = dict(name=SHORT[src], distance_pc=d, distance_err_pc=derr,
                   porb_min=PORB[src])
        for band, keys, lam in (('W1', ('W1', 'W1cw'), W1_LAM),
                                ('W2', ('W2', 'W2cw'), W2_LAM)):
            mags = [(b[k]['mag'], b[k]['catalog']) for k in keys
                    if k in b and not b[k]['upper_limit']]
            if not mags:
                continue
            m_faint, cat = max(mags)
            m_bright = min(mags)[0]
            alam = A_RATIO[band] * 3.1 * EBV[src]
            mu = 5 * np.log10(d / 10.0)
            res[band] = dict(m_faintest=m_faint, from_catalog=cat,
                             m_brightest=m_bright,
                             epoch_spread=m_faint - m_bright,
                             extinction=alam, dist_mod=mu,
                             M_abs_limit=m_faint - alam - mu)
        M_lim = res['W1']['M_abs_limit']

        # ---- (1) dynamical limit ------------------------------------
        dyn_excl, ms_donor = [], None
        if p_hr:
            for r in seq:
                if roche_radius(r['M'], p_hr) < r['R'] / MAX_INFLATION:
                    dyn_excl.append(r['spt'])
            ms_donor = donor_mass_from_period(seq, p_hr)

        # ---- (2) photometric limit ----------------------------------
        phot_excl, allowed, table = [], [], []
        for r in seq:
            if p_hr:
                R2 = float(roche_radius(r['M'], p_hr))
                if R2 < r['R'] / MAX_INFLATION:
                    table.append(dict(spt=r['spt'], teff=r['teff'],
                                      mass=r['M'], R2=R2, M_W1_pred=np.nan,
                                      verdict='dynamically excluded'))
                    continue
            else:
                R2 = r['R']
            M_pred = r['M_W1'] - 5 * np.log10(R2 / r['R'])
            ok = M_pred > M_lim
            (allowed if ok else phot_excl).append(r['spt'])
            table.append(dict(spt=r['spt'], teff=r['teff'], mass=r['M'],
                              R2=R2, M_W1_pred=M_pred,
                              W1_pred=M_pred + res['W1']['dist_mod'],
                              verdict='allowed' if ok else
                              'excluded by W1'))
        res['dynamical_excluded'] = dyn_excl
        res['photometric_excluded'] = phot_excl
        res['allowed'] = allowed
        res['earliest_allowed'] = allowed[0] if allowed else None
        res['ms_donor_at_period'] = (
            dict(spt=ms_donor['spt'], mass=ms_donor['M'], teff=ms_donor['teff'],
                 R2=float(roche_radius(ms_donor['M'], p_hr)),
                 M_W1=ms_donor['M_W1'],
                 W1_pred=ms_donor['M_W1'] - 5 * np.log10(
                     roche_radius(ms_donor['M'], p_hr) / ms_donor['R'])
                 + res['W1']['dist_mod'])
            if ms_donor else None)
        res['table'] = table
        out[src] = res

    for src, r in out.items():
        print(f'=== {r["name"]} ===')
        print(f'  d = {r["distance_pc"]:.0f} +- {r["distance_err_pc"]:.0f} pc,'
              f'  P_orb = {r["porb_min"] if r["porb_min"] else "unknown"} min')
        for band in ('W1', 'W2'):
            if band in r:
                q = r[band]
                print(f'  {band}: faintest {q["m_faintest"]:.2f} '
                      f'({q["from_catalog"]}), epoch spread '
                      f'{q["epoch_spread"]:+.2f} mag, A={q["extinction"]:.3f}'
                      f'  =>  donor M_{band} > {q["M_abs_limit"]:.2f}')
        if r['dynamical_excluded']:
            print(f'  dynamically excluded (donor cannot fit the Roche lobe):'
                  f' {r["dynamical_excluded"][0]}-{r["dynamical_excluded"][-1]}')
        if r['photometric_excluded']:
            print(f'  excluded by W1 (would be too bright): '
                  f'{r["photometric_excluded"][0]}'
                  f'-{r["photometric_excluded"][-1]}')
        else:
            print('  excluded by W1: none (all surviving types are fainter '
                  'than the limit)')
        print(f'  earliest allowed donor: {r["earliest_allowed"]}')
        m = r['ms_donor_at_period']
        if m:
            print(f'  expected main-sequence donor at this period: {m["spt"]}'
                  f' (M={m["mass"]:.2f} Msun, Teff={m["teff"]:.0f} K, '
                  f'R={m["R2"]:.3f} Rsun) => W1 = {m["W1_pred"]:.2f}, i.e. '
                  f'{m["W1_pred"]-r["W1"]["m_faintest"]:+.2f} mag relative to '
                  f'the observed flux')
        print()

    with open(f'{ROOT}/data/donor_limits.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f'written {ROOT}/data/donor_limits.json')


if __name__ == '__main__':
    main()
