#!/usr/bin/env python3
"""
lightcurve_analysis.py — ZTF light-curve characterisation for the 4 sources.

Per source:
  1. quality cuts (catflags==0, magerr<0.30), per band (zg/zr/zi);
  2. long-term light curve and accretion-state inspection: rolling median,
     identification of low-state intervals (>0.8 mag below the bright-state
     median); slow state changes removed by per-band rolling-median
     detrending BEFORE any period search (established workflow: never run
     the period search on data containing state-change/outburst trends);
  3. Lomb-Scargle on the detrended data, per band and combined,
     frequency grid 2-40 cycles/day; spectral window function; bootstrap
     false-alarm levels; explicit alias map (f +/- 1, 2 c/d, 2f, f/2);
  4. phase-folded light curves at the adopted period (phase-binned medians).

Folded plots are written for user review before the paper adopts them.
"""

import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import pubstyle
pubstyle.apply()

warnings.filterwarnings('ignore')

BASE = os.path.join(os.environ.get('CYC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
OUT = os.environ.get('CYC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    'J0005': dict(name='DESI J000558.72+294103.8',
                  csv=f'{BASE}/ZTF_RA1.49466_DEC29.68439.csv'),
    'J0022': dict(name='DESI J002253.23+134040.7',
                  csv=f'{BASE}/ZTF_RA5.72177_DEC13.67799.csv'),
    'J0749': dict(name='DESI J074917.11+365427.9',
                  csv=f'{BASE}/ZTF_RA117.32130_DEC36.90776.csv',
                  fold_2p=True),   # catalogue P_orb = 0.0558375 d = 2x LS peak
    'J0035': dict(name='LAMOST J003553.36+433341.4',
                  csv=f'{BASE}/ZTF_RA8.9724_DEC43.5615.csv'),
}

FMIN, FMAX = 2.0, 40.0      # cycles / day  (36 min .. 12 h)


def load_ztf(csv):
    import csv as _csv
    rows = {'zg': [], 'zr': [], 'zi': []}
    with open(csv) as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            try:
                flags = int(r['catflags'], 0)
                mag = float(r['mag'])
                err = float(r['magerr'])
                hjd = float(r['hjd'])
            except (ValueError, KeyError):
                continue
            if flags != 0 or err <= 0 or err > 0.30:
                continue
            band = r['filtercode']
            if band in rows:
                rows[band].append((hjd, mag, err))
    out = {}
    for b, lst in rows.items():
        if len(lst) >= 20:
            a = np.array(sorted(lst))
            out[b] = dict(t=a[:, 0], m=a[:, 1], e=a[:, 2])
    return out


def rolling_median(t, m, window_days=25.0):
    """Centred rolling median; robust to gaps (uses points within window)."""
    out = np.empty_like(m)
    for i, ti in enumerate(t):
        sel = np.abs(t - ti) < window_days / 2
        out[i] = np.median(m[sel])
    return out


def detect_states(t, m, bright_quantile=0.25, low_offset=0.8):
    """Bright-state reference = 25th-percentile magnitude (brightest quartile).
    Points whose rolling median is >low_offset mag fainter are 'low state'."""
    ref = np.quantile(m, bright_quantile)
    rm = rolling_median(t, m, 40.0)
    low = rm > ref + low_offset
    return ref, rm, low


def bootstrap_fap(t, dm, e, freq, n_boot=200, seed=3):
    rng = np.random.default_rng(seed)
    maxp = np.empty(n_boot)
    for i in range(n_boot):
        sh = rng.permutation(dm)
        p = LombScargle(t, sh, e).power(freq)
        maxp[i] = np.nanmax(p)
    return np.quantile(maxp, [0.95, 0.99, 0.999])


def alias_partners(f):
    """Common ZTF aliases of frequency f (cycles/day)."""
    out = {}
    for k in (1.0, 2.0):
        out[f'f+{k:.0f}'] = f + k
        if f - k > 0:
            out[f'f-{k:.0f}'] = f - k
    out['2f'] = 2 * f
    out['f/2'] = f / 2
    out['1-day beat (1-f)'] = abs(1 - f)
    return out


def analyse(src):
    meta = SOURCES[src]
    bands = load_ztf(meta['csv'])
    print(f'=== {src}: {meta["name"]} ===')
    for b, d in bands.items():
        print(f'  {b}: {len(d["t"])} pts, '
              f'{d["t"].max()-d["t"].min():.0f} d baseline, '
              f'<mag>={np.median(d["m"]):.2f}')

    figdir = os.path.join(OUT, 'lightcurves')
    os.makedirs(figdir, exist_ok=True)

    # ---------------- long-term LC + state detection ----------------
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colors = dict(zg='seagreen', zr='crimson', zi='goldenrod')
    state_info = {}
    detrended = {}
    for b, d in bands.items():
        ref, rm, low = detect_states(d['t'], d['m'])
        state_info[b] = dict(bright_ref=float(ref),
                             n_low=int(low.sum()), n_tot=len(low))
        ax.errorbar(d['t'] - 2458000, d['m'], yerr=d['e'], fmt='.', ms=3,
                    lw=0.5, color=colors[b], alpha=0.55, label=b)
        ax.plot(d['t'] - 2458000, rm, '-', color=colors[b], lw=1.2,
                alpha=0.9)
        # detrend with rolling median (removes state changes / slow trends)
        detrended[b] = dict(t=d['t'], dm=d['m'] - rm, e=d['e'], low=low)
    ax.invert_yaxis()
    ax.set_xlabel('HJD - 2458000  [d]')
    ax.set_ylabel('mag')
    ax.legend(fontsize=8)
    ax.text(0.01, 0.95, meta['name'], transform=ax.transAxes, va='top', fontsize=9)
    fig.savefig(f'{figdir}/{src}_longterm.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---------------- period search ----------------
    freq = np.linspace(FMIN, FMAX, 200000)
    results = {}
    fig, axes = plt.subplots(len(bands) + 1, 1,
                             figsize=(9, 2.3 * (len(bands) + 1)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    band_peaks = {}
    for i, (b, d) in enumerate(detrended.items()):
        ls = LombScargle(d['t'], d['dm'], d['e'])
        p = ls.power(freq)
        fap95, fap99, fap999 = bootstrap_fap(d['t'], d['dm'], d['e'],
                                             freq[::20])
        ipk = np.nanargmax(p)
        fpk = freq[ipk]
        band_peaks[b] = dict(f=float(fpk), P_min=float(1440.0 / fpk),
                             power=float(p[ipk]), fap99=float(fap99),
                             fap999=float(fap999))
        ax = axes[i]
        ax.plot(freq, p, lw=0.5, color=colors[b])
        ax.axhline(fap99, ls=':', color='gray', lw=0.8)
        ax.axhline(fap999, ls='--', color='gray', lw=0.8)
        ax.plot(fpk, p[ipk], 'v', color='k', ms=6)
        ax.set_ylabel(f'LS power ({b})', fontsize=8)
        ax.text(0.99, 0.9, f'peak: {fpk:.4f} c/d = {1440/fpk:.2f} min',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        print(f'  {b}: peak f={fpk:.5f} c/d  P={1440/fpk:.3f} min '
              f'power={p[ipk]:.3f} (99% FAP={fap99:.3f}, 99.9%={fap999:.3f})')

    # combined (bands merged after per-band normalisation by amplitude)
    tt = np.concatenate([d['t'] for d in detrended.values()])
    mm = np.concatenate([d['dm'] / max(np.std(d['dm']), 1e-3)
                         for d in detrended.values()])
    ee = np.concatenate([d['e'] / max(np.std(d['dm']), 1e-3)
                         for d in detrended.values()])
    ls = LombScargle(tt, mm, ee)
    p = ls.power(freq)
    ipk = np.nanargmax(p)
    fpk = freq[ipk]
    fap95, fap99, fap999 = bootstrap_fap(tt, mm, ee, freq[::20])
    ax = axes[-1]
    ax.plot(freq, p, lw=0.5, color='navy')
    ax.axhline(fap99, ls=':', color='gray', lw=0.8)
    ax.axhline(fap999, ls='--', color='gray', lw=0.8)
    ax.plot(fpk, p[ipk], 'v', color='k', ms=6)
    ax.set_ylabel('LS power (g+r)', fontsize=8)
    ax.set_xlabel('frequency  [cycles/day]')
    ax.text(0.99, 0.9, f'peak: {fpk:.4f} c/d = {1440/fpk:.2f} min',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    print(f'  combined: f={fpk:.5f} c/d  P={1440/fpk:.3f} min '
          f'power={p[ipk]:.3f} (99% FAP={fap99:.3f})')

    # top-5 combined peaks for alias bookkeeping
    from scipy.signal import find_peaks
    pk_idx, _ = find_peaks(p, height=fap99, distance=2000)
    top = sorted(pk_idx, key=lambda i: -p[i])[:5]
    top_peaks = [dict(f=float(freq[i]), P_min=float(1440 / freq[i]),
                      power=float(p[i])) for i in top]
    print('  top combined peaks:')
    for tp in top_peaks:
        print(f'    f={tp["f"]:8.4f}  P={tp["P_min"]:7.2f} min  '
              f'power={tp["power"]:.3f}')
    fig.savefig(f'{figdir}/{src}_periodogram.pdf', bbox_inches='tight')
    plt.close(fig)

    # spectral window function
    figw, axw = plt.subplots(figsize=(8, 2.4))
    pw = LombScargle(tt, np.ones_like(mm), fit_mean=False,
                     center_data=False).power(freq)
    axw.plot(freq, pw, lw=0.5, color='k')
    axw.set_xlabel('frequency [cycles/day]')
    axw.set_ylabel('window power')
    figw.savefig(f'{figdir}/{src}_window.pdf', bbox_inches='tight')
    plt.close(figw)

    # alias map for the adopted peak
    aliases = alias_partners(fpk)
    alias_power = {}
    for k, fa in aliases.items():
        if FMIN <= fa <= FMAX:
            ia = np.argmin(np.abs(freq - fa))
            j0 = max(ia - 1500, 0)
            j1 = min(ia + 1500, len(freq))
            alias_power[k] = dict(f=float(fa),
                                  power=float(np.nanmax(p[j0:j1])))

    # band-to-band consistency
    fs = [bp['f'] for bp in band_peaks.values()]
    consistent = (max(fs) - min(fs)) * 1440 / fpk**2 < 0.5 if len(fs) > 1 \
        else None

    # ---------------- folded light curve ----------------
    P_day = 1.0 / fpk
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for b, d in detrended.items():
        ph = ((d['t'] / P_day) % 1.0)
        ph2 = np.concatenate([ph, ph + 1])
        dm2 = np.concatenate([d['dm'], d['dm']])
        axs[0].plot(ph2, dm2, '.', ms=2.5, color=colors[b], alpha=0.45,
                    label=b)
        # phase-binned medians
        nb = 20
        cent, med = [], []
        for k in range(nb):
            m = (ph >= k / nb) & (ph < (k + 1) / nb)
            if m.sum() >= 3:
                cent.append((k + 0.5) / nb)
                med.append(np.median(d['dm'][m]))
        cent = np.array(cent)
        med = np.array(med)
        axs[0].plot(np.concatenate([cent, cent + 1]),
                    np.concatenate([med, med]), 'o-', ms=4,
                    color=colors[b], lw=1.4, mec='k', mew=0.4)
    axs[0].invert_yaxis()
    axs[0].set_xlabel('phase')
    axs[0].set_ylabel(r'$\Delta$mag (detrended)')
    axs[0].set_title(f'P = {1440*P_day:.3f} min', fontsize=10)
    axs[0].legend(fontsize=8)

    # second panel: fold at the strongest alias (for comparison/review)
    alt = max(alias_power.items(), key=lambda kv: kv[1]['power']) \
        if alias_power else None
    if alt is not None:
        fa = alt[1]['f']
        Pa = 1.0 / fa
        for b, d in detrended.items():
            ph = ((d['t'] / Pa) % 1.0)
            axs[1].plot(np.concatenate([ph, ph + 1]),
                        np.concatenate([d['dm'], d['dm']]), '.', ms=2.5,
                        color=colors[b], alpha=0.45)
        axs[1].invert_yaxis()
        axs[1].set_xlabel('phase')
        axs[1].set_title(f'strongest alias {alt[0]}: '
                         f'P = {1440*Pa:.3f} min', fontsize=10)
    axs[0].text(0.02, 0.95, meta['name'], transform=axs[0].transAxes, va='top', fontsize=9)
    fig.savefig(f'{figdir}/{src}_folded.pdf', bbox_inches='tight')
    plt.close(fig)

    # orbital-period fold (2x the LS peak) for double-humped systems
    if meta.get('fold_2p'):
        Pl = 2.0 * P_day
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for b, d in detrended.items():
            ph = ((d['t'] / Pl) % 1.0)
            ax.plot(np.concatenate([ph, ph + 1]),
                    np.concatenate([d['dm'], d['dm']]), '.', ms=2.5,
                    color=colors[b], alpha=0.4, label=b)
            nb = 30
            cent, med = [], []
            for k in range(nb):
                m = (ph >= k / nb) & (ph < (k + 1) / nb)
                if m.sum() >= 3:
                    cent.append((k + 0.5) / nb)
                    med.append(np.median(d['dm'][m]))
            cent, med = np.array(cent), np.array(med)
            ax.plot(np.concatenate([cent, cent + 1]),
                    np.concatenate([med, med]), 'o-', ms=4,
                    color=colors[b], lw=1.4, mec='k', mew=0.4)
        ax.invert_yaxis()
        ax.set_xlabel('phase')
        ax.set_ylabel(r'$\Delta$mag')
        ax.set_title(f'{meta["name"]} folded at '
                     f'$2P_{{\\rm phot}}$ = {2880*P_day:.3f} min',
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig(f'{figdir}/{src}_folded_2P.pdf', bbox_inches='tight')
        plt.close(fig)

    out = dict(source=src, name=meta['name'],
               n_points={b: len(d['t']) for b, d in bands.items()},
               states=state_info,
               band_peaks=band_peaks,
               combined_peak=dict(f=float(fpk), P_min=float(1440 / fpk),
                                  power=float(p[ipk]),
                                  fap99=float(fap99), fap999=float(fap999)),
               top_peaks=top_peaks,
               alias_power=alias_power,
               band_consistent=bool(consistent) if consistent is not None
               else None)
    with open(os.path.join(OUT, 'data', f'{src}_lc_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print()
    return out


if __name__ == '__main__':
    import sys as _sys
    os.makedirs(os.path.join(OUT, 'data'), exist_ok=True)
    targets = _sys.argv[1:] if len(_sys.argv) > 1 else list(SOURCES)
    for s in targets:
        analyse(s)
