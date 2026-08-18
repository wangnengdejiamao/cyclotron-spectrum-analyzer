#!/usr/bin/env python3
"""sed_figure.py — archival UV-to-mid-IR SEDs of the four targets.

Referee (general comment): "it would be beneficial to plot the SED of all 4
systems, in particular showing any available IR photometry (WISE, Spitzer,
SPHEREx etc), and placing upper limits on the donor type."

Each panel shows
  * the dereddened survey spectrum and the joint continuum+cyclotron model
    of Fig. 3, extrapolated outside the fitted 3950-9300 A window;
  * every archival photometric point between the far-UV and 5 um, with the
    epoch (catalogue) distinguished by the symbol, because none of it is
    simultaneous with the spectrum;
  * the ZTF g and r bright/faint range, which sets the scale of the
    accretion-state changes and therefore the systematic floor on any
    SED-based inference;
  * the most luminous donor still allowed by the mid-IR (Sect. donor
    limits), as the upper limit on the donor type.

Outputs figures/sed_panels.pdf and the numbers behind it.
"""
import csv as _csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle
import joint_pipeline as jp
from cyclotron_m2 import cal_cy_spec

pubstyle.apply()
C = pubstyle.COLORS
ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FR = os.path.join(ROOT, 'validation_candidates', 'full_refit', 'data')
BASE = os.environ.get('CYC_RAW', os.path.join(ROOT, 'data', 'raw'))
KEV_J = 1.602176634e-16

SHORT = {'J0005': 'DESI J0005+2941', 'J0022': 'DESI J0022+1340',
         'J0749': 'DESI J0749+3654', 'J0035': 'LAMOST J0035+4333'}
ORDER = ['J0005', 'J0022', 'J0749', 'J0035']

ZTF = {'J0005': f'{BASE}/ZTF_RA1.49466_DEC29.68439.csv',
       'J0022': f'{BASE}/ZTF_RA5.72177_DEC13.67799.csv',
       'J0749': f'{BASE}/ZTF_RA117.32130_DEC36.90776.csv',
       'J0035': (f'{BASE}/每个源分析/LAMOST_J003553.36+433341.4/'
                 'ZTF_RA8.9724_DEC43.5615.csv')}
ZTF_LAM = {'zg': 4746.0, 'zr': 6366.0, 'zi': 7829.0}

# band: (lambda_eff [A], zero point [Jy])   AB bands use 3631 Jy
ZP = {'FUV': (1528.0, 3631.0), 'NUV': (2271.0, 3631.0),
      'uSDSS': (3557.0, 3631.0), 'gSDSS': (4702.0, 3631.0),
      'rSDSS': (6175.0, 3631.0), 'iSDSS': (7490.0, 3631.0),
      'zSDSS': (8946.0, 3631.0),
      'gP1': (4849.0, 3631.0), 'rP1': (6201.0, 3631.0),
      'iP1': (7535.0, 3631.0), 'zP1': (8674.0, 3631.0),
      'yP1': (9628.0, 3631.0),
      'J': (12350.0, 1594.0), 'H': (16620.0, 1024.0), 'Ks': (21590.0, 666.7),
      'W1': (33526.0, 309.54), 'W2': (46028.0, 171.79),
      'W3': (115608.0, 31.674), 'W4': (220883.0, 8.363),
      'W1cw': (33526.0, 309.54), 'W2cw': (46028.0, 171.79)}

# marker style per catalogue
STYLE = {'GALEX': dict(marker='^', mfc='violet', ms=5.5),
         'SDSS16': dict(marker='o', mfc='none', ms=4.5),
         'PanSTARRS': dict(marker='o', mfc='0.25', ms=4.5),
         'AllWISE': dict(marker='s', mfc='gold', ms=5.0),
         'CatWISE': dict(marker='D', mfc='none', ms=4.2)}


def mag_to_flam(band, mag):
    lam, zp = ZP[band]
    fnu = zp * 1e-23 * 10 ** (-0.4 * mag)
    return lam, fnu * 2.998e18 / lam ** 2


def a_lambda(lam_AA, ebv):
    """A_lambda [mag]: CCM89 optically, Wang & Chen (2019) ratios in the IR."""
    lam = float(lam_AA)
    if lam < 33000.0:
        return float(jp.ccm89_alam_av(np.array([lam]))[0]) * 3.1 * ebv
    ratio = {33526.0: 0.039, 46028.0: 0.026}.get(round(lam, 0), 0.02)
    return ratio * 3.1 * ebv


def wide_koester():
    cache = f'{ROOT}/data/koester_cache_wide.npz'
    import scipy.interpolate as si
    if os.path.exists(cache):
        z = np.load(cache)
        g = jp.KoesterGrid.__new__(jp.KoesterGrid)
        g.teffs, g.loggs, g.wave = z['teffs'], z['loggs'], z['wave']
        g.interp = si.RegularGridInterpolator(
            (g.teffs, g.loggs, g.wave), z['cube'], method='linear',
            bounds_error=False, fill_value=0.0)
        return g
    g = jp.KoesterGrid(jp.KOESTER_DIR, wave_grid=np.arange(1150., 60000., 10.))
    np.savez_compressed(cache, teffs=g.teffs, loggs=g.loggs, wave=g.wave,
                        cube=g.interp.values)
    return g


def ztf_range(src):
    """2-98 per cent magnitude range per ZTF band (accretion states)."""
    out = {}
    path = ZTF.get(src)
    if not path or not os.path.exists(path):
        return out
    rows = {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            try:
                if int(r['catflags'], 0) != 0:
                    continue
                m, e = float(r['mag']), float(r['magerr'])
            except (ValueError, KeyError):
                continue
            if e <= 0 or e > 0.30:
                continue
            rows.setdefault(r['filtercode'], []).append(m)
    for b, v in rows.items():
        if len(v) > 30 and b in ZTF_LAM:
            lo, hi = np.percentile(v, [2, 98])
            out[b] = (ZTF_LAM[b], lo, hi)
    return out


def model_curves(src, wave, kg):
    """Model components on `wave`.  The Koester grid stops at 3 um, so the
    photosphere is continued as a Rayleigh-Jeans tail beyond 2.9 um."""
    js = json.load(open(os.path.join(FR, f'{src}_full_refit.json')))
    p = np.array(js['adopted_params'])
    a = np.array(js['adopted_amps'])
    band = np.linspace(3950.0, 9300.0, 400)

    wd = a[0] * kg.shape(wave, p[0], p[1])
    w_anchor = 29000.0
    f_anchor = float(a[0] * kg.shape(np.array([w_anchor]), p[0], p[1])[0])
    far = wave > w_anchor
    wd[far] = f_anchor * (w_anchor / wave[far]) ** 4      # Rayleigh-Jeans

    sp = a[1] * jp.planck(wave, p[2]) / jp.planck(band, p[2]).max()
    cyn = cal_cy_spec(band * 1e-10, p[4] * KEV_J, p[3] * 100.0,
                      np.deg2rad(p[5]), 10.0 ** p[6]).max()
    cy = a[2] * cal_cy_spec(wave * 1e-10, p[4] * KEV_J, p[3] * 100.0,
                            np.deg2rad(p[5]), 10.0 ** p[6]) / cyn
    return wd, sp, cy, p


def donor_points(src, dl):
    """Photometric points of the donor.

    Where the orbital period is known this is the main-sequence donor that
    the period-density relation demands; where it is not (J0022) it is the
    most luminous donor the mid-IR still permits."""
    r = dl[src]
    if r.get('ms_donor_at_period'):
        spt = r['ms_donor_at_period']['spt']
        kind = 'expected'
    else:
        spt = r['earliest_allowed']
        kind = 'limit'
    row = next(t for t in r['table'] if t['spt'] == spt)
    mu = r['W1']['dist_mod']
    # EEM absolute magnitudes of that type, scaled to the Roche radius
    seq = {}
    for line in open(f'{ROOT}/data/eem_dwarf_sequence.txt'):
        c = line.split()
        if len(c) > 30 and c[0] == spt:
            def g(i):
                try:
                    return float(c[i])
                except ValueError:
                    return np.nan
            seq = dict(M_J=g(21), M_Ks=g(22), ks_w1=g(23), w1_w2=g(24),
                       R=g(6), teff=g(1))
            break
    if not seq or not np.isfinite(seq['M_Ks']):
        return spt, [], None, kind
    dm = -5 * np.log10(row['R2'] / seq['R'])       # inflation to Roche radius
    mags = {'J': seq['M_J'] + dm + mu, 'Ks': seq['M_Ks'] + dm + mu}
    mags['W1'] = mags['Ks'] - (seq['ks_w1'] if np.isfinite(seq['ks_w1']) else 0)
    mags['W2'] = mags['W1'] - (seq['w1_w2'] if np.isfinite(seq['w1_w2']) else 0)
    pts = []
    for b, m in mags.items():
        lam, fl = mag_to_flam(b, m)
        pts.append((lam, fl))
    return spt, sorted(pts), seq.get('teff'), kind


def main():
    kg = wide_koester()
    phot = json.load(open(f'{ROOT}/data/sed_photometry.json'))
    dl = json.load(open(f'{ROOT}/data/donor_limits.json'))
    spx_path = f'{ROOT}/data/spherex/spherex_binned.json'
    spx = json.load(open(spx_path)) if os.path.exists(spx_path) else {}
    wave = np.unique(np.concatenate([np.geomspace(1250.0, 56000.0, 900),
                                 np.arange(3000.0, 30000.0, 8.0)]))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=True)
    report = {}
    for ax, src in zip(axes.ravel(), ORDER):
        meta = jp.SOURCES[src]
        ebv = meta['ebv']
        wd, sp, cy, p = model_curves(src, wave, kg)
        total = wd + sp + cy

        # LAMOST is in relative flux: anchor the model to the mean ZTF r
        scale = 1.0
        if src == 'J0035':
            lam_r, f_r = mag_to_flam('rSDSS', 19.05)
            scale = f_r / max(float(np.interp(lam_r, wave, total)), 1e-300)

        w_s, f_s, _ = jp.load_spectrum(meta)
        sel = (w_s > 3700) & (w_s < 9400)
        ax.plot(w_s[sel], w_s[sel] * f_s[sel] * scale, color='0.72', lw=0.3,
                zorder=1)
        ax.axvspan(3950.0, 9300.0, color='0.90', zorder=0)
        infit = (wave >= 3950.0) & (wave <= 9300.0)
        for arr, col, ls, lw in ((total, C['model'], '-', 1.3),
                                 (wd, C['wd'], '--', 0.8),
                                 (sp, C['spot'], '--', 0.8),
                                 (cy, C['cyc'], '-', 0.9)):
            y = wave * arr * scale
            ax.plot(wave, y, ls, color=col, lw=0.75 * lw, alpha=0.22,
                    zorder=3)
            ax.plot(wave[infit], y[infit], ls, color=col, lw=lw, zorder=4)

        # --- archival photometry, dereddened -------------------------
        vals = []
        for band, info in phot[src]['bands'].items():
            if band not in ZP:
                continue
            lam, flam = mag_to_flam(band, info['mag'])
            corr = 10 ** (0.4 * a_lambda(lam, ebv))
            y = lam * flam * corr
            st = STYLE.get(info['catalog'], dict(marker='o', mfc='k', ms=4))
            if info['upper_limit']:
                ax.plot(lam, y, marker='v', ms=4.0, mfc='none', mec='0.45',
                        mew=0.7, ls='none', zorder=5)
                continue
            e = info['err'] or 0.0
            ylo = y * (1 - 10 ** (-0.4 * e)) if e else 0.0
            yhi = y * (10 ** (0.4 * e) - 1) if e else 0.0
            ax.errorbar([lam], [y], yerr=[[ylo], [yhi]], ls='none',
                        ecolor='k', elinewidth=0.7, capsize=1.2,
                        mec='k', mew=0.7, zorder=6, **st)
            vals.append(y)

        # --- SPHEREx binned forced photometry ------------------------
        if src in spx:
            for b in spx[src]['bins']:
                lam = b['lam_um'] * 1e4
                fnu = b['f_uJy'] * 1e-29                 # erg/s/cm2/Hz
                efn = b['e_uJy'] * 1e-29
                y = lam * fnu * 2.998e18 / lam ** 2
                ey = lam * efn * 2.998e18 / lam ** 2
                corr = 10 ** (0.4 * a_lambda(lam, ebv))
                y, ey = y * corr, ey * corr
                if y - ey > 0:
                    ax.errorbar([lam], [y], yerr=[ey], ls='none',
                                marker='h', ms=4.0, mfc='mediumturquoise',
                                mec='teal', mew=0.6, ecolor='teal',
                                elinewidth=0.6, capsize=1.0, zorder=5)
                    vals.append(max(y - ey, 1e-30))
                else:                                    # <2 sigma: 2s limit
                    ax.plot(lam, y + 2 * ey, marker='v', ms=3.6, mfc='none',
                            mec='teal', mew=0.7, ls='none', zorder=5)

        # --- ZTF accretion-state range -------------------------------
        for b, (lam, m_lo, m_hi) in ztf_range(src).items():
            if b != 'zr':
                continue
            corr = 10 ** (0.4 * a_lambda(lam, ebv))
            y_hi = lam * mag_to_flam('rSDSS', m_lo)[1] * corr
            y_lo = lam * mag_to_flam('rSDSS', m_hi)[1] * corr
            ax.plot([lam, lam], [y_lo, y_hi], color='crimson', lw=3.0,
                    alpha=0.30, solid_capstyle='butt', zorder=7)
            ax.plot([lam], [np.sqrt(y_lo * y_hi)], marker='_', ms=6,
                    color='crimson', mew=1.2, ls='none', zorder=7)
            vals += [y_lo, y_hi]

        # --- donor upper limit ---------------------------------------
        spt, pts, teff, kind = donor_points(src, dl)
        if pts:
            lam_d = np.array([q[0] for q in pts])
            f_d = np.array([q[1] for q in pts])
            ax.plot(lam_d, lam_d * f_d, ':', marker='*', ms=6.5,
                    color='saddlebrown', mfc='peachpuff', mew=0.5, lw=0.9,
                    zorder=5)
            vals += list(lam_d * f_d)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1250, 56000)
        ref = np.nanpercentile(w_s[sel] * f_s[sel] * scale, 99.0)
        ax.set_ylim(max(min(vals) * 0.35, ref * 2e-3), ref * 4)
        ax.text(0.03, 0.94, SHORT[src], transform=ax.transAxes, va='top',
                fontsize=8.5)
        tag = (rf'donor at $P_{{\rm orb}}$: {spt[:-1]} V'
               if kind == 'expected' else
               rf'donor $\leq$ {spt[:-1]} V')
        ax.text(0.03, 0.055, tag, transform=ax.transAxes, va='bottom',
                fontsize=7.0, color='saddlebrown')
        report[src] = dict(donor=spt, kind=kind, donor_teff=teff)

    for ax in axes[1]:
        ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\lambda F_\lambda$ [erg s$^{-1}$ cm$^{-2}$]')

    handles = [
        Line2D([0], [0], color=C['model'], lw=1.3, label='total model'),
        Line2D([0], [0], color=C['wd'], lw=0.9, ls='--', label='WD'),
        Line2D([0], [0], color=C['spot'], lw=0.9, ls='--', label='hot spot'),
        Line2D([0], [0], color=C['cyc'], lw=1.0, label='cyclotron'),
        Line2D([0], [0], marker='^', color='k', mfc='violet', ls='none',
               ms=5, label='GALEX'),
        Line2D([0], [0], marker='o', color='k', mfc='0.25', ls='none',
               ms=4.5, label='Pan-STARRS'),
        Line2D([0], [0], marker='o', color='k', mfc='none', ls='none',
               ms=4.5, label='SDSS'),
        Line2D([0], [0], marker='s', color='k', mfc='gold', ls='none',
               ms=5, label='AllWISE'),
        Line2D([0], [0], marker='D', color='k', mfc='none', ls='none',
               ms=4.2, label='CatWISE'),
        Line2D([0], [0], marker='h', color='teal', mfc='mediumturquoise',
               ls='none', ms=4.5, label='SPHEREx (binned)'),
        Line2D([0], [0], color='crimson', lw=3.0, alpha=0.30,
               label='ZTF $r$ state range'),
        Line2D([0], [0], color='saddlebrown', lw=0.9, ls=':', marker='*',
               mfc='peachpuff', ms=6.5, label='donor (labelled)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=7, frameon=False,
               fontsize=6.4, bbox_to_anchor=(0.52, 1.005),
               handlelength=1.6, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.24, hspace=0.09, left=0.105, right=0.985,
                        top=0.885, bottom=0.095)
    fig.savefig(f'{ROOT}/figures/sed_panels.pdf')
    plt.close(fig)
    print('sed_panels.pdf saved')
    for s, r in report.items():
        print(f'  {s}: donor {r["donor"]} ({r["kind"]}, '
              f'Teff {r["donor_teff"]:.0f} K)')


if __name__ == '__main__':
    main()
