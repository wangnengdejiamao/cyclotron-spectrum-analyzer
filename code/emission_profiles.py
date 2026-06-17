#!/usr/bin/env python3
"""
emission_profiles.py — line-profile decomposition of the strongest
emission lines (H-alpha, H-beta) of the four targets.

Each line is modelled in velocity space as a local straight continuum
plus one or two Gaussians.  The two-component (narrow + broad) fit is
adopted over the single Gaussian when it is justified by an F-test;
otherwise the single Gaussian is shown.  In polars the narrow component
traces low-velocity / irradiated-donor gas and the broad component the
magnetically channelled accretion stream, although in these orbit-
averaged survey coadds the components are blurred by orbital motion.

Outputs:
  data/emission_profiles.json    per-line single/double Gaussian fits
  figures/emission_profiles.pdf  profile montage (rows: sources;
                                 cols: H-alpha, H-beta), in the style
                                 of phase-resolved polar line fits
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle
from joint_pipeline import SOURCES, load_spectrum

pubstyle.apply()
OUT = os.environ.get('CYC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_KMS = 299792.458
VWIN = 2200.0            # +/- velocity window fitted (km/s)
INSTR = {'desi': 90.0, 'lamost': 165.0}     # instrumental FWHM (km/s)

PLINES = {'Halpha': 6562.8, 'Hbeta': 4861.3}
LAB = {'Halpha': r'H$\alpha$', 'Hbeta': r'H$\beta$'}


def _g1s(v, a, v0, s):                       # single Gaussian, no continuum
    return a * np.exp(-0.5 * ((v - v0) / s) ** 2)


def _g2s(v, an, vn, sn, ab, vb, sb):         # narrow + broad, no continuum
    return (an * np.exp(-0.5 * ((v - vn) / sn) ** 2)
            + ab * np.exp(-0.5 * ((v - vb) / sb) ** 2))


def _despike(vv, ff, ee):
    """Reject narrow cosmic-ray / sky-residual spikes: points more than
    4x the local MAD above a 7-point median are dropped (two passes)."""
    from scipy.ndimage import median_filter
    for _ in range(2):
        med = median_filter(ff, size=7, mode='nearest')
        mad = 1.4826 * np.median(np.abs(ff - med)) + 1e-30
        keep = (ff - med) < 4.0 * mad
        vv, ff, ee = vv[keep], ff[keep], ee[keep]
    return vv, ff, ee


def _vbin(vv, ff, ee, bw=70.0):
    """Bin onto a uniform velocity grid (inverse-variance weighted) to
    suppress pixel noise; lines here are 400-900 km/s wide so a 70 km/s
    bin is loss-free."""
    edges = np.arange(-VWIN, VWIN + bw, bw)
    vb, fb, eb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (vv >= lo) & (vv < hi)
        if s.sum() == 0:
            continue
        wgt = 1.0 / ee[s] ** 2
        vb.append(np.sum(vv[s] * wgt) / wgt.sum())
        fb.append(np.sum(ff[s] * wgt) / wgt.sum())
        eb.append(np.sqrt(1.0 / wgt.sum()))
    return np.array(vb), np.array(fb), np.array(eb)


def fit_line(w, f, e, rest, meta_kind='desi'):
    v = (w - rest) / rest * C_KMS
    m = np.abs(v) <= VWIN
    if m.sum() < 12:
        return None
    vv, ff, ee = v[m], f[m], np.maximum(e[m], 1e-30)
    vv, ff, ee = _despike(vv, ff, ee)
    vv, ff, ee = _vbin(vv, ff, ee)
    if len(vv) < 12:
        return None
    # fixed local continuum: a straight line through the outer wings, so the
    # Gaussian must account for the whole line (core AND wings) rather than
    # letting a free continuum slope absorb the broad wings
    edge = np.abs(vv) > 0.62 * VWIN
    cc = np.polyfit(vv[edge], ff[edge], 1) if edge.sum() >= 4 \
        else np.array([0.0, np.median(ff)])
    resid = ff - np.polyval(cc, vv)
    pk = resid.max()
    # robust line centroid (flux-weighted first moment of the clipped core)
    core = np.abs(vv) < 1500.0
    rc = np.clip(resid[core], 0, None)
    vc = float(np.sum(vv[core] * rc) / np.sum(rc)) if rc.sum() > 0 else 0.0
    vc = float(np.clip(vc, -400, 400))
    # ---- single Gaussian on the continuum-subtracted line ----
    try:
        p1, _ = curve_fit(_g1s, vv, resid, p0=[pk, vc, 350.0], sigma=ee,
                          absolute_sigma=True, maxfev=20000,
                          bounds=([0, vc - 300, 90],
                                  [np.inf, vc + 300, VWIN]))
        chi1 = np.sum(((resid - _g1s(vv, *p1)) / ee) ** 2)
    except Exception:
        return None
    res = dict(v=vv.tolist(), f=ff.tolist(), e=ee.tolist(),
               cont=[float(cc[0]), float(cc[1])],
               single=dict(p=[float(x) for x in p1], chi2=float(chi1),
                           dof=len(vv) - 3),
               fwhm_single=float(2.3548 * abs(p1[2])),
               v0_single=float(p1[1]))
    # ---- double Gaussian (narrow + broad) on the residual ----
    try:
        p0d = [pk * 0.7, vc, 180.0, pk * 0.4, vc, 800.0]
        p2, _ = curve_fit(_g2s, vv, resid, p0=p0d, sigma=ee,
                          absolute_sigma=True, maxfev=40000,
                          bounds=([0, vc - 300, 90, 0, vc - 700, 350],
                                  [np.inf, vc + 300, 320, np.inf,
                                   vc + 700, VWIN]))
        chi2v = np.sum(((resid - _g2s(vv, *p2)) / ee) ** 2)
        dof2 = len(vv) - 6
        F = ((chi1 - chi2v) / 3) / (chi2v / dof2) if chi2v > 0 else 0.0
        pF = 1.0 - stats.f.cdf(F, 3, dof2) if F > 0 else 1.0
        an, vn, sn, ab, vb, sb = p2
        if abs(sn) > abs(sb):
            an, vn, sn, ab, vb, sb = ab, vb, sb, an, vn, sn
        fwn, fwb = 2.3548 * abs(sn), 2.3548 * abs(sb)
        fbroad = abs(ab * sb) / (abs(an * sn) + abs(ab * sb) + 1e-30)
        res['double'] = dict(p=[float(x) for x in (an, vn, sn, ab, vb, sb)],
                             chi2=float(chi2v), dof=dof2,
                             F=float(F), p_value=float(pF),
                             fwhm_narrow=float(fwn), fwhm_broad=float(fwb),
                             broad_flux_frac=float(fbroad))
        # Adopt the two-component model only when it is a *resolved*
        # narrow+broad profile, not a fit to an unresolved spike or to the
        # continuum curvature: require a significant F-test, a narrow
        # component clearly above the instrumental width, a broad component
        # carrying real flux but not spanning the whole window, and the two
        # widths well separated.
        instr = INSTR[meta_kind]
        res['adopt'] = ('double' if (pF < 0.005
                                     and fwn > max(2.5 * instr, 280.0)
                                     and 400 < fwb < 2300
                                     and fbroad > 0.15 and fwb > 1.8 * fwn)
                        else 'single')
    except Exception:
        res['adopt'] = 'single'
    return res


def analyse(src):
    meta = SOURCES[src]
    w, f, e = load_spectrum(meta)
    out = {}
    for key, rest in PLINES.items():
        if rest < w.min() + 30 or rest > w.max() - 30:
            continue
        r = fit_line(w, f, e, rest, meta_kind=meta['kind'])
        if r:
            out[key] = r
    return out


def main():
    allres = {s: analyse(s) for s in SOURCES}
    dump = {s: {k: {kk: vv for kk, vv in r.items()
                    if kk not in ('v', 'f', 'e')}
                for k, r in d.items()} for s, d in allres.items()}
    with open(f'{OUT}/data/emission_profiles.json', 'w') as fh:
        json.dump(dump, fh, indent=2)

    print(f"{'src':6s} {'line':6s} {'adopt':7s} {'FWHMn':>6s} {'FWHMb':>6s} "
          f"{'pF':>8s}")
    for s, d in allres.items():
        for k, r in d.items():
            if r['adopt'] == 'double':
                print(f"{s:6s} {k:6s} {r['adopt']:7s} "
                      f"{r['double']['fwhm_narrow']:6.0f} "
                      f"{r['double']['fwhm_broad']:6.0f} "
                      f"{r['double']['p_value']:8.1e}")
            else:
                print(f"{s:6s} {k:6s} {r['adopt']:7s} "
                      f"{r['fwhm_single']:6.0f}    ---  "
                      f"{r.get('double', {}).get('p_value', float('nan')):8.1e}")
    make_figure(allres)


def make_figure(allres):
    srcs = list(SOURCES)
    cols = list(PLINES)
    fig, axes = plt.subplots(len(srcs), len(cols), figsize=(6.8, 8.6))
    for i, s in enumerate(srcs):
        for j, key in enumerate(cols):
            ax = axes[i, j]
            r = allres[s].get(key)
            if r is None:
                ax.set_visible(False)
                continue
            vv = np.array(r['v']); ff = np.array(r['f'])
            vg = np.linspace(vv.min(), vv.max(), 600)
            cont = np.polyval(r['cont'], vg)
            ax.plot(vv, ff, 'k.', ms=2.2, label='data')
            if r['adopt'] == 'double':
                an, vn, sn, ab, vb, sb = r['double']['p']
                ax.plot(vg, cont + _g2s(vg, an, vn, sn, ab, vb, sb), '-',
                        color='crimson', lw=1.4, label='total')
                ax.plot(vg, cont + an * np.exp(-0.5 * ((vg - vn) / sn) ** 2),
                        '--', color='green', lw=1.0, label='narrow')
                ax.plot(vg, cont + ab * np.exp(-0.5 * ((vg - vb) / sb) ** 2),
                        '--', color='royalblue', lw=1.0, label='broad')
            else:
                ax.plot(vg, cont + _g1s(vg, *r['single']['p']), '-',
                        color='crimson', lw=1.4, label='total')
            ax.axvline(0, color='0.7', lw=0.5, ls=':')
            ax.set_xlim(-VWIN, VWIN)
            if i == 0:
                ax.set_title(LAB[key], fontsize=10)
            if j == 0:
                ax.set_ylabel(SOURCES[s]['name'].split()[0] + '\n'
                              + SOURCES[s]['name'].split()[1], fontsize=7)
            if i < len(srcs) - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6.5, loc='upper right', handlelength=1.4,
                       framealpha=0.9)
    for ax in axes[-1]:
        ax.set_xlabel(r'Velocity [km\,s$^{-1}$]', fontsize=8)
    fig.text(0.005, 0.5, r'$F_\lambda$ (dereddened; arbitrary for LAMOST)',
             rotation=90, va='center', fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.05,
                        hspace=0.12, wspace=0.22)
    fig.savefig(f'{OUT}/figures/emission_profiles.pdf')
    plt.close(fig)
    print('emission_profiles.pdf saved')


if __name__ == '__main__':
    main()
