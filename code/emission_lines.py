#!/usr/bin/env python3
"""
emission_lines.py — measure the optical emission lines of the four
targets (Balmer series, He I, and the high-excitation He II 4686 line)
and derive the standard magnetic-CV diagnostics:

  * equivalent widths (EW) and, for the flux-calibrated DESI spectra,
    line fluxes and luminosities;
  * the He II 4686 / H-beta EW ratio and EW(H-beta), the Silber (1992)
    magnetic-activity diagnostic;
  * the Balmer decrement (Halpha/H-beta, Hgamma/H-beta), which is
    inverted (optically thick) in many polars;
  * the emission-line FWHM (a kinematic width, here orbit-averaged and
    instrument-broadened).

Each line is measured by fitting a straight local continuum to two
side-bands and (a) directly integrating the continuum-subtracted line
within a window (EW, flux), and (b) fitting a single Gaussian for the
width.  Uncertainties are propagated from the per-pixel errors by a
Monte-Carlo resampling.

Outputs:
  data/emission_lines.json   per-source measurements and ratios
  data/emission_lines.csv    flat table
  figures/emission_lines.pdf line-profile montage
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle
from joint_pipeline import SOURCES, load_spectrum

pubstyle.apply()
OUT = os.environ.get('CYC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_KMS = 299792.458

# Instrumental FWHM (km/s) used to deconvolve the line widths.
# DESI R~2500-4000 over the optical; LAMOST low-res R~1800.
INSTR_FWHM = {'desi': 90.0, 'lamost': 165.0}

# line: (rest wavelength, line half-window, [(c1lo,c1hi),(c2lo,c2hi)] sidebands)
LINES = {
    'Hbeta':    (4861.3, 30.0, [(4775, 4805), (4905, 4945)]),
    'HeII4686': (4685.7, 22.0, [(4610, 4638), (4730, 4762)]),
    'Hgamma':   (4340.5, 26.0, [(4250, 4288), (4392, 4430)]),
    'Hdelta':   (4101.7, 22.0, [(4035, 4062), (4150, 4182)]),
    'HeI4471':  (4471.5, 16.0, [(4420, 4448), (4498, 4525)]),
    'Halpha':   (6562.8, 34.0, [(6445, 6485), (6650, 6690)]),
    'HeI6678':  (6678.2, 18.0, [(6605, 6640), (6705, 6742)]),
    'HeI5876':  (5875.6, 13.0, [(5805, 5840), (5905, 5945)]),  # Na D blend on red
}
# Pretty labels for the table / figure.  Keep the ion labels explicit:
# Matplotlib text extraction can mangle small-caps Roman numerals.
LABEL = {'Hbeta': r'H$\beta$', 'HeII4686': r'He II $\lambda4686$',
         'Hgamma': r'H$\gamma$', 'Hdelta': r'H$\delta$',
         'HeI4471': r'He I $\lambda4471$', 'Halpha': r'H$\alpha$',
         'HeI6678': r'He I $\lambda6678$', 'HeI5876': r'He I $\lambda5876$'}


def _gauss(x, a, x0, sig, c0, c1):
    return c0 + c1 * (x - x0) + a * np.exp(-0.5 * ((x - x0) / sig) ** 2)


def measure_line(w, f, e, rest, hw, sidebands, n_mc=400):
    """Return dict with EW, flux, FWHM and errors for one line.
    EW>0 for emission. flux in the spectrum's flux units * Angstrom."""
    # local continuum from the two sidebands (robust straight line)
    cm = np.zeros_like(w, bool)
    for lo, hi in sidebands:
        cm |= (w >= lo) & (w <= hi)
    if cm.sum() < 4:
        return None
    A = np.vstack([np.ones(cm.sum()), w[cm] - rest]).T
    coef, *_ = np.linalg.lstsq(A, f[cm], rcond=None)
    cont = coef[0] + coef[1] * (w - rest)
    clevel = coef[0]                       # continuum at the rest wavelength
    if not np.isfinite(clevel) or clevel <= 0:
        return None
    lm = (w >= rest - hw) & (w <= rest + hw)
    if lm.sum() < 5:
        return None
    wl, resid, el = w[lm], (f - cont)[lm], e[lm]
    dl = np.gradient(wl)
    flux = np.sum(resid * dl)
    ew = flux / clevel                     # >0 emission
    # Gaussian for the width (free centroid).  Normalize the flux first:
    # DESI fluxes are ~1e-15, too small for curve_fit's default tolerances,
    # which otherwise "converges" immediately and leaves the width at p0.
    # The width is in wavelength units, independent of the y-scale.
    try:
        from scipy.optimize import curve_fit
        scale = abs(clevel)
        yl, eyl = f[lm] / scale, np.maximum(el, 1e-30) / scale
        a0 = max((yl - np.median(yl)).max(), 1e-3)
        p0 = [a0, rest, 4.0, np.median(yl), coef[1] / scale]
        popt, _ = curve_fit(_gauss, wl, yl, p0=p0, sigma=eyl,
                            absolute_sigma=False, maxfev=10000,
                            bounds=([0, rest - hw, 0.5, -np.inf, -np.inf],
                                    [np.inf, rest + hw, hw, np.inf, np.inf]))
        sig, cen = popt[2], popt[1]
        fwhm_obs = 2.3548 * sig / cen * C_KMS
        vshift = (cen - rest) / rest * C_KMS
    except Exception:
        sig = np.nan
        fwhm_obs, vshift = np.nan, np.nan
    # Monte-Carlo errors on EW and flux
    ews, fluxes = [], []
    for _ in range(n_mc):
        fm = f + np.random.normal(0, 1, len(f)) * e
        cf, *_ = np.linalg.lstsq(A, fm[cm], rcond=None)
        ct = cf[0] + cf[1] * (w - rest)
        cl = cf[0]
        if cl <= 0:
            continue
        fx = np.sum((fm - ct)[lm] * dl)
        fluxes.append(fx)
        ews.append(fx / cl)
    ew_err = float(np.std(ews)) if ews else np.nan
    flux_err = float(np.std(fluxes)) if fluxes else np.nan
    return dict(rest=rest, ew=float(ew), ew_err=ew_err,
                flux=float(flux), flux_err=flux_err,
                cont=float(clevel), fwhm_obs=float(fwhm_obs),
                vshift=float(vshift))


def analyse(src):
    meta = SOURCES[src]
    w, f, e = load_spectrum(meta)
    res = {}
    for key, (rest, hw, sb) in LINES.items():
        if rest < w.min() + 20 or rest > w.max() - 20:
            continue
        m = measure_line(w, f, e, rest, hw, sb)
        if m is not None:
            res[key] = m
    # diagnostics
    out = dict(name=meta['name'], kind=meta['kind'], lines=res)
    def ew(k):
        return res[k]['ew'] if k in res else np.nan
    def ewe(k):
        return res[k]['ew_err'] if k in res else np.nan
    hb, he2 = ew('Hbeta'), ew('HeII4686')
    out['EW_Hbeta'] = hb
    out['HeII_Hbeta'] = he2 / hb if hb and np.isfinite(hb) else np.nan
    # propagate ratio error
    if np.isfinite(he2) and np.isfinite(hb) and hb != 0:
        out['HeII_Hbeta_err'] = abs(he2 / hb) * np.hypot(
            ewe('HeII4686') / he2 if he2 else np.nan, ewe('Hbeta') / hb)
    out['silber_magnetic'] = bool(np.isfinite(hb) and hb > 20
                                  and np.isfinite(out['HeII_Hbeta'])
                                  and out['HeII_Hbeta'] > 0.4)
    # Balmer decrement from fluxes (DESI) or EW (relative-flux LAMOST)
    use = 'flux' if meta['kind'] == 'desi' else 'ew'
    def val(k):
        return res[k][use] if k in res else np.nan
    out['balmer_meas'] = use
    out['Halpha_Hbeta'] = val('Halpha') / val('Hbeta') \
        if 'Halpha' in res and 'Hbeta' in res else np.nan
    out['Hgamma_Hbeta'] = val('Hgamma') / val('Hbeta') \
        if 'Hgamma' in res and 'Hbeta' in res else np.nan
    out['balmer_inverted'] = bool(np.isfinite(out['Halpha_Hbeta'])
                                  and out['Halpha_Hbeta'] < 2.0)
    # line luminosities (DESI only; flux in 1e-17 already scaled to erg/s/cm2/A)
    if meta['kind'] == 'desi' and meta['dist_pc']:
        d_cm = meta['dist_pc'] * 3.0857e18
        fourpid2 = 4 * np.pi * d_cm ** 2
        for k in res:
            res[k]['lum'] = res[k]['flux'] * fourpid2   # erg/s
            res[k]['lum_err'] = res[k]['flux_err'] * fourpid2
    # instrumental deconvolution of FWHM
    instr = INSTR_FWHM[meta['kind']]
    for k in res:
        fo = res[k]['fwhm_obs']
        res[k]['fwhm_kms'] = float(np.sqrt(max(fo ** 2 - instr ** 2, 0.0))) \
            if np.isfinite(fo) else np.nan
    return out


def main():
    results = {s: analyse(s) for s in SOURCES}
    with open(f'{OUT}/data/emission_lines.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    # flat CSV
    rows = ['source,line,rest_A,EW_A,EW_err,flux,flux_err,FWHM_kms,vshift_kms']
    for s, r in results.items():
        for k, m in r['lines'].items():
            rows.append(f"{s},{k},{m['rest']:.1f},{m['ew']:.2f},"
                        f"{m['ew_err']:.2f},{m['flux']:.3e},{m['flux_err']:.3e},"
                        f"{m.get('fwhm_kms', float('nan')):.0f},"
                        f"{m['vshift']:.0f}")
    open(f'{OUT}/data/emission_lines.csv', 'w').write('\n'.join(rows) + '\n')

    # console summary
    print(f"{'src':6s} {'EW(Hb)':>8s} {'HeII/Hb':>9s} {'Ha/Hb':>7s} "
          f"{'silber':>7s} {'inv.Balmer':>10s}")
    for s, r in results.items():
        print(f"{s:6s} {r['EW_Hbeta']:8.1f} {r['HeII_Hbeta']:9.2f} "
              f"{r['Halpha_Hbeta']:7.2f} {str(r['silber_magnetic']):>7s} "
              f"{str(r['balmer_inverted']):>10s}")

    make_figure(results)


def make_figure(results):
    srcs = list(SOURCES)
    fig, axes = plt.subplots(len(srcs), 2, figsize=(7.2, 8.0),
                             gridspec_kw=dict(width_ratios=[1.7, 1.0]))
    blue_lines = [('Hdelta', 4101.7), ('Hgamma', 4340.5),
                  ('HeI4471', 4471.5), ('HeII4686', 4685.7),
                  ('Hbeta', 4861.3)]
    red_lines = [('Halpha', 6562.8), ('HeI6678', 6678.2)]
    line_color = '#b2182b'
    label_box = dict(facecolor='white', edgecolor='none', alpha=0.78, pad=0.25)
    for i, s in enumerate(srcs):
        meta = SOURCES[s]
        w, f, e = load_spectrum(meta)
        axL, axR = axes[i, 0], axes[i, 1]
        for ax, (lo, hi), marks in [(axL, (4050, 4935), blue_lines),
                                    (axR, (6440, 6750), red_lines)]:
            m = (w >= lo) & (w <= hi)
            ax.plot(w[m], f[m], color='0.18', lw=0.55)
            ymax = np.nanpercentile(f[m], 99.3)
            ymin = np.nanpercentile(f[m], 1.0)
            pad = 0.12 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlim(lo, hi)
            for key, lam in marks:
                if key in results[s]['lines']:
                    ax.axvline(lam, color=line_color, lw=0.6, ls=':', alpha=0.65)
                    if i == 0:
                        ax.text(lam, 0.985, LABEL[key],
                                transform=ax.get_xaxis_transform(), fontsize=6.3,
                                ha='center', va='top', color=line_color,
                                bbox=label_box)
        axL.text(0.015, 0.93, meta['name'], transform=axL.transAxes,
                 fontsize=7.5, va='top',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.4))
        hb = results[s]['EW_Hbeta']
        rr = results[s]['HeII_Hbeta']
        axR.text(0.96, 0.93,
                 f"EW(H$\\beta$) = {hb:.0f} $\\AA$\nHe II/H$\\beta$ = {rr:.2f}",
                 transform=axR.transAxes, fontsize=6.5, va='top', ha='right',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=0.5))
        if i < len(srcs) - 1:
            axL.set_xticklabels([]); axR.set_xticklabels([])
    axes[-1, 0].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    axes[-1, 1].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
    fig.text(0.012, 0.5, r'$F_\lambda$ (dereddened; arbitrary for LAMOST)',
             rotation=90, va='center', fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.06,
                        hspace=0.10, wspace=0.18)
    fig.savefig(f'{OUT}/figures/emission_lines.pdf')
    plt.close(fig)
    print('emission_lines.pdf saved')


if __name__ == '__main__':
    main()
