#!/usr/bin/env python3
"""make_referee_tables.py — LaTeX tables added in the referee revision.

  tab:sed        archival UV-to-mid-IR photometry of the four targets
  tab:continuum  best-fit continuum parameters of the adopted solutions,
                 their internal (in)consistency as white dwarfs, and the
                 physically constrained refit

Writes manuscript_aa/submission/tab_sed.tex and tab_continuum.tex, which
main_aa.tex \\input{}s.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pipeline as jp
from wd_prior_refit import r_of_m, logg_of_m

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
SUB = os.environ.get('CYC_TABLES', os.path.join(ROOT, 'tables'))
FR = os.path.join(ROOT, 'validation_candidates', 'full_refit', 'data')
WP = os.path.join(ROOT, 'data', 'wd_prior')
G_CGS, MSUN = 6.674e-8, 1.989e33

ORDER = ['J0005', 'J0022', 'J0749', 'J0035']
LABEL = {'J0005': 'DESI~J0005$+$2941', 'J0022': 'DESI~J0022$+$1340',
         'J0749': 'DESI~J0749$+$3654', 'J0035': 'LAMOST~J0035$+$4333'}
DIST = {'J0005': 604.0, 'J0022': 583.0, 'J0749': 946.0, 'J0035': None}

# rows of the SED table, in wavelength order
ROWS = [('FUV', 'GALEX FUV', 1528), ('NUV', 'GALEX NUV', 2271),
        ('uSDSS', 'SDSS $u$', 3557), ('gSDSS', 'SDSS $g$', 4702),
        ('gP1', 'PS1 $g$', 4849), ('rSDSS', 'SDSS $r$', 6175),
        ('rP1', 'PS1 $r$', 6201), ('iSDSS', 'SDSS $i$', 7490),
        ('iP1', 'PS1 $i$', 7535), ('zP1', 'PS1 $z$', 8674),
        ('zSDSS', 'SDSS $z$', 8946), ('yP1', 'PS1 $y$', 9628),
        ('W1', 'AllWISE $W1$', 33526), ('W1cw', 'CatWISE $W1$', 33526),
        ('W2', 'AllWISE $W2$', 46028), ('W2cw', 'CatWISE $W2$', 46028)]


def sed_table():
    ph = json.load(open(f'{ROOT}/data/sed_photometry.json'))
    out = [r'\begin{table*}[ht]',
           r'\caption{Archival photometry of the four targets.'
           r'\label{tab:sed}}', r'\centering',
           r'\begin{tabular}{llcccc}', r'\hline\hline',
           r'Band & $\lambda_{\rm eff}$ & ' +
           ' & '.join(LABEL[s].replace('~', '\\,') for s in ORDER) + r' \\',
           r' & (\AA) & (mag) & (mag) & (mag) & (mag) \\', r'\hline']
    for key, lab, lam in ROWS:
        cells = []
        for s in ORDER:
            b = ph[s]['bands'].get(key)
            if b is None or b['upper_limit']:
                cells.append(r'\ldots')
            else:
                cells.append(f"${b['mag']:.2f}\\pm{b['err']:.2f}$")
        if all(c == r'\ldots' for c in cells):
            continue
        out.append(f'{lab} & {lam} & ' + ' & '.join(cells) + r' \\')
    out += [r'\hline', r'\end{tabular}',
            r'\tablefoot{Magnitudes are as catalogued (AB for GALEX, SDSS '
            r'and Pan-STARRS, Vega for WISE) and are \emph{not} '
            r'dereddened; the SED of Fig.~\ref{fig:sed} applies the '
            r'extinction of Table~\ref{tab:sample}. '
            r'None of the four targets is detected by 2MASS, by the UKIRT '
            r'Hemisphere Survey or by Spitzer. The AllWISE $W3$ and $W4$ '
            r'entries are upper limits in all four cases and are omitted. '
            r'The two independent WISE reductions (AllWISE, epochs '
            r'2010--2011; CatWISE2020, epochs 2010--2018) and the two '
            r'independent optical surveys are listed separately because '
            r'their differences measure the accretion-state variability '
            r'rather than the photometric error (Sect.~\ref{sec:sed}).}',
            r'\end{table*}']
    return '\n'.join(out)


def continuum_table():
    out = [r'\begin{table*}[ht]',
           r'\caption{Continuum parameters of the adopted fits and of the '
           r'physically constrained refit.\label{tab:continuum}}',
           r'\centering', r'\begin{tabular}{llccccccc}', r'\hline\hline',
           r'Source & Fit & $T_{\rm wd}$ & $\log g$ & $R_{\rm wd}$ & '
           r'$M_{\rm wd}$ & $T_{\rm spot}$ & WD/spot/cyc & $B$ \\',
           r' & & (K) & (cgs) & ($R_\odot$) & ($M_\odot$) & (K) & '
           r'(per cent) & (MG) \\', r'\hline']
    summary = {}
    for s in ORDER:
        js = json.load(open(os.path.join(FR, f'{s}_full_refit.json')))
        wp = json.load(open(os.path.join(WP, f'{s}_wdprior_branches.json')))
        p = np.array(js['adopted_params'])
        a = np.array(js['adopted_amps'])
        d = DIST[s]
        if d is not None:
            R = np.sqrt(a[0]) * d * jp.PC_CM
            m_impl = 10 ** p[1] * R ** 2 / G_CGS / MSUN
            rcell = f'{R/jp.R_SUN_CM:.4f}'
            if m_impl > 1.4:
                mcell = r'\textbf{' + f'{m_impl:.0f}' + '}'
            elif m_impl < 0.1:
                mcell = r'\textbf{' + f'{m_impl:.3f}' + '}'
            else:
                mcell = f'{m_impl:.2f}'
        else:
            rcell = mcell = r'\ldots'
            m_impl = np.nan
        fr = component_fractions(s, p, a)
        out.append(
            f'{LABEL[s]} & free & {p[0]:.0f} & {p[1]:.2f} & {rcell} & '
            f'{mcell} & {p[2]:.0f} & '
            f'{100*fr[0]:.0f}/{100*fr[1]:.0f}/{100*fr[2]:.0f} & '
            f'{p[3]:.2f} ' + r'\\')
        row = wp['branches'][0]
        if d is None:      # relative flux: radius and mass are meaningless
            gcell, rcon, mcon = r'\ldots', r'\ldots', r'\ldots'
        else:
            gcell = f'{row["logg"]:.2f}'
            rcon = f'{row["R_wd_Rsun"]:.4f}'
            mcon = f'{row["M_wd"]:.2f}'
        out.append(
            f' & constrained & {row["T_wd"]:.0f} & {gcell} & '
            f'{rcon} & {mcon} & '
            f'{row["T_spot"]:.0f} & '
            f'{100*row["frac_int"]["wd"]:.0f}/'
            f'{100*row["frac_int"]["spot"]:.0f}/'
            f'{100*row["frac_int"]["cyc"]:.0f} & '
            f'{row["B"]:.2f} ' + r'\\')
        summary[s] = dict(m_implied=m_impl, B_free=float(p[3]),
                          B_con=row['B'], dB=row['B'] - float(p[3]),
                          chi2_free=js['chi2_red'] * (js['n_bins'] - 10),
                          dchi2=row['chi2'] - min(
                              q['chi2'] for q in js['solutions']
                              if 'chi2' in q),
                          alt=[dict(B_start=q['B_start'], B=q['B'])
                               for q in wp['branches'][1:]])
        out.append(r'\hline' if s != ORDER[-1] else '')
    foot = (
        r"\tablefoot{The `free' rows are the adopted solutions of "
        r"Table~\ref{tab:results}, in which $T_{\rm wd}$, $\log g$ and the "
        r"solid angle $s_{\rm wd}=(R_{\rm wd}/d)^2$ vary independently. "
        r"$R_{\rm wd}$ is then the radius implied by $s_{\rm wd}$ at the "
        r"Gaia distance, and $M_{\rm wd}=gR_{\rm wd}^2/G$ the mass implied "
        r"by that radius together with the fitted gravity. Masses in bold "
        r"lie outside the white dwarf range, so the free fit does not "
        r"return a physical white dwarf (Sect.~\ref{sec:wdprior}). "
        r"The `constrained' rows impose the \citet{Nauenberg1972} "
        r"mass--radius relation with $M_{\rm wd}\in[0.6,1.0]\,M_\odot$, "
        r"$T_{\rm wd}\in[8000,20000]$\,K and the Gaia distance, so "
        r"$\log g$, $R_{\rm wd}$ and $s_{\rm wd}$ all follow from "
        r"$M_{\rm wd}$. LAMOST~J0035$+$4333 has a relative flux scale, so "
        r"only the temperature priors apply to it and neither gravity, "
        r"radius nor mass can be inferred. Its free-fit hot-spot "
        r"temperature is a legacy solution that predates the "
        r"$5\times10^{4}$\,K cap now enforced by the released code, "
        r"and is superseded by the constrained row. "
        r"The penultimate column gives the band-integrated flux fraction "
        r"of the three components over 3950--9300\,\AA, and the last "
        r"column the recovered field.}")
    out += [r'\hline', r'\end{tabular}', foot,
            r'\end{table*}']
    return '\n'.join(out), summary


def component_fractions(src, p, a):
    kg = component_fractions.kg
    from cyclotron_m2 import cal_cy_spec
    w = np.linspace(3950.0, 9300.0, 600)
    wd = a[0] * kg.shape(w, p[0], p[1])
    sp = a[1] * jp.planck(w, p[2]) / jp.planck(
        np.linspace(3950, 9300, 400), p[2]).max()
    cyn = cal_cy_spec(np.linspace(3950e-10, 9300e-10, 400), p[4] * jp.KEV_J,
                      p[3] * 100.0, np.deg2rad(p[5]), 10.0 ** p[6]).max()
    cy = a[2] * cal_cy_spec(w * 1e-10, p[4] * jp.KEV_J, p[3] * 100.0,
                            np.deg2rad(p[5]), 10.0 ** p[6]) / cyn
    ii = np.array([np.trapz(x, w) for x in (wd, sp, cy)])
    return ii / ii.sum()


if __name__ == '__main__':
    component_fractions.kg = jp.KoesterGrid(jp.KOESTER_DIR)
    with open(os.path.join(SUB, 'tab_sed.tex'), 'w') as fh:
        fh.write(sed_table() + '\n')
    tab, summary = continuum_table()
    with open(os.path.join(SUB, 'tab_continuum.tex'), 'w') as fh:
        fh.write(tab + '\n')
    with open(os.path.join(ROOT, 'data', 'wd_prior_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    print('tab_sed.tex and tab_continuum.tex written\n')
    for s, r in summary.items():
        print(f'{s}: B {r["B_free"]:.2f} -> {r["B_con"]:.2f} '
              f'(dB={r["dB"]:+.2f} MG), dchi2={r["dchi2"]:+.1f}, '
              f'implied free mass {r["m_implied"]:.1f} Msun, '
              f'alt branches {r["alt"]}')
