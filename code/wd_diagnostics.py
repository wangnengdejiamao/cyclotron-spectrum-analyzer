#!/usr/bin/env python3
"""wd_diagnostics.py — referee check: are the fitted WD parameters physical?

For every science target this reports, for the adopted full-refit solution:
  * the fitted T_wd, log g and the implied radius R_wd = sqrt(s_wd) * d,
  * the mass implied by (log g, R_wd)  -- the internal consistency test,
  * the flux fraction of each model component at 5500 A and integrated
    over the fitted band,
and compares them with a physical WD (M = 0.6-1.0 Msun on a
mass-radius relation, Teff = 8000-20000 K) at the Gaia distance.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pipeline as jp

ROOT = os.environ.get('CYC_ROOT') or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
FR = os.path.join(ROOT, 'validation_candidates', 'full_refit', 'data')
G_CGS = 6.674e-8
MSUN = 1.989e33
SRCS = ['J0005', 'J0022', 'J0749', 'J0035']


def r_of_m(m_msun):
    """Nauenberg (1972) zero-temperature WD mass-radius relation [cm]."""
    m_ch = 5.816 / 2.0 ** 2          # Chandrasekhar mass for mu_e = 2
    x = (m_msun / m_ch) ** (2.0 / 3.0)
    return 7.83e8 * np.sqrt(1.0 - x) / (m_msun / m_ch) ** (1.0 / 3.0)


def main():
    kg = jp.KoesterGrid(jp.KOESTER_DIR)
    print('\nNauenberg M-R relation:')
    for m in (0.6, 0.7, 0.8, 0.9, 1.0):
        R = r_of_m(m)
        print(f'  M={m:.2f} Msun  R={R:.3e} cm = {R/jp.R_SUN_CM:.4f} Rsun  '
              f'log g={np.log10(G_CGS*m*MSUN/R**2):.3f}')

    for src in SRCS:
        meta = jp.SOURCES[src]
        js = json.load(open(os.path.join(FR, f'{src}_full_refit.json')))
        p = np.array(js['adopted_params'])
        a = np.array(js['adopted_amps'])
        d = meta['dist_pc']
        print(f'\n=== {src} ({meta["name"]}) ===')
        print(f'  fitted T_wd={p[0]:.0f} K  log g={p[1]:.2f}  '
              f'T_spot={p[2]:.0f} K')
        print(f'  amps: s_wd={a[0]:.3e}  s_spot={a[1]:.3e}  A_cyc={a[2]:.3e}')
        print(f'  s_wd bounds: {js["s_wd_bounds"][0]:.3e} '
              f'{js["s_wd_bounds"][1]:.3e}'
              f'  (at bound: '
              f'{"LOWER" if a[0] <= 1.02*js["s_wd_bounds"][0] else ""}'
              f'{"UPPER" if a[0] >= 0.98*js["s_wd_bounds"][1] else ""})')
        if d is not None:
            R = np.sqrt(a[0]) * d * jp.PC_CM
            m_impl = 10 ** p[1] * R ** 2 / G_CGS / MSUN
            print(f'  implied R_wd={R:.3e} cm = {R/jp.R_SUN_CM:.4f} Rsun '
                  f'(at d={d} pc)')
            print(f'  ==> mass implied by (log g, R): {m_impl:.2f} Msun '
                  f'{"*** UNPHYSICAL ***" if m_impl > 1.4 else ""}')
            # what a physical WD would contribute
            for m_wd in (0.6, 0.8, 1.0):
                Rp = r_of_m(m_wd)
                s_p = (Rp / (d * jp.PC_CM)) ** 2
                for T in (10000.0, 15000.0, 20000.0):
                    lg = np.log10(G_CGS * m_wd * MSUN / Rp ** 2)
                    fl = s_p * kg.shape(np.array([5500.0]), T,
                                        min(lg, 9.49))[0]
                    print(f'     physical WD M={m_wd:.1f} T={T:.0f}: '
                          f'F(5500)={fl:.3e}')
                    break

        # component flux fractions
        w = np.linspace(3950.0, 9300.0, 600)
        wd = a[0] * kg.shape(w, p[0], p[1])
        spb = jp.planck(w, p[2])
        spn = jp.planck(np.linspace(3950, 9300, 400), p[2]).max()
        sp = a[1] * spb / spn
        from cyclotron_m2 import cal_cy_spec
        cyr = cal_cy_spec(w * 1e-10, p[4] * jp.KEV_J, p[3] * 100.0,
                          np.deg2rad(p[5]), 10.0 ** p[6])
        cyn = cal_cy_spec(np.linspace(3950e-10, 9300e-10, 400),
                          p[4] * jp.KEV_J, p[3] * 100.0,
                          np.deg2rad(p[5]), 10.0 ** p[6]).max()
        cy = a[2] * cyr / cyn
        tot = wd + sp + cy
        i55 = np.argmin(np.abs(w - 5500.0))
        print(f'  at 5500 A: WD {100*wd[i55]/tot[i55]:5.1f}%  '
              f'spot {100*sp[i55]/tot[i55]:5.1f}%  '
              f'cyc {100*cy[i55]/tot[i55]:5.1f}%')
        iw, isp, ic = [np.trapz(x, w) for x in (wd, sp, cy)]
        it = iw + isp + ic
        print(f'  band-integrated: WD {100*iw/it:5.1f}%  spot {100*isp/it:5.1f}%'
              f'  cyc {100*ic/it:5.1f}%')


if __name__ == '__main__':
    main()
