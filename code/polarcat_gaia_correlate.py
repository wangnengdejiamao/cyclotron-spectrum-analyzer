import warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io.votable import parse
from scipy import stats

t = parse('/Users/ljm/Desktop/cyc/polar_sample/2025a_polars.vot').get_first_table().to_table()
P = np.asarray(t['P_orb'].filled(np.nan), float)
B = np.asarray(t['B1'].filled(np.nan), float)
yr = np.asarray(t['YDiscovery'].filled(np.nan), float)
ecl = np.asarray(t['ecl_len'].filled(np.nan), float)

def report(x, y, nx, ny):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        print(f'{nx} vs {ny}: N={m.sum()} too few'); return
    rho, p = stats.spearmanr(x[m], y[m])
    r, pp = stats.pearsonr(x[m], y[m])
    print(f'{nx} vs {ny}: N={m.sum()}  Spearman rho={rho:+.3f} (p={p:.4f})  Pearson r={r:+.3f} (p={pp:.4f})')

print('=== PolarCat internal correlations (242 polars) ===')
report(np.log10(P), B, 'log P_orb', 'B')
report(P, B, 'P_orb', 'B')
report(yr, B, 'YDiscovery', 'B')      # newer discoveries find stronger fields?
report(ecl, B, 'ecl_len', 'B')
report(np.log10(P), ecl, 'log P_orb', 'ecl_len')

# split by period gap, with bootstrap CIs on median
def med_ci(x, n=2000):
    bs = [np.median(np.random.choice(x, len(x))) for _ in range(n)]
    return np.median(x), np.percentile(bs, 16), np.percentile(bs, 84)
mPB = np.isfinite(P) & np.isfinite(B)
Pf, Bf = P[mPB], B[mPB]
for name, sel in [('P<129 (below gap)', Pf<129), ('129-191 (gap)', (Pf>=129)&(Pf<=191)),
                  ('P>191 (above gap)', Pf>191)]:
    if sel.sum()>=4:
        m_, lo_, hi_ = med_ci(Bf[sel])
        print(f'  {name}: N={sel.sum()}, median B = {m_:.1f} [{lo_:.1f}, {hi_:.1f}] MG')
