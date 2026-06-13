"""Shared matplotlib style for all paper figures (ApJ-like)."""
import matplotlib as mpl

COLORS = dict(data='0.45', model='#c1272d', wd='#2b6cb0', spot='#e69f00',
              cyc='#1b7837', resid='#1f3a93',
              srcs=['#c1272d', '#2b6cb0', '#1b7837', '#e69f00'])


def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })
