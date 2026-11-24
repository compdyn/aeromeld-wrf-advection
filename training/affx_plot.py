#!/usr/bin/env python
"""
Example decoded populations of affx.py runs, in the style of the WRF paper's
diagnostic panels: for six test populations of one seed (the lowest, median
and highest frozen fraction at -24 C, and the lowest, median and highest
scattering at 1000 nm), one row each with the number and mass size
distributions, the CCN spectrum, the frozen fraction (log) and the
scattering and absorption (log). True values solid black; each run's decoded
values dashed. Non-positive decoded values are marked with x on the log axes
(at the bottom edge), as they would be blank in a log-scale map.

Decodes from the saved run files: <name>_s<seed>.pt (A, b or their raw
softplus parameters, and the element scales s) and <name>_s<seed>_Z.npy.

Usage: python affx_plot.py OUT.pdf DIR/NAME [DIR/NAME ...] [--seed 0]
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('out')
p.add_argument('runs', nargs='+')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--cache', default='/pscratch/sd/m/mwest/aeromeld/affx/cache.npz')
p.add_argument('--splits', default='/pscratch/sd/m/mwest/aeromeld/models/test_idx.npz')
args = p.parse_args()

BLOCKS = ['M', 'N', 'C', 'S', 'A', 'F']
SIZES = {'M': 225, 'N': 15, 'C': 100, 'S': 8, 'A': 8, 'F': 100}
OFFS = dict(zip(BLOCKS, np.cumsum([0] + [SIZES[k] for k in BLOCKS])[:-1]))
blk = lambda X, k: X[..., OFFS[k]:OFFS[k] + SIZES[k]]

c = np.load(args.cache)
X, ntot = c['X'], c['ntot']
tst = np.sort(np.load(args.splits)['idx'][args.seed])

def decode(run):
    st = torch.load(f'{run}_s{args.seed}.pt', map_location='cpu')
    sp = torch.nn.functional.softplus
    if 'A_eff' in st:                       # runs from after the --mono / --simplex options
        A, b = st['A_eff'], st['b_eff']
    else:
        A = sp(st['Araw']) if 'Araw' in st else st['A']
        b = sp(st['braw']) if 'braw' in st else st['b']
    Z = torch.from_numpy(np.load(f'{run}_s{args.seed}_Z.npy')).double()
    return ((Z @ A.double().T + b.double()) * st['s'].double()).numpy()

dec = {r.split('/')[-1]: decode(r) for r in args.runs}
ff24 = blk(X, 'F')[:, 40]                   # temperature index 40 of 100 on -40..0 C: -24 C
sc1000 = blk(X, 'S')[:, 7]
pick = []
for v in [ff24, sc1000]:
    o = tst[np.argsort(v[tst])]
    pick += [o[0], o[len(o) // 2], o[-1]]
labels = ['lowest FF(-24C)', 'median FF(-24C)', 'highest FF(-24C)', 'lowest scat(1000nm)', 'median scat(1000nm)', 'highest scat(1000nm)']

d_edges = 10 ** (-8.75 + 0.25 * np.arange(16)) * 1e6     # um
d_mid = np.sqrt(d_edges[:-1] * d_edges[1:])
s_pct = np.exp(np.linspace(np.log(1e-3), np.log(1e-1), 100)) * 100
temps = np.linspace(-40, 0, 100)
wave = np.linspace(300, 1000, 8)

fig, axs = plt.subplots(len(pick), 6, figsize=(17, 2.6 * len(pick)), constrained_layout=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for row, (i, lab) in enumerate(zip(pick, labels)):
    curves = [('true', X[i], 'k', '-')] + [(n, d[i], colors[j % 10], '--') for j, (n, d) in enumerate(dec.items())]
    for name, x, col, ls in curves:
        N = blk(x, 'N') * ntot[i] / 0.25 * 1e-6                       # dN/dlog10D, cm^-3
        Mt = blk(x, 'M').reshape(15, 15).sum(0) * ntot[i] / 0.25 * 1e9  # dM/dlog10D, ug m^-3
        axs[row, 0].step(d_mid, N, where='mid', color=col, ls=ls, label=name)
        axs[row, 1].step(d_mid, Mt, where='mid', color=col, ls=ls)
        axs[row, 2].plot(s_pct, blk(x, 'C'), color=col, ls=ls)
        for k, ax, xs, scale in [('F', axs[row, 3], temps, 1.0), ('S', axs[row, 4], wave, ntot[i] * 1e6),
                                 ('A', axs[row, 5], wave, ntot[i] * 1e6)]:
            v = blk(x, k) * scale
            pos = v > 0
            ax.plot(xs[pos], v[pos], color=col, ls=ls)
            if (~pos).any():
                ax.plot(xs[~pos], np.full((~pos).sum(), np.nan), 'x', color=col)
                ax.scatter(xs[~pos], np.zeros((~pos).sum()), marker='x', color=col, transform=ax.get_xaxis_transform(), clip_on=False)
    for ax in axs[row, :2]:
        ax.set_xscale('log')
    axs[row, 2].set_xscale('log'); axs[row, 2].set_ylim(-0.05, 1.1)
    for ax in axs[row, 3:]:
        ax.set_yscale('log')
    axs[row, 0].set_ylabel(lab, fontsize=9)
for ax, t in zip(axs[0], ['dN/dlogD (cm-3)', 'dM/dlogD (ug m-3)', 'CCN/CN vs s (%)', 'frozen fraction vs T (C)',
                          'scattering (Mm-1) vs wavelength', 'absorption (Mm-1) vs wavelength']):
    ax.set_title(t, fontsize=9)
axs[0, 0].legend(fontsize=7)
fig.suptitle(f'seed {args.seed}: x = non-positive decoded values (blank on a log map)', fontsize=10)
fig.savefig(args.out)
print('wrote', args.out)
