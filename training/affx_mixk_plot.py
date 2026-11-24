#!/usr/bin/env python
"""
Error against the number K of mixed test populations, from affx_mixk.py
--json files: the geometric mean of the paper's eight metrics, the
the same without the three mass metrics (number-based: CCN, scattering,
absorption, frozen fraction, N), the decoding-mixtures error, and the WRF
paper's diagnostics (CCN, scattering, absorption, frozen fraction) and the
speciated mass distribution. Lines = mean over seeds, bands = seed min-max.
Dashed (nonlinear decoders): their decoded populations, mixed (dm), i.e. what
they would give if they mixed exactly.

Usage: python affx_mixk_plot.py OUT.pdf JSON:NAME:LABEL [JSON:NAME:LABEL ...] [--seeds 0 1 2]
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

argv = sys.argv[1:]
keep = None                                                         # --seeds: only these seeds (paired)
if '--seeds' in argv:
    i = argv.index('--seeds'); keep = [int(a) for a in argv[i + 1:]]; argv = argv[:i]
out, specs = argv[0], argv[1:]
PANELS = [('geo', 'geometric mean, 8 metrics'), ('num', 'geometric mean, number-based 5'),
          ('comm', 'decoding mixtures (geo)'), ('ccn_cdf', 'CCN'), ('logqs_pop', 'scattering'),
          ('logqa_pop', 'absorption'), ('logfrznfrac_tmp', 'frozen fraction'), ('m_chmprthst', 'speciated mass')]
NUM = ['ccn_cdf', 'logqs_pop', 'logqa_pop', 'logfrznfrac_tmp', 'n_prthst']   # 'num': without the mass metrics


def val(r, key):
    return np.exp(np.mean([np.log(r[k]) for k in NUM])) if key == 'num' else r[key]
COLORS = ['#1f4e9c', '#c0392b', '#e08e0b', '#2a9d5c', '#7d3c98', '#555555']

fig, axs = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True)
allK = set()
for ci, spec in enumerate(specs):
    fn, name, label = spec.split(':', 2)
    d = json.load(open(fn))[name]
    if keep is not None:
        ix = [i for i, si in enumerate(d['seeds']) if si in keep]
        d = dict(seeds=[d['seeds'][i] for i in ix], K={k: [v[i] for i in ix] for k, v in d['K'].items()})
    Ks = sorted(int(k) for k in d['K'])
    allK |= set(Ks)
    col = COLORS[ci % len(COLORS)]
    label = f'{label} ({len(d["seeds"])})'
    for ax, (key, title) in zip(axs.flat, PANELS):
        kind = 'comm' if key == 'comm' else 'mix'
        mkey = 'geo' if key in ('geo', 'comm') else key
        v = np.array([[100 * val(s[kind], mkey) for s in d['K'][str(K)]] for K in Ks])
        if key == 'comm' and np.abs(v).max() < 1e-6:
            ax.plot(Ks, v.mean(1), color=col, lw=1.8, label=label)
            continue
        ax.plot(Ks, v.mean(1), color=col, lw=1.8, marker='o', ms=3, label=label)
        if v.shape[1] > 1:
            ax.fill_between(Ks, v.min(1), v.max(1), color=col, alpha=0.15, lw=0)
        if key != 'comm':
            dm = np.array([[100 * val(s['dm'], mkey) for s in d['K'][str(K)]] for K in Ks])
            if np.abs(dm - v).max() > 1e-6:
                ax.plot(Ks, dm.mean(1), color=col, lw=1.2, ls='--')
for ax, (key, title) in zip(axs.flat, PANELS):
    ax.set_title(title, fontsize=10)
    ax.set_xscale('log', base=2)
    tk = [k for k in sorted(allK) if (k & (k - 1) == 0 and (max(allK) <= 16 or k.bit_length() % 2)) or k == max(allK)]
    ax.set_xticks(tk); ax.set_xticklabels([str(k) for k in tk], fontsize=8)
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
for ax in axs[1]:
    ax.set_xlabel('populations mixed, K')
for ax in axs[:, 0]:
    ax.set_ylabel('test error (%)')
axs[0, 0].legend(fontsize=8, frameon=False)
fig.text(0.5, 0.005, 'solid: decode(mixed latent) vs true mixture; dashed: decoded populations, mixed; '
         'bands: seed min-max; (n) seeds', ha='center', fontsize=8, color='#444444')
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(out)
print('wrote', out)
