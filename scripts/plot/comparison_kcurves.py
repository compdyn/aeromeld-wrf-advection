#!/usr/bin/env python
"""comparison_kcurves: WRF-AeroMELD's decoder and the sectional and modal schemes on mixtures of
K test populations.

The overall error (mean of the six diagnostics) against the true mixture, on the same Dirichlet
mixtures of K test populations, from single populations (K = 1) to the whole test set (5000):
the decoder of WRF-AeroMELD (trained on mixtures of up to 16, the size basis's test-set file,
mixing_training.MIXK_JSON: notes/affx_stats/mixk_r39_lr.json for drydiam; mean over its ten
training seeds) and the schemes (comparison.KCURVES: baselines/data/drydiam/kcurves_baselines.json
for drydiam; the same ten test splits, no training: each population's bins or modes computed from its particles,
mixed in the representation). Lines: means; bands: the range over the ten seeds or splits.

Products: none.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from mixing_training import SIX, finish_k, kcurve  # noqa: E402
from paperfigs import comparison, figures  # noqa: E402
from paperfigs.comparison import SCHEMES  # noqa: E402
from paperfigs.fig_palettes import SINGLE_COLUMN  # noqa: E402

INK = '#1a1a1a'


def baseline_kcurve(key):
    d = json.loads(comparison.KCURVES.read_text())[key]
    Ks = sorted(int(k) for k in d['K'])
    v = np.array([[100 * np.mean([r['mix'][k] for k in SIX]) for r in d['K'][str(K)]] for K in Ks]).T
    return np.array(Ks), v


def draw():
    have = json.loads(comparison.KCURVES.read_text())
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 3.0),
                           gridspec_kw=dict(left=.15, right=.97, top=.97, bottom=.37))
    handles = []
    for k, (name, _, color, marker, ls) in SCHEMES.items():
        if k == 'aeromeld':
            K, v = kcurve('twin_k16')
        elif k in have:
            K, v = baseline_kcurve(k)
        else:
            continue
        ax.fill_between(K, v.min(0), v.max(0), color=color, alpha=.16, lw=0, zorder=1)
        ax.plot(K, v.mean(0), color=color, ls=ls, lw=1.2, marker=marker, ms=3.2, mfc=color, mec=INK, mew=.4,
                zorder=3)
        handles.append(Line2D([], [], color=color, ls=ls, lw=1.2, marker=marker, ms=4, mfc=color, mec=INK, mew=.4,
                              label=name if k != 'aeromeld' else 'WRF-AeroMELD (its decoder)'))
        print('%-9s ' % k + '  '.join('K=%d %.1f' % (kk, m) for kk, m in zip(K, v.mean(0)) if kk in (1, 3, 16, 5000)))
    finish_k(ax, ticks=(1, 4, 16, 64, 256, 1024, 5000))
    ax.set_xlabel('Number of test populations mixed, $K$')
    ax.set_ylim(0, None)
    ax.set_ylabel('Overall error (%)')   # the caption: against the true mixture
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, 0.0),
               fontsize=6.5, handletextpad=.4, columnspacing=1.2)
    return fig


def make(run=None):
    """For make_all.py: fixed numbers, whatever the run."""
    return figures.save(draw(), 'comparison_kcurves')


def main():
    args = figures.cli(__doc__).parse_args()
    figures.setup(args)
    make()


if __name__ == '__main__':
    main()
