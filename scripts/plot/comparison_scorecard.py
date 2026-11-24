#!/usr/bin/env python
"""comparison_scorecard: WRF-AeroMELD against the sectional and modal schemes, as a table.

Mean error over per-source transport's mixed cells at 12 h (the paper's error,
paperfigs/errors.py), per diagnostic and overall, for WRF-AeroMELD and the
comparison schemes, each carried by WRF's transport with separate limiters
(baselines/data/transport_terms.json). Rows:
  the six diagnostics;
  the overall error (mean of the six) and the overall error without the frozen fraction;
  the overall error split into transport error e(Q4, Q2) and representation error e(Q2, Q0).
Each row's lowest value is bold; the cells are shaded by their ratio to the row's lowest value
(log scale, capped at CAP). Column headers give the advected tracers per grid cell.

Products: none; reads the baseline study's numbers (paperfigs/comparison.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from paperfigs import comparison, errors, figures  # noqa: E402
from paperfigs.comparison import NO_FROZEN, SCHEMES  # noqa: E402
from paperfigs.fig_palettes import TWO_COLUMN  # noqa: E402

TRACERS = comparison.tracers()
#: (case in transport_terms.json, column label, family). Missing cases are skipped.
COLUMNS = tuple((k, name.replace('WRF-AeroMELD', 'WRF-\nAeroMELD'), fam) for k, (name, fam, *_) in SCHEMES.items())
#: Row blocks: (key, label, term). Keys are diagnostics, 'overall' or NO_FROZEN.
#: The size distributions are binned by dry diameter, with every species' mass, water included (the paper's basis, drydiam;
#: PAPERFIGS_BASIS=wet or dry reads an earlier basis: paperfigs/comparison.py).
NAME = 'comparison_scorecard' + comparison.SUFFIX
BLOCKS = ([(k, errors.LABELS[k], 'total') for k in errors.SIX],
          [('overall', 'Overall error', 'total'),
           (NO_FROZEN, 'Overall without\nfrozen fraction', 'total')],
          [('overall', 'Transport error', 'transport'),
           ('overall', 'Representation error', 'representation')])
CAP = 30.0
GAP = 0.45
INK = '#1a1a1a'
CMAP = LinearSegmentedColormap.from_list('scorecard', ['#ffffff', '#fde0c5', '#f4a582', '#d6604d', '#8e1b26'])


def fmt(v):
    p = 100 * v
    return '%.0f%%' % p if p >= 9.95 else '%.1f%%' % p


def table(columns=COLUMNS):
    have = comparison.present([c for c, *_ in columns])
    cols = [c for c in columns if c[0] in have]
    rows = [r for block in BLOCKS for r in block]
    v = np.array([[comparison.error(c, k, t) for c, *_ in cols] for k, _, t in rows])
    return cols, rows, v


def draw(columns=COLUMNS):
    cols, rows, v = table(columns)
    nc, nr = len(cols), len(rows)
    # y position of each row (top to bottom), with gaps between blocks
    ys, y = [], 0.0
    for b, block in enumerate(BLOCKS):
        for _ in block:
            ys.append(y)
            y += 1
        y += GAP
    ys = np.array(ys)
    fig = plt.figure(figsize=(TWO_COLUMN, 3.3))
    left, width = .315, .675
    ax = fig.add_axes([left, .13, width, .66])
    best = v.min(axis=1, keepdims=True)
    ratio = np.clip(v / best, 1, CAP)
    level = np.log(ratio) / np.log(CAP)
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(plt.Rectangle((j + .03, ys[i] + .04), .94, .92, color=CMAP(level[i, j]), lw=0))
            win = fmt(v[i, j]) == fmt(best[i, 0])
            ax.text(j + .5, ys[i] + .5, fmt(v[i, j]), ha='center', va='center', fontsize=7.5,
                    fontweight='bold' if win else 'normal', color='white' if level[i, j] > .62 else INK)
    for i, (_, label, _) in enumerate(rows):
        ax.text(-.08, ys[i] + .5, label, ha='right', va='center', fontsize=7.5, color=INK, linespacing=.95,
                fontweight='bold' if label == 'Overall error' else 'normal')
    for j, (c, label, fam) in enumerate(cols):
        ax.text(j + .5, -.62, label, ha='center', va='bottom', fontsize=7.5, color=INK, linespacing=.95,
                fontweight='bold' if c == 'aeromeld' else 'normal')
        ax.text(j + .5, -.12, '%d tracers' % TRACERS[c], ha='center', va='bottom', fontsize=6.5,
                color='#5c5c5c')
    # family brackets above the headers
    fams = {}
    for j, (_, _, fam) in enumerate(cols):
        if fam:
            fams.setdefault(fam, []).append(j)
    for fam, js in fams.items():
        x0, x1 = min(js) + .08, max(js) + .92
        ax.plot([x0, x1], [-1.75, -1.75], color='#8c8c8c', lw=.6, clip_on=False)
        ax.text((x0 + x1) / 2, -1.85, fam, ha='center', va='bottom', fontsize=7, color='#5c5c5c')
    ax.set_xlim(0, nc)
    ax.set_ylim(ys[-1] + 1, 0)
    ax.axis('off')
    # colour key
    cax = fig.add_axes([left + width * .25, .045, width * .5, .025])
    t = np.linspace(0, 1, 256)
    cax.imshow(t[None, :], aspect='auto', cmap=CMAP, extent=(0, 1, 0, 1))
    ticks = [1, 3, 10, 30]
    cax.set_xticks([np.log(r) / np.log(CAP) for r in ticks], ['%g×' % r for r in ticks])
    cax.set_yticks([])
    cax.tick_params(labelsize=6.5, length=2, pad=1.5)
    for s in cax.spines.values():
        s.set_linewidth(.5)
    cax.set_title("Ratio to the row's lowest error (lowest in bold)", fontsize=6.5, pad=2.5, color='#4d4d4d')
    for (c, *_), col in zip(cols, v.T):
        print('%-10s ' % c + ' '.join('%6.2f' % (100 * x) for x in col))
    return fig


def make(run=None):
    """For make_all.py: fixed numbers, whatever the run."""
    return figures.save(draw(), NAME)


def main():
    args = figures.cli(__doc__).parse_args()
    figures.setup(args)
    make()


if __name__ == '__main__':
    main()
