#!/usr/bin/env python
"""summary_errors: the recommended configuration's error in every mixed cell, per diagnostic.

One violin per diagnostic of the paper's error (paperfigs/errors.py: number and species mass
size distributions, CCN spectrum, frozen fraction, scattering and absorption spectra), over
per-source transport's mixed cells at the end of the run (12 h), for the decoder trained on
mixtures of up to 16 at its representative seed (data/decomposition.json) with separate
limiters, against the particle-resolved reference. Above each violin, the mean over the cells
(the number the text and the other figures use). Replaces summary_symmetric_error (single
conditions of four spectra, SMAPE), 2026-11-13.

The violins are kernel densities of log10(e) on a linear axis with decade labels.

Products: none; reads data/decomposition.json and data/transport/all<seed>_12h.
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from paperfigs import config, errors, figures  # noqa: E402
from paperfigs.fig_palettes import TWO_COLUMN  # noqa: E402
from summary_symmetric_error import INK, draw_violins, violin_key  # noqa: E402

RECIPE = config.recipe('twin_k16')
HOURS = 12.0
Y_LIMITS = (-4, 0)


def per_cell(hours=HOURS):
    d = json.loads((config.DATA_DIR / 'decomposition.json').read_text())
    seed = d['models'][RECIPE]['representative_seed']
    model = '%s.%d.0' % (RECIPE, seed)
    r = errors.decompose(model, config.REPLAY_12H % seed, hours, keys=['sep:twin_k16'], per_cell=True)
    return model, r['cells']['sep:twin_k16']['total']


def draw(model, e):
    fig, ax = plt.subplots(figsize=(TWO_COLUMN, 2.6))
    fig.subplots_adjust(left=.11, right=.99, bottom=.2, top=.86)
    xs = np.arange(len(errors.SIX))
    vals = [np.log10(np.maximum(e[k], 1e-6)) for k in errors.SIX]
    draw_violins(ax, vals, xs)
    means = [np.mean(e[k]) for k in errors.SIX]
    # the mean of each violin, written above it (2026-11-13: keep the row clear of the top
    # tick label; 2026-11-14: no marker for the mean on the violin)
    for xp, m in zip(xs, means):
        ax.text(xp, 1.06, '%.1f%%' % (100 * m), transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize='small', color=INK)
    ax.text(-.6, 1.06, 'Mean:', transform=ax.get_xaxis_transform(), ha='right', va='bottom', fontsize='small',
            color=INK)
    lo, hi = Y_LIMITS
    ax.set_ylim(lo, hi)
    decades = np.arange(lo, hi + 1)
    ax.yaxis.set_major_locator(FixedLocator(decades))
    # in percent, as every other figure: 0.01, 0.1, 1, 10, 100
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '%g' % (100 * 10.0 ** round(v))))
    ax.yaxis.set_minor_locator(FixedLocator([k + np.log10(m) for k in decades[:-1] for m in range(2, 10)]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', which='major', color='#d9d9d9', lw=.6, zorder=0)
    ax.grid(axis='y', which='minor', color='#ececec', lw=.4, zorder=0)
    ax.set_ylabel('Error per cell (%)')
    ax.set_xlim(-.6, len(xs) - .4)
    ax.set_xticks(xs, [errors.LABELS[k].replace(' distribution', '\ndistribution').replace(' spectrum', '\nspectrum')
                       for k in errors.SIX])
    ax.tick_params(axis='x', length=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    handles, labels = violin_key()
    # one row along the bottom decade (0.01-0.1 %), which no violin reaches; the caption's words
    ax.legend(handles, labels, loc='lower right', handler_map={tuple: HandlerTuple(ndivide=1)},
              frameon=False, ncol=3, fontsize=7, handlelength=1.4, columnspacing=.8, handletextpad=.4,
              borderaxespad=.2)
    print('%s, %d mixed cells: mean %s' % (model, len(e[errors.SIX[0]]),
                                          ', '.join('%s %.2f%%' % (errors.SHORT[k], 100 * np.mean(e[k]))
                                                    for k in errors.SIX)))
    return fig


def make(run=None):
    return figures.save(draw(*per_cell()), 'summary_errors')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    figures.setup(args)
    make()


if __name__ == '__main__':
    main()
