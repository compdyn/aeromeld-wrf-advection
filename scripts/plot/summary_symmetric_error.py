#!/usr/bin/env python
"""summary_symmetric_error: violins of AeroMELD's symmetric relative error against the exact answer.

One violin per condition (CCN/CN at four s_env, scattering and absorption at
400 and 1000 nm, frozen fraction at four temperatures), over the MIXED grid
cells: where at least two populations each hold analysis.MIX_MIN_FRAC of the
PartMC number. Those are the cells whose error is in question, because
AeroMELD carries a mixture as the number-weighted average of latents and the
decoder never saw such averages in training. --all-cells plots every cell, to
summary_symmetric_error_all. The number above each violin is SMAPE (%).

The violins are kernel densities of log10(e) on a linear axis with decade
labels; e itself on a log axis would squash the lower decades into a spike.
The statistics come from paperfigs.analysis, which the metrics table uses too.
From read_wrf_data.ipynb cell 47.

2026-11-12: against the exact answer (paperfigs.reference via analysis.compute),
over per-source transport's mixed cells; before, against WRF-PartMC.

Products: aeromeld_domain; the exact reference (paperfigs.reference).
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')   # the KDEs go through BLAS; see PLAN.md

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from paperfigs import analysis, figures  # noqa: E402
from paperfigs.fig_palettes import ANNOTATION_COLOR, TWO_COLUMN  # noqa: E402

# Achromatic on purpose: the warm and cool hue families belong to the sampling
# points and the latent populations.
VIOLIN_FACE = '#b3b3b3'
INK = ANNOTATION_COLOR
MUTED = '#5c5c5c'
#: Decade axis of log10(e): bottom fixed at 1e-3, top above e's ceiling of 2.
Y_LIMITS = (-3, 1)


def draw_violins(ax, values, xs):
    """Violins at `xs` with p05-p95 (thin), IQR (thick) and median (white dot)."""
    parts = ax.violinplot(values, positions=xs, widths=.8, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor('none')
        body.set_alpha(1)
    for xp, d in zip(xs, values):
        p05, p25, p50, p75, p95 = np.percentile(d, [5, 25, 50, 75, 95])
        ax.vlines(xp, p05, p95, color=INK, lw=1, zorder=3)
        ax.vlines(xp, p25, p75, color=INK, lw=4, zorder=3)
        ax.plot(xp, p50, 'o', ms=4.5, mfc='white', mec=INK, mew=.8, zorder=4)


def stat_row(ax, header, texts, xs, y=1.02):
    for xp, t in zip(xs, texts):
        ax.text(xp, y, t, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize='small', color=INK)
    if header is not None:
        ax.text(-.01, y, header, transform=ax.transAxes, ha='right', va='bottom',
                fontsize='small', color=MUTED)


def violin_key():
    return ([Patch(facecolor=VIOLIN_FACE, edgecolor='none'),
             (Line2D([], [], color=INK, lw=1),
              Line2D([], [], color=INK, lw=4, solid_capstyle='butt'),
              Line2D([], [], ls='none', marker='o', ms=4.5, mfc='white', mec=INK, mew=.8))],
            ['Grid cells', '5th–95th percentiles / interquartile range / median'])


def finish_condition_axes(ax, xs, ticks, spans):
    ax.set_xlim(xs[0] - .6, xs[-1] + .6)
    ax.set_xticks(xs, ticks)
    ax.tick_params(axis='x', length=0)
    for label, x0, x1 in spans:
        ax.text(.5 * (x0 + x1), -.17, label, transform=ax.get_xaxis_transform(),
                ha='center', va='top')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)


def make(run, mixed_only=True):
    conds, stats, mixed, pops = analysis.compute(run, mixed_only)
    positions, spans = analysis.layout(conds)

    fig, ax = plt.subplots(figsize=(TWO_COLUMN, 2.8))
    fig.subplots_adjust(left=.1, right=.99, bottom=.27, top=.91)
    draw_violins(ax, [s['log10_e'] for s in stats], positions)
    stat_row(ax, 'SMAPE', [('%.1f%%' if s['smape_pct'] < 10 else '%.0f%%') % s['smape_pct']
                           for s in stats], positions)
    lo, hi = Y_LIMITS
    ax.set_ylim(lo, hi)
    decades = np.arange(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r'$10^{%d}$' % round(v)))
    ax.yaxis.set_minor_locator(FixedLocator(
        [k + np.log10(m) for k in decades for m in range(2, 10) if k + np.log10(m) < np.log10(2)]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', which='major', color='#d9d9d9', lw=.6, zorder=0)
    ax.grid(axis='y', which='minor', color='#ececec', lw=.4, zorder=0)
    ax.set_ylabel(r'SMAPE term $e$ per cell')
    finish_condition_axes(ax, positions, [c['tick'] for c in conds], spans)
    ax.legend(*violin_key(), loc='lower right', handler_map={tuple: HandlerTuple(ndivide=1)},
              frameon=False, ncol=2)

    print('%d of %d cells mixed (populations %s, >= %g%% each); plotted: %s'
          % (mixed.sum(), mixed.size, ', '.join(map(str, pops)), 100 * analysis.MIX_MIN_FRAC,
             'mixed cells' if mixed_only else 'every cell'))
    print('%-34s %6s %8s %9s %9s %9s %9s' % ('', 'cells', 'SMAPE', 'p05', 'median', 'p95',
                                            'unmixed'))
    for c, s in zip(conds, stats):
        print('%-28s %-5s %6d %7.2f%% %9.2e %9.2e %9.2e %8.2f%%'
              % (c['label_plain'], c['tick'].replace('$', ''), s['n_cells'], s['smape_pct'],
                 s['e_p05'], s['e_median'], s['e_p95'], s['smape_unmixed_pct']))
    return figures.save(fig, 'summary_symmetric_error' if mixed_only
                        else 'summary_symmetric_error_all')


def main(argv=None):
    ap = figures.cli(__doc__)
    ap.add_argument('--all-cells', action='store_true', help='every cell, not just mixed ones')
    args = ap.parse_args(argv)
    make(figures.setup(args), mixed_only=not args.all_cells)


if __name__ == '__main__':
    main()
