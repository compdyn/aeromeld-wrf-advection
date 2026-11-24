#!/usr/bin/env python
"""model_comparison: AeroMELD's error against PartMC for several ML models, condition by condition.

  (a) SMAPE over the mixed cells
  (b) AeroMELD / PartMC ratio over the same cells: median (marker) and
      p05-p95 (whisker), on a log axis

The conditions are the violin figure's (summary_symmetric_error): CCN/CN at four
s_env, scattering and absorption at 400 and 1000 nm, frozen fraction at four
temperatures, at the final instant. Each model keeps one offset within every
condition's slot, so a model is a fixed column of markers across the figure.

(a) ranks the models but cannot say how far off or in which direction: e is
symmetric and saturates at 2 (SMAPE 200%), so a 10x and a 1000x miss score
alike. (b) gives both. The ratio keeps AeroMELD's sign: where it decodes to zero
or below (the affine decoder in physical units does, in up to half the mixed
cells), those cells rank lowest instead of dropping out, a whisker that runs
off the bottom edge ends in a caret, and the share of such cells is printed at
the foot of the slot. Every number comes from paperfigs.analysis, the function
behind the violins and data/metrics.csv, so the three cannot disagree.

Runs: the registry's [sets] paper_compare, in its order, by default; --set NAME
or --runs A,B,C for another selection, which goes to
graphics/by_model/_compare/ unless --out says otherwise. make_all makes the
paper_compare figure only for the main run.

Products: aeromeld_domain (each run), partmc_diag_domain, partmc_fields.
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from paperfigs import analysis, config, fig_palettes, figures, registry  # noqa: E402
from paperfigs.fig_palettes import (ANNOTATION_COLOR, MODEL_STYLES, TWO_COLUMN,  # noqa: E402
                                    create_label_box, model_style)
from summary_symmetric_error import finish_condition_axes  # noqa: E402

SET = 'paper_compare'
INK = ANNOTATION_COLOR
MUTED = '#5c5c5c'
SLOT_SHADE = '#f2f2f2'
#: Width of a condition's slot that the models share, in condition spacings.
SLOT = 0.8
SMAPE_TICKS = (2, 5, 10, 20, 50, 100, 200)
#: SMAPE axis: its top is SMAPE's ceiling (e <= 2).
SMAPE_LIM = (2, 200)
RATIO_LIM = (1e-2, 1e4)


def collect(runs):
    """[(run, conditions, stats)], one per run, from analysis.compute; checks
    that every run was scored on the same conditions."""
    out = []
    for run in runs:
        conds, stats, _, _ = analysis.compute(run, mixed_only=True)
        out.append((run, conds, stats))
    ref = [(c['group'], c['tick']) for c in out[0][1]]
    for run, conds, _ in out[1:]:
        if [(c['group'], c['tick']) for c in conds] != ref:
            raise ValueError('%s was scored on other conditions than %s' % (run.key, out[0][0].key))
    return out


def styles(results):
    """(colour, marker) per run, from fig_palettes.MODEL_STYLES."""
    out, spare = [], 0
    for run, _, _ in results:
        out.append(model_style(run.model.key, spare))
        spare += run.model.key not in MODEL_STYLES
    return out


def decade_axis(ax, lo, hi, ticks=None):
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    if ticks is None:
        ticks = 10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1)
        fmt = lambda v, _: r'$10^{%d}$' % round(np.log10(v))
    else:
        fmt = lambda v, _: '%g' % v
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', which='major', color='#d9d9d9', lw=.6, zorder=0)
    ax.set_axisbelow(True)


def draw(results):
    conds = results[0][1]
    positions, spans = analysis.layout(conds)
    positions = np.asarray(positions, dtype=float)
    n = len(results)
    offsets = (np.arange(n) - (n - 1) / 2) * SLOT / n

    fig, (ax_s, ax_r) = plt.subplots(2, 1, figsize=(TWO_COLUMN, 4.7), sharex=True)
    fig.subplots_adjust(left=.1, right=.99, bottom=.235, top=.95, hspace=.2)
    for ax in (ax_s, ax_r):
        for x in positions[::2]:
            ax.axvspan(x - .5, x + .5, color=SLOT_SHADE, lw=0, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    decade_axis(ax_s, *SMAPE_LIM, ticks=SMAPE_TICKS)
    decade_axis(ax_r, *RATIO_LIM)
    ax_r.axhline(1, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    lo, hi = RATIO_LIM

    handles = []
    for (run, _, stats), (color, marker), dx in zip(results, styles(results), offsets):
        xs = positions + dx
        mk = dict(marker=marker, ms=4.5, mfc=color, mec=INK, mew=.4, ls='none', zorder=4,
                  clip_on=False)
        ax_s.plot(xs, [s['smape_pct'] for s in stats], **mk)
        for x, s in zip(xs, stats):
            p05, p50, p95 = s['ratio_p05'], s['ratio_median'], s['ratio_p95']
            if p95 > lo:
                ax_r.vlines(x, max(p05, lo), min(p95, hi), color=color, lw=1.1, zorder=3)
            if p05 <= lo:              # runs off the bottom: a caret marks the cut
                ax_r.plot(x, lo, marker=7, ms=4, color=color, ls='none', zorder=4, clip_on=False)
            if p50 > lo:
                ax_r.plot(x, p50, **mk)
            if s['n_nonpositive']:     # beside the whisker, not on it
                ax_r.text(x + .06, lo * 1.3, r'$\leq$0: %.0f%%' % (100 * s['n_nonpositive'] / s['ratio_cells']),
                          rotation=90, ha='left', va='bottom', fontsize=5.5, color=MUTED)
        handles.append(Line2D([], [], label=run.model.label, **mk))

    ax_s.set_ylabel('SMAPE (%)')
    ax_r.set_ylabel('AeroMELD / PartMC')
    ax_s.tick_params(axis='x', labelbottom=False, length=0)
    finish_condition_axes(ax_r, positions, [c['tick'] for c in conds], spans)
    create_label_box(ax_s, '(a) SMAPE over mixed cells')
    create_label_box(ax_r, '(b) Ratio: median, p05–p95')
    fig.legend(handles=handles, loc='lower center', ncol=min(3, len(handles)), frameon=False,
               bbox_to_anchor=(.55, 0), handletextpad=.2, columnspacing=.8, fontsize=6.5)
    return fig


def report(results):
    conds = results[0][1]
    print('%-26s %-10s %8s %10s %10s %10s %6s' % ('run', 'condition', 'SMAPE', 'p05', 'median',
                                                  'p95', '<=0'))
    for run, _, stats in results:
        for c, s in zip(conds, stats):
            print('%-26s %-10s %7.1f%% %10.3g %10.3g %10.3g %6d'
                  % (run.key, '%s %s' % (c['group'][:4], c['tick'].replace('$', '')),
                     s['smape_pct'], s['ratio_p05'], s['ratio_median'], s['ratio_p95'],
                     s['n_nonpositive']))


def make(run, runs=None):
    """The paper_compare figure when `run` is the main run (make_all), or the
    figure for an explicit list of registry Runs."""
    if runs is None:
        if run.role != 'main':
            print('model_comparison: skipped for %s; it compares the [sets] %s runs and is made '
                  'with the main run' % (run.key, SET))
            return None
        runs = registry.load().select(set_name=SET)
    if len(runs) < 2:
        raise ValueError('a comparison needs at least two runs, got %d' % len(runs))
    results = collect(runs)
    report(results)
    return figures.save(draw(results), 'model_comparison')


def main(argv=None):
    ap = figures.cli(__doc__)
    ap.add_argument('--set', default=SET, help='a [sets] entry of registry.toml (default %(default)s)')
    ap.add_argument('--runs', help='comma-separated run keys, instead of --set')
    args = ap.parse_args(argv)
    run = figures.setup(args)
    reg = registry.load()
    if args.runs:
        runs = reg.select(keys=args.runs.split(','))
    else:
        runs = reg.select(set_name=args.set)
    if (args.runs or args.set != SET) and args.out is None:
        fig_palettes.FIGURE_DIR = str(config.BY_MODEL_GRAPHICS / '_compare')
    make(run, runs)


if __name__ == '__main__':
    main()
