#!/usr/bin/env python
"""comparison_tracers: each scheme's overall error against the number of tracers it advects per grid cell.

The overall error (mean over the six diagnostics) at 12 h over the mixed cells, for WRF-AeroMELD
and the sectional and modal schemes carried by WRF's transport (paperfigs/comparison.py), against
the advected tracers per grid cell (number plus species masses per bin or mode; WRF-AeroMELD N and
N z) on a log axis, since doubling the tracers doubles the transport's cost. A dashed line joins
the two schemes of each family (modal: MAM4 and MATRIX; sectional: MOSAIC 8 and TOMAS-40), the
family's trade-off of tracers against error; WRF-AeroMELD is a single point. Each point's label sits
next to it, without a leader line.

2026-11-18: labels are the scheme's name only, beside the point (a three-line variant
with the tracers and overall error under the name was tried and not kept); the families read
"Modal models" and "Sectional models", the sectional one above its line.

2026-11-15: the open markers (the overall error without the frozen fraction) and
their legend are gone; the x axis is a plain log axis (10, 100, 1000, with minor ticks) instead of
ticks at the schemes' own counts; the family lines are new; y from 0, "Overall error (%)".

Products: none; reads the baseline study's numbers (paperfigs/comparison.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter  # noqa: E402

from paperfigs import comparison, figures  # noqa: E402
from paperfigs.comparison import SCHEMES  # noqa: E402
from paperfigs.fig_palettes import SINGLE_COLUMN  # noqa: E402

INK, GRID = '#1a1a1a', '#e3e3e3'
XLIM = (5, 2000)
YLIM = (0, 25)
NAME_SIZE = 6.5
#: Each scheme's label: offset (points) of the label from the marker, and its alignment (ha, va).
LABEL_AT = {'aeromeld': ((5, 0), ('left', 'center')),
            'mam4l': ((0, 5), ('center', 'bottom')),
            'matrix1': ((5, 0), ('left', 'center')),
            'sec8': ((0, 5), ('center', 'bottom')),
            'tomas40': ((5, 0), ('left', 'center'))}
#: The families: the two schemes joined, the line's colour (the darker of the pair's), and where
#: its name goes: a fraction along the line from the first scheme, and an offset (points) and
#: alignment from there.
FAMILIES = (('Modal\nmodels', ('mam4l', 'matrix1'), '#542788', .5, (-6, -3), ('right', 'top')),
            ('Sectional\nmodels', ('sec8', 'tomas40'), '#8c510a', .5, (4, 2), ('left', 'bottom')))


def draw():
    tr = comparison.tracers()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.35),
                           gridspec_kw=dict(left=.15, right=.97, top=.97, bottom=.18))
    keys = comparison.present()
    err = {k: 100 * comparison.error(k) for k in keys}
    for name, pair, color, at, (dx, dy), (ha, va) in FAMILIES:
        if not all(k in err for k in pair):
            continue
        x = np.array([tr[k] for k in pair], float)
        y = np.array([err[k] for k in pair])
        ax.plot(x, y, color=color, lw=.9, ls=(0, (4, 2.5)), zorder=2)
        # the name at a fraction along the line (log x, linear y)
        xm = 10 ** (np.log10(x[0]) + at * np.diff(np.log10(x))[0])
        ym = y[0] + at * (y[1] - y[0])
        ax.annotate(name, (xm, ym), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va,
                    fontsize=6.5, color=color, style='italic', linespacing=1.0, multialignment=ha)
    for k in keys:
        name, _, color, marker, _ = SCHEMES[k]
        x, y = tr[k], err[k]
        ax.scatter([x], [y], s=30, marker=marker, color=color, edgecolor='white', lw=.5, zorder=5, clip_on=False)
        (dx, dy), (ha, va) = LABEL_AT[k]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va,
                    fontsize=NAME_SIZE, color=INK, zorder=6, annotation_clip=False)
        print('%-9s %4d tracers  overall error %5.2f%%' % (k, x, y))
    ax.set_xscale('log')
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '%d' % v))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Advected tracers per grid cell')
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel('Overall error (%)')
    ax.grid(which='major', color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def make(run=None):
    return figures.save(draw(), 'comparison_tracers' + comparison.SUFFIX)


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    figures.setup(args)
    make()


if __name__ == '__main__':
    main()
