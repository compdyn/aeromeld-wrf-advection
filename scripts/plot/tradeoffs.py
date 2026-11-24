#!/usr/bin/env python
"""tradeoffs: the two trade-offs of the paper, in the case at the end of the run (12 h).

Since 2026-11-15 two single-panel figures, one per appendix: tradeoffs_transport (a) and
tradeoffs_decoders (b), untagged; the combined two-panel figure, tradeoffs, is still made.

(a) Transport schemes, for WRF-AeroMELD's decoder (decoder 3, trained on mixtures of up to 16):
the fraction error (sources carried in the wrong proportions) against the off-mixture error (the
latent vector pushed off the mixtures of the sources). (b) Decoders, on the exact mixtures of
per-source transport: the reconstruction error (decoding each source population) against the
averaging error (decoding a mixture differs from mixing the decoded sources).

Both are the paper's error (paperfigs/errors.py), the mean over the six diagnostics, averaged
over the mixed cells; each marker is the median over the ten training seeds (the seeds differ
little; the text gives their spread), and each label gives the median of the stage's error
measured directly (transport error, representation error). The two axes of a panel are identical (same
range and ticks). The grey diagonals mark a constant sum of the two terms, which bounds the
stage's error (exactly: each diagnostic's error is a metric, so it obeys the triangle inequality,
and so does its mean over cells and diagnostics; paperfigs/errors.py); the one from the
upper-left corner says so. The recommended choice is boxed.

Since 2026-11-13: medians only (the seeds were small dots), every label on a leader
line, the affine decoder a filled circle (its old triangle sat half below the axis), axes
matched, the panel tags lifted clear of the diagonals.

Products: none; reads data/decomposition.json (scripts/decomposition.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.offsetbox as moffset  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from paperfigs import config, figures  # noqa: E402
from paperfigs.fig_palettes import AFFX_STYLES, TWO_COLUMN  # noqa: E402

INK = '#262626'
GRID = '#e6e6e6'
DIAG = '#bdbdbd'
DATA = Path(__import__('os').environ.get('DECOMPOSITION_JSON', config.REPO / 'data' / 'decomposition.json'))
MAIN = config.recipe('twin_k16')
#: (key, label, colour, marker): the paper's numbered transport schemes and decoders, named as in
#: Section 2 (2026-11-13: the same names everywhere).
REC = '\n(WRF-AeroMELD)'
SCHEMES = (('donor', 'Transport scheme 1:\ndonor-cell', '#b2182b', 'o'),
           ('shB', 'Transport scheme 2:\nshared limiter', '#2166ac', 's'),
           ('sep', 'Transport scheme 3:\nseparate limiters' + REC, '#4393c3', 'D'))
DECODERS = ((config.recipe('twin_none'), 'Decoder 1:\nno mixtures'), (config.recipe('twin_pairs'), 'Decoder 2:\npairs'),
            (config.recipe('twin_k16'), 'Decoder 3:\nup to 16' + REC), (config.recipe('twin_k32'), 'Decoder 4:\nup to 32'),
            (config.recipe('ax_rel1'), 'Decoder 5:\naffine'))
RECOMMENDED = {'a': 'sep', 'b': config.recipe('twin_k16')}
#: Each panel: axis limit, tick step, diagonal step, the stage's error (the label of the corner
#: diagonal, which bounds it: the error is at most the sum of its two terms, by the triangle
#: inequality).
PANELS = {'a': (12, 2, 4, 'transport error'), 'b': (10, 2, 5, 'representation error')}
#: Each label: position in data coordinates, alignment, and where on the text its leader line
#: starts (relpos, axes fraction of the text box).
LABELS = {'donor': ((11.9, 3.0), 'right', (1, 0)), 'shB': ((5.0, 1.1), 'left', (0, 0)),
          'sep': ((0.3, 4.4), 'left', (.15, 0)),
          config.recipe('twin_none'): ((5.6, 7.5), 'left', (0, .5)), config.recipe('twin_pairs'): ((0.85, 4.4), 'left', (.7, 0)),
          config.recipe('twin_k16'): ((5.25, 4.9), 'left', (0, 0)), config.recipe('twin_k32'): ((5.3, 2.75), 'left', (0, .5)),
          config.recipe('ax_rel1'): ((9.8, 0.75), 'right', (1, 0))}


def values(seeds, scheme, term):
    return 100 * np.array([seeds[k][scheme][term]['overall'] for k in sorted(seeds, key=int)])


def point(ax, x, y, colour, marker, label, value, boxed, where, stage):
    # the recommended choice on top: decoders 3 and 4 nearly coincide
    ax.scatter([x], [y], s=42, color=colour, marker=marker, edgecolor='white', lw=.6, zorder=6.5 if boxed else 6,
               clip_on=False)
    if boxed:
        ax.scatter([x], [y], s=190, marker='s', facecolor='none', edgecolor=INK, lw=.9, zorder=7, clip_on=False)
    (tx, ty), ha, relpos = where
    ax.annotate('%s\n%s %.1f%%' % (label, stage, value), (x, y), xytext=(tx, ty), textcoords='data', ha=ha,
                va='bottom', fontsize=6.5, color=INK, linespacing=1.0, zorder=8, annotation_clip=False,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'),
                arrowprops=dict(arrowstyle='-', color='#8c8c8c', lw=.5, shrinkA=0, shrinkB=6 if boxed else 4,
                                relpos=relpos))


def frame(ax, panel, xlabel, ylabel):
    lim, tick, step, words = PANELS[panel]
    for c in range(step, 2 * lim + 1, step):
        ax.plot([0, c], [c, 0], color=DIAG, lw=.6, ls=(0, (3, 2)), zorder=1)
        x0 = max(0.0, c - lim)                     # the line's upper-left end in the panel
        if lim - x0 > .12 * lim:
            d = .02 * lim
            ax.text(x0 + d, c - x0 - d, '%s \u2264 %d%%' % (words, c) if c == lim else '%d%%' % c,
                    fontsize=5.5, color='#8c8c8c',
                    rotation=-45, rotation_mode='anchor', ha='left', va='bottom', zorder=2)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(tick))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)


def tag(ax, text):
    """The boxed panel tag, above the top right corner, clear of the diagonals."""
    box = moffset.AnchoredOffsetbox(loc='lower right', child=moffset.TextArea(text, textprops=dict(size=7)), pad=.2, frameon=True,
                                    bbox_to_anchor=(1.0, 1.035), bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(box)


def draw_transport(ax, d):
    """(a): the transport schemes, off-mixture against fraction error."""
    main = d['models'][MAIN]['seeds']
    frame(ax, 'a', 'Fraction error (%)', 'Off-mixture error (%)')
    for s, label, colour, marker in SCHEMES:
        point(ax, np.median(values(main, s, 'fraction')), np.median(values(main, s, 'offmixture')), colour,
              marker, label, np.median(values(main, s, 'transport')), s == RECOMMENDED['a'], LABELS[s],
              'transport error')


def draw_decoders(ax, d):
    """(b): the decoders, averaging against reconstruction error."""
    models = d['models']
    frame(ax, 'b', 'Reconstruction error (%)', 'Averaging error (%)')
    for m, label in DECODERS:
        seeds = models[m]['seeds']
        colour, marker = AFFX_STYLES[config.base_name(m)]
        point(ax, np.median(values(seeds, 'oracle', 'reconstruction')),
              np.median(values(seeds, 'oracle', 'mixture')), colour, marker, label,
              np.median(values(seeds, 'oracle', 'decoder')), m == RECOMMENDED['b'], LABELS[m],
              'representation\nerror')


def draw(d):
    """The two panels side by side, tagged (a) and (b): the old combined figure."""
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, .58 * TWO_COLUMN))
    fig.subplots_adjust(left=.085, right=.975, bottom=.13, top=.9, wspace=.3)
    draw_transport(axes[0], d)
    draw_decoders(axes[1], d)
    tag(axes[0], '(a) Transport scheme error trade-offs')
    tag(axes[1], '(b) Decoder error trade-offs')
    return fig


#: A single panel's figure (inches): the combined figure's square axes (2.13 in), so the labels,
#: placed in data coordinates, and the font sizes print as they did there. Margins: the y label
#: and ticks on the left, the x label and ticks below, room above for the top tick label.
SINGLE_AXES, SINGLE_LEFT, SINGLE_RIGHT, SINGLE_BOTTOM, SINGLE_TOP = 2.13, .5, .1, .42, .12


def draw_single(d, panel):
    """One panel, untagged, as its own figure (each goes to its own appendix)."""
    w = SINGLE_LEFT + SINGLE_AXES + SINGLE_RIGHT
    h = SINGLE_BOTTOM + SINGLE_AXES + SINGLE_TOP
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([SINGLE_LEFT / w, SINGLE_BOTTOM / h, SINGLE_AXES / w, SINGLE_AXES / h])
    (draw_transport if panel == 'a' else draw_decoders)(ax, d)
    return fig


def make(run=None):
    d = json.loads(DATA.read_text())
    return [figures.save(draw_single(d, 'a'), 'tradeoffs_transport'),
            figures.save(draw_single(d, 'b'), 'tradeoffs_decoders')]
    # The combined two-panel figure (draw) is no longer in the paper (2026-11-15: one panel per appendix).


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    figures.setup(args)
    make()


if __name__ == '__main__':
    main()
