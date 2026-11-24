#!/usr/bin/env python
"""diagnostic_panels_ABC: AeroMELD against PartMC at the three sampling points, final instant.

The paper's fig:point-panels. Three blocks, one per sampling point, stacked top
to bottom, each in an outlined box. A block's eight panels (draw_point):
(a) number and (b) total mass size distribution, (c) CCN spectrum (activated fraction),
(d) frozen fraction, (e) PartMC and (f) AeroMELD per-species mass
distributions, stacked, (g) scattering and (h) absorption.
Each block's header names its point, with a swatch of the point's colour (the
colour of its lines and of its marker in the domain figure) and its population
mix by number at the final instant, from partmc_fields.

Panels are lettered on through the blocks: A (a)-(h), B (i)-(p), C (q)-(x).
The columns are the same quantities in every block, so only the bottom block
carries the x-axis titles; the upper blocks keep their tick labels. One legend
for all three: species colours, and the reference solid / AeroMELD dotted with open markers.

Sized to share a page with a short caption: 6.69 x 9.92 in, i.e. 5.5 x 8.2 in
at the paper's text width (5.5 x 9.25 in text block). No text overlaps
(checked against every label's extent, 2026-11-09).

A test of SHAPE and COMPOSITION. Every extensive PartMC quantity is rescaled to
the decoded cell's total number, k = sum(AeroMELD n_prthst) / PartMC ntot,
because the latent vector is intensive: a WRF-ML cell pairs the shape it
carries with its own ntot, which the decoder passes through essentially
exactly. The activated and frozen fractions are intensive and never rescaled.

Both size distributions are densities, dN/dlog10(D) and dM/dlog10(D), each on
its own bin grid (AeroMELD 0.25 decade from 10^-8.75 m; PartMC 0.2 decade from
10^-9 m), drawn as stairs against their own edges. Wavelength grids differ too
(see diagnostic_ML_vs_PR).
History. Ported from read_wrf_data.ipynb cell 27 (diagnostic_panel_0/1/2.pdf
there) as three single-point figures, diagnostic_panel_{A,B,C}, with these
fixes:
  * units: everything is per kg of dry air (the decoder is handed WRF-ML's
    mixing ratio, and PartMC is rescaled onto it); the notebook labelled the
    axes m^-3. Mass is in ug kg^-1, which also removes the 1e-8 offset text
    that overprinted the (f) tag.
  * the (a)/(b) axis tops follow the curves drawn, not the unscaled PartMC
    curve (per m^3), which a rescaled curve could overshoot.
  * the stacked mass panels (e)/(f) draw their last bin; fill_between was
    given the left edges only, so the last bin had zero width.
  * layout: wspace .62 -> .8 for the two-line optics labels, left .075 -> .09
    so the widest tick labels keep the (a) label on the page, and all four
    size panels share the 10^-3..10 um axis ((a)/(b) were autoscaled).
  * AeroMELD's decoded dN and per-species dM are clipped at zero before use
    (curves, the (f) stack, and k): the decoder returns small negative bins,
    which a stack cannot hold. The aeromeld_points product keeps them.
  * (g) and (h) use fig_palettes.SCATTERING_RANGE and ABSORPTION_RANGE,
    shared with every optics figure (2026-11-09; (h) was 1e-3..1e1); since
    2026-11-15 three decades each, 1e0-1e3 and 1e-1-1e2 (every line fits:
    scattering 1.8-79, absorption 0.11-2.7).
  * 2026-11-09, for consistency with the other figures: axis labels,
    frozen-fraction range (was 1e-14..1) and ticks, and wavelength ticks from
    fig_palettes; the diameter axes say "Wet diameter"; a grid in every
    panel, as in latent_populations.
  * 2026-11-12: the reference is the exact answer (paperfigs.reference: the
    source populations' particles mixed in per-source transport's fractions,
    on the decoder's wet-diameter bins), not WRF-PartMC; "PartMC" in the
    variable names below now means that reference. Before that, the same day:
    the diameter axes said "Diameter": WRF-PartMC's diagnostics
    files bin the particles by DRY diameter, the decoder by wet diameter.
    Binned by dry diameter, the exact mixture at C has a mass-median diameter
    of 0.25 um (WRF-PartMC 0.26), by wet diameter 0.52 um (AeroMELD 0.52);
    at A and B the aerosol is nearly dry and the two agree.
  * 2026-11-09: the three became this one figure, in place of the paper's
    Figures 5-7; the single-point figures are no longer made. The
    size-distribution y-labels went onto two lines for it.

Products: aeromeld_points, partmc_diag_points, partmc_fields.
"""

import sys
from pathlib import Path
from string import ascii_lowercase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.lines  # noqa: E402
import matplotlib.offsetbox as moffset  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib.ticker import MaxNLocator  # noqa: E402

from paperfigs import common, config, figures, products, reference  # noqa: E402
from paperfigs.fig_palettes import (ABSORPTION_RANGE, CCN_LABEL, DIAMETER_LABEL,  # noqa: E402
                                    FROZEN_FRACTION_LABEL, FROZEN_FRACTION_RANGE,
                                    FROZEN_FRACTION_TICKS, SCATTERING_RANGE, SPECIES_COLORS,
                                    SUPERSATURATION_LABEL, TEMPERATURE_LABEL, TWO_COLUMN,
                                    WAVELENGTH_LABEL, WAVELENGTH_TICKS, create_label_box,
                                    decade_ticks, point_color, point_lw, species_label)

M_TO_UM = 1e6
UG_PER_KG = 1e9              # kg kg-1 -> ug kg-1
#: Both sides are binned by wet diameter (the decoder's bins), since the
#: reference is the exact answer (2026-11-12).
DIAM_LABEL = DIAMETER_LABEL
#: Diameter axis of all four size-distribution panels (um).
DIAM_LIM_UM = (1e-3, 10)
#: On two lines, like the optics labels: on one line they are taller than the
#: panels of the combined figure and ran into the rows above and below.
MASS_LABEL = r'$dM/d\log_{10}D$' '\n' r'($\mathrm{\mu}$g kg$^{-1}$)'
NUMBER_LABEL = r'$dN/d\log_{10}D$' '\n' r'(kg$^{-1}$)'
OPTICS_UNITS = r'(Mm$^{-1}$ m$^3$ kg$^{-1}$)'


def ceil_to_half(x):
    """x rounded up to a half step of its own decade (an axis top)."""
    step = 0.5 * 10 ** np.floor(np.log10(x))
    return np.ceil(x / step) * step


#: One block's panel tags, in panel order: [[number, total mass, CCN, frozen
#: fraction], [PartMC stack, AeroMELD stack, scattering, absorption]].
TAGS = 'abcdefgh'


def draw_point(axes, ml, pmc, i_loc, tags=TAGS, xlabels=True):
    """One point's eight panels into a 2 x 4 array of axes, from the two products
    already at one instant. `tags` are the eight panel letters in that order;
    xlabels=False leaves the axis titles off (the tick labels stay), for a
    block whose columns are titled by the block below it.

    Returns (style handles, species handles): PartMC and AeroMELD lines, and
    the species patches, for the caller's legends."""
    ml_edges = 10 ** common.log10_edges(ml) * M_TO_UM
    pmc_edges = 10 ** common.log10_edges(pmc) * M_TO_UM
    ml_width = np.diff(common.log10_edges(ml))
    ml_dN = np.maximum(ml['dNdlogDp'].values[i_loc], 0.0)    # clipped: see the docstring
    k = float(np.sum(ml_dN * ml_width) / pmc['ntot'].values[i_loc])
    print('point %s: exact extensive quantities x %.3f (decoded / exact number)'
          % (config.LOC_LABELS[i_loc], k))

    color = point_color(i_loc)
    pmc_kw = dict(color=color, ls='-', lw=point_lw(i_loc), label='Reference')
    ml_kw = dict(color=color, ls=':', lw=1.3, label='WRF-AeroMELD')
    # Open markers on the dotted line (as latent_populations), so WRF-AeroMELD stays visible
    # where it lies on the reference.
    # clip_on=False: the spectra's end markers sit on the axes' edges (s = 0.1 %, -40 C, 300 and
    # 1000 nm), where clipping hid half of each (2026-11-13)
    mk = dict(marker='o', ms=3.0, mfc='white', mec=color, mew=.8, clip_on=False)
    centres = lambda e: np.sqrt(e[1:] * e[:-1])

    def xlabel(ax, text):
        if xlabels:
            ax.set_xlabel(text)

    # (a) number size distribution
    ax = axes[0][0]
    pmc_dN = pmc['dNdlogDp'].values[i_loc]
    ax.stairs(k * pmc_dN, pmc_edges, baseline=None, **pmc_kw)
    ax.stairs(ml_dN, ml_edges, baseline=None, **ml_kw)
    ax.plot(centres(ml_edges)[::2], ml_dN[::2], ls='none', zorder=4, **mk)
    xlabel(ax, DIAM_LABEL)
    ax.set_ylabel(NUMBER_LABEL)
    ax.set_xscale('log')
    ax.set_xlim(DIAM_LIM_UM)
    ax.set_ylim(bottom=0, top=ceil_to_half(max((k * pmc_dN).max(), ml_dN.max())))
    create_label_box(ax, '(%s)' % tags[0])
    handles, labels = ax.get_legend_handles_labels()
    style_handles = dict(zip(labels, handles))

    # (b) total mass size distribution
    ax = axes[0][1]
    pmc_M = UG_PER_KG * k * pmc['dMdlogDp'].values[i_loc]    # (species, bin), VAE order
    ml_M = UG_PER_KG * np.maximum(ml['dMdlogDp'].values[i_loc], 0.0)
    ax.stairs(pmc_M.sum(0), pmc_edges, baseline=None, **pmc_kw)
    ax.stairs(ml_M.sum(0), ml_edges, baseline=None, **ml_kw)
    ax.plot(centres(ml_edges)[::2], ml_M.sum(0)[::2], ls='none', zorder=4, **mk)
    xlabel(ax, DIAM_LABEL)
    ax.set_ylabel(MASS_LABEL)
    ax.set_xscale('log')
    ax.set_xlim(DIAM_LIM_UM)
    ax.set_ylim(bottom=0, top=ceil_to_half(max(pmc_M.sum(0).max(), ml_M.sum(0).max())))
    create_label_box(ax, '(%s)' % tags[1])

    # (c) CCN spectrum
    ax = axes[0][2]
    ax.plot(100 * pmc['eps'].values, pmc['ccn_cdf'].values[i_loc], **pmc_kw)
    ax.plot(100 * ml['eps'].values, ml['ccn_cdf'].values[i_loc], markevery=12, **mk, **ml_kw)
    xlabel(ax, SUPERSATURATION_LABEL)
    # two lines: on one, "Activated fraction" reaches up to the block's header
    ax.set_ylabel(CCN_LABEL.replace(' ', '\n'))
    ax.set_xscale('log')
    ax.set_ylim([0, 1])
    ax.set_xlim([.1, 10])
    create_label_box(ax, '(%s)' % tags[2])

    # (d) frozen fraction
    ax = axes[0][3]
    ax.plot(pmc['temperature'].values, pmc['frozen_fraction'].values[i_loc], **pmc_kw)
    ax.plot(ml['temperature'].values, ml['frozen_fraction'].values[i_loc], markevery=12, **mk, **ml_kw)
    xlabel(ax, TEMPERATURE_LABEL)
    ax.set_ylabel(FROZEN_FRACTION_LABEL)      # one line: on two it runs into the CCN panel
    ax.set_yscale('log')
    ax.set_xlim([-40, 0])
    ax.set_ylim(FROZEN_FRACTION_RANGE)
    ax.set_yticks(FROZEN_FRACTION_TICKS)
    create_label_box(ax, '(%s)' % tags[3])

    # (g)/(h) optics
    pmc_wl, ml_wl = 1e9 * pmc['wavelength'].values, 1e9 * ml['wavelength'].values
    for ax, var, ylab, ylim, tag in (
            (axes[1][2], 'scattering', 'Scattering\n' + OPTICS_UNITS, SCATTERING_RANGE, tags[6]),
            (axes[1][3], 'absorption', 'Absorption\n' + OPTICS_UNITS, ABSORPTION_RANGE, tags[7])):
        ax.plot(pmc_wl, 1e6 * k * pmc[var].values[i_loc], ms=3, **pmc_kw)
        ax.plot(ml_wl, 1e6 * ml[var].values[i_loc], **mk, **ml_kw)
        xlabel(ax, WAVELENGTH_LABEL)
        ax.set_ylabel(ylab)
        ax.set_ylim(ylim)
        create_label_box(ax, '(%s)' % tag)
        ax.set_yscale('log')
        ax.set_yticks(decade_ticks(*ylim))
        ax.set_xlim([300, 1000])
        ax.set_xticks(WAVELENGTH_TICKS, ['400', '', '1000'])         # 700 unlabelled: room at 5.5 in

    # (e)/(f) speciated mass, stacked
    mass_max = 0.0
    for ax, vals, edges in ((axes[1][0], pmc_M, pmc_edges), (axes[1][1], ml_M, ml_edges)):
        baseline = np.zeros(vals.shape[1])
        for i_spec, name in enumerate(config.ML_SPECIES):
            top = baseline + vals[i_spec]
            # step='post' holds each value until the next x: repeat the last
            # value at the last right edge so the last bin has its width.
            ax.fill_between(edges, np.append(baseline, baseline[-1]), np.append(top, top[-1]),
                            step='post', label=name, color=SPECIES_COLORS[name], ec='black')
            baseline = top
        mass_max = max(mass_max, baseline.max())
        ax.set_xlim(DIAM_LIM_UM)
        xlabel(ax, DIAM_LABEL)
        ax.set_xscale('log')
    # one y-label for the two stacks, which share their axis: the (f) tag, "WRF-AeroMELD", reaches
    # over where (f)'s own label would be
    axes[1][0].set_ylabel(MASS_LABEL)
    # ticks no higher than 85% of the axis, so that the wide panel tags above the stacks never
    # cover the top tick label (they reach left of the axes)
    ticks = [t for t in MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(0, mass_max) if t <= mass_max]
    for ax in axes[1][:2]:
        ax.set_yticks(ticks)
        ax.set_ylim(bottom=0, top=max(1.08 * mass_max, ticks[-1] / .85))
    create_label_box(axes[1][0], '(%s) Reference' % tags[4])
    create_label_box(axes[1][1], '(%s) WRF-AeroMELD' % tags[5])
    handles, labels = axes[1][1].get_legend_handles_labels()
    species_handles = dict(zip(labels, handles))
    for ax in np.ravel(axes):
        ax.grid(True)            # every panel, as in latent_populations
    return style_handles, species_handles


def load_points(run):
    """(AeroMELD, exact answer) at the points at the final instant, grids checked."""
    ml = common.at_time(products.load('aeromeld_points', run=run.key), config.TIMESTAMP_FINAL)
    pmc = common.at_time(reference.points(run), config.TIMESTAMP_FINAL)
    if not np.allclose(ml['eps'], pmc['eps'], rtol=1e-6):
        raise ValueError('the CCN supersaturation grids differ')
    if not np.allclose(ml['temperature'], pmc['temperature'], atol=1e-4):
        raise ValueError('the frozen-fraction temperature grids differ')
    if list(ml['species'].values) != list(pmc['species'].values):
        raise ValueError('the two products order species differently')
    return ml, pmc


BOX_EDGE = dict(edgecolor='0.45', lw=0.8)    # the outline around each block

# Vertical layout, in inches from the top of the figure.
TOP_PAD = 0.02                   # above the first box
BOX_PAD = 0.07                   # inside a box, above its header
HEADER = 0.22                    # the header line
TAG_ROOM = 0.16                  # panel tags and offset texts above each axes
AXES_H = 0.58                    # 8.08 in in all, to fit the 9.25 in text height with its caption
GAP_TICKS = 0.40                 # between a block's rows: tick labels and tags
GAP_TITLES = 0.62                # the same, plus x-axis titles (bottom block)
BELOW_TICKS = 0.32               # under an upper block's second row, to its box
BELOW_TITLES = 0.46              # under the bottom block's second row, to its box
BLOCK_GAP = 0.08                 # between boxes
LEGEND_H = 0.66                  # the legends, 0.1 in below the last box (they touched it)
N_BLOCKS = len(config.LOC_LABELS)
#: Figure height (inches), from the layout above: 8.08 in.
FIG_H = (TOP_PAD + N_BLOCKS * (BOX_PAD + HEADER + TAG_ROOM + 2 * AXES_H)
         + (N_BLOCKS - 1) * (GAP_TICKS + BELOW_TICKS + BLOCK_GAP)
         + GAP_TITLES + BELOW_TITLES + LEGEND_H)

#: Box edges and the axes grid, as fractions of the figure width.
BOX_X0, BOX_X1 = 0.006, 0.994
LEFT, RIGHT, WSPACE = .135, .93, 1.15   # inside the box; room for the two-line y-labels (5.5 in page width)


def source_mass_per_particle(model):
    """Mass per particle of each source population (POP_SCENARIOS order), from the sources'
    particle-resolved speciated mass distributions (the populations product of `model`; the
    particle-resolved part is the same for every model)."""
    from paperfigs import errors
    pop = products.load('populations', model=model)
    if list(pop['scenario'].values) != list(config.POP_SCENARIOS):
        raise ValueError('populations product scenarios %s' % pop['scenario'].values)
    return errors.source_vectors(pop, 'pmc')[:, :225].sum(1)


def population_mix(ref, i_loc, mass_pp):
    """'73% urban, 27% background by mass (56%, 44% by number)': the point's mass fractions by
    source population (per-source transport, `ref` at the final instant, weighted by each
    source's mass per particle), largest first, those under 0.5% left out, and the same
    populations' number fractions. The points are chosen by mass (checks/pick_points.py)."""
    f = dict(zip(reference.SCENARIOS, ref['f_source'].values[i_loc]))
    num = np.array([f[s] for s in config.POP_SCENARIOS])
    w = num * mass_pp
    mass = w / w.sum()
    order = [k for k in np.argsort(-mass) if mass[k] >= 0.005]
    by_mass = ', '.join('%.0f%% %s' % (100 * mass[k], config.POP_LABELS[k].lower()) for k in order)
    by_number = ', '.join('%.0f' % (100 * num[k]) for k in order)
    return '%s by mass (%s%% by number)' % (by_mass, by_number)


def block_header(fig, y_top_in, i_loc, text):
    """Point swatch, bold point name and its population mix, at the top left of
    the block whose top edge is y_top_in inches below the figure top."""
    swatch = moffset.DrawingArea(16, 6)
    swatch.add_artist(matplotlib.lines.Line2D([0, 16], [3, 3], color=point_color(i_loc), lw=3,
                                              solid_capstyle='butt'))
    row = moffset.HPacker(children=[
        swatch,
        moffset.TextArea('Point %s' % config.LOC_LABELS[i_loc], textprops=dict(weight='bold')),
        moffset.TextArea(text, textprops=dict(size=7.5))], align='center', pad=0, sep=5)
    y = 1 - (y_top_in + BOX_PAD + HEADER / 2) / FIG_H
    fig.add_artist(moffset.AnchoredOffsetbox(
        loc='center left', child=row, frameon=False, pad=0, borderpad=0,
        bbox_to_anchor=(BOX_X0 + 0.012, y), bbox_transform=fig.transFigure))


def draw(ml, pmc, mass_pp):
    fig = plt.figure(figsize=(TWO_COLUMN, FIG_H))
    n_loc = len(config.LOC_LABELS)
    y = TOP_PAD
    style = species = None
    for i_loc in range(n_loc):
        last = i_loc == n_loc - 1
        gap, below = (GAP_TITLES, BELOW_TITLES) if last else (GAP_TICKS, BELOW_TICKS)
        block_top = y
        axes_top = block_top + BOX_PAD + HEADER + TAG_ROOM
        axes_bottom = axes_top + 2 * AXES_H + gap
        block_bottom = axes_bottom + below

        fig.add_artist(mpatches.FancyBboxPatch(
            (BOX_X0, 1 - block_bottom / FIG_H), BOX_X1 - BOX_X0, (block_bottom - block_top) / FIG_H,
            boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure,
            facecolor='none', zorder=-10, **BOX_EDGE))
        block_header(fig, block_top, i_loc, population_mix(pmc, i_loc, mass_pp))

        gs = fig.add_gridspec(2, 4, left=LEFT, right=RIGHT, wspace=WSPACE,
                              top=1 - axes_top / FIG_H, bottom=1 - axes_bottom / FIG_H,
                              hspace=gap / AXES_H)
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)])
        tags = ascii_lowercase[8 * i_loc: 8 * i_loc + 8]
        style, species = draw_point(axes, ml, pmc, i_loc, tags=tags, xlabels=last)
        y = block_bottom + BLOCK_GAP

    # One legend for the three blocks. Colour says the point (block headers);
    # the line style says the model, so its key is drawn in a neutral grey.
    legend_top = 1 - (y - BLOCK_GAP + 0.1) / FIG_H
    if config.SIZE_BASIS == 'dry':            # the dry distributions hold no water
        species = {k: v for k, v in species.items() if k != 'H2O'}
    fig.legend(species.values(), [species_label(k) for k in species], loc='upper center', ncol=5,
               bbox_to_anchor=(0.35, legend_top), borderaxespad=0)
    key = [matplotlib.lines.Line2D([], [], color='0.3', lw=1.2, ls='-', label='Reference'),
           matplotlib.lines.Line2D([], [], color='0.3', lw=1.3, ls=':', marker='o', ms=3.0, mfc='white',
                                   mew=.8, label='WRF-AeroMELD')]
    fig.legend(handles=key, loc='upper center', ncol=1, bbox_to_anchor=(0.85, legend_top), borderaxespad=0)
    return fig


def make(run):
    ml, pmc = load_points(run)
    return figures.save(draw(ml, pmc, source_mass_per_particle(run.model.key)), 'diagnostic_panels_ABC')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
