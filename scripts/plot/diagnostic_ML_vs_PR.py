#!/usr/bin/env python
"""diagnostic_ML_vs_PR: AeroMELD and PartMC spectra at the three sampling points.

Columns: config.TIMESTAMPS_SPECTRA (00:00, 02:00, 04:00). Rows: CCN/CN fraction
against s_env, frozen fraction against temperature, scattering and absorption
against wavelength. PartMC solid, AeroMELD dashed, one colour per point.

Optics are per kg of dry air on both sides (PartMC x its own ALT) and are NOT
rescaled to the decoded number: this figure compares absolute values, the
diagnostic panels compare shapes. Each side is drawn on its own wavelength grid
(ML 300-1000 nm by 100; PartMC 300, 400, 550, 600, 1000 nm).
From read_wrf_data.ipynb cell 45, with three changes: the times are picked by
timestamp, not by integer index [0, 8, 16]; PartMC's CCN lines get the point's
line width like every other PartMC line (they had matplotlib's default); and the
optics y-labels are on two lines, as the one-line versions overprinted.
The optics y-ranges are fig_palettes.SCATTERING_RANGE and ABSORPTION_RANGE,
shared with every optics figure.
2026-11-09, for consistency with the other figures: PartMC solid and AeroMELD
dashed (the notebook had them the other way round); axis labels, frozen-fraction
range and ticks, and wavelength ticks from fig_palettes; optics labelled every
decade; 7.6 in tall instead of 6.5, as the rows were squashed; a grid in
every panel, as in latent_populations.
Later on 2026-11-09: laid out in inches (LEFT ... ROW_GAP below) with golden
panels, 1.68 x 1.04 in (were 1.62 x 1.08, ratio 1.50); the margins shrank to
the labels (0.3 in had been empty above the titles, 0.13 in left, 0.19 in
right), so the figure is 7.15 in tall; and the y-labels are aligned.

Products: aeromeld_points, partmc_diag_points.
"""
import sys
from pathlib import Path
from string import ascii_lowercase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from paperfigs import common, config, figures, products  # noqa: E402
from paperfigs.fig_palettes import (ABSORPTION_RANGE, CCN_LABEL,  # noqa: E402
                                    FROZEN_FRACTION_LABEL, FROZEN_FRACTION_RANGE,
                                    FROZEN_FRACTION_TICKS, SCATTERING_RANGE,
                                    SUPERSATURATION_LABEL, TEMPERATURE_LABEL, TWO_COLUMN,
                                    WAVELENGTH_LABEL, WAVELENGTH_TICKS, create_label_box,
                                    decade_ticks, point_color, point_lw)

M_INV_TO_MM_INV = 1e6

# Layout, in inches. The margins hold just the labels, and every panel is
# golden (width / height = 1.618).
GOLDEN = (1 + 5 ** 0.5) / 2
LEFT = 0.79             # two-line y-labels and the 10^-15 tick labels
RIGHT = 0.18            # the last column's "1000" and "10^1" tick labels
TOP = 0.22              # column titles and panel tags
BOTTOM = 0.975          # the last row's x-axis titles and the legend
COL_GAP = 0.34          # the CCN row's "10^1" | "10^-1" at the column ends; at
                        # 0.25 they ran together
ROW_GAP = 0.60          # x tick labels and titles, then the next row's tags
AXES_W = (TWO_COLUMN - LEFT - RIGHT - 2 * COL_GAP) / 3
AXES_H = AXES_W / GOLDEN
FIG_H = TOP + 4 * AXES_H + 3 * ROW_GAP + BOTTOM


def make(run):
    ml_all = products.load('aeromeld_points', run=run.key)
    pmc_all = products.load('partmc_diag_points')
    if not np.allclose(ml_all['eps'], pmc_all['eps'], rtol=1e-6):
        raise ValueError('the CCN supersaturation grids differ')
    if not np.allclose(ml_all['temperature'], pmc_all['temperature'], atol=1e-4):
        raise ValueError('the frozen-fraction temperature grids differ')

    fig, axes = plt.subplots(4, 3, figsize=(TWO_COLUMN, FIG_H), sharex='row', sharey='row',
                             gridspec_kw={'left': LEFT / TWO_COLUMN, 'right': 1 - RIGHT / TWO_COLUMN,
                                          'top': 1 - TOP / FIG_H, 'bottom': BOTTOM / FIG_H,
                                          'wspace': COL_GAP / AXES_W, 'hspace': ROW_GAP / AXES_H})
    for i_time, ts in enumerate(config.TIMESTAMPS_SPECTRA):
        ml, pmc = common.at_time(ml_all, ts), common.at_time(pmc_all, ts)
        axes[0][i_time].set_title('$t$ = %.1f h' % float(ml['hours']), fontsize=8)
        ml_eps, pmc_eps = 100 * ml['eps'].values, 100 * pmc['eps'].values
        ml_wl, pmc_wl = 1e9 * ml['wavelength'].values, 1e9 * pmc['wavelength'].values
        for i_loc, label in enumerate(config.LOC_LABELS):
            c, lw = point_color(i_loc), point_lw(i_loc)
            # PartMC solid, AeroMELD dashed, as in every other figure. PartMC is
            # drawn first so it heads each point's legend column.
            pmc_kw = dict(color=c, lw=lw, label='PartMC Point %s' % label)
            ml_kw = dict(color=c, lw=lw, ls='--', label='AeroMELD Point %s' % label)
            axes[0][i_time].plot(pmc_eps, pmc['ccn_cdf'].values[i_loc], **pmc_kw)
            axes[0][i_time].plot(ml_eps, ml['ccn_cdf'].values[i_loc], **ml_kw)
            axes[1][i_time].plot(pmc['temperature'], pmc['frozen_fraction'].values[i_loc], **pmc_kw)
            axes[1][i_time].plot(ml['temperature'], ml['frozen_fraction'].values[i_loc], **ml_kw)
            for row, var in ((2, 'scattering'), (3, 'absorption')):
                axes[row][i_time].plot(pmc_wl, M_INV_TO_MM_INV * pmc[var].values[i_loc], **pmc_kw)
                axes[row][i_time].plot(ml_wl, M_INV_TO_MM_INV * ml[var].values[i_loc], **ml_kw)
        axes[0][i_time].set_xlabel(SUPERSATURATION_LABEL)
        axes[1][i_time].set_xlabel(TEMPERATURE_LABEL)
        axes[2][i_time].set_xlabel(WAVELENGTH_LABEL)
        axes[3][i_time].set_xlabel(WAVELENGTH_LABEL)
    for ax, letter in zip(axes.flatten(), ascii_lowercase):
        create_label_box(ax, '(%s)' % letter)
        ax.grid(True)            # every panel, as in latent_populations and the point panels

    axes[0][0].set_ylabel(CCN_LABEL)
    axes[0][0].set_xscale('log')
    axes[0][0].set_ylim([0, 1])
    axes[0][0].set_xlim([.1, 10])
    axes[1][0].set_ylabel(FROZEN_FRACTION_LABEL)
    axes[1][0].set_xlim([-40, 0])
    axes[1][0].set_yscale('log')
    axes[1][0].set_ylim(FROZEN_FRACTION_RANGE)
    axes[1][0].set_yticks(FROZEN_FRACTION_TICKS)
    axes[2][0].set_ylabel('Scattering\n' r'(Mm$^{-1}$ m$^3$ kg$^{-1}$)')
    axes[2][0].set_yscale('log')
    axes[2][0].set_ylim(SCATTERING_RANGE)
    axes[2][0].set_yticks(decade_ticks(*SCATTERING_RANGE))
    axes[2][0].set_xlim([300, 1000])
    axes[2][0].set_xticks(WAVELENGTH_TICKS)
    axes[3][0].set_ylabel('Absorption\n' r'(Mm$^{-1}$ m$^3$ kg$^{-1}$)')
    axes[3][0].set_yscale('log')
    axes[3][0].set_ylim(ABSORPTION_RANGE)
    axes[3][0].set_xlim([300, 1000])
    axes[3][0].set_xticks(WAVELENGTH_TICKS)
    axes[3][0].set_yticks(decade_ticks(*ABSORPTION_RANGE))
    for r in (2, 3):
        axes[r][0].yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=10))
        axes[r][0].yaxis.set_minor_formatter(NullFormatter())
    fig.align_ylabels(axes[:, 0])

    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    return figures.save(fig, 'diagnostic_ML_vs_PR')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
