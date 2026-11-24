#!/usr/bin/env python
"""frozen_fraction: frozen-fraction maps at four temperatures, AeroMELD vs the exact answer.

Rows: AeroMELD, PartMC, AeroMELD / PartMC. Columns: the temperature-grid
points nearest analysis.SMAPE_TEMP_C (-40, -30, -20, -10 C), the violins' and
model comparison's frozen-fraction conditions, picked as analysis.conditions
picks them. Surface, final instant.

The third row is a RATIO: frozen fraction spans decades, where a difference
would show only the bright end. Both sides share one temperature grid (checked).
From read_wrf_data.ipynb cell 22, whose third row was labelled "Difference"
although it showed the ratio.

Changed 2026-11-09: the columns were the grid points nearest -40, -24, -16 and
-8 C (indices 0, 40, 60, 80, the notebook's); now the violins' temperatures,
as the CCN and optics maps use the violins' s_env and wavelengths. The colour
scale moved down a decade, 1e-11..1e-1 -> 1e-12..1e-2 (still 10 decades): at
-8 C, 88 % of the cells were below 1e-11; at the new temperatures every cell
is within 3.4e-12..2.8e-2. The colour bar says "Frozen fraction" (no "(1)").
Also 2026-11-09: the ratio bar's label is "AeroMELD / PartMC", as in
optics_ML_vs_PR (it was a stacked fraction that ran past the figure's right
edge), and both bars sit 0.005 further left, clear of the edge. The column
titles' temperature is inside the math, for a minus sign rather than a hyphen
and a single space after "=" (matplotlib already spaces "=" in math).

2026-11-12: the reference is the exact answer (paperfigs.reference), not
WRF-PartMC; rows "WRF-AeroMELD", "Exact", and the comparison's colour scale
narrowed to the much smaller differences.
Products: aeromeld_domain, partmc_diag_domain, grid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import analysis, figures, products, reference  # noqa: E402
from paperfigs.fig_palettes import FROZEN_FRACTION_LABEL, TWO_COLUMN  # noqa: E402

#: Colour scale of the AeroMELD and PartMC rows (10 decades).
FROZEN_NORM = (1e-12, 1e-2)


def make(run):
    ml = products.load('aeromeld_domain', run=run.key)
    pmc = reference.domain(run, ml.attrs['timestamp'])        # the exact answer
    grid = products.load('grid')
    figures.same_instant(ml, pmc)
    temp = pmc['temperature'].values
    if not np.allclose(temp, ml['temperature'].values, atol=1e-4):
        raise ValueError('AeroMELD and reference temperature grids differ')
    crs = figures.map_crs(grid)
    temp_index = [int(np.argmin(np.abs(temp - T))) for T in analysis.SMAPE_TEMP_C]

    norm = mcolors.LogNorm(*FROZEN_NORM)
    ratio_norm = mcolors.LogNorm(vmin=1 / 3, vmax=3)
    fig, axes = plt.subplots(3, len(temp_index), figsize=(TWO_COLUMN, 3.6),
                             subplot_kw={'projection': crs})
    fig.subplots_adjust(left=.05, right=.8, bottom=.02, top=.93, wspace=.06, hspace=.08)
    for i_col, i_T in enumerate(temp_index):
        f_ml = ml['frozen_fraction'].values[:, :, i_T]
        f_pmc = pmc['frozen_fraction'].values[:, :, i_T]
        pcm = figures.plot_field(axes[0][i_col], grid, f_ml, norm=norm)
        figures.plot_field(axes[1][i_col], grid, f_pmc, norm=norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = f_ml / f_pmc
        pcm_ratio = figures.plot_field(axes[2][i_col], grid, ratio, norm=ratio_norm,
                                       cmap='RdBu_r')
        axes[0][i_col].set_title(r'$T = %.0f\,^{\circ}$C' % temp[i_T])
    figures.row_labels(axes[:, 0], ['WRF-AeroMELD', 'Reference', 'Ratio'])

    # Colour bars at x = .82 (at the 5.5 in page width): room for the 10^-12 tick labels and
    # the "Frozen fraction" label.
    pos0, pos1 = axes[1, -1].get_position(), axes[0, -1].get_position()
    cax = fig.add_axes([.82, pos0.y0, .022, pos1.y1 - pos0.y0])
    fig.colorbar(pcm, cax=cax, orientation='vertical').set_label(FROZEN_FRACTION_LABEL)
    pos0 = axes[2, -1].get_position()
    cax = fig.add_axes([.82, pos0.y0, .022, pos0.y1 - pos0.y0])
    cb = fig.colorbar(pcm_ratio, cax=cax, orientation='vertical')
    cb.set_label('WRF-AeroMELD /\nreference')
    cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_ticks([1 / 3, .5, 1, 2, 3], labels=['1/3', '1/2', '1', '2', '3'])
    return figures.save(fig, 'frozen_fraction')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
