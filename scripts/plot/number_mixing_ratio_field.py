#!/usr/bin/env python
"""number_mixing_ratio_field: total number mixing ratio, AeroMELD vs PartMC, final instant.

(a) WRF-ML (AeroMELD) ntot, (b) WRF-PartMC ntot, (c) their difference, at the
surface, both in # per kg dry air (PartMC's TOT_NUM_CONC x its ALT). Panels
(a)/(b) share a colour scale up to the 99th percentile of both; (c) is
symmetric about zero at the 99th percentile of |difference|.
From read_wrf_data.ipynb cell 13.
2026-11-09: each colour bar's power of ten moved into its label ("10^9 # kg^-1
dry air", figures.scaled_colorbar, as in domain); the offset text ("1e8")
overprinted the longer difference label. The bars moved up 0.015, as the
labels reached below the figure.

Products: wrfml_fields, partmc_fields, grid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import common, config, figures, products  # noqa: E402
from paperfigs.fig_palettes import TWO_COLUMN, create_label_box  # noqa: E402


#: Bottom of both colour bars (figure fraction). 0.15 put the labels'
#: descenders 0.01 in below the figure; 0.165 leaves ~0.03 in.
CBAR_Y = 0.165


def draw(grid, f_ml, f_pmc):
    vmax = np.nanpercentile(np.concatenate([f_pmc.ravel(), f_ml.ravel()]), 99)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    diff = f_ml - f_pmc
    dmax = max(np.nanpercentile(np.abs(diff), 99), 1e-30)
    dnorm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)

    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, TWO_COLUMN / 2.5),
                             subplot_kw={'projection': figures.map_crs(grid)})
    fig.subplots_adjust(left=.025, right=.975, bottom=.2, top=.95)
    m0 = figures.plot_field(axes[0], grid, f_ml, norm=norm)
    figures.plot_field(axes[1], grid, f_pmc, norm=norm)
    pos0, pos1 = axes[0].get_position(), axes[1].get_position()
    cax = fig.add_axes([pos0.x0, CBAR_Y, pos1.x1 - pos0.x0, .05])
    figures.scaled_colorbar(fig, m0, cax, figures.NUMBER_MIXING_RATIO_LABEL)
    m2 = figures.plot_field(axes[2], grid, diff, norm=dnorm, cmap='RdBu_r')
    pos2 = axes[2].get_position()
    cax = fig.add_axes([pos2.x0, CBAR_Y, pos2.x1 - pos2.x0, .05])
    figures.scaled_colorbar(fig, m2, cax, r'Difference ($10^{%d}$ # kg$^{-1}$ dry air)')
    create_label_box(axes[0], '(a) AeroMELD')
    create_label_box(axes[1], '(b) PartMC')
    create_label_box(axes[2], '(c) AeroMELD$-$PartMC')
    return fig


def make(run):
    ml = products.load('wrfml_fields', run=run.key)
    fields = products.load('partmc_fields')
    grid = products.load('grid')
    ts = ml.attrs['timestamp']
    i = common.time_index([str(t) for t in fields['instant'].values], ts)
    return figures.save(draw(grid, ml['ntot'].values, fields['ntot'].values[i]),
                        'number_mixing_ratio_field')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
