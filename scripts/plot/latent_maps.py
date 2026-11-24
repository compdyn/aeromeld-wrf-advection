#!/usr/bin/env python
"""latent_space_normalized: maps of the WRF-ML latent variables at the final instant.

3 x 3 maps of z_n at the surface, each on its own colour scale rounded out to
'nice' bounds. A 9-latent run shows all nine; a larger latent space shows the
nine with the largest spatial spread, labelled with their real index
(common.latents_to_show, shared with latent_timeseries).
From read_wrf_data.ipynb cell 31, which drew them at 03:00; every domain figure
is now at the final instant (04:00).
2026-11-09: laid out in inches (LEFT ... COL_GAP below), each map's height
from the grid's shape and the figure's height from the maps. matplotlib's
default subplot margins had left 0.95 in empty at the left and 0.4-0.5 in on
the other sides; the maps are now 1.59 in wide (were 1.18).

Products: wrfml_fields, grid.
"""
import sys
from pathlib import Path
from string import ascii_lowercase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import common, figures, products  # noqa: E402
from paperfigs.fig_palettes import TWO_COLUMN, create_label_box  # noqa: E402


# Layout, in inches.
LEFT, RIGHT = 0.03, 0.02
TOP, BOTTOM = 0.04, 0.09        # BOTTOM: the last row's lowest colour-bar tick label
TAG_ROOM = 0.17                 # the panel tag above each map
ROW_GAP = 0.06                  # under a row, above the next row's tags
MAP_TO_CBAR = 0.05
CBAR_W = 0.12
CBAR_TICKS = 0.40               # colour-bar ticks and labels ("-0.04")
COL_GAP = 0.08


def choose_fmt(vmin, vmax):
    span = vmax - vmin
    return '%.1f' if span >= 1 else '%.2f' if span >= 0.1 else '%.3f'


def nice_bounds(vmin, vmax, nice_steps=(1, 2, 2.5, 5, 10)):
    """vmin rounded down and vmax up to a multiple of a 'nice' step sized to
    the span, so a colour bar has at most about ten steps."""
    span = vmax - vmin
    if span == 0:
        return vmin, vmax
    base = 10 ** np.floor(np.log10(span))
    for step in nice_steps:
        if span / (step * base) <= 10:
            nice_step = step * base
            break
    else:
        nice_step = nice_steps[-1] * base
    return np.floor(vmin / nice_step) * nice_step, np.ceil(vmax / nice_step) * nice_step


def draw(grid, z, latent_ids):
    """z: (latent, south_north, west_east); latent_ids: the 1-based n of each z_n."""
    show = common.latents_to_show(z)
    crs = figures.map_crs(grid)
    n_y, n_x = grid['XLAT'].shape
    # Each map's height follows from its width and the grid's shape (square
    # cells), and the figure's height from the maps.
    cell_w = MAP_TO_CBAR + CBAR_W + CBAR_TICKS
    map_w = (TWO_COLUMN - LEFT - RIGHT - 3 * cell_w - 2 * COL_GAP) / 3
    map_h = map_w * n_y / n_x
    fig_h = TOP + 3 * (TAG_ROOM + map_h) + 2 * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(TWO_COLUMN, fig_h))

    def rect(x_in, y_top_in, w_in):
        """[x0, y0, w, h] in figure fraction, from inches measured from the
        left and from the TOP, for one map row."""
        return [x_in / TWO_COLUMN, 1 - (y_top_in + map_h) / fig_h, w_in / TWO_COLUMN,
                map_h / fig_h]

    for count, k in enumerate(show):
        row, col = divmod(count, 3)
        x0 = LEFT + col * (map_w + cell_w + COL_GAP)
        y_top = TOP + TAG_ROOM + row * (map_h + ROW_GAP + TAG_ROOM)
        ax = fig.add_axes(rect(x0, y_top, map_w), projection=crs)
        field = z[k]
        vmin, vmax = nice_bounds(field.min(), field.max())
        mesh = figures.plot_field(ax, grid, field, norm=mcolors.Normalize(vmin, vmax))
        create_label_box(ax, '(%s) $z_{%d}$' % (ascii_lowercase[count], latent_ids[k]))
        cax = fig.add_axes(rect(x0 + map_w + MAP_TO_CBAR, y_top, CBAR_W))
        cax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        cax.yaxis.set_major_formatter(mticker.FormatStrFormatter(choose_fmt(vmin, vmax)))
        fig.colorbar(mesh, cax=cax, orientation='vertical')
    return fig


def make(run):
    ml = products.load('wrfml_fields', run=run.key)
    grid = products.load('grid')
    return figures.save(draw(grid, ml['z'].values, ml['latent'].values),
                        'latent_space_normalized')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
