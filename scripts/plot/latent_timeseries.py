#!/usr/bin/env python
"""latent_timeseries: the WRF-ML latent variables over time at the sampling points.

3 x 3 panels, one per latent (the same nine latent_space_normalized shows, chosen
from the final-instant maps), each with one line per sampling point.
From read_wrf_data.ipynb cell 33, with these changes: each point is read at
its own cell (the notebook read the cell one step north-east: 1-based WRF
indices used as 0-based numpy indices); the nine latents are chosen at the
final instant, as the maps are; a legend for the points (the notebook had
none); left margin .1 -> .12, so the third row's y-label stays on the page.

Products: wrfml_fields.
"""
import sys
from pathlib import Path
from string import ascii_lowercase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from paperfigs import common, config, figures, products  # noqa: E402
from paperfigs.fig_palettes import (TIME_LABEL, TWO_COLUMN, create_label_box,  # noqa: E402
                                    point_color, point_lw)


def draw(hours, z_points, latent_ids, show):
    """z_points: (latent, point, time); show: 0-based latents to draw."""
    fig, axes = plt.subplots(3, 3, figsize=(TWO_COLUMN, 4), sharex=True,
                             gridspec_kw={'bottom': 0.2, 'top': .95, 'left': .12,
                                          'right': .95, 'wspace': .5, 'hspace': .5})
    for count, (ax, k) in enumerate(zip(axes.flatten(), show)):
        for i_loc, label in enumerate(config.LOC_LABELS):
            ax.plot(hours, z_points[k, i_loc], color=point_color(i_loc), lw=point_lw(i_loc),
                    label='Point %s' % label)
        ax.set_xlim([hours[0], hours[-1]])
        create_label_box(ax, '(%s) $z_{%d}$' % (ascii_lowercase[count], latent_ids[k]))
    for ax in axes[-1]:
        ax.set_xlabel(TIME_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel('Latent value $z$')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    return fig


def make(run):
    ml = products.load('wrfml_fields', run=run.key)
    fig = draw(ml['hours'].values, ml['z_points'].values, ml['latent'].values,
               common.latents_to_show(ml['z'].values))
    return figures.save(fig, 'latent_timeseries')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
