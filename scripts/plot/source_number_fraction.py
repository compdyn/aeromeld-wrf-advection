#!/usr/bin/env python
"""source_number_fraction: number fraction of each aerosol population at the sampling points.

One panel per point: the fraction of the number from the
background (scenario 561: the IC base and the boundary inflow), the urban IC
box (635) and the emissions (812), over the run.
From read_wrf_data.ipynb cell 51, with these changes:
  * each point is read at its own cell (the notebook read the cell one step
    north-east: 1-based WRF indices used as 0-based numpy indices);
  * populations are grouped by scenario. The notebook summed sources 000-003
    and called 003 alone "Emissions", leaving out 004, the second emission
    source (its num_flux is zero in this run, so nothing changes today);
  * the time axis comes from the file's timestamps, not linspace(0, 4, 25),
    which assumed 10-min output;
  * panels are labelled with their point, and the lines take the populations'
    colours (fig_palettes.POP_COLORS, as in latent_populations) rather than
    matplotlib's default cycle, whose orange read as a sampling point.

2026-11-12: from per-source transport in the run's replay
(paperfigs.reference), every 15 minutes, not WRF-PartMC's particle tags.

Products: populations (via paperfigs.reference), data/transport/<replay>.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import config, figures, reference  # noqa: E402
from paperfigs.fig_palettes import (TIME_LABEL, TWO_COLUMN, create_label_box,  # noqa: E402
                                    pop_color, pop_lw)

#: (label, scenario) of each line, in config.POP_SCENARIOS order, so each takes
#: its population's colour.
POPULATIONS = tuple(zip(config.POP_LABELS, config.POP_SCENARIOS))


def draw(hours, n_source, scenario):
    """n_source: (source, point, time) number, any one basis; scenario: (source,)."""
    total = n_source.sum(0)
    left, right, H = 0.095, 0.985, 1.949
    fig, axes = plt.subplots(1, n_source.shape[1], figsize=(TWO_COLUMN, H),
                             gridspec_kw={'bottom': 0.7 / H, 'top': 1 - .17/ H, 'right': right,
                                          'left': left, 'wspace': .25, 'hspace': .75})
    for i_loc, ax in enumerate(axes):
        for i_pop, (label, scen) in enumerate(POPULATIONS):
            ax.plot(hours, n_source[scenario == scen, i_loc].sum(0) / total[i_loc], label=label,
                    color=pop_color(i_pop), lw=pop_lw(i_pop))
        create_label_box(ax, '(%s) Point %s' % ('abc'[i_loc], config.LOC_LABELS[i_loc]))
        ax.set_ylim([0, 1.05])
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim([hours[0], hours[-1]])
        ax.set_xticks(np.arange(0, hours[-1] + .1, 4))       # 0, 4, 8, 12 h (2026-11-15)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    axes[0].set_ylabel('Source fraction')
    return fig


def make(run):
    pts = reference.points(run)                     # per-source transport of the run's replay
    scen = np.asarray(reference.SCENARIOS)
    unknown = sorted(set(scen.tolist()) - {s for _, s in POPULATIONS})
    if unknown:
        raise ValueError('sources from scenarios %s belong to no population' % unknown)
    fig = draw(pts['hours'].values, np.moveaxis(pts['f_source'].values, -1, 0), scen)
    return figures.save(fig, 'source_number_fraction')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
