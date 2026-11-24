#!/usr/bin/env python
"""computational_cost: memory and compute of WRF-PartMC against WRF-AeroMELD (log bars).

  (a) quantities advected per grid cell
  (b) core-hours (core-minutes until 2026-11-14; the text uses core-hours)

No product: the numbers are typed in below, each with its source. They are the
notebook's (read_wrf_data.ipynb cell 49), ported as is. The layout is tighter
than the notebook's (smaller margins, golden-ratio panels); the content is the
same.

Where the numbers come from (2026-11-12, the runs of the paper).

  (a) WRF-PartMC 105,000 values = 5000 computational particles per cell
        (num_particles in its namelist.input) x 21 species (aero_data.dat),
        the particle masses alone, before weights and ids.
      WRF-AeroMELD 10 values = 9 latents + N.
  (b) core-minutes of the whole four-hour run, meteorology included (drawn in core-hours):
      WRF-PartMC 166,233 = 4:48:36 x 576 cores (SLURM job 1116966 on the authors' cluster,
        2026-11-02, the run that wrote the paper's aerosols file).
      WRF-AeroMELD 189 = 88.5 s x 128 cores: the mean of the per-tracer
        WRF-AeroMELD runs on one Perlmutter CPU node, 91 s (job 61166420) and
        86 s (job 61177092), 68.2 and 69.5 s of it time stepping.
      WRF (no aerosol) 127 = 59.5 s x 128 cores: the same run directory with
        chem_opt = 0 and no emission input, 59 s (job 61176723) and 60 s
        (job 61177093), 43.8 and 43.5 s time stepping.
      WRF-PartMC ran on another cluster, so its ratio to the others is
      approximate. Decoding (seconds for every surface cell) and WRF-PartMC's
      own diagnostics are not included.

Products: none.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from paperfigs import figures  # noqa: E402
from paperfigs.fig_palettes import TWO_COLUMN, create_label_box  # noqa: E402

#: (a) quantities advected per grid cell; sources in the module docstring.
MEMORY = (('WRF-\nPartMC', 105000), ('WRF-\nAeroMELD', 10))
#: (b) core-minutes of the whole run; sources in the module docstring.
CORE_MINUTES = (('WRF-\nPartMC', 166233), ('WRF-\nAeroMELD', 189), ('WRF, no\naerosol', 127))

# Neutral: model identity is line style elsewhere; blue/orange are the pop/point ramps.
BAR_COLORS = ('#8c8c8c', '#2b2b2b', '#d0d0d0')
INK, MUTED = '#0b0b0b', '#5c5c5c'


def log_bar_panel(ax, items, title, ylabel, verb, floor=1):
    """Log bars of (label, value) from `floor`; the first two are compared by an arrow. Values
    are written to the nearest integer, below 10 to one decimal."""
    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items], float)
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.55, color=BAR_COLORS[:len(vals)], zorder=2)
    ax.set_yscale('log')
    ax.set_ylim(floor, 10 ** np.ceil(np.log10(vals.max()) + 0.6))
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.35, f'{v:.1f}' if v < 10 else f'{v:,.0f}', ha='center', va='bottom',
                fontsize='small')
    hi, lo = vals[0], vals[1]
    ax.annotate('', xy=(0.5, lo), xytext=(0.5, hi),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    for yv in (hi, lo):
        ax.plot([0.275, 0.725], [yv, yv], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    r = float(f'{hi / lo:.1g}')
    ax.text(0.56, np.sqrt(hi * lo), rf'$\sim${r:,.0f}× {verb}', ha='left', va='center',
            fontsize='small', fontweight='bold', linespacing=1.1)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.55, len(vals) - 0.45)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(ylabel)
    create_label_box(ax, title)               # boxed, as every other figure's panel tags
    ax.tick_params(colors=MUTED, labelcolor=INK)


def draw(memory=MEMORY, core_minutes=CORE_MINUTES):
    # ~0.04 in of white on every side; the left margin and the gap hold the y-labels ((a)'s on two
    # lines, 2026-11-14), which leaves 2.3 x 1.69 in panels.
    H = 2.3
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, H),
                             gridspec_kw={'left': 0.125, 'right': 0.992, 'bottom': 0.36 / H,
                                          'top': 1 - 0.21 / H, 'wspace': 0.3})
    log_bar_panel(axes[0], memory, '(a) Advected quantities', 'Quantities advected\nper grid cell', 'fewer')
    core_hours = [(k, v / 60) for k, v in core_minutes]
    log_bar_panel(axes[1], core_hours, '(b) Computational cost', 'Core-hours, 4-h run', 'cheaper',
                  floor=.1)
    return fig


def make(run):
    """`run` is accepted for make_all; the figure does not depend on it."""
    return figures.save(draw(), 'computational_cost')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
