#!/usr/bin/env python
"""diagnostic_maps: four diagnostics over the domain, WRF-AeroMELD against the particle-resolved reference.

Columns: activated fraction at s = 0.3 %, frozen fraction at T = -20 C, scattering and absorption at
700 nm. Rows: WRF-AeroMELD, the particle-resolved reference (paperfigs.reference: per-source
transport of the source populations' particle-resolved diagnostics), and their ratio. Lowest model
level, end of the run (12 h), the main run, as the three figures it replaces (2026-11-15):
CCN_ML_vs_PR, frozen_fraction and optics_ML_vs_PR (plot/ccn_maps.py, frozen_fraction.py,
optics_maps.py, kept but no longer made), whose loading, conditions and colour maps it reuses.

  * s = 0.3 %: both spectra interpolated to it on their own grid (as ccn_maps).
  * T = -20 C: the temperature-grid point nearest it, -20.2 C (the grid steps 0.4 C; as
    frozen_fraction), titled -20 C as there.
  * 700 nm: a point of both wavelength grids, taken exactly (as optics_maps).
  * The value rows: viridis, each column on its own colour bar under the two rows. Activated
    fraction CCN_RANGE, 0.35-0.7 (2026-11-18: on 0-1 the maps were washed out; at 12 h both
    rows span 0.39-0.67, the background about 0.40); frozen fraction 1e-10-1e-5 (one temperature spans 7e-10-1.4e-6, so
    frozen_fraction's ten decades, for four temperatures, would leave the map nearly uniform);
    scattering and absorption the shared ranges (fig_palettes.SCATTERING_RANGE, ABSORPTION_RANGE),
    in Mm^-1 m^3 kg^-1 (per kg of dry air).
  * The ratio row: WRF-AeroMELD / reference, all four on one diverging log scale, RATIO_RANGE
    (0.8-1.25, symmetric in log), with one colour bar. Over the mixed cells it holds 99.8 % or more
    of the cells of every column.

The ratio's median and 5th-95th percentiles over the mixed cells (reference.mixed: at least two
populations each hold 5 % of the number), and its median over the cells of the background population
alone (at least 99 % of the number), are printed, for the text.

Products: aeromeld_domain, grid, populations (the reference), data/transport/<replay>.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import common, figures, products, reference  # noqa: E402
from paperfigs.fig_palettes import ABSORPTION_RANGE, SCATTERING_RANGE, TWO_COLUMN, decade_ticks  # noqa: E402

SAT = 0.003                     # s as a fraction
TEMP_C = -20.0
WL_NM = 700
M_INV_TO_MM_INV = 1e6
CCN_RANGE = (0.35, 0.7)
CCN_TICKS = (0.4, 0.5, 0.6, 0.7)
RATIO_RANGE = (0.8, 1.25)
RATIO_TICKS = (0.8, 0.9, 1, 1.1, 1.25)
OPTICS_UNITS = r'Mm$^{-1}$ m$^3$ kg$^{-1}$'

# Layout (inches).
LEFT, RIGHT, WSPACE = .36, .04, .07      # room on the left for the rotated row names
TITLE_H = .36                            # the two-line column titles
HSPACE = .05                             # between the two value rows
VALUE_KEY_H = .5                         # under them: each column's colour bar, ticks and units
RATIO_KEY_H = .48                        # under the ratio row: its colour bar, ticks and label
BAR_H = .085


def columns(ml, pmc):
    """[(title, AeroMELD field, reference field, norm, ticks, tick labels, bar label, extend)]."""
    temp = pmc['temperature'].values
    if not np.allclose(temp, ml['temperature'].values, atol=1e-4):
        raise ValueError('AeroMELD and reference temperature grids differ')
    i_t = int(np.argmin(np.abs(temp - TEMP_C)))
    ml_nm, pmc_nm = ml['wavelength'].values * 1e9, pmc['wavelength'].values * 1e9
    ccn = [common.interp_along_last_axis(d['ccn_cdf'].values, d['eps'].values, SAT) for d in (ml, pmc)]
    frozen = [d['frozen_fraction'].values[:, :, i_t] for d in (ml, pmc)]

    def optics(var):
        return [M_INV_TO_MM_INV * d[var].values[:, :, common.exact_index(nm, WL_NM)]
                for d, nm in ((ml, ml_nm), (pmc, pmc_nm))]

    log_labels = lambda ticks: [r'$10^{%d}$' % round(np.log10(t)) for t in ticks]
    frozen_ticks = decade_ticks(1e-10, 1e-5)
    return [
        ('Activated fraction\n' r'$s = %.1f$%%' % (100 * SAT), *ccn, mcolors.Normalize(*CCN_RANGE),
         CCN_TICKS, ['%g' % t for t in CCN_TICKS], '', 'neither'),
        ('Frozen fraction\n' r'$T = %.0f\,^{\circ}$C' % temp[i_t], *frozen, mcolors.LogNorm(1e-10, 1e-5),
         frozen_ticks, [lab if k % 2 == 0 else '' for k, lab in enumerate(log_labels(frozen_ticks))], '',
         'neither'),
        ('Scattering\n' r'$\lambda = %d$ nm' % WL_NM, *optics('scattering'), mcolors.LogNorm(*SCATTERING_RANGE),
         decade_ticks(*SCATTERING_RANGE), log_labels(decade_ticks(*SCATTERING_RANGE)), OPTICS_UNITS, 'max'),
        ('Absorption\n' r'$\lambda = %d$ nm' % WL_NM, *optics('absorption'), mcolors.LogNorm(*ABSORPTION_RANGE),
         decade_ticks(*ABSORPTION_RANGE), log_labels(decade_ticks(*ABSORPTION_RANGE)), OPTICS_UNITS, 'max'),
    ]


def draw(grid, cols, mixed, background=None):
    crs = figures.map_crs(grid)
    n = len(cols)
    W = TWO_COLUMN
    pw = (W - LEFT - RIGHT - (n - 1) * WSPACE) / n
    ph = pw * grid['XLAT'].shape[0] / grid['XLAT'].shape[1]
    H = TITLE_H + 2 * ph + HSPACE + VALUE_KEY_H + ph + RATIO_KEY_H
    fig = plt.figure(figsize=(W, H))
    row_top = (TITLE_H, TITLE_H + ph + HSPACE, TITLE_H + 2 * ph + HSPACE + VALUE_KEY_H)     # from the top
    axes = np.array([[fig.add_axes([(LEFT + c * (pw + WSPACE)) / W, 1 - (top + ph) / H, pw / W, ph / H],
                                   projection=crs) for c in range(n)] for top in row_top])
    ratio_norm = mcolors.LogNorm(*RATIO_RANGE)
    stats = []
    for c, (title, f_ml, f_ref, norm, ticks, labels, units, extend) in enumerate(cols):
        mesh = figures.plot_field(axes[0, c], grid, f_ml, norm=norm)
        figures.plot_field(axes[1, c], grid, f_ref, norm=norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = f_ml / f_ref
        ratio_mesh = figures.plot_field(axes[2, c], grid, ratio, norm=ratio_norm, cmap='RdBu_r')
        axes[0, c].set_title(title, fontsize=7.5, pad=3, linespacing=1.15)
        # the column's colour bar, under its two value rows
        bw = .82 * pw
        x0 = LEFT + c * (pw + WSPACE) + (pw - bw) / 2
        y_top = row_top[1] + ph + .08
        cax = fig.add_axes([x0 / W, 1 - (y_top + BAR_H) / H, bw / W, BAR_H / H])
        cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend=extend, extendfrac=.06)
        cb.ax.xaxis.set_minor_locator(mticker.NullLocator())
        cb.set_ticks(list(ticks), labels=list(labels))
        cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
        if units:
            cb.set_label(units, fontsize=6.5, labelpad=2)
        r = ratio[mixed]
        r = r[np.isfinite(r)]
        stats.append((title.replace('\n', ', '), np.median(r), *np.percentile(r, [5, 95]),
                      np.mean((r < RATIO_RANGE[0]) | (r > RATIO_RANGE[1])),
                      np.median(ratio[background]) if background is not None else np.nan))
    for ax, label in zip(axes[:, 0], ('WRF-AeroMELD', 'Particle-resolved\nreference', 'Ratio')):
        ax.annotate(label, xy=(0, .5), xycoords='axes fraction', xytext=(-4, 0), textcoords='offset points',
                    rotation=90, ha='right', va='center', fontsize=7.5, linespacing=1.1, annotation_clip=False)
    # one colour bar for the four ratio maps, under the middle two columns
    bw = 2 * pw + WSPACE
    y_top = row_top[2] + ph + .09
    cax = fig.add_axes([(W - RIGHT + LEFT - bw) / 2 / W, 1 - (y_top + BAR_H) / H, bw / W, BAR_H / H])
    cb = fig.colorbar(ratio_mesh, cax=cax, orientation='horizontal', extend='both', extendfrac=.04)
    cb.ax.xaxis.set_minor_locator(mticker.NullLocator())
    cb.set_ticks(list(RATIO_TICKS), labels=['%g' % t for t in RATIO_TICKS])
    cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cb.set_label('WRF-AeroMELD / reference', fontsize=6.5, labelpad=2)
    print('ratio WRF-AeroMELD / reference over the %d mixed cells:' % mixed.sum())
    for name, med, p5, p95, out, bg in stats:
        print('  %-34s median %.3f, 5th-95th percentiles %.3f-%.3f; outside %g-%g: %.2f %%; '
              'background-only cells (>= 99 %% of the number) median %.3f'
              % (name, med, p5, p95, *RATIO_RANGE, 100 * out, bg))
    return fig


def make(run):
    ml = products.load('aeromeld_domain', run=run.key)
    ts = ml.attrs['timestamp']
    pmc = reference.domain(run, ts)                         # the particle-resolved reference
    grid = products.load('grid')
    figures.same_instant(ml, pmc)
    background = reference.fractions(run, ts)[0] >= .99      # the background population alone (text)
    return figures.save(draw(grid, columns(ml, pmc), reference.mixed(run, ts), background), 'diagnostic_maps')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
