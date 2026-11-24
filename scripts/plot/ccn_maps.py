#!/usr/bin/env python
"""CCN_ML_vs_PR: activated-fraction maps at four supersaturations, AeroMELD vs the exact answer.

Rows: AeroMELD, PartMC, AeroMELD - PartMC. Columns: s = 0.1, 0.3, 0.6,
1.0 %. Surface, final instant. Paper label: fig:ccn-fraction.

PartMC is the domain diagnostics' ccn_cdf (computed from the particles), not
the aerosols file's ccn_* fields. Both spectra are interpolated to each s_env
on their own grid. From read_wrf_data.ipynb cell 43.
2026-11-09: the column titles' number is inside the math, which removes a
double space after "=" (matplotlib already spaces "=" in math).

2026-11-12: the reference is the exact answer (paperfigs.reference), not
WRF-PartMC; rows "WRF-AeroMELD", "Exact", and the comparison's colour scale
narrowed to the much smaller differences.
Products: aeromeld_domain, partmc_diag_domain, grid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from paperfigs import analysis, common, figures, products, reference  # noqa: E402
from paperfigs.fig_palettes import CCN_LABEL, TWO_COLUMN  # noqa: E402

SAT_TARGETS = analysis.SMAPE_SAT       # s_env as fractions; the violins' CCN columns


def make(run):
    ml = products.load('aeromeld_domain', run=run.key)
    pmc = reference.domain(run, ml.attrs['timestamp'])        # the exact answer
    grid = products.load('grid')
    figures.same_instant(ml, pmc)
    crs = figures.map_crs(grid)

    norm = mcolors.Normalize(0, 1)
    diff_norm = mcolors.Normalize(-.05, .05)
    fig, axes = plt.subplots(3, len(SAT_TARGETS), figsize=(TWO_COLUMN, 3.6),
                             subplot_kw={'projection': crs})
    fig.subplots_adjust(left=.05, right=.8, bottom=.02, top=.93, wspace=.06, hspace=.08)
    for i_col, sat in enumerate(SAT_TARGETS):
        f_ml = common.interp_along_last_axis(ml['ccn_cdf'].values, ml['eps'].values, sat)
        f_pmc = common.interp_along_last_axis(pmc['ccn_cdf'].values, pmc['eps'].values, sat)
        pcm = figures.plot_field(axes[0][i_col], grid, f_ml, norm=norm)
        figures.plot_field(axes[1][i_col], grid, f_pmc, norm=norm)
        pcm_diff = figures.plot_field(axes[2][i_col], grid, f_ml - f_pmc, norm=diff_norm,
                                      cmap='RdBu_r')
        axes[0][i_col].set_title(r'$s = %3.1f$%%' % (sat * 100))
    figures.row_labels(axes[:, 0], ['WRF-AeroMELD', 'Reference', 'Difference'])

    pos0, pos1 = axes[1, -1].get_position(), axes[0, -1].get_position()
    cax = fig.add_axes([.82, pos0.y0, .022, pos1.y1 - pos0.y0])
    fig.colorbar(pcm, cax=cax, orientation='vertical').set_label(CCN_LABEL)
    pos0 = axes[2, -1].get_position()
    cax = fig.add_axes([.82, pos0.y0, .022, pos0.y1 - pos0.y0])
    cb = fig.colorbar(pcm_diff, cax=cax, orientation='vertical')
    cb.set_label('WRF-AeroMELD $-$\nreference')
    cb.set_ticks([-.05, 0, .05], labels=['$-$0.05', '0', '0.05'])
    return figures.save(fig, 'CCN_ML_vs_PR')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
