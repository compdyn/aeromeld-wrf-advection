#!/usr/bin/env python
"""optics_ML_vs_PR: scattering and absorption maps at 400 and 1000 nm, AeroMELD vs the exact answer.

Scattering in the left two columns, absorption in the right two, each with its
own colour bar over the AeroMELD and PartMC rows beneath it: 4 decades each,
absorption's one decade below scattering's (fig_palettes.SCATTERING_RANGE and
ABSORPTION_RANGE, shared with every optics figure). The third row is
AeroMELD / PartMC on one scale. Surface, final instant; both sides per kg of
dry air.

Each side is indexed on its OWN wavelength grid, exactly: they share 300, 400,
600 and 1000 nm but not 500/550. From read_wrf_data.ipynb cell 44.
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
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import analysis, common, figures, products, reference  # noqa: E402
from paperfigs.fig_palettes import ABSORPTION_RANGE, SCATTERING_RANGE, TWO_COLUMN  # noqa: E402

WL_TARGETS_NM = analysis.SMAPE_WL_NM   # the violins' optics columns
M_INV_TO_MM_INV = 1e6

#: (product variable, label, colour scale in Mm^-1 m^3 kg^-1). At 04:00 the
#: cells span 0.29-1500 (scattering) and 0.034-79 (absorption) at these
#: wavelengths, so the brightest scattering cells saturate.
QUANTITIES = (
    ('scattering', 'Scattering', mcolors.LogNorm(*SCATTERING_RANGE)),
    ('absorption', 'Absorption', mcolors.LogNorm(*ABSORPTION_RANGE)),
)


def make(run):
    ml = products.load('aeromeld_domain', run=run.key)
    pmc = reference.domain(run, ml.attrs['timestamp'])        # the exact answer
    grid = products.load('grid')
    figures.same_instant(ml, pmc)
    crs = figures.map_crs(grid)
    ml_nm, pmc_nm = ml['wavelength'].values * 1e9, pmc['wavelength'].values * 1e9

    ratio_norm = mcolors.LogNorm(vmin=.8, vmax=1.25)
    n_wl = len(WL_TARGETS_NM)
    fig_h = 4.85                         # inches; colour-bar offsets below are / fig_h
    fig = plt.figure(figsize=(TWO_COLUMN, fig_h))
    gs = fig.add_gridspec(3, 2 * n_wl + 1, width_ratios=[1] * n_wl + [.3] + [1] * n_wl,
                          left=.06, right=.975, bottom=.145, top=.835, wspace=.05, hspace=.06)
    axes = np.array([[[fig.add_subplot(gs[i_row, i_q * (n_wl + 1) + i_wl], projection=crs)
                       for i_wl in range(n_wl)]
                      for i_q in range(len(QUANTITIES))]
                     for i_row in range(3)])            # [row, quantity, wavelength]

    for i_q, (var, name, norm) in enumerate(QUANTITIES):
        for i_wl, wl in enumerate(WL_TARGETS_NM):
            f_ml = M_INV_TO_MM_INV * ml[var].values[:, :, common.exact_index(ml_nm, wl)]
            f_pmc = M_INV_TO_MM_INV * pmc[var].values[:, :, common.exact_index(pmc_nm, wl)]
            pcm = figures.plot_field(axes[0, i_q, i_wl], grid, f_ml, norm=norm)
            figures.plot_field(axes[1, i_q, i_wl], grid, f_pmc, norm=norm)
            pcm_ratio = figures.plot_field(axes[2, i_q, i_wl], grid, f_ml / f_pmc,
                                           norm=ratio_norm, cmap='RdBu_r')
            axes[0, i_q, i_wl].set_title(r'$\lambda = %d$ nm' % wl)
        pos0, pos1 = axes[0, i_q, 0].get_position(), axes[0, i_q, -1].get_position()
        cax = fig.add_axes([pos0.x0, pos0.y1 + .25 / fig_h, pos1.x1 - pos0.x0, .1 / fig_h])
        cb = fig.colorbar(pcm, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cb.set_label(r'%s (Mm$^{-1}$ m$^3$ kg$^{-1}$)' % name)
    figures.row_labels(axes[:, 0, 0], ['WRF-AeroMELD', 'Reference', 'Ratio'])

    pos0, pos1 = axes[2, 0, -1].get_position(), axes[2, -1, 0].get_position()
    cax = fig.add_axes([pos0.x0, pos0.y0 - .22 / fig_h, pos1.x1 - pos0.x0, .1 / fig_h])
    cb = fig.colorbar(pcm_ratio, cax=cax, orientation='horizontal')
    cb.set_label('WRF-AeroMELD / reference')
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_ticks([.8, .9, 1, 1.1, 1.25], labels=['0.8', '0.9', '1', '1.1', '1.25'])
    return figures.save(fig, 'optics_ML_vs_PR')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
