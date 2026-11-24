#!/usr/bin/env python
"""domain: the WRF-PartMC domain, emission sources and sampling points.

(a) the total number mixing ratio at the start, (b) the near-surface wind
averaged over the whole run, with streamlines. Both panels outline the
emission sources (where num_flux > 0) and mark the sampling points A, B, C.
The run-mean wind is the honest summary: the median speed doubles (about 7 to
14 m/s) by 04 UTC while each cell's direction holds steady (|mean wind| / mean
speed has a median of 0.98). U and V are grid-relative and the map projection
is the WRF grid itself, so they need no rotation.
From read_wrf_data.ipynb cell 15.
2026-11-09: (a)'s power of ten is in its colour-bar label ("10^10 # kg^-1 dry
air", figures.scaled_colorbar), not an offset text, as in
number_mixing_ratio_field.
2026-11-12: latitude and longitude every GRID_STEP degrees, labelled on the
left of (a) and below both panels.

2026-11-12: no WRF-PartMC. (a) is the number of the run's replay at the
start (uniform background plus the urban box, as WRF-AeroMELD and per-source
transport start), (b) the run-mean lowest-level wind of the WRF 4.7.1 run
whose fluxes the replays use (data/sources/wind_mean.npz: U and V destaggered,
averaged over the 17 15-minute outputs, 00-04 UTC).

Products: grid; data/transport/<replay>, data/sources/wind_mean.npz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cartopy.crs as ccrs  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import common, config, figures, products, reference  # noqa: E402
from paperfigs.fig_palettes import (ANNOTATION_STYLES, MAP_ANNOTATION_COLOR,  # noqa: E402
                                    MAP_ANNOTATION_STROKE, TWO_COLUMN, create_label_box,
                                    halo, halo_for, point_color)

LEFT = .055                    # room for (a)'s latitude labels
WGAP = .04                     # between the two maps
TOP_MARGIN = .07               # room for the (a)/(b) label boxes
CBAR_HEIGHT = .25              # colour bars, and the longitude labels above them
GRID_STEP = 2                  # degrees between labelled latitude/longitude lines


def hhmm(timestamp):
    return timestamp[11:16]


def plot_grid_box(ax, grid, j_min, j_max, i_min, i_max, **kwargs):
    """Outline of the 0-based inclusive cell range, through the cell centres."""
    xlong, xlat = grid['XLONG'].values, grid['XLAT'].values
    i_range = np.arange(i_min, i_max + 1)
    j_range = np.arange(j_min, j_max + 1)
    j_edge = np.concatenate([np.full(i_range.shape, j_min), j_range,
                             np.full(i_range.shape, j_max), j_range[::-1]])
    i_edge = np.concatenate([i_range, np.full(j_range.shape, i_max),
                             i_range[::-1], np.full(j_range.shape, i_min)])
    ax.plot(xlong[j_edge, i_edge], xlat[j_edge, i_edge], transform=ccrs.PlateCarree(), **kwargs)


def draw(grid, f_start, u, v, times, points):
    """`points`: 0-based (j, i) array indices of the sampling points."""
    n_y, n_x = grid['XLAT'].shape
    right = 1 - .025
    panel_w = (right - LEFT - WGAP) / 2
    height = panel_w / (n_x / n_y) / (1 - TOP_MARGIN - CBAR_HEIGHT)
    crs = figures.map_crs(grid)
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, TWO_COLUMN * height),
                             subplot_kw={'projection': crs})
    fig.subplots_adjust(left=LEFT, right=right, bottom=CBAR_HEIGHT, top=1 - TOP_MARGIN,
                        wspace=WGAP / panel_w)

    def cbar_axes(ax):
        pos = ax.get_position()
        return fig.add_axes([pos.x0, 0.11, pos.width, 0.05])

    m0 = figures.plot_field(axes[0], grid, f_start,
                            norm=mcolors.Normalize(vmin=f_start.min(), vmax=f_start.max()))
    figures.scaled_colorbar(fig, m0, cbar_axes(axes[0]), figures.NUMBER_MIXING_RATIO_LABEL)

    # Light end of a neutral ramp: viridis is taken by (a), the orange and blue
    # ramps belong to the points and populations, and black to geography.
    speed = np.hypot(u, v)
    wind_cmap = mcolors.LinearSegmentedColormap.from_list(
        'wind', matplotlib.colormaps['Greys'](np.linspace(0.05, 0.55, 256)))
    m1 = figures.plot_field(axes[1], grid, speed, cmap=wind_cmap,
                            norm=mcolors.Normalize(vmin=0, vmax=np.nanpercentile(speed, 99)))
    fig.colorbar(m1, cax=cbar_axes(axes[1]), orientation='horizontal',
                 extend='max').set_label(r'Wind speed (m s$^{-1}$)')
    xy = crs.transform_points(ccrs.PlateCarree(), grid['XLONG'].values, grid['XLAT'].values)
    axes[1].streamplot(xy[..., 0], xy[..., 1], u, v, transform=crs, density=1.4, color='0.2',
                       linewidth=.5, arrowsize=.6, zorder=3)

    boxes = grid['source_box'].values               # (source, [i_min, i_max, j_min, j_max])
    xlong, xlat = grid['XLONG'].values, grid['XLAT'].values
    for ax in axes:
        for s, (i_min, i_max, j_min, j_max) in enumerate(boxes):
            if i_min < 0:
                continue                            # this source never emits
            plot_grid_box(ax, grid, j_min, j_max, i_min, i_max,
                          color=MAP_ANNOTATION_COLOR, linewidth=1.4,
                          linestyle=ANNOTATION_STYLES[s % len(ANNOTATION_STYLES)],
                          path_effects=halo(2.6, MAP_ANNOTATION_STROKE), zorder=4,
                          label='Emission source')
        geodetic = ccrs.PlateCarree()._as_mpl_transform(ax)
        for i_loc, (j, i) in enumerate(points):
            color = point_color(i_loc)
            lon, lat = xlong[j, i], xlat[j, i]
            ax.scatter([lon], [lat], transform=ccrs.PlateCarree(), color=color, s=45,
                       edgecolor='white', linewidth=1.0, zorder=6,
                       path_effects=halo(2.0, MAP_ANNOTATION_STROKE))
            ax.annotate(config.LOC_LABELS[i_loc], xy=(lon, lat), xycoords=geodetic,
                        xytext=(-11, -11), textcoords='offset points', zorder=6,
                        color=color, fontweight='bold', path_effects=halo_for(color, 2.2))

    for k, ax in enumerate(axes):
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color='0.5',
                          linewidth=.4, alpha=.6, linestyle=':', zorder=2,
                          xlocs=np.arange(-130, -110, GRID_STEP), ylocs=np.arange(30, 46, GRID_STEP))
        gl.top_labels = gl.right_labels = False
        gl.left_labels = k == 0
        gl.xlabel_style = gl.ylabel_style = {'size': 'small'}
        gl.rotate_labels = False

    axes[0].legend(loc='lower left')
    create_label_box(axes[0], '(a) Number mixing ratio, %s UTC' % hhmm(times[0]))
    create_label_box(axes[1], '(b) Mean wind, %s–%s UTC' % (hhmm(times[0]), hhmm(times[-1])))
    return fig


def make(run):
    grid = products.load('grid')
    s = reference.replay(run)
    f_start = reference.number(run, config.TIMESTAMP_START, s)
    with np.load(config.REPO / 'data' / 'sources' / 'wind_mean.npz') as w:
        u, v, times = w['u_mean'], w['v_mean'], [str(t) for t in w['times']]
    points = [common.point_index(p) for p in config.LOCATIONS]
    fig = draw(grid, f_start, u, v, times, points)
    return figures.save(fig, 'domain')


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    make(figures.setup(args))


if __name__ == '__main__':
    main()
