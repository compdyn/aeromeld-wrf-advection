#!/usr/bin/env python
"""domain_evolution: the aerosol over the domain through the run (candidate Figure 2).

From per-source transport in a replay (default the main run's, config.REPLAY_12H of the
representative seed; any replay works, the fields are the reference's): at each
snapshot time, the total number mixing ratio, and the composition by source
population as a colour, the three source fractions mixing three colours
(background, urban, emission) in linear light, with a ternary key. The
run-mean surface wind (data/sources/wind_mean.npz, the 4-hour WRF run) and
the sampling points complete the set-up.

--option picks the layout, to compare:
  number       wind + number at each time
  composition  wind + composition at each time
  both         number row over composition row (no wind panel)

Usage (from scripts/): python plot/domain_evolution.py [--replay dd5_12h] [--hours 0,2,4,6,8,10,12]
                       [--option composition] [--out DIR] [--format png]

Products: grid; data/transport/<replay>, data/sources/wind_mean.npz.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cartopy.crs as ccrs  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paperfigs import common, config, figures, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402
from paperfigs.fig_palettes import (MAP_ANNOTATION_COLOR, MAP_ANNOTATION_STROKE, POP_COLORS,  # noqa: E402
                                    POP_MAP_COLORS, TWO_COLUMN, apply_map_features, halo, point_color,
                                    use_paper_style)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

#: Background, urban, emission: the populations' line colours with the background
#: lightened (fig_palettes.POP_MAP_COLORS); urban and emission far apart in hue,
#: so that their mixture (purple) is distinct.
POP_COLOURS = POP_MAP_COLORS
POP_NAMES = config.POP_LABELS
GRID_STEP = 2                  # degrees between latitude/longitude lines
NUMBER_NORM = mcolors.LogNorm(5e8, 1e10)
URBAN_LEVELS = (2e9, 5e9, 8e9)    # kg-1: contours of the number in the 0-h panel (the urban Gaussian)
#: The emission rectangle, outlined in the 0-h panel: the emission population's colour, dashed.
EMISSION_LINE = dict(color=POP_COLORS[2], lw=.8, ls=(0, (3, 1.6)))


def emission_box(ax, grid):
    """Dashed outline of the cells where the emission population is emitted (the grid
    product's source_box, 0-based inclusive), drawn along the outer cell edges."""
    i0, i1, j0, j1 = (int(v) for v in grid['source_box'].values[0])
    lon, lat = grid['XLONG'].values, grid['XLAT'].values
    # cell-edge corners, half a cell outside the outermost centres (linear extrapolation)
    corner = lambda a, j, i, dj, di: a[j, i] + .5 * (a[j, i] - a[j - dj, i - di])
    pts = [(j0, i0, 1, 1), (j0, i1, 1, -1), (j1, i1, -1, -1), (j1, i0, -1, 1)]
    xs = [corner(lon, j, i, dj, di) for j, i, dj, di in pts]
    ys = [corner(lat, j, i, dj, di) for j, i, dj, di in pts]
    n = 30                                          # points per side, so the edges follow the projection
    lo, la = [], []
    for k in range(4):
        t = np.linspace(0, 1, n)
        lo += list(xs[k] + t * (xs[(k + 1) % 4] - xs[k])); la += list(ys[k] + t * (ys[(k + 1) % 4] - ys[k]))
    ax.plot(lo, la, transform=ccrs.PlateCarree(), zorder=5, **EMISSION_LINE)


def mix_rgb(f, colours=POP_COLOURS):
    """(3, ...) fractions -> (..., 3) RGB, mixing the colours in linear light."""
    lin = np.array([mcolors.to_rgb(c) for c in colours]) ** 2.2
    return np.clip(np.einsum('s...,sc->...c', np.clip(f, 0, 1), lin), 0, 1) ** (1 / 2.2)


def rgb_map(ax, grid, rgb):
    lon, lat = grid['XLONG'].values, grid['XLAT'].values
    ax.pcolormesh(lon, lat, rgb[..., 0], color=rgb.reshape(-1, 3), transform=ccrs.PlateCarree(),
                  shading='auto', rasterized=True)
    ax.set_extent([*ax.get_xlim(), *ax.get_ylim()], crs=ax.projection)
    apply_map_features(ax)


def ternary_key(ax):
    """A triangle of the colours (an image, so that it scales), each corner
    labelled by its population: background and urban beside the base, emission
    above the apex."""
    h = np.sqrt(3) / 2
    x, y = np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, h, 261))
    f_emi = y / h
    f_urb = x - f_emi / 2
    f = np.stack([1 - f_urb - f_emi, f_urb, f_emi])
    rgba = np.concatenate([mix_rgb(f), (f.min(0) >= -1e-9)[..., None].astype(float)], -1)
    ax.imshow(rgba, extent=(0, 1, 0, h), origin='lower', interpolation='bilinear', zorder=1)
    ax.plot([0, 1, .5, 0], [0, 0, h, 0], color='0.35', lw=.5, zorder=2)
    kw = dict(fontsize=6.5)
    ax.text(-0.06, 0.0, POP_NAMES[0], ha='right', va='center', **kw)
    ax.text(1.06, 0.0, POP_NAMES[1], ha='left', va='center', **kw)
    ax.text(0.5, h + 0.06, POP_NAMES[2], ha='center', va='bottom', **kw)
    ax.set_xlim(-1.25, 2.25); ax.set_ylim(-0.08, h + 0.3); ax.set_aspect('equal'); ax.set_axis_off()


#: Where each point's letter sits, offset (points) from its marker, and its alignment: A and B to
#: the west, C (near the panel's bottom edge, and just south-east of B) to the east.
POINT_LABEL_AT = (((-5, 0), 'right'), ((-5, 0), 'right'), ((5, 0), 'left'))


def mark_points(ax, grid):
    """The sampling points, all three in one style (2026-11-15: A and C had white outlines
    and B a dark one): the point's colour, edged in ink, and an ink letter on a white halo, which
    read alike on the whole grey range of the wind map."""
    lon, lat = grid['XLONG'].values, grid['XLAT'].values
    geodetic = ccrs.PlateCarree()._as_mpl_transform(ax)
    for i_loc, loc in enumerate(config.LOCATIONS):
        j, i = common.point_index(loc)
        ax.scatter([lon[j, i]], [lat[j, i]], transform=ccrs.PlateCarree(), color=point_color(i_loc), s=16,
                   edgecolor=MAP_ANNOTATION_STROKE, linewidth=.6, zorder=6)
        (dx, dy), ha = POINT_LABEL_AT[i_loc % len(POINT_LABEL_AT)]
        ax.annotate(config.LOC_LABELS[i_loc], xy=(lon[j, i], lat[j, i]), xycoords=geodetic, xytext=(dx, dy),
                    textcoords='offset points', ha=ha, va='center', fontsize=6.5, color=MAP_ANNOTATION_STROKE,
                    fontweight='bold', path_effects=halo(1.6, MAP_ANNOTATION_COLOR), zorder=7)


def wind_panel(ax, grid, wind='wind_mean_12h.npz'):
    with np.load(config.REPO / 'data' / 'sources' / wind) as w:
        u, v = w['u_mean'], w['v_mean']
    speed = np.hypot(u, v)
    cmap = mcolors.LinearSegmentedColormap.from_list('wind', matplotlib.colormaps['Greys'](np.linspace(.05, .55, 256)))
    m = figures.plot_field(ax, grid, speed, cmap=cmap, norm=mcolors.Normalize(0, np.nanpercentile(speed, 99)))
    crs = ax.projection
    xy = crs.transform_points(ccrs.PlateCarree(), grid['XLONG'].values, grid['XLAT'].values)
    ax.streamplot(xy[..., 0], xy[..., 1], u, v, transform=crs, density=.9, color='0.2', linewidth=.4,
                  arrowsize=.45, zorder=3)
    mark_points(ax, grid)
    return m


def gridlines(ax, left, bottom, label_step=GRID_STEP):
    """Dotted latitude/longitude lines every GRID_STEP degrees (as the earlier
    domain figure), labelled every label_step degrees on the left and bottom edges."""
    kw = dict(x_inline=False, y_inline=False, color='0.5', linewidth=.4, alpha=.6, linestyle=':', zorder=2)
    ax.gridlines(draw_labels=False, xlocs=np.arange(-130, -110, GRID_STEP), ylocs=np.arange(30, 46, GRID_STEP), **kw)
    # longitude labels from 122 W: the 124 W line meets the bottom edge at the left
    # corner, where its label would touch the neighbouring panel's 118 W
    gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-122, -110, label_step),
                      ylocs=np.arange(30, 46, label_step), **{**kw, 'linestyle': '-', 'alpha': 0})
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = left, bottom
    gl.xlabel_style = gl.ylabel_style = {'size': 5.5}
    gl.xpadding = gl.ypadding = 2
    gl.rotate_labels = False


def draw(grid, s, hours, option):
    n_src = lambda it: np.stack([s['oracle'][it].astype(float)[list(g)].sum(0) for g in T.ORACLE_GROUPS])
    crs = figures.map_crs(grid)
    if option == 'both':
        nrow, ncol = 2, len(hours)
    else:
        n_maps = len(hours) + 1                         # + the wind panel
        ncol = 4 if n_maps > 4 else n_maps
        nrow = int(np.ceil(n_maps / ncol))
    # layout in inches: lat labels on the left, a title over each row, lon labels
    # under the bottom row, and the keys in a strip under the maps
    W, left, right, wsp = TWO_COLUMN, .34, .04, .06
    title_h, lon_h = .17, .17
    key_h = .86 if option == "composition" else .72     # composition: room for the emission rectangle's key
    pw = (W - left - right - (ncol - 1) * wsp) / ncol
    ph = pw * grid['XLAT'].shape[0] / grid['XLAT'].shape[1]
    H = nrow * (title_h + ph) + lon_h + key_h
    fig = plt.figure(figsize=(W, H))
    axes = []
    for r in range(nrow):
        y0 = H - (r + 1) * (title_h + ph)
        for c in range(ncol):
            axes.append(fig.add_axes([(left + c * (pw + wsp)) / W, y0 / H, pw / W, ph / H], projection=crs))
    label_step = GRID_STEP if pw > 1.2 else 2 * GRID_STEP
    k = 0
    if option != 'both':
        m_wind = wind_panel(axes[0], grid)
        axes[0].set_title('Mean wind, 0–%g h' % hours[-1], fontsize=7, pad=2)
        k = 1
    m_num = None
    for h in hours:
        it = T.time_index(s, h)
        n = n_src(it)
        tot = n.sum(0)
        if option in ('number', 'both'):
            ax = axes[k]; k += 1
            m_num = figures.plot_field(ax, grid, tot, norm=NUMBER_NORM)
            ax.set_title('%g h' % h, fontsize=7, pad=2)
        if option in ('composition', 'both'):
            ax = axes[k if option == 'composition' else k - 1 + len(hours)]
            if option == 'composition':
                k += 1
            rgb_map(ax, grid, mix_rgb(n / tot))
            if h == hours[0]:
                # The initial urban box is uniform in composition but not in number: its number
                # is a Gaussian, 1e10 kg-1 at the centre falling to the background's 1e9 at
                # the edge. Contours of the number show it.
                ax.contour(grid['XLONG'].values, grid['XLAT'].values, tot, levels=URBAN_LEVELS,
                           colors='white', linewidths=.5, transform=ccrs.PlateCarree())
                emission_box(ax, grid)
            if option == 'composition':
                ax.set_title('%g h' % h, fontsize=7, pad=2)
    n_used = k if option != 'both' else 2 * len(hours)
    for ax in axes[n_used:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n_used]):
        r, c = divmod(i, ncol)
        last_row = r == nrow - 1 or i + ncol >= n_used
        gridlines(ax, left=c == 0, bottom=last_row, label_step=label_step)
    # the keys, side by side in the strip under the maps
    keys = []
    if m_num is not None:
        keys.append(('cbar', m_num, 'Number mixing ratio (kg$^{-1}$)', 'both'))
    if option != 'both':
        keys.append(('cbar', m_wind, 'Mean wind speed, 0–%g h (m s$^{-1}$)' % hours[-1], 'max'))
    if option in ('composition', 'both'):
        keys.append(('ternary',))
    slot = (W - left - right) / len(keys)
    for i, key in enumerate(keys):
        x_mid = left + (i + .5) * slot
        if key[0] == 'cbar':
            cw = min(1.9, .8 * slot)
            cax = fig.add_axes([(x_mid - cw / 2) / W, (key_h - .3 if option == "composition" else key_h * .55) / H, cw / W, .085 / H])
            cb = fig.colorbar(key[1], cax=cax, orientation='horizontal', extend=key[3])
            cb.set_label(key[2], fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6)
            if i == 0 and option == 'composition':
                x_legend = x_mid
        else:
            kw_ = 2.3
            tax = fig.add_axes([(x_mid - kw_ / 2) / W, .02 / H, kw_ / W, (key_h - .04) / H])
            ternary_key(tax)
    if option == 'composition':
        # under the first colour bar, level with the ternary key's Background and Urban labels
        # (the triangle's base is y = 0 in the key's data coordinates)
        fig.legend(handles=[Line2D([], [], label='Emission rectangle', **EMISSION_LINE)],
                   loc='center', bbox_to_anchor=(x_legend / W, 0.),
                   bbox_transform=blended_transform_factory(fig.transFigure, tax.transData),
                   frameon=False, fontsize=6.5, handlelength=2.6, borderpad=0)
    return fig


def main_replay():
    """The replay of the paper's main run (config.REPLAY_12H of the representative seed in
    data/decomposition.json; 2026-11-19, dd5_12h). Per-source transport, all this figure reads, is
    the same in every replay (rep6_12h's equals dd5_12h's at every hour shown)."""
    import json
    dec = json.loads((config.DATA_DIR / 'decomposition.json').read_text())['models']
    return config.REPLAY_12H % dec[config.recipe('twin_k16')]['representative_seed']


def make(run=None, replay=None, hours=(0, 2, 4, 6, 8, 10, 12), option='composition'):
    grid = products.load('grid')
    s = T.load(replay or main_replay())
    return figures.save(draw(grid, s, hours, option), 'domain_evolution' + ('' if option == 'composition' else '_' + option))


def main(argv=None):
    ap = figures.cli(__doc__)
    ap.add_argument('--replay', default=None, help='default: the main run\'s replay')
    ap.add_argument('--hours', default='0,2,4,6,8,10,12')
    ap.add_argument('--option', default='composition', choices=('number', 'composition', 'both'))
    args = ap.parse_args(argv)
    figures.setup(args)
    use_paper_style()
    make(replay=args.replay, hours=[float(h) for h in args.hours.split(',')], option=args.option)


if __name__ == '__main__':
    main()
