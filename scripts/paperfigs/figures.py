"""Plumbing shared by the plot scripts: command line, output directory, maps.

Every plot script takes

    --run KEY        the registry run to plot (default: the role = "main" run)
    --out DIR        write here instead of the default directory
    --format EXT     pdf (default) or png, e.g. for a quick look

The main run's figures go to the paper (config.PAPER_GRAPHICS); any other
run's go to config.BY_MODEL_GRAPHICS/<run key>/, so the paper's copies are
only ever written by the main run.
"""
import argparse
from pathlib import Path

import numpy as np

from . import common, config, fig_palettes, registry


def cli(doc):
    ap = argparse.ArgumentParser(description=doc.strip().split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=doc)
    ap.add_argument('--run', help='registry run key (default: the main run)')
    ap.add_argument('--out', type=Path, help='output directory')
    ap.add_argument('--format', default='pdf', choices=('pdf', 'png'))
    return ap


def setup(args):
    """Resolve the run, point save_figure at the right directory, apply the
    paper style. Returns the registry Run.

    Also selects the non-interactive Agg backend: the scripts only write files,
    and an interactive backend can change the output. macOS's snaps the figure
    to whole screen pixels, so a 170 mm figure came out 0.2 pt narrower."""
    import matplotlib
    matplotlib.use('Agg')
    reg = registry.load()
    run = reg.run(args.run) if args.run else reg.main()
    if args.out is not None:
        out = args.out
    elif run.role == 'main':
        out = config.PAPER_GRAPHICS
    else:
        out = config.BY_MODEL_GRAPHICS / run.key
    fig_palettes.FIGURE_DIR = str(out)
    fig_palettes.FIGURE_FORMAT = args.format
    fig_palettes.use_paper_style()
    return run


def save(fig, name):
    """save_figure under the configured directory and format; prints the path."""
    import matplotlib.pyplot as plt
    path = fig_palettes.save_figure(fig, name)
    plt.close(fig)
    print('wrote', path)
    return path


# --- maps ----------------------------------------------------------------------

def map_crs(grid):
    """The WRF Lambert conformal projection, from the grid product's attributes."""
    import cartopy.crs as ccrs
    a = grid.attrs
    return ccrs.LambertConformal(
        central_longitude=float(a['proj_STAND_LON']),
        central_latitude=float(a['proj_MOAD_CEN_LAT']),
        standard_parallels=(float(a['proj_TRUELAT1']), float(a['proj_TRUELAT2'])),
        globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000))


def plot_field(ax, grid, field, *, norm=None, cmap='viridis', rasterized=True):
    """pcolormesh of a (south_north, west_east) field on the WRF grid, with the
    map features. Returns the QuadMesh for a colour bar. The mesh is rasterized
    by default: drawn as vectors, nine maps make a 4 MB PDF."""
    import cartopy.crs as ccrs
    mesh = ax.pcolormesh(grid['XLONG'].values, grid['XLAT'].values, np.asarray(field),
                         transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                         rasterized=rasterized)
    # Pin the autoscaled view (the grid's cell edges) as an explicit extent.
    # Same view, but cartopy then clips the global LAND/OCEAN polygons to it:
    # without it, saving one map took ~48 s, with it 0.3 s.
    ax.set_extent([*ax.get_xlim(), *ax.get_ylim()], crs=ax.projection)
    fig_palettes.apply_map_features(ax)
    return mesh


#: The total-number colour bar's label in domain and number_mixing_ratio_field,
#: for scaled_colorbar.
NUMBER_MIXING_RATIO_LABEL = r'Total number mixing ratio ($10^{%d}$ # kg$^{-1}$ dry air)'


def scaled_colorbar(fig, mappable, cax, label, **kwargs):
    """Horizontal colour bar with its power of ten in the label instead of
    matplotlib's offset text ("1e8"), which sits on the label's line at the
    bar's right end and overprints a long label. `label` takes the exponent as
    %d; it is floor(log10) of the norm's largest |limit|. Returns the Colorbar."""
    import matplotlib.ticker as mticker
    norm = mappable.norm
    power = int(np.floor(np.log10(max(abs(norm.vmin), abs(norm.vmax)))))
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal', **kwargs)
    fmt = mticker.ScalarFormatter()
    fmt.set_powerlimits((power, power))      # always this exponent
    cb.formatter = fmt
    cb.update_ticks()
    cb.ax.xaxis.get_offset_text().set_visible(False)
    cb.set_label(label % power)
    return cb


def row_labels(axes, labels):
    """Rotated row names to the left of the first column."""
    for ax, label in zip(axes, labels):
        ax.annotate(label, xy=(-0.1, 0.5), xycoords='axes fraction', rotation=90,
                    va='center', ha='center', annotation_clip=False)


same_instant = common.same_instant      # the map scripts call it from here
