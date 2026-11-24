#!/usr/bin/env python
"""The transport schemes in the case: the paper's figures from the 12-hour offline replays.

All from the offline replays of the CARES case on WRF's own mass fluxes (transport/,
data/transport/all<seed>_12h, transport/rep12_array.sbatch), against the particle-resolved
reference (per-source transport of the source populations' particle-resolved diagnostics), with
the paper's error and its four terms (paperfigs/errors.py; data/decomposition*.json from
scripts/decomposition.py). The schemes: donor-cell, the shared limiter and separate limiters.

  schemes_fronts  the urban source fraction at 12 h around the urban plume: per-source
                  transport and the three schemes (the recommended decoder's representative
                  seed; the fractions implied by the latent vector).
  schemes_time    the overall error through the run of each scheme and of per-source transport
                  (the representation error alone), for the recommended decoder's representative seed
                  (data/decomposition_time.json).
  schemes_terms   the four terms and the overall error through the run, one panel per scheme
                  (appendix).
  schemes_models  the overall error at 12 h of every decoder under every scheme, median over
                  ten seeds with the range, and each decoder's own error (appendix).

The two trade-offs (fraction against off-mixture error, reconstruction against mixture
error) are plot/tradeoffs.py. Dropped 2026-11-13: schemes_budget (the old three-term budget,
RMS log ratios) and schemes_tradeoff (latent-space distances), replaced by tradeoffs.

Usage (from scripts/): python plot/transport_schemes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.offsetbox as moffset  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from paperfigs import config, errors, figures, mixing, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402
from paperfigs.fig_palettes import SINGLE_COLUMN, TWO_COLUMN, create_label_box  # noqa: E402

INK, GRID = '#1a1a1a', '#e3e3e3'
HOURS = 12.0
RECIPE = config.recipe('twin_k16')
#: (scheme, label, colour), from first order to the most accurate
SCHEMES = (('donor', 'Donor-cell', '#b2182b'),
           ('shB', 'Shared limiter', '#2166ac'),
           ('sep', 'Separate limiters', '#4393c3'))
REFERENCE = ('oracle', 'Reference:\nper-source transport', '#7f7f7f')   # its schemes_fronts title
DECODERS = tuple((config.recipe(b), lab) for b, lab in zip(config.BASES, ('No mixtures', 'Pairs', 'Up to 16',
                                                                         'Up to 32', 'Affine')))
#: The plume region shown in schemes_fronts (0-based south_north, west_east ranges).
BOX = (slice(0, 110), slice(40, 135))


def decomposition():
    return json.loads((config.DATA_DIR / 'decomposition.json').read_text())


def main_model():
    """(registry key, replay) of the recommended decoder's representative seed."""
    seed = decomposition()['models'][RECIPE]['representative_seed']
    return '%s.%d.0' % (RECIPE, seed), config.REPLAY_12H % seed


def scheme_key(scheme, model):
    return '%s:%s' % (scheme, T.run_model(model)) if scheme in ('shB', 'sep') else scheme


def schemes_fronts(run=None):
    import cartopy.crs as ccrs
    model, replay = main_model()
    s = T.load(replay)
    it = T.time_index(s, HOURS)
    grid = products.load('grid')
    crs = figures.map_crs(grid)
    z_star = products.load('populations', model=model)['z'].values
    panels = []
    for k, lab, _ in (REFERENCE,) + SCHEMES:
        key = scheme_key(k, model)
        if k == 'sep':
            f = mixing.implied_fractions(T.latent(s, key, z_star, T.run_model(model), it), z_star)[0]
        else:
            f = T.fractions(s, key, it)
        title = lab if k == 'oracle' else SCHEME_LINES[k]
        panels.append((title, f[1]))
    # inches: the maps 1.65 in tall, 0.53 in above them for the four-line title, 0.6 in below
    H = 2.78
    fig, axes = plt.subplots(1, len(panels), figsize=(TWO_COLUMN, H), subplot_kw={'projection': crs},
                             gridspec_kw=dict(left=.005, right=.995, top=1 - .53 / H, bottom=.6 / H,
                                              wspace=.03))
    lon, lat = grid['XLONG'].values, grid['XLAT'].values
    j, i = BOX
    corners = crs.transform_points(ccrs.PlateCarree(), lon[j, i].ravel(), lat[j, i].ravel())
    ext = [corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()]
    for ax, (title, f) in zip(axes, panels):
        mesh = figures.plot_field(ax, grid, np.clip(f, 0, 1), cmap='viridis')
        mesh.set_clim(0, 1)
        ax.set_extent(ext, crs=crs)
        ax.set_title(title, fontsize=7, pad=2)
    cax = fig.add_axes([.3, .325 / H, .4, .0875 / H])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    cb.set_label('Urban source fraction after %g h' % HOURS, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return figures.save(fig, 'schemes_fronts')


#: Legend names of the schemes (the paper's numbered list), and of per-source transport.
SCHEME_NAMES = {'donor': 'Transport scheme 1: donor-cell', 'shB': 'Transport scheme 2: shared limiter',
                'sep': 'Transport scheme 3: separate limiters\n(WRF-AeroMELD)',
                'oracle': 'Per-source transport (representation error)'}
#: The same names wrapped for the narrow panels of schemes_fronts and the tags of schemes_terms.
SCHEME_LINES = {'donor': 'Transport scheme 1:\ndonor-cell', 'shB': 'Transport scheme 2:\nshared limiter',
                'sep': 'Transport scheme 3:\nseparate limiters\n(WRF-AeroMELD)'}
#: The four terms in schemes_terms: (term, colour, line style). Not the schemes' colours, which
#: name the panels' schemes in schemes_time.
TERM_STYLES = (('offmixture', '#762a83', ':'), ('fraction', '#b35806', '-'),
               ('mixture', '#8c8c8c', '--'), ('reconstruction', '#4d4d4d', '-.'))


def _time_record():
    model, replay = main_model()
    rec = json.loads((config.DATA_DIR / 'decomposition_time.json').read_text())['%s:%s' % (model, replay)]
    h = np.array(rec['hours'])
    return rec, h, (lambda sch, term: 100 * np.array([x[term]['overall'] for x in rec['schemes'][sch]]))


def _time_axis(ax, h):
    ax.grid(color=GRID, lw=.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, h[-1])
    ax.set_xticks(np.arange(0, h[-1] + .1, 2))
    ax.set_ylim(0, None)
    ax.set_xlabel('Time (h)', fontsize=7)


def schemes_time(run=None):
    """The overall error of each transport scheme through the run, with decoder 3 (the recommended
    decoder) at its representative seed, and per-source transport's (the decoder error alone,
    what remains with exact transport). Since 2026-11-13 one panel: the mixed-cell counts
    went to the text, the four terms to the appendix (schemes_terms)."""
    rec, h, val = _time_record()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.75),
                           gridspec_kw=dict(left=.14, right=.97, top=.97, bottom=.43))
    for k, _, colour in SCHEMES:
        ax.plot(h, val(k, 'total'), color=colour, lw=1.4, label=SCHEME_NAMES[k])
    ax.plot(h, val('oracle', 'total'), color=REFERENCE[2], lw=1.0, ls='--', label=SCHEME_NAMES['oracle'])
    ax.set_ylabel('Overall error (%)', fontsize=7)
    _time_axis(ax, h)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower left', ncol=1, frameon=False, fontsize=6.5,
               bbox_to_anchor=(.02, 0), handlelength=2.4)
    return figures.save(fig, 'schemes_time')


def schemes_terms(run=None):
    """Appendix: the four terms of the error through the run for each transport scheme (one
    panel each, decoder 3 at its representative seed), with the overall error. The decoder's two
    terms are the same in every panel."""
    rec, h, val = _time_record()
    # inches: the axes 1.27 in tall, 0.44 in above them for the three-line tag of (c)
    H = 2.56
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, H), sharey=True,
                             gridspec_kw=dict(left=.075, right=.985, top=1 - .44 / H, bottom=.846 / H,
                                              wspace=.12))
    for ax, (k, _, _) in zip(axes, SCHEMES):
        for term, colour, ls in TERM_STYLES:
            ax.plot(h, val(k, term), color=colour, ls=ls, lw=1.2, label=errors.TERM_LABELS[term])
        ax.plot(h, val(k, 'total'), color=INK, lw=1.4, label=errors.TERM_LABELS['total'])
        _time_axis(ax, h)
        letter = 'abc'[[x[0] for x in SCHEMES].index(k)]
        tag = moffset.AnchoredOffsetbox(loc='lower left', pad=.2, frameon=True, borderpad=0,
                                        child=moffset.TextArea('(%s) %s' % (letter, SCHEME_LINES[k]),
                                                               textprops=dict(size=6.5)),
                                        bbox_to_anchor=(0, 1.02), bbox_transform=ax.transAxes)
        ax.add_artist(tag)
    axes[0].set_ylabel('Error (%)', fontsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(.53, 0), handlelength=2.4, columnspacing=1.5)
    return figures.save(fig, 'schemes_terms')


def schemes_models(run=None):
    """The overall error at 12 h of every decoder (groups) under every scheme (bars): median over
    the ten seeds with the range, and the decoder's own error (per-source transport) as a tick."""
    d = decomposition()['models']
    # inches: the axes 1.74 in tall, 1.05 in below them for the ticks and the legend
    H = 2.87
    fig, ax = plt.subplots(figsize=(TWO_COLUMN, H))
    fig.subplots_adjust(left=.075, right=.995, top=1 - .078 / H, bottom=1.05 / H)
    width = .8 / len(SCHEMES)
    for i, (k, lab, colour) in enumerate(SCHEMES):
        x = np.arange(len(DECODERS)) + (i - (len(SCHEMES) - 1) / 2) * width
        v = [100 * np.array([s[k]['total']['overall'] for s in d[m]['seeds'].values()]) for m, _ in DECODERS]
        ax.bar(x, [np.median(a) for a in v], width * .92, color=colour, label=lab, zorder=2)
        ax.vlines(x, [a.min() for a in v], [a.max() for a in v], color=INK, lw=.7, zorder=3)
    for j, (m, _) in enumerate(DECODERS):
        dec = np.median([100 * s['oracle']['total']['overall'] for s in d[m]['seeds'].values()])
        ax.plot([j - .45, j + .45], [dec, dec], color=INK, lw=.9, ls='--', zorder=4)
    rec = '\n(WRF-AeroMELD)'
    ax.set_xticks(np.arange(len(DECODERS)), ['Decoder %d:\n%s%s' % (j + 1, lab.lower(), rec if m == RECIPE else '')
                                             for j, (m, lab) in enumerate(DECODERS)], fontsize=7)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.yaxis.set_major_locator(MultipleLocator(5))    # integer tick labels: the y label stays inside
    ax.set_ylabel('Overall error (%)', fontsize=7)
    ax.grid(axis='y', color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [Patch(facecolor=c, label=SCHEME_NAMES[k]) for k, _, c in SCHEMES]
    handles.append(Line2D([], [], color=INK, ls='--', lw=.9, label=SCHEME_NAMES['oracle']))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=6.5,
               bbox_to_anchor=(.53, -.005), handlelength=1.6, columnspacing=1.2)
    return figures.save(fig, 'schemes_models')


def make(_run=None):
    """The paper's figures (make_all.py): fixed models and replays, whatever the run."""
    return [schemes_fronts(), schemes_time(), schemes_terms(), schemes_models()]


def main(argv=None):
    args = figures.cli(__doc__).parse_args(argv)
    figures.setup(args)
    return make()


if __name__ == '__main__':
    main()
