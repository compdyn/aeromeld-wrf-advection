#!/usr/bin/env python
"""Candidate figures (2026-11-09): spectrum errors and their attribution.

Not paper figures yet: options to choose between. Each diagnostic is one
spectrum error (analysis.spectra: CCN/CN over s_env 0.1-1 %, scattering and
absorption over 300-1000 nm, frozen fraction over -40 to -10 C), not a value
per supersaturation, wavelength or temperature. The attribution is
paperfigs.mixing, which needs only existing products. Written to
graphics/candidates/ unless --out says otherwise.

  cand_summary_spectra       (a) each mixed cell's spectrum error, as violins, with
                             SMAPE; (b) SMAPE and (c) AeroMELD/PartMC ratio for
                             each of the five decoders. Could replace both the
                             summary (violin) and model-comparison figures.
  cand_attribution_bars      mean and RMS of each error term, per spectrum
  cand_attribution_violins   each term's distribution over mixed cells
  cand_attribution_maps      maps of the total and each term, per spectrum
  cand_attribution_decoders  RMS of each term for each of the five decoders
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from model_comparison import styles  # noqa: E402
from paperfigs import config, figures, mixing, products, registry  # noqa: E402
from paperfigs.fig_palettes import TWO_COLUMN, create_label_box  # noqa: E402
from summary_symmetric_error import INK, MUTED, VIOLIN_FACE  # noqa: E402

#: Error terms, in plotting order, and their colours: four categorical slots of
#: the dataviz reference palette (aqua, violet, yellow, red), checked with its
#: validator (light mode: CVD adjacent dE >= 15.3, normal >= 20.8; two slots are
#: under 3:1 against white, so bars carry an ink edge and a legend). The number
#: term (optics only) is grey; the total is ink.
TERM_ORDER = ('mixing', 'transport', 'endmember', 'sampling', 'number')
TERM_COLORS = {'mixing': '#1baf7a', 'transport': '#6250d6', 'endmember': '#eda100',
               'sampling': '#e34948', 'number': '#9a9a9a'}
LOG_LIM = 0.5          # diverging map limits, log10 ratio (a factor of ~3.2)


def xlabels(spectra):
    return ['%s\n%s' % (s['name'], s['span']) for s in spectra]


def finish_x(ax, spectra, labels=True):
    xs = np.arange(len(spectra))
    ax.set_xlim(-.6, len(spectra) - .4)
    ax.set_xticks(xs, xlabels(spectra) if labels else [''] * len(spectra))
    ax.tick_params(axis='x', length=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    return xs


def mixed_values(m, sp, key):
    a = (sp['total'] if key == 'total' else sp['terms'][key])[m['mixed']]
    return a[np.isfinite(a)]


def rms(a):
    return float(np.sqrt(np.mean(a ** 2))) if a.size else np.nan


def term_legend(fig, y=1.0, terms=TERM_ORDER, total_patch=False):
    if total_patch:
        handles = [plt.Rectangle((0, 0), 1, 1, fc=INK, alpha=.3, label=mixing.TERM_LABELS['total'])]
    else:
        handles = [Line2D([], [], ls='none', marker='D', ms=4, mfc='white', mec=INK, mew=.8,
                          label=mixing.TERM_LABELS['total'])]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=TERM_COLORS[t], ec=INK, lw=.3,
                              label=mixing.TERM_LABELS[t]) for t in terms]
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(.54, y), handlelength=1.2, columnspacing=1)


def term_bars(ax, m, stat, width=.15):
    spectra = m['spectra']
    xs = np.arange(len(spectra))
    for j, t in enumerate(TERM_ORDER):
        vals = [stat(mixed_values(m, sp, t)) for sp in spectra]
        ax.bar(xs + (j - 2) * width, vals, width=width * .9, color=TERM_COLORS[t], ec=INK, lw=.3,
               zorder=2)
    ax.plot(xs, [stat(mixed_values(m, sp, 'total')) for sp in spectra], ls='none', marker='D',
            ms=4, mfc='white', mec=INK, mew=.8, zorder=3)
    ax.axhline(0, color=INK, lw=.6)
    ax.grid(axis='y', color='#d9d9d9', lw=.6)


# --- the candidates ------------------------------------------------------------------

def cand_summary_spectra(results):
    """(a) main run's per-cell spectrum errors; (b) SMAPE and (c) ratio per decoder."""
    main = results[0][1]
    spectra = main['spectra']
    fig = plt.figure(figsize=(TWO_COLUMN, 3.0))
    ax = fig.add_axes([.09, .28, .38, .62])
    vals = [np.log10(np.maximum(sp['error'][main['mixed']], 1e-4)) for sp in spectra]
    xs = np.arange(len(spectra))
    parts = ax.violinplot(vals, positions=xs, widths=.75, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor('none')
        body.set_alpha(1)
    for xp, d in zip(xs, vals):
        p05, p25, p50, p75, p95 = np.percentile(d, [5, 25, 50, 75, 95])
        ax.vlines(xp, p05, p95, color=INK, lw=1, zorder=3)
        ax.vlines(xp, p25, p75, color=INK, lw=4, zorder=3)
        ax.plot(xp, p50, 'o', ms=4.5, mfc='white', mec=INK, mew=.8, zorder=4)
    for xp, sp in zip(xs, spectra):
        ax.text(xp, 1.02, '%.0f%%' % (100 * np.mean(sp['error'][main['mixed']])),
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
    ax.text(-.02, 1.02, 'SMAPE', transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
            color=MUTED)
    ax.set_ylim(-2.5, np.log10(2))
    ax.yaxis.set_major_locator(FixedLocator([-2, -1, 0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r'$10^{%d}$' % round(v)))
    ax.set_ylabel('Spectrum error (mean $e$)')
    finish_x(ax, spectra)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.grid(axis='y', color='#e6e6e6', lw=.5)
    create_label_box(ax, '(a) %s' % results[0][0].model.label)

    styl = styles([(r, None, None) for r, _ in results])
    n = len(results)
    offs = (np.arange(n) - (n - 1) / 2) * .13
    ax_s = fig.add_axes([.56, .6, .42, .3])
    ax_r = fig.add_axes([.56, .28, .42, .27])
    handles = []
    for (run, m), (color, marker), dx in zip(results, styl, offs):
        mk = dict(marker=marker, ms=4, mfc=color, mec=INK, mew=.4, ls='none', zorder=4)
        smape = [100 * np.nanmean(sp['error'][m['mixed']]) for sp in m['spectra']]
        ax_s.plot(xs + dx, smape, **mk)
        for x, sp in zip(xs + dx, m['spectra']):
            r = 10 ** mixed_values(m, sp, 'total')
            p05, p50, p95 = np.percentile(r, [5, 50, 95])
            ax_r.vlines(x, max(p05, 1e-2), min(p95, 1e2), color=color, lw=1)
            ax_r.plot(x, p50, **mk)
        handles.append(Line2D([], [], label=run.model.label, **mk))
    ax_s.set_yscale('log')
    ax_s.set_ylim(3, 200)
    ax_s.set_yticks([5, 10, 20, 50, 100], ['5', '10', '20', '50', '100'])
    ax_s.set_ylabel('SMAPE (%)', fontsize=7)
    finish_x(ax_s, spectra, labels=False)
    ax_s.grid(axis='y', color='#e6e6e6', lw=.5)
    create_label_box(ax_s, '(b)')
    ax_r.set_yscale('log')
    ax_r.set_ylim(.1, 100)
    ax_r.axhline(1, color=MUTED, lw=.8, ls=(0, (3, 2)))
    ax_r.set_ylabel('Ratio (geometric\nmean over spectrum)', fontsize=7)
    finish_x(ax_r, spectra)
    ax_r.tick_params(axis='x', labelsize=6)
    create_label_box(ax_r, '(c)')
    fig.legend(handles=handles, loc='lower center', ncol=min(3, n), frameon=False, fontsize=7,
               bbox_to_anchor=(.5, 0), handletextpad=.3, columnspacing=1.2)
    return figures.save(fig, 'cand_summary_spectra')


def cand_attribution_bars(m):
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 2.9),
                             gridspec_kw={'wspace': .3, 'top': .82, 'bottom': .2, 'left': .09,
                                          'right': .99})
    term_bars(axes[0], m, lambda a: float(np.mean(a)) if a.size else np.nan)
    axes[0].set_ylabel(r'Mean $\log_{10}$ ratio (bias)')
    term_bars(axes[1], m, rms)
    axes[1].set_ylabel(r'RMS $\log_{10}$ ratio (spread)')
    for ax, tag in zip(axes, ('(a) Bias: the terms add up', '(b) Spread')):
        finish_x(ax, m['spectra'])
        ax.tick_params(axis='x', labelsize=6)
        create_label_box(ax, tag)
    term_legend(fig)
    return figures.save(fig, 'cand_attribution_bars')


def cand_attribution_violins(m):
    rows = ('total', 'mixing', 'transport', 'endmember', 'sampling')
    spectra = m['spectra']
    fig, axes = plt.subplots(1, len(spectra), figsize=(TWO_COLUMN, 3.0), sharey=True,
                             gridspec_kw={'wspace': .08, 'top': .78, 'bottom': .06, 'left': .11,
                                          'right': .99})
    for ax, sp in zip(axes, spectra):
        vals = [np.clip(mixed_values(m, sp, t), -1, 1) for t in rows]
        xs = np.arange(len(rows))
        parts = ax.violinplot(vals, positions=xs, widths=.8, showextrema=False)
        for body, t in zip(parts['bodies'], rows):
            body.set_facecolor(INK if t == 'total' else TERM_COLORS[t])
            body.set_alpha(.3 if t == 'total' else .75)
            body.set_edgecolor('none')
        for xp, d in zip(xs, vals):
            p05, p25, p50, p75, p95 = np.percentile(d, [5, 25, 50, 75, 95])
            ax.vlines(xp, p05, p95, color=INK, lw=.8, zorder=3)
            ax.vlines(xp, p25, p75, color=INK, lw=3, zorder=3)
            ax.plot(xp, p50, 'o', ms=3.5, mfc='white', mec=INK, mew=.7, zorder=4)
        ax.axhline(0, color=INK, lw=.6)
        ax.set_xticks(xs, [])
        ax.tick_params(axis='x', length=0)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis='y', color='#e6e6e6', lw=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title('%s\n%s' % (sp['name'], sp['span']), fontsize=7)
    axes[0].set_ylabel(r'$\log_{10}$ ratio, WRF-AeroMELD / WRF-PartMC', fontsize=7)
    term_legend(fig, y=1.0, terms=TERM_ORDER[:4], total_patch=True)
    return figures.save(fig, 'cand_attribution_violins')


def cand_attribution_maps(m, grid):
    cols = ('total', 'mixing', 'transport', 'endmember', 'sampling')
    spectra = m['spectra']
    crs = figures.map_crs(grid)
    fig, axes = plt.subplots(len(spectra), len(cols), figsize=(TWO_COLUMN, 6.3),
                             subplot_kw={'projection': crs},
                             gridspec_kw={'wspace': .04, 'hspace': .08, 'left': .08, 'right': .99,
                                          'top': .93, 'bottom': .11})
    norm = mcolors.TwoSlopeNorm(0, -LOG_LIM, LOG_LIM)
    for r, sp in enumerate(spectra):
        for k, t in enumerate(cols):
            field = sp['total'] if t == 'total' else sp['terms'][t]
            pcm = figures.plot_field(axes[r, k], grid, field, norm=norm, cmap='RdBu_r')
            if r == 0:
                axes[r, k].set_title(mixing.TERM_LABELS[t].replace(' ', '\n', 1)
                                     if t != 'total' else 'Total', fontsize=7)
        figures.row_labels(axes[r:r + 1, 0], [sp['name']])
    cax = fig.add_axes([.3, .06, .4, .018])
    fig.colorbar(pcm, cax=cax, orientation='horizontal', extend='both').set_label(
        r'Mean $\log_{10}$ ratio over the spectrum, WRF-AeroMELD / WRF-PartMC', fontsize=7)
    return figures.save(fig, 'cand_attribution_maps')


def cand_attribution_decoders(results):
    spectra = results[0][1]['spectra']
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COLUMN, 4.6),
                             gridspec_kw={'wspace': .18, 'hspace': .5, 'top': .9, 'bottom': .14,
                                          'left': .08, 'right': .99})
    n = len(results)
    width = .15
    for ax, i_sp in zip(axes.flat, range(len(spectra))):
        xs = np.arange(n)
        top = 0
        for j, t in enumerate(TERM_ORDER):
            vals = np.array([rms(mixed_values(m, m['spectra'][i_sp], t)) for _, m in results])
            ax.bar(xs + (j - 2) * width, vals, width=width * .9, color=TERM_COLORS[t], ec=INK,
                   lw=.3, zorder=2)
            top = max(top, np.nanmax(vals))
        tot = [rms(mixed_values(m, m['spectra'][i_sp], 'total')) for _, m in results]
        ax.plot(xs, tot, ls='none', marker='D', ms=4, mfc='white', mec=INK, mew=.8, zorder=3)
        lim = 1.0 if spectra[i_sp]['group'] == 'frozen_fraction' else .35
        ax.set_ylim(0, lim)
        for bar in ax.patches:
            if bar.get_height() > lim:
                ax.text(bar.get_x() + bar.get_width() / 2, lim * .97, '%.1f' % bar.get_height(),
                        ha='center', va='top', fontsize=5.5, rotation=90)
        for x, v in zip(xs, tot):
            if v > lim:
                ax.text(x, lim * 1.01, '%.1f' % v, ha='center', va='bottom', fontsize=6)
        ax.set_xticks(xs, [short_model(r) for r, _ in results], fontsize=6)
        ax.tick_params(axis='x', length=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#d9d9d9', lw=.6)
        ax.set_axisbelow(True)
        create_label_box(ax, '(%s) %s, %s' % ('abcd'[i_sp], spectra[i_sp]['name'], spectra[i_sp]['span']))
    axes[0, 0].set_ylabel(r'RMS $\log_{10}$ ratio')
    axes[1, 0].set_ylabel(r'RMS $\log_{10}$ ratio')
    term_legend(fig)
    return figures.save(fig, 'cand_attribution_decoders')


def short_model(run):
    """Two-line tick label for a decoder."""
    m = run.model
    if m.backend == 'affx':                      # the paper's own models
        return {'twin_none': 'Nonlinear\nno mixing', 'twin_pairs': 'Nonlinear\n2-way',
                'twin_k16': 'Nonlinear\nup to 16-way', 'twin_k32': 'Nonlinear\nup to 32-way',
                'ax_rel1': 'Affine', 'ax_k16': 'Affine\nup to 16-way'}.get(m.key.split('.')[0].removesuffix('_of'),
                                                                  m.label)
    if m.family.startswith('affine'):
        return 'Affine\n' + m.label.split(',', 1)[1].strip()
    return 'MLP\n$d_z=%d$%s' % (m.n_latent, ', reg.' if 'regular' in m.label else '')


def make(run, compare_runs):
    results = [(r, mixing.compute(r)) for r in compare_runs]
    main = next(m for r, m in results if r.key == run.key)
    results.sort(key=lambda rm: rm[0].key != run.key)        # main run first
    grid = products.load('grid')
    return [cand_summary_spectra(results), cand_attribution_bars(main),
            cand_attribution_violins(main), cand_attribution_maps(main, grid),
            cand_attribution_decoders(results)]


def main(argv=None):
    ap = figures.cli(__doc__)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = config.REPO / 'graphics' / 'candidates'
    run = figures.setup(args)
    make(run, registry.load().select(set_name='paper_compare'))


if __name__ == '__main__':
    main()
