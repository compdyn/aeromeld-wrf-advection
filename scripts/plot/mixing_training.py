#!/usr/bin/env python
"""How much mixture training the nonlinear decoder needs, and the affine decoder.

Three figures for the paper's sections "How important is mixing?" and "What
about a true affine model?" (notes/affx_story.md, section 8). All use every
seed of each model (ten): lines or large markers are the mean over seeds, bands
or small dots the seeds themselves.

  mixing_kcurves     test populations mixed K at a time (K = 1 to the whole
                     test set, Dirichlet weights): the error against the true
                     mixture, for the nonlinear decoder trained with no
                     mixtures, pairs, up to 16 and up to 32 populations, and the
                     affine AX-rel1. From notes/affx_stats/*.json
                     (training/affx_mixk.py). Dropped 2026-11-11 (the text gives
                     the numbers, all in the JSON): a panel of the
                     decoding-mixtures part ('comm'), and the figure
                     affine_kcurves (main model against AX-rel1 per diagnostic).
  mixing_wrf         the case's runs of all five models, ten seeds each: (a)
                     spectrum error over the mixed cells at 04 UTC, (b) RMS of
                     the decoding-mixtures term of the attribution (zero for the
                     affine decoder). From data/seed_stats.json (seed_stats.py).
                     Since 2026-11-12: the offline replays with separate
                     limiters, against the exact answer (TRANSPORT, REFERENCE).

Errors on the test populations are the paper's error (training/paper_errors.py
logratio_metrics, as paperfigs/errors.py for the case), averaged over the populations or
mixtures; the overall error is the mean over the six diagnostics (number and species mass
size distributions, CCN, frozen fraction, scattering, absorption). Since 2026-11-13 from
mixk_r33_lr.json (affx_mixk.py --logratio); before, the geometric mean over eight
diagnostics of the AeroMELD paper's metrics (mixk_r33.json).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from paperfigs import config, figures, registry  # noqa: E402
from paperfigs.fig_palettes import AFFX_STYLES, SINGLE_COLUMN, TWO_COLUMN, create_label_box  # noqa: E402

STATS = config.REPO / 'notes' / 'affx_stats'
INK, GRID = '#1a1a1a', '#e3e3e3'

#: model name -> (stats json, name in it); every one has ten seeds.
#: The paper's models are those retrained on the optics-corrected targets (round 33,
#: notes/pmc_diagnostics_check.md); ORIGINAL = True draws the first set instead.
ORIGINAL = False
#: Since 2026-11-19 the final rebuild (round 39: dry diameter, every mass including water; mixk_r39_lr.json;
#: config.SUFFIX). Before: round 37 (2026-11-16, water left out), round 33 (wet diameter).
MIXK_JSON = {'_dd': 'mixk_r39_lr.json', '_dry': 'mixk_r37_lr.json', '_of': 'mixk_r33_lr.json'}[config.SUFFIX]
KCURVES = {b: (MIXK_JSON, config.recipe(b)) for b in config.BASES} if not ORIGINAL else {
    'twin_none': ('mixk_r30.json', 'NLk0-mix0'),
    'twin_pairs': ('mixk_bigK_10s.json', 'NL-mix10'),
    'twin_k16': ('mixk_r31.json', 'NL-mix10-k16'),
    'twin_k32': ('mixk_r32.json', 'NL-mix10-k32'),
    'ax_rel1': ('mixk_bigK_10s.json', 'AX-rel1'),
}
TWINS = ('twin_none', 'twin_pairs', 'twin_k16', 'twin_k32')
FIVE = TWINS + ('ax_rel1',)
#: The paper's numbered decoders (Section 2.3), as there.
LABELS = {'twin_none': 'Decoder 1: no mixtures', 'twin_pairs': 'Decoder 2: pairs',
          'twin_k16': 'Decoder 3: up to 16 (WRF-AeroMELD)', 'twin_k32': 'Decoder 4: up to 32',
          'ax_rel1': 'Decoder 5: affine'}
#: The six diagnostics of the paper's error (paperfigs/errors.py SIX); 'six' is their mean.
SIX = ('n_prthst', 'm_chmprthst', 'ccn_cdf', 'logfrznfrac_tmp', 'logqs_pop', 'logqa_pop')

def kcurve(name, metric='six', kind='mix'):
    """(K (n,), values (seeds, n)) in %."""
    fn, key = KCURVES[name]
    d = json.loads((STATS / fn).read_text())[key]
    Ks = sorted(int(k) for k in d['K'])
    val = (lambda m: np.mean([m[k] for k in SIX])) if metric == 'six' else (lambda m: m[metric])
    v = np.array([[100 * val(r[kind]) for r in d['K'][str(K)]] for K in Ks]).T
    if v.shape[0] != 10:
        raise ValueError('%s %s: %d seeds, expected 10' % (fn, key, v.shape[0]))
    return np.array(Ks), v


def style(name):
    color, marker = AFFX_STYLES[name]
    return color, marker


def draw_kcurve(ax, name, metric='six', kind='mix', ls='-', band=True):
    K, v = kcurve(name, metric, kind)
    if kind == 'comm' and name == 'ax_rel1':
        return
    color, marker = style(name)
    if band:
        ax.fill_between(K, v.min(0), v.max(0), color=color, alpha=.18, lw=0, zorder=1)
    ax.plot(K, v.mean(0), ls=ls, color=color, lw=1.2, marker=marker, ms=3.2, mfc=color, mec=INK,
            mew=.4, zorder=3)


def finish_k(ax, ylabel=None, ticks=(1, 2, 4, 16, 64, 256, 1024, 5000)):
    ax.set_xscale('log')
    ax.set_xlim(.8, 7000)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(['all' if t == 5000 else str(t) for t in ticks])
    ax.grid(color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def legend(fig, names, y=0.0, ncol=None):
    handles = [Line2D([], [], color=style(n)[0], marker=style(n)[1], mfc=style(n)[0], mec=INK, mew=.4,
                      ms=4, lw=1.2, label=LABELS[n]) for n in names]
    fig.legend(handles=handles, loc='lower center', ncol=ncol or len(handles), frameon=False,
               bbox_to_anchor=(.5, y), handletextpad=.4, columnspacing=1.3)


def mixing_kcurves():
    fig, a = plt.subplots(figsize=(SINGLE_COLUMN, 3.35),
                          gridspec_kw=dict(left=.15, right=.97, top=.97, bottom=.42))
    for n in FIVE:
        draw_kcurve(a, n)
    finish_k(a, ticks=(1, 4, 16, 64, 256, 1024, 5000))
    a.set_xlabel('Number of test populations mixed, $K$')
    a.set_ylim(0, None)
    a.set_ylabel('Overall error (%)')   # the caption: against the true mixture
    legend(fig, FIVE, ncol=1)
    return figures.save(fig, 'mixing_kcurves')


#: Which runs mixing_wrf shows (seed_stats.json 'transport'): the offline replays with
#: separate limiters, the paper's AeroMELD since 2026-11-12, or 'online' (WRF-AeroMELD).
TRANSPORT = 'sep'
#: Against what: 'exact' (per-source transport of the sources' particle-resolved
#: diagnostics; replays only) or 'partmc' (WRF-PartMC).
REFERENCE = 'exact'


def seed_stats():
    return json.loads((config.DATA_DIR / 'seed_stats.json').read_text())


def mixing_wrf():
    st = {k: r for k, r in seed_stats().items() if r.get('transport', 'online') == TRANSPORT}
    spectra = 'spectra_exact' if REFERENCE == 'exact' else 'spectra'
    groups = ['ccn', 'scattering', 'absorption', 'frozen_fraction']
    any_run = next(iter(st.values()))['spectra']
    fig, axs = plt.subplots(2, 4, figsize=(TWO_COLUMN, 3.9),
                            gridspec_kw=dict(left=.1, right=.99, top=.93, bottom=.17, wspace=.34,
                                             hspace=.3))
    rng = np.random.default_rng(0)
    for j, g in enumerate(groups):
        for row, (key, ylab) in enumerate((('smape', 'Error against the\n%s (%%)' % (
                'exact answer' if REFERENCE == 'exact' else 'WRF-PartMC')),
                                           ('rms_mixing', 'Decoding mixtures,\nRMS $\\log_{10}$ ratio'))):
            ax = axs[row, j]
            for i, n in enumerate(FIVE):
                vals = np.array([r[spectra][g][key] for r in st.values()
                                 if r['name'] == (n if ORIGINAL else config.recipe(n))])
                if len(vals) != 10:
                    raise ValueError('%s: %d seeds in seed_stats.json' % (n, len(vals)))
                color, marker = style(n)
                ax.plot(i + rng.uniform(-.18, .18, len(vals)), vals, ls='none', marker=marker, ms=2.6,
                        mfc=color, mec='none', alpha=.55, zorder=2)
                ax.plot(i, vals.mean(), ls='none', marker=marker, ms=6, mfc=color, mec=INK, mew=.6, zorder=3)
            ax.set_xlim(-.6, len(FIVE) - .4)
            ax.set_xticks([])                     # the legend names the models
            ax.grid(axis='y', color=GRID, lw=.5)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            if row == 0:
                ax.set_yscale('log')
                if REFERENCE == 'exact':
                    lo, hi = (1, 10) if g == 'ccn' else (2, 100)
                    ticks = [1, 2, 5, 10] if g == 'ccn' else [2, 5, 10, 20, 50, 100]
                else:
                    lo, hi = (3, 20) if g == 'ccn' else (10, 200)
                    ticks = [3, 5, 10, 20] if g == 'ccn' else [10, 20, 50, 100, 200]
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_yticklabels([str(t) for t in ticks])
                create_label_box(ax, '(%s) %s' % ('abcd'[j], any_run[g]['name']))
            else:
                ax.set_ylim(0, None)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                create_label_box(ax, '(%s)' % 'efgh'[j])
            if j == 0:
                ax.set_ylabel(ylab)
    legend(fig, FIVE, y=0, ncol=3)
    return figures.save(fig, 'mixing_wrf')


def make(_run=None):
    # mixing_wrf dropped 2026-11-12: the models are compared on the test set
    # (mixing_kcurves); the CARES case compares transport schemes (plot/transport_schemes.py).
    return [mixing_kcurves()]


def main(argv=None):
    ap = figures.cli(__doc__)
    args = ap.parse_args(argv)
    if args.out is None:                       # candidates until adopted into the paper
        args.out = config.REPO / 'graphics' / 'candidates' / 'mixing'
    run = figures.setup(args)
    make(run)


if __name__ == '__main__':
    main()
