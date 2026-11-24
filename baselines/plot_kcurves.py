#!/usr/bin/env python
"""The paper's Figure 7 (mixing_kcurves: test populations mixed K at a time, error against the true
mixture) with the sectional and modal baselines added.

The decoders are drawn by scripts/plot/mixing_training.py's own functions (imported, unchanged:
draw_kcurve, finish_k, the labels and AFFX_STYLES), from notes/affx_stats/mixk_r33_lr.json. The
baselines come from baselines/data/kcurves_baselines.json (baselines/kcurves.py, the same mixtures
and the same error), drawn the same way: line = mean over the split seeds of the six-diagnostic
mean error of the 'mix' comparison, band = min-max over the seeds.

Styles: the decoders keep the paper's (mixing_training.py: green ramp, magenta affine); the
baselines take baselines/style.py's shared STYLES (sectional browns dashed, modal purples dash-dot).
Shown (coordinator, 2026-11-14): sec20w (the coag paper's 20 wet bins, the best sectional) and sec8
(MOSAIC 8 bins); mam4, and mam4a only if its curve differs visibly from mam4's (MAM_A_VISIBLE: by
more than 0.5 percentage points or 5 % at some K). 'prdiag' (diag.py on the true particles, the
floor of the comparison) is a thin grey dotted line in the log-scale versions only.

Writes baselines/figures/mixing_kcurves_baselines.{pdf,png} (linear axis, as the paper),
mixing_kcurves_baselines_log.{pdf,png} (log axis, so the decoders and the baselines both read) and
mixing_kcurves_baselines_all.{pdf,png} (log axis, every baseline computed).

Usage: MPLBACKEND=Agg .venv/bin/python baselines/plot_kcurves.py [--json baselines/data/kcurves_baselines.json]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'scripts'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402


def load_mixing_training():
    spec = importlib.util.spec_from_file_location('mixing_training', REPO / 'scripts' / 'plot' / 'mixing_training.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MT = load_mixing_training()
from paperfigs import fig_palettes  # noqa: E402

sys.path.insert(0, str(HERE))
from style import STYLES  # noqa: E402

#: name -> (label, colour, marker, linestyle), from baselines/style.py; in drawing/legend order.
ORDER = ('sec20w', 'sec8', 'sec20', 'sec4', 'mam4', 'mam4a', 'mam4l')
BASELINES = {n: (STYLES[n]['label'], STYLES[n]['color'], STYLES[n]['marker'], STYLES[n]['ls']) for n in ORDER}
MAIN = ('sec20w', 'sec8', 'mam4', 'mam4a')
FLOOR = ('prdiag', 'Particle diagnostics of the reference particles\n(floor of the comparison)', '#8c8c8c')


def bcurve(d, name, metric='six', kind='mix'):
    """(K (n,), values (seeds, n)) in % from the baselines JSON."""
    e = d[name]
    Ks = sorted(int(k) for k in e['K'])
    val = (lambda m: np.mean([m[k] for k in MT.SIX])) if metric == 'six' else (lambda m: m[metric])
    v = np.array([[100 * val(r[kind]) for r in e['K'][str(K)]] for K in Ks]).T
    return np.array(Ks), v


def draw_baseline(ax, d, name, kind='mix'):
    label, color, marker, ls = BASELINES[name]
    K, v = bcurve(d, name, kind=kind)
    if v.shape[0] > 1:
        ax.fill_between(K, v.min(0), v.max(0), color=color, alpha=.18, lw=0, zorder=1)
    ax.plot(K, v.mean(0), ls=ls, color=color, lw=1.2, marker=marker, ms=3.0, mfc=color, mec=MT.INK,
            mew=.4, zorder=3)


def mam4a_visible(d, tol_abs=0.5, tol_rel=0.05):
    """Whether mam4a's mean curve differs visibly from mam4's, compared on the splits both have
    (the split-to-split spread at large K is larger than their difference)."""
    if 'mam4a' not in d or 'mam4' not in d:
        return False
    common = sorted(set(d['mam4a']['seeds']) & set(d['mam4']['seeds']))
    def mean(n):
        e = d[n]
        idx = [e['seeds'].index(si) for si in common]
        return np.array([np.mean([100 * np.mean([e['K'][K][i]['mix'][k] for k in MT.SIX]) for i in idx])
                         for K in sorted(e['K'], key=int)])
    a, b = mean('mam4a'), mean('mam4')
    return bool(np.any(np.abs(a - b) > np.maximum(tol_abs, tol_rel * b)))


def shown(d, every=False):
    if every:
        return [n for n in ORDER if n in d]
    return [n for n in MAIN if n in d and (n != 'mam4a' or mam4a_visible(d))]


def handles(names_dec, names_base, floor=False):
    h = []
    for n in names_dec:
        c, mk = MT.style(n)
        h.append(Line2D([], [], color=c, marker=mk, mfc=c, mec=MT.INK, mew=.4, ms=4, lw=1.2, label=MT.LABELS[n]))
    for n in names_base:
        label, c, mk, ls = BASELINES[n]
        h.append(Line2D([], [], color=c, marker=mk, mfc=c, mec=MT.INK, mew=.4, ms=3.5, lw=1.2, ls=ls, label=label))
    if floor:
        h.append(Line2D([], [], color=FLOOR[2], lw=.9, ls=':', label=FLOOR[1]))
    return h


def figure(d, log=False, every=False):
    names = shown(d, every)
    floor = log and FLOOR[0] in d
    rows = len(MT.FIVE) + len(names) + (2 if floor else 0)
    leg = 0.155 * rows + 0.05                   # legend height (in)
    h = 2.25 + 0.5 + leg                        # axes + x label + legend
    fig, a = plt.subplots(figsize=(fig_palettes.SINGLE_COLUMN, h),
                          gridspec_kw=dict(left=.15, right=.97, top=1 - .08 / h, bottom=(leg + .5) / h))
    for n in MT.FIVE:
        MT.draw_kcurve(a, n)
    for n in names:
        draw_baseline(a, d, n)
    if floor:
        K, v = bcurve(d, FLOOR[0])
        a.plot(K, v.mean(0), ls=':', color=FLOOR[2], lw=.9, zorder=2)
    MT.finish_k(a, ticks=(1, 4, 16, 64, 256, 1024, 5000))
    a.set_xlabel('Number of test populations mixed, $K$')
    a.set_ylabel('Error against the true mixture (%)')
    if log:
        a.set_yscale('log')
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        lo = 0.1 if floor else 1
        top = max([50] + [bcurve(d, n)[1].max() * 1.15 for n in names])
        a.set_ylim(lo, top)
        ticks = [t for t in ticks if lo <= t <= top]
        a.yaxis.set_major_locator(FixedLocator(ticks))
        a.yaxis.set_minor_locator(NullLocator())
        a.set_yticklabels([('%g' % t) for t in ticks])
    else:
        a.set_ylim(0, None)
    fig.legend(handles=handles(MT.FIVE, names, floor), loc='lower center', ncol=1, frameon=False,
               bbox_to_anchor=(.5, 0), handletextpad=.4, fontsize='small', handlelength=2.6)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--json', default=str(HERE / 'data' / 'kcurves_baselines.json'))
    ap.add_argument('--out', default=str(HERE / 'figures'))
    args = ap.parse_args(argv)
    d = json.loads(Path(args.json).read_text())
    fig_palettes.use_paper_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print('mam4a drawn in the main figures:', mam4a_visible(d))
    for log, every, tag in ((False, False, ''), (True, False, '_log'), (True, True, '_all')):
        fig = figure(d, log, every)
        stem = 'mixing_kcurves_baselines' + tag
        for ext in ('pdf', 'png'):
            fig.savefig(out / ('%s.%s' % (stem, ext)), dpi=300)
            print('wrote', out / ('%s.%s' % (stem, ext)))
        plt.close(fig)
    # the numbers behind the figure, for the notes
    print(tables(d))


def row(d, n, Ks, kind='mix', metric='six'):
    """The baseline's mean over splits at each K of Ks ('-' where it has no such K)."""
    Kb, v = bcurve(d, n, metric, kind)
    m = dict(zip(Kb.tolist(), v.mean(0)))
    return ' | '.join(('%.2f' % m[K]) if K in m else '-' for K in Ks)


def tables(d):
    """Markdown tables of the numbers behind the figure (for notes_kcurves.md)."""
    names = [n for n in BASELINES if n in d]
    Ks = MT.kcurve(MT.FIVE[0])[0].tolist()
    L = []
    L.append('Overall error (mean of the six diagnostics, %, mean over splits), against the dataset truth (mix):\n')
    L.append('| model | ' + ' | '.join('K=%d' % K for K in Ks) + ' |')
    L.append('|---|' + '---|' * len(Ks))
    for n in MT.FIVE:
        L.append('| %s | ' % MT.LABELS[n] + ' | '.join('%.2f' % x for x in MT.kcurve(n)[1].mean(0)) + ' |')
    for n in ([FLOOR[0]] if FLOOR[0] in d else []) + names:
        lab = BASELINES[n][0] if n in BASELINES else 'prdiag (diag.py on the true particles)'
        L.append('| %s | ' % lab + row(d, n, Ks) + ' |')
    for kind, title in (('mix_pr', 'against diag.py on the true particles (mix_pr: the representation alone)'),
                        ('dm', 'reconstruction, each population represented then mixed (dm), vs dataset truth'),
                        ('comm', 'averaging, mixing inside bins/modes (comm)')):
        have = [n for n in names if kind in d[n]['K']['1'][0]]
        if not have:
            continue
        L.append('\n' + title[0].upper() + title[1:] + ':\n')
        L.append('| model | ' + ' | '.join('K=%d' % K for K in Ks) + ' |')
        L.append('|---|' + '---|' * len(Ks))
        for n in have:
            L.append('| %s | ' % BASELINES[n][0] + row(d, n, Ks, kind) + ' |')
    sel = [K for K in (1, 3, 16, 5000) if K in Ks]
    L.append('\nPer diagnostic (mix, %%, mean over splits) at K = %s:\n' % ', '.join(map(str, sel)))
    L.append('| model | K | ' + ' | '.join(MT.SIX) + ' |')
    L.append('|---|---|' + '---|' * len(MT.SIX))
    for n in ('twin_k16', 'ax_rel1'):
        for K in sel:
            i = Ks.index(K)
            L.append('| %s | %d | ' % (MT.LABELS[n].split(' (')[0], K) +
                     ' | '.join('%.2f' % MT.kcurve(n, k)[1].mean(0)[i] for k in MT.SIX) + ' |')
    for n in ([FLOOR[0]] if FLOOR[0] in d else []) + names:
        for K in sel:
            L.append('| %s | %d | ' % (n, K) + ' | '.join(row(d, n, [K], metric=k) for k in MT.SIX) + ' |')
    return '\n'.join(L)

if __name__ == '__main__':
    main()
