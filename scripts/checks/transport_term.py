#!/usr/bin/env python
"""What the attribution's transport term is made of (notes/transport_term.md).

The transport term compares WRF-AeroMELD's source fractions f_ml (where the
advected latent sits in the sources' triangle) with WRF-PartMC's f_pr (from its
particle tags). Is their difference the benchmark's finite particle count, or a
real difference in how the two models transport the source air? For one run
(default the main run), over the mixed cells at 04 UTC:

  1. RMS of d = f_ml - f_pr per source, against the binomial noise
     sqrt(f (1 - f) / N_p) that N_p particles per cell would give, and the
     N_p that noise alone would need.
  2. Spatial structure: lag-1 correlation of d between neighbouring cells
     (white noise: 0), and the RMS of d after averaging both fields over
     k x k boxes (white noise: down by k).
  3. Front sharpness: cells that are mixed (largest fraction < 0.95, 0.8) in
     each model, and the mean gradient of the background fraction.
  4. The sources' frozen fraction (PartMC), which says why frozen fraction has
     the largest transport term.
  5. Control: the total number N. Both models advect it with the same scheme
     (WRF's 5th-order monotonic advection); only the source fractions differ in
     scheme (WRF-AeroMELD moves the latent donor-cell with N's fluxes,
     WRF-PartMC advects each source's number on its own). If the codes'
     difference is the composition step, N agrees and the fractions do not.

Writes notes/transport_check/fractions.png (f_pr, f_ml and d for each source).

Usage (from scripts/): python checks/transport_term.py [--run KEY]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

from paperfigs import config, mixing, products, registry  # noqa: E402

OUT = config.REPO / 'notes' / 'transport_check'


def lag_corr(a, mask):
    """Correlation of a between north and east neighbours, both in mask."""
    out = []
    for a1, a2, m in ((a[:-1], a[1:], mask[:-1] & mask[1:]), (a[:, :-1], a[:, 1:], mask[:, :-1] & mask[:, 1:])):
        out.append(np.corrcoef(a1[m], a2[m])[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default=None, help='registry run key (default: the online run twin_k16_of.0.0_5e7)')
    args = ap.parse_args()
    reg = registry.load()
    run = reg.runs[args.run or 'twin_k16_of.0.0_5e7']        # the online run notes/transport_term.md is about
    m = mixing.compute(run)
    f_pr, f_ml, mixed = m['f_pr'], m['f_ml'], m['mixed']
    d = f_ml - f_pr
    rms = lambda a: float(np.sqrt(np.nanmean(a ** 2)))
    print('run %s, %s, %d mixed cells' % (run.key, m['timestamp'], mixed.sum()))

    print('\n1. Source-fraction difference d = f_ml - f_pr against binomial noise')
    for s, (sc, name) in enumerate(zip(config.POP_SCENARIOS, config.POP_LABELS)):
        var = f_pr[s] * (1 - f_pr[s])
        print('  %-10s mean %+.3f  RMS %.3f   noise RMS at 1000 / 5000 particles %.4f / %.4f   '
              'particles per cell noise would need %.0f'
              % (name, np.nanmean(d[s][mixed]), rms(d[s][mixed]), np.sqrt(var[mixed].mean() / 1000),
                 np.sqrt(var[mixed].mean() / 5000), var[mixed].mean() / np.mean(d[s][mixed] ** 2)))

    print('\n2. Spatial structure (white noise: correlation 0, box average divides RMS by k)')
    for s, name in enumerate(config.POP_LABELS):
        sm = [rms((uniform_filter(np.nan_to_num(f_ml[s]), k, mode='nearest')
                   - uniform_filter(np.nan_to_num(f_pr[s]), k, mode='nearest'))[mixed]) for k in (1, 3, 5, 9)]
        print('  %-10s neighbour correlation of d (N, E) %.2f %.2f   (f_pr %.2f %.2f, f_ml %.2f %.2f)   '
              'RMS of d, box 1/3/5/9: %s' % ((name,) + tuple(lag_corr(d[s], mixed)) + tuple(lag_corr(f_pr[s], mixed))
                                            + tuple(lag_corr(f_ml[s], mixed)) + (' '.join('%.3f' % v for v in sm),)))

    print('\n3. Front sharpness')
    for name, f in (('WRF-PartMC', f_pr), ('WRF-AeroMELD', f_ml)):
        mx = np.nanmax(f, 0)
        gy, gx = np.gradient(np.nan_to_num(f[0]))
        print('  %-12s cells with largest fraction < 0.95: %d, < 0.8: %d;  mean |grad f_background| '
              'over mixed cells %.3f per cell' % (name, (mx < .95).sum(), (mx < .8).sum(), np.hypot(gx, gy)[mixed].mean()))

    print('\n4. Frozen fraction of the sources (PartMC)')
    pop = products.load('populations', model=run.model.key)
    ff = mixing._source_spectra(pop, m['timestamp'])[1]['frozen_fraction']
    for t in (-40, -35, -30, -25):
        v = ff.sel(temperature=t, method='nearest').values
        print('  %d C: ' % t + '  '.join('%s %.2e' % (n, x) for n, x in zip(config.POP_LABELS, v)))

    for sp in m['spectra']:
        t = sp['terms']['transport'][mixed]
        print('  transport term, %-15s mean %+.3f RMS %.3f' % (sp['group'], np.nanmean(t), rms(t)))

    print('\n5. Control: total number N (same advection scheme in both) against the fractions')
    n_ml = products.load('aeromeld_domain', run=run.key)['ntot'].values.astype(float)
    fields = products.load('partmc_fields')
    n_pr = fields['ntot'].sel(instant=m['timestamp']).values.astype(float)
    grad = lambda a: np.hypot(*np.gradient(np.nan_to_num(a)))[mixed].mean()
    pairs = [('log10 N', np.log10(n_pr), np.log10(n_ml))] + [
        ('f ' + name, f_pr[s], f_ml[s]) for s, name in enumerate(config.POP_LABELS)]
    for name, a_pr, a_ml in pairs:
        dd = a_ml - a_pr
        if name == 'log10 N':                 # a constant offset is not a transport difference
            dd = dd - np.median(dd[mixed])
        print('  %-14s RMS difference / spread of the WRF-PartMC field %.2f   correlation %.3f   '
              'mean |gradient| per cell: WRF-PartMC %.3f, WRF-AeroMELD %.3f'
              % (name, rms(dd[mixed]) / a_pr[mixed].std(), np.corrcoef(a_pr[mixed], a_ml[mixed])[0, 1],
                 grad(a_pr), grad(a_ml)))

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(3, 3, figsize=(11, 10))
    for s, name in enumerate(config.POP_LABELS):
        for j, (t, a, cm, lim) in enumerate((('WRF-PartMC $f$', f_pr[s], 'viridis', (0, 1)),
                                            ('WRF-AeroMELD $f$', f_ml[s], 'viridis', (0, 1)),
                                            ('difference', d[s], 'RdBu_r', (-.4, .4)))):
            ax = axs[s, j]
            im = ax.imshow(a, origin='lower', cmap=cm, vmin=lim[0], vmax=lim[1])
            ax.contour(mixed, [.5], colors='k', linewidths=.4)
            ax.set_title('%s: %s' % (name, t), fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle('Source fractions at %s, lowest level (black: the mixed cells)' % m['timestamp'], fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fractions.png', dpi=90)
    print('\nwrote', OUT / 'fractions.png')


if __name__ == '__main__':
    main()
