#!/usr/bin/env python
"""The transport appendix's fraction-space numbers for one 12-h replay (2026-11-16, the dry rebuild):

  - separate limiters: RMS (over the mixed cells) of the implied source fractions minus per-source
    transport's, per source; the latent's distance from the nearest mixture, as a fraction of the
    spacing of the source latents (RMS and 99th percentile); the lowest implied fraction;
  - the mean gradient of the urban fraction over the mixed cells, per scheme, against per-source
    transport's (donor-cell's is "less than half");
  - the number carried as one tracer against per-source transport's sum: RMS relative difference
    over the domain and over the mixed cells (independent of the decoder).

Mixed cells: at least two sources with number fraction >= 0.05 under per-source transport
(analysis.mixed_cells' rule, as transport_terms and decomposition use).

Usage (from scripts/): python checks/transport_stats.py [--replay dry1_12h] [--zsrc zsrc_dry/zsrc_s1.npz]
                       [--model twin_k16] [--hours 12]
  the wet paper's main run: --replay all1_12h --zsrc zsrc_of/zsrc_s1.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import config, mixing  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replay', default=config.REPLAY_12H % 1)
    ap.add_argument('--zsrc', default=None, help='source latents, relative to data/wrfml_runs')
    ap.add_argument('--model', default='twin_k16')
    ap.add_argument('--hours', type=float, default=12.0)
    args = ap.parse_args()
    seed = int(''.join(c for c in args.replay.split('_')[0] if c.isdigit()))
    zsrc = args.zsrc or ('zsrc%s/zsrc_s%d.npz' % (config.SUFFIX, seed))
    with np.load(config.DATA_DIR / 'wrfml_runs' / zsrc) as f:
        z_star = f[args.model + '_z'].astype(float)
    s = T.load(args.replay)
    it = T.time_index(s, args.hours)
    f_ref = T.fractions(s, 'oracle', it)
    mixed = (f_ref >= 0.05).sum(0) >= 2
    edge = mixing.edge_length(z_star)
    rms = lambda a, m=mixed: float(np.sqrt(np.nanmean(a[m] ** 2)))
    print('%s, %s, %g h: %d mixed cells; source-latent spacing %.4g' % (args.replay, args.model, args.hours,
                                                                       mixed.sum(), edge))
    key = 'sep:' + args.model
    z = T.latent(s, key, z_star, args.model, it)
    f, _ = mixing.implied_fractions(z, z_star)
    zb = np.tensordot(z_star.T, mixing.f_bar(f), axes=1)
    dist = np.linalg.norm(z - zb, axis=0) / edge
    print('separate limiters: RMS f - f_oracle (bg, urban, emis) %s; max |df| %.3f'
          % (' '.join('%.4f' % rms(f[k] - f_ref[k]) for k in range(3)), np.nanmax(np.abs(f - f_ref)[:, mixed])))
    print('  distance from the nearest mixture / spacing: RMS %.2f%%, 99th pct %.2f%%, max %.2f%%; '
          'lowest implied fraction %.3f, highest %.3f'
          % (100 * rms(dist), 100 * np.nanpercentile(dist[mixed], 99), 100 * np.nanmax(dist[mixed]),
             np.nanmin(f[:, mixed]), np.nanmax(f[:, mixed])))

    def grad_urban(fr):
        gy, gx = np.gradient(np.nan_to_num(fr[1]))
        return float(np.hypot(gx, gy)[mixed].mean())

    g0 = grad_urban(f_ref)
    out = ['per source %.4f' % g0]
    for k, lab in (('donor', 'donor-cell'), ('shB:' + args.model, 'shared limiter')):
        if k in s:
            g = grad_urban(T.fractions(s, k, it))
            out.append('%s %.4f (%.2f)' % (lab, g, g / g0))
    out.append('separate limiters %.4f (%.2f)' % (grad_urban(f), grad_urban(f) / g0))
    print('mean |grad| of the urban fraction over the mixed cells:', ', '.join(out))
    n_sep, n_or = T.number(s, key, it), T.number(s, 'oracle', it)
    r = n_sep / n_or - 1
    allc = np.isfinite(r)
    print('number, one tracer vs per-source sum: RMS relative difference %.2f%% (domain), %.2f%% (mixed)'
          % (100 * rms(r, allc), 100 * rms(r)))


if __name__ == '__main__':
    main()
