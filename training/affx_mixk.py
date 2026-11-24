#!/usr/bin/env python
"""
Error against the number of mixed populations, for affx.py runs (affine or
nonlinear decoders): the paper's eight test metrics and their geometric mean
on mixtures of K test populations, K = 1 (single populations) to 16 (or more;
K up to the test-set size, 5000: Dirichlet mixtures of the whole test set).

Mixtures as in affx.py: K distinct test populations of the seed, Dirichlet(1)
volume fractions, converted to number fractions alpha with the populations'
n_tot; the true mixture is sum_k alpha_k x_k (per particle; exact for every
diagnostic), the latent of the mixture is sum_k alpha_k z_k (exact for the
particle-mean encoder). Three comparisons per mixture:

    mix    decode(sum alpha z)      vs  sum alpha x        what transport gives
    dm     sum alpha decode(z)      vs  sum alpha x        the decoded populations, mixed
    comm   decode(sum alpha z)      vs  sum alpha decode(z)  decoding-mixtures error (0 if affine)

For K = 1 all three reduce to the single-population test error (comm = 0).
The same mixtures (per seed) are used for every run, so runs compare paired.

A run argument NAME=DIR/RUN1,DIR/RUN2,... pools the seeds of several runs
(e.g. seeds 0-2 and 3-9 trained as separate jobs) under NAME. --json writes
every number.

--pct adds the tails: percentiles (50, 90, 99) and max over the mixtures of
each metric and of the per-mixture geometric means (geo1, num1). The affine
bound ||e_mix|| <= sum alpha ||e_k|| holds per mixture, so it speaks to the tails.

Usage: python affx_mixk.py [NAME=]DIR/RUN[,DIR/RUN...] ... [--K 1 2 3 4 8 16] [--nmix 2000] [--md] [--per-seed] [--pct] [--json OUT]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_errors import logratio_metrics, paper_metrics, ROWS

p = argparse.ArgumentParser()
p.add_argument('runs', nargs='+')
p.add_argument('--K', type=int, nargs='+', default=[1, 2, 3, 4, 8, 16])
p.add_argument('--nmix', type=int, default=2000)
p.add_argument('--md', action='store_true')
p.add_argument('--per-seed', action='store_true')
p.add_argument('--json', default=None)
p.add_argument('--pct', action='store_true', help='also tails: percentiles and max over the mixtures of each metric, '
               'and of the per-mixture geometric means (geo1: 8 metrics, num1: number-based 5)')
p.add_argument('--cache', default='/pscratch/sd/m/mwest/aeromeld/affx/cache.npz')
p.add_argument('--splits', default='/pscratch/sd/m/mwest/aeromeld/models/test_idx.npz')
p.add_argument('--logratio', action='store_true', help='the WRF-AeroMELD paper\'s errors: half the RMS log '
               'ratio on the logarithmic rows (paper_errors.logratio_metrics)')
args = p.parse_args()
METRICS = logratio_metrics if args.logratio else paper_metrics

BLOCKS = [('M', 225, (15, 15), True), ('N', 15, (1, 15), True), ('C', 100, (1, 100), False),
          ('S', 8, (1, 8), True), ('A', 8, (1, 8), True), ('F', 100, (1, 100), False)]
DSNAME = dict(M='m_chmprthst', N='n_prthst', C='ccn_cdf', S='qs_pop', A='qa_pop', F='frznfrac_tmp')
GEO = ['ccn_cdf', 'logqs_pop', 'logqa_pop', 'logfrznfrac_tmp', 'm_chmprthst', 'n_prthst', 'm_prthst', 'm_chmprt']
SHORT = dict(ccn_cdf='CCN', logqs_pop='scat', logqa_pop='abs', logfrznfrac_tmp='FF', m_chmprthst='Mspec',
             n_prthst='N', m_prthst='Mtot', m_chmprt='Mbulk')
c = np.load(args.cache)
X, ntot = c['X'], c['ntot']
splits = np.load(args.splits)
sp = lambda v: np.logaddexp(0, v)


def units(Xpp, n):
    out, o = {}, 0
    for k, size, shape, ext in BLOCKS:
        v = Xpp[:, o:o + size].reshape(-1, *shape)
        out[DSNAME[k]] = v * n[:, None, None] if ext else v
        o += size
    return out


NUM = ['ccn_cdf', 'logqs_pop', 'logqa_pop', 'logfrznfrac_tmp', 'n_prthst']   # number-based (no mass metrics)
PCT = (50, 90, 99)


def metrics(Xo, Xr, n):
    e = {ROWS[k]: v for k, v in METRICS(units(Xo, n), units(Xr, n)).items()}
    m = {k: float(np.mean(e[k])) for k in GEO}
    m['geo'] = float(np.exp(np.mean([np.log(m[k]) for k in GEO])))
    if args.pct:                                                    # tails over the populations / mixtures
        lg = lambda ks: np.exp(np.mean([np.log(np.maximum(e[k], 1e-12)) for k in ks], 0))  # per-mixture geo
        for name, arr in [(k, e[k]) for k in GEO] + [('geo1', lg(GEO)), ('num1', lg(NUM))]:
            for p in PCT:
                m[f'{name}_p{p}'] = float(np.percentile(arr, p))
            m[f'{name}_max'] = float(np.max(arr))
        m['geo1_mean'], m['num1_mean'] = float(lg(GEO).mean()), float(lg(NUM).mean())
    return m


def decoder(run, si):
    """decode(z (n, d)) -> x (n, 456), physical per-particle units."""
    st = torch.load(f'{run}_s{si}.pt', map_location='cpu')
    W = lambda k: st[k].double().numpy()
    s = W('s')
    if st.get('dec', 'affine') != 'mlp':
        A, b = W('A_eff'), W('b_eff')
        return lambda z: (z @ A.T + b) * s
    layers = sorted({int(k.split('.')[1]) for k in st if k.startswith('dmlp.') and k.endswith('.weight')})
    mono = st.get('mono', False)

    def dec(z):
        h = z
        for i, li in enumerate(layers):
            h = h @ W(f'dmlp.{li}.weight').T + W(f'dmlp.{li}.bias')
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        if st.get('bounded', False):
            # softmax heads (affx.py --bounded): N shares, C cumulative from low s, F cumulative from warm
            def smax(a):
                a = a - a.max(-1, keepdims=True); e = np.exp(a); return e / e.sum(-1, keepdims=True)
            n_ = h[..., :-2]
            M_, N_, C_, S_, A_, F_ = (n_[..., 0:225], n_[..., 225:240], n_[..., 240:340], n_[..., 340:348],
                                      n_[..., 348:356], n_[..., 356:456])
            wc = smax(np.concatenate([C_, h[..., -2:-1]], -1))[..., :100]
            wf = smax(np.concatenate([F_, h[..., -1:]], -1))[..., :100]
            return np.concatenate([sp(M_) * s[0:225], smax(N_), np.cumsum(wc, -1), sp(S_) * s[340:348],
                                   sp(A_) * s[348:356], np.flip(np.cumsum(np.flip(wf, -1), -1), -1)], -1)
        v = sp(h)
        parts, o = [], 0
        for k, size, _, _ in BLOCKS:
            vk = v[:, o:o + size]
            if mono and k in ('C', 'F'):
                vk = vk @ W('G' + k).T + (W('cC') if k == 'C' else 0)
            parts.append(vk); o += size
        return np.concatenate(parts, 1) * s
    return dec


rows = []
for arg in args.runs:
    name, prefixes = (arg.split('=', 1) if '=' in arg else (os.path.basename(arg), arg))
    pairs = []                                                      # (run prefix, seed)
    for run in prefixes.split(','):
        pairs += [(run, int(m.group(1))) for f in glob.glob(f'{run}_s*.pt')
                  if (m := re.fullmatch(re.escape(run) + r'_s(\d+)\.pt', f))]
    pairs.sort(key=lambda t: t[1])
    seeds = [si for _, si in pairs]
    assert len(set(seeds)) == len(seeds), f'{name}: a seed twice: {seeds}'
    res = {K: [] for K in args.K}
    for run, si in pairs:
        dec = decoder(run, si)
        Z = np.load(f'{run}_s{si}_Z.npy').astype(np.float64)
        tst = np.sort(splits['idx'][si])
        Xd = dec(Z[tst])                                             # decoded test populations
        if os.path.exists(f'{run}_s{si}_X.npy'):                     # check the numpy decoder
            Xrun = np.load(f'{run}_s{si}_X.npy')[tst[:50]]
            err = (np.abs(Xd[:50] - Xrun) / (np.abs(Xrun) + 1e-30)).max()
            assert err < 1e-3, f'{run} s{si}: numpy decoder differs from the run by {err:.1e}'
        for K in args.K:
            rng = np.random.RandomState(4321 + 97 * si + K)
            if K == 1:
                m = metrics(X[tst], Xd, ntot[tst])
                res[K].append(dict(mix=m, dm=m, comm={k: 0.0 for k in m}))
                continue
            pick = np.stack([rng.choice(len(tst), K, replace=False) for _ in range(args.nmix)])
            lam = rng.dirichlet(np.ones(K), size=args.nmix)
            nk = ntot[tst][pick]
            al = lam * nk; nm = al.sum(1); al /= nm[:, None]
            Wm = np.zeros((args.nmix, len(tst)))                      # mixing weights over the test set
            np.put_along_axis(Wm, pick, al, 1)                        # (memory O(nmix ntest) for any K)
            mo = Wm @ X[tst]
            mdm = Wm @ Xd
            mdec = dec(Wm @ Z[tst])
            res[K].append(dict(mix=metrics(mo, mdec, nm), dm=metrics(mo, mdm, nm), comm=metrics(mdm, mdec, nm)))
    rows.append((name, seeds, res))

for name, seeds, res in rows:
    print(f'\n{name}: {len(seeds)} seeds {seeds}')
    hdr = ['K', 'mix geo', '(min-max)', 'dm geo', 'comm geo'] + [SHORT[k] for k in GEO]
    if args.md:
        print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for K in args.K:
        r = res[K]
        g = [100 * x['mix']['geo'] for x in r]
        vals = [f'{np.mean(g):.2f}', f'{min(g):.2f}-{max(g):.2f}', f'{100 * np.mean([x["dm"]["geo"] for x in r]):.2f}',
                f'{100 * np.mean([x["comm"]["geo"] for x in r]):.2f}'] + \
               [f'{100 * np.mean([x["mix"][k] for x in r]):.1f}' for k in GEO]
        print(('| ' + ' | '.join([str(K)] + vals) + ' |') if args.md else '  '.join([f'K={K:2d}'] + vals))
        if args.per_seed:
            print('   per seed (mix geo): ' + ' '.join(f'{x:.2f}' for x in g))
        if args.pct:
            t = lambda kind, k: f'{100 * np.mean([x[kind][k] for x in r]):.2f}'
            print('   tails (mix, per-mixture geo / num, p50 p90 p99 max): ' +
                  ' '.join(t('mix', f'geo1_{q}') for q in ('p50', 'p90', 'p99', 'max')) + ' / ' +
                  ' '.join(t('mix', f'num1_{q}') for q in ('p50', 'p90', 'p99', 'max')) +
                  ' | comm: ' + ' '.join(t('comm', f'geo1_{q}') for q in ('p50', 'p90', 'p99', 'max')))

if args.json:
    import json
    with open(args.json, 'w') as fp:
        json.dump({name: dict(seeds=seeds, K={str(K): v for K, v in res.items()}) for name, seeds, res in rows}, fp)
    print('wrote', args.json)
