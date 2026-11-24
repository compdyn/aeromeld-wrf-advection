#!/usr/bin/env python
"""optfix_cache: affx.py's training cache with the corrected optics (cache_optfix.npz).

The particle dataset's optics use the geometric cross-section of the particle's shell; the
corrected values (the whole particle, pi/4 d^2) are computed by partnn's
dataset_build/recompute_diagnostics.py (diag_fixed.npz). This script replaces the scattering (S)
and absorption (A) blocks of affx.py's cache (cache.npz, written by affx.py from the PM10 dataset
on its first run) with them, per particle (divided by n_tot, as affx.py stores the extensive
blocks), after checking that the cache's S and A blocks are the dataset's. Everything else is
unchanged. The output is the input of dataset/dry_cache.py.

This is the script form of the one-off command that wrote cache_optfix.npz on 2026-11-11; it
reproduces that file (checked with --check).

Usage (CPU; numpy + h5py):
    python optfix_cache.py --cache cache.npz --fixed diag_fixed.npz --data 04_bwchisamp_pm10.h5 \
        --out cache_optfix.npz [--check EXISTING.npz]
"""
import argparse

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--cache', required=True, help="affx.py's cache of the PM10 dataset (cache.npz)")
p.add_argument('--fixed', required=True, help='recompute_diagnostics.py output (diag_fixed.npz)')
p.add_argument('--data', required=True, help='the PM10 dataset (04_bwchisamp_pm10.h5)')
p.add_argument('--out', required=True)
p.add_argument('--check', default=None, help='an existing cache_optfix.npz to compare with')
args = p.parse_args()

#: affx.py's block layout: M (225), N (15), C (100), then S (8) and A (8)
S0, A0 = 340, 348

c = np.load(args.cache)
X, ntot = c['X'].copy(), c['ntot']
d = np.load(args.fixed)
with h5py.File(args.data, 'r') as f:
    qs = f['qs_pop'][:].reshape(len(ntot), 8)
    qa = f['qa_pop'][:].reshape(len(ntot), 8)
assert np.allclose(X[:, S0:S0 + 8], qs / ntot[:, None], rtol=1e-12), 'the cache S block is not the dataset qs_pop'
assert np.allclose(X[:, A0:A0 + 8], qa / ntot[:, None], rtol=1e-12), 'the cache A block is not the dataset qa_pop'
X[:, S0:S0 + 8] = d['qs_pop'] / ntot[:, None]
X[:, A0:A0 + 8] = d['qa_pop'] / ntot[:, None]
np.savez(args.out, X=X, ntot=ntot, m=c['m'], i_snr=c['i_snr'])
print('wrote', args.out, '; median ratio of the new to the old S and A:',
      np.median(X[:, S0:S0 + 8] / (qs / ntot[:, None])), np.median(X[:, A0:A0 + 8] / (qa / ntot[:, None])))
if args.check:
    e = np.load(args.check)
    for k in ('X', 'ntot', 'm', 'i_snr'):
        print(k, 'identical' if np.array_equal(e[k], np.load(args.out)[k]) else 'DIFFERENT')
