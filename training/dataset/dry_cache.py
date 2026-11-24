#!/usr/bin/env python
"""dry_cache: affx.py's training cache with the two size distributions binned by DRY diameter.

2026-11-16: how do the comparison figures change if every number and mass size distribution
is binned by dry diameter, the usual coordinate in atmospheric science? The comparison schemes'
dry distributions come from their representative particles (baselines/dist_checks.py); for
WRF-AeroMELD, whose decoder outputs wet distributions, a decoder is retrained on dry targets with
the encoder frozen (affx.py --init-from --freeze-enc --cache <this file's output>), so the latents
and the transport runs are unchanged.

Reads an affx.py cache (X, ntot, m, i_snr; default cache_optfix.npz, the paper's targets) and the
dataset's particles in float64 (m_prtchm, w_prt), and writes a copy whose M (15 species x 15 bins)
and N (15 bins) blocks are binned by dry diameter, (6/pi sum_{k != H2O} m_k / rho_k)^(1/3); H2O is
excluded from the diameter and from M (its row is zero). Everything else (CCN, optics, frozen
fraction, the particles the encoder reads) is unchanged.

Check first: the WET blocks recomputed here with the same code (all 15 species in the diameter)
must match the cache's, which pins the species order, the densities, the bins and the
normalization (M in kg per particle, N the fraction of particles in each bin).

Bins as baselines/diag.py (vendor vae_partmc_utils): np.histogram on 20 logspace edges from
10^-8.75 m to 100 um, [lo, hi), the first 15 kept (to 10 um); a particle outside them is in no bin
but still counts in the per-particle normalization.

2026-11-19 (the final choice): --basis drydiam bins by dry diameter as above but keeps every
species' mass in M, water included (its row is the particles' water in each dry-diameter bin), for
training round 39 (cache_drydiam.npz). Its extra check: N and M's 14 non-water rows must equal those
of --dry-cache (cache_dry.npz, the same binning), and M's water row must hold the water in the bins.

Usage (on a Perlmutter CPU node; numpy + h5py):
    python dry_cache.py [--basis dry|drydiam] [--cache IN.npz] [--data DATA.h5] [--out OUT.npz]
"""
import argparse
import json
import time

import h5py
import numpy as np

SPECIES = ('SO4', 'NO3', 'Cl', 'NH4', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'Na', 'OIN', 'OC', 'BC',
           'MOC', 'H2O')
#: kg m-3, the particle files' (= extract_diagnostics.DEFAULT_CHEM_DENSITY; baselines/diag.RHO)
RHO = np.array([1800., 1800., 2200., 1800., 1400., 1400., 1400., 1400., 1400., 2200., 2600., 1000., 1800.,
                1000., 1000.])
I_H2O = SPECIES.index('H2O')
DRY = np.arange(15) != I_H2O
N_BINS_ALL, NB = 19, 15
EDGES = np.logspace(-8.75, -4.0, N_BINS_ALL + 1)
I_M, I_N = slice(0, 225), slice(225, 240)          # affx.py BLOCKS: M (225), N (15), C, S, A, F

p = argparse.ArgumentParser()
p.add_argument('--cache', default='/pscratch/sd/m/mwest/aeromeld/affx/cache_optfix.npz')
p.add_argument('--data', default='/pscratch/sd/m/mwest/aeromeld/data/04_bwchisamp_pm10.h5')
p.add_argument('--basis', default='dry', choices=('dry', 'drydiam'),
               help="dry: H2O excluded from the diameter and from M (round 37); drydiam: H2O excluded from "
                    "the diameter only, every species' mass in M (round 39)")
p.add_argument('--out', default=None, help='default cache_dry.npz or cache_drydiam.npz next to --cache')
p.add_argument('--dry-cache', default='/pscratch/sd/m/mwest/aeromeld/affx/cache_dry.npz',
               help='--basis drydiam: the round-37 cache its N and non-water M rows must reproduce')
p.add_argument('--chunk', type=int, default=500)
args = p.parse_args()
if args.out is None:
    args.out = '/pscratch/sd/m/mwest/aeromeld/affx/cache_%s.npz' % args.basis


def blocks(m, w, basis):
    """Particles m (P, n, 15) kg, weights w (P, n) -> M (P, 15, 15) kg per particle and N (P, 15)
    the fraction of particles, binned by 'wet' or dry diameter: 'dry' (the H2O row of M is 0) or
    'drydiam' (every species' mass in M, water included)."""
    P = len(m)
    w = np.where(w > 0, w, 0.0)
    m = np.where((w[..., None] > 0) & (m > 0), m, 0.0)
    w = np.where(m.sum(-1) == 0.0, 0.0, w)                        # vendor clean_prtcls
    tot = w.sum(-1, keepdims=True)
    wn = w / np.where(tot > 0, tot, 1.0)
    sel = np.ones(15, bool) if basis == 'wet' else DRY              # the species in the diameter
    D = (6.0 / np.pi * (m[..., sel] / RHO[sel]).sum(-1)) ** (1.0 / 3.0)
    sel = DRY if basis == 'dry' else np.ones(15, bool)              # the species in M
    b = np.searchsorted(EDGES, D, side='right') - 1
    b = np.where(D == EDGES[-1], N_BINS_ALL - 1, b)
    ok = (b >= 0) & (b < NB) & (wn > 0)
    idx = (np.arange(P)[:, None] * NB + np.where(ok, b, 0)).ravel()
    wv = np.where(ok, wn, 0.0)
    N = np.bincount(idx, weights=wv.ravel(), minlength=P * NB).reshape(P, NB)
    M = np.zeros((P, 15, NB))
    for k in np.flatnonzero(sel):
        M[:, k] = np.bincount(idx, weights=(wv * m[..., k]).ravel(), minlength=P * NB).reshape(P, NB)
    return M, N


t0 = time.time()
c = np.load(args.cache)
X, ntot, i_snr = c['X'], c['ntot'], c['i_snr']
n_pop = len(ntot)
print('cache %s: %d populations, X %s (%.0f s)' % (args.cache, n_pop, X.shape, time.time() - t0), flush=True)
Xd = X.copy()
dev = {'M': [], 'N': []}
mass_wet, mass_dry = np.zeros(n_pop), np.zeros(n_pop)
with h5py.File(args.data, 'r') as f:
    assert f['n_tot'].shape[0] == n_pop or np.prod(f['n_tot'].shape) == n_pop
    mds, wds = f['m_prtchm'], f['w_prt']
    for a in range(0, n_pop, args.chunk):
        e = min(a + args.chunk, n_pop)
        m = mds[a:e].astype(np.float64).reshape(e - a, -1, 15)
        w = wds[a:e].astype(np.float64).reshape(e - a, -1)
        Mw, Nw = blocks(m, w, 'wet')
        Md, Nd = blocks(m, w, args.basis)
        # the check: recomputed wet blocks against the cache's, relative to each population's max
        Mc, Nc = X[a:e, I_M].reshape(e - a, 15, NB), X[a:e, I_N]
        dev['M'].append(np.abs(Mw - Mc).max(axis=(1, 2)) / np.maximum(np.abs(Mc).max(axis=(1, 2)), 1e-300))
        dev['N'].append(np.abs(Nw - Nc).max(axis=1) / np.maximum(np.abs(Nc).max(axis=1), 1e-300))
        Xd[a:e, I_M] = Md.reshape(e - a, -1)
        Xd[a:e, I_N] = Nd
        mass_wet[a:e], mass_dry[a:e] = Mw.sum(axis=(1, 2)), Md.sum(axis=(1, 2))
        if a % (20 * args.chunk) == 0:
            print('  %6d / %d (%.0f s)' % (e, n_pop, time.time() - t0), flush=True)
dM, dN = np.concatenate(dev['M']), np.concatenate(dev['N'])
check = {'wet_M_max_rel_dev': float(dM.max()), 'wet_M_pops_above_1e-6': int((dM > 1e-6).sum()),
         'wet_N_max_rel_dev': float(dN.max()), 'wet_N_pops_above_1e-6': int((dN > 1e-6).sum()),
         'wet_M_rel_dev_p99': float(np.percentile(dM, 99)), 'wet_N_rel_dev_p99': float(np.percentile(dN, 99)),
         'dry_over_wet_mass_in_bins_median': float(np.median(mass_dry / np.maximum(mass_wet, 1e-300))),
         'dry_N_in_bins_median': float(np.median(Xd[:, I_N].sum(1))),
         'wet_N_in_bins_median': float(np.median(X[:, I_N].sum(1)))}
check['basis'] = args.basis
if args.basis == 'drydiam':
    # the same dry-diameter binning as the round-37 cache: N and the 14 non-water rows of M identical
    with np.load(args.dry_cache) as cd:
        Xr = cd['X']
    Mn, Mr = Xd[:, I_M].reshape(n_pop, 15, NB), Xr[:, I_M].reshape(n_pop, 15, NB)
    check.update(dry_cache=args.dry_cache,
                 vs_dry_cache_N_max_abs_diff=float(np.abs(Xd[:, I_N] - Xr[:, I_N]).max()),
                 vs_dry_cache_M_nonwater_max_abs_diff=float(np.abs(Mn[:, DRY] - Mr[:, DRY]).max()),
                 dry_cache_water_row_max=float(np.abs(Mr[:, I_H2O]).max()),
                 vs_dry_cache_other_blocks_equal=bool(np.array_equal(Xd[:, 240:], Xr[:, 240:])),
                 water_fraction_of_mass_in_bins_median=float(np.median(Mn[:, I_H2O].sum(-1) / np.maximum(Mn.sum((1, 2)), 1e-300))),
                 water_fraction_of_mass_in_bins_p90=float(np.percentile(Mn[:, I_H2O].sum(-1) / np.maximum(Mn.sum((1, 2)), 1e-300), 90)),
                 pops_with_water=int((Mn[:, I_H2O].sum(-1) > 0).sum()))
    del Xr
print(json.dumps(check, indent=1), flush=True)
if check['wet_M_rel_dev_p99'] > 1e-4 or check['wet_N_rel_dev_p99'] > 1e-4:
    raise SystemExit('the recomputed wet blocks do not match the cache: not writing %s' % args.out)
if args.basis == 'drydiam' and (check['vs_dry_cache_N_max_abs_diff'] > 0 or check['vs_dry_cache_M_nonwater_max_abs_diff'] > 0
                                or check['dry_cache_water_row_max'] > 0 or not check['vs_dry_cache_other_blocks_equal']):
    raise SystemExit('the dry-diameter blocks differ from %s: not writing %s' % (args.dry_cache, args.out))
np.savez(args.out, X=Xd, ntot=ntot, m=c['m'], i_snr=i_snr)
with open(args.out[:-4] + '_check.json', 'w') as fo:
    json.dump(dict(check, cache=args.cache, data=args.data, out=args.out), fo, indent=1)
print('wrote %s (%.0f s)' % (args.out, time.time() - t0), flush=True)
