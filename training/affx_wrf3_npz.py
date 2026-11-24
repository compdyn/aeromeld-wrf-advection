#!/usr/bin/env python
"""
Inputs for affx_paperfigs.py (run on Perlmutter): for one seed of each
affx.py run, the latents of the three WRF source populations (our dataset's
samples of scenarios 561, 635, 812 at PartMC output index 13 = i_t 12), the
decoder A, b in physical per-particle units, and the samples' true
diagnostics. Latents are recomputed from the particles with the run's encoder
(checked against the run's saved latents) twice: from the whole sample (`_z`)
and without its particles above 2.5 um (`_z_nocoarse`; only Urban has one, a
5 um dust particle with 97.5% of the sample's mass). The original files, which the
WRF runs were started from, have no particle above 2.5 um.

--samples NPZ (from resample_wrf3.py): use those samples (particles and
diagnostics) instead of the dataset's; the encoder is still checked against
the run's saved latents on the dataset's samples.

Nonlinear decoders (affx.py --dec mlp) have no A, b: their decoder is applied
here in numpy (checked against the run's own decoded values), to the source
latents (`_xsrc`) and, with --lam (mixing weights lambda of a WRF-AeroMELD
run, from the paper's products: lam_dom (3, sn, we), lam_pts (3, point,
time)), to the mixed latents z = sum_k lambda_k z*_k of every cell
(`_xdom`: C, S, A, F only, columns 240-455; `_xpts`: all; per particle,
physical units). Affine models are decoded where they are used (A, b).

Usage: python affx_wrf3_npz.py OUT.npz KEY=RUN_PREFIX [KEY=RUN_PREFIX ...] [--seed 0] [--samples NPZ] [--lam NPZ]
"""
import argparse
import json
import os
import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('runs', nargs='+')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data', default='/pscratch/sd/m/mwest/aeromeld/data/04_bwchisamp_pm10.h5')
ap.add_argument('--cache', default='/pscratch/sd/m/mwest/aeromeld/affx/cache.npz')
ap.add_argument('--splits', default='/pscratch/sd/m/mwest/aeromeld/models/test_idx.npz')
ap.add_argument('--samples', default=None)
ap.add_argument('--lam', default=None)
args = ap.parse_args()

POPS = [14037, 15887, 20312]
c = np.load(args.cache)
with h5py.File(args.data, 'r') as f:
    m_ds = f['m_prtchm'][POPS]                                # (3, 1000, 15) kg
    rho = f['chem_rho'][:]
    wave_len = f['wave_len'][:]
if args.samples:
    smp = np.load(args.samples)
    m, X, ntot = smp['m'], smp['X'], smp['ntot']
    assert np.allclose(smp['X_ds'], c['X'][POPS], rtol=1e-6, atol=0), 'resample_wrf3 diagnostics differ from the dataset\'s'
else:
    m, X, ntot = m_ds, c['X'][POPS], c['ntot'][POPS]
D = (6 / np.pi * (m / rho).sum(-1)) ** (1 / 3)
keep = D <= 2.5e-6
print('particles above 2.5 um per source (dropped for _z_nocoarse):', (~keep).sum(1).tolist())
out = dict(X=X, ntot=ntot, pops=np.array(POPS), wave_len=wave_len)
sp = lambda v: np.logaddexp(0, v)
si = args.seed
for kv in args.runs:
    key, run = kv.split('=')
    st = torch.load(f'{run}_s{si}.pt', map_location='cpu')
    W = lambda n: st[n].double().numpy()
    nonlin = st.get('dec', 'affine') == 'mlp'
    if nonlin:
        A, b, pos = None, None, True
    elif 'A_eff' in st:
        A, b, pos = W('A_eff'), W('b_eff'), 'Araw' in st
    else:                                                     # round 1 runs
        A, b = (sp(W('Araw')), sp(W('braw'))) if 'Araw' in st else (W('A'), W('b'))
        pos = 'Araw' in st
    s = W('s')
    d = np.load(f'{run}_s{si}_Z.npy', mmap_mode='r').shape[1]

    def decode(z):
        """z (..., d) -> x (..., 456), physical per-particle units."""
        if not nonlin:
            return z @ (s[:, None] * A).T + s * b
        h = z
        layers = sorted({int(k.split('.')[1]) for k in st if k.startswith('dmlp.') and k.endswith('.weight')})
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
        for k, n_ in [('M', 225), ('N', 15), ('C', 100), ('S', 8), ('A', 8), ('F', 100)]:
            vk = v[..., o:o + n_]
            if st.get('mono', False) and k in ('C', 'F'):
                vk = vk @ W('G' + k).T + (W('cC') if k == 'C' else 0)
            parts.append(vk); o += n_
        return np.concatenate(parts, -1) * s
    if 'u_shift' in st:
        sh, sc = W('u_shift'), W('u_scale')
    else:                                                     # round 1: affx.py's input scaling, recomputed
        sp_ = np.load(args.splits)
        trn = np.setdiff1d(np.arange(len(c['ntot'])), sp_['idx'][si])
        with h5py.File(args.data, 'r') as f:
            mt = np.sqrt(f['m_prtchm'][trn[::10]].astype(np.float32).reshape(-1, 15)).astype(np.float64)
        sh, sc = mt.mean(0), np.maximum(mt.std(0, ddof=1), 1e-30)
    Zrun = np.load(f'{run}_s{si}_Z.npy')[POPS].astype(np.float64)
    def phi(mi):
        u = (np.sqrt(mi) - sh) / sc
        h = np.maximum(u @ W('mlp.0.weight').T + W('mlp.0.bias'), 0)
        h = np.maximum(h @ W('mlp.2.weight').T + W('mlp.2.bias'), 0)
        f_ = (h @ W('mlp.4.weight').T + W('mlp.4.bias'))[:, :d]
        return sp(f_) if pos else f_
    z_ds = np.stack([phi(m_ds[i]).mean(0) for i in range(3)])
    err = np.abs(z_ds - Zrun).max() / np.abs(Zrun).max()
    z_all = np.stack([phi(m[i]).mean(0) for i in range(3)])
    z_nc = np.stack([phi(m[i])[keep[i]].mean(0) for i in range(3)])
    assert err < 1e-4, f'{run}: latents differ from the run\'s by {err:.1e}'
    if nonlin:
        # the numpy decoder against the run's own decoded values (seed's test populations, a few)
        Xrun = np.load(f'{run}_s{si}_X.npy')
        Zr = np.load(f'{run}_s{si}_Z.npy').astype(np.float64)
        jj = np.arange(0, len(Zr), 997)[:20]
        ed = (np.abs(decode(Zr[jj]) - Xrun[jj]) / (np.abs(Xrun[jj]) + 1e-30 * np.abs(Xrun[jj]).max())).max()
        print(f'{key}: numpy decoder vs the run\'s decoded values: max rel. difference {ed:.1e}')
        assert ed < 1e-3, ed
    else:
        out[key + '_A'], out[key + '_b'] = s[:, None] * A, s * b
    out[key + '_z'], out[key + '_z_nocoarse'] = z_all, z_nc
    out[key + '_xsrc'] = decode(z_all)
    if args.lam and nonlin:                                   # affine ones are decoded where they are used
        lam = np.load(args.lam)
        # domain: only C, S, A, F (columns 240-455), the four spectra the domain products hold
        out[key + '_xdom'] = decode(np.einsum('kij,kd->ijd', lam['lam_dom'], z_all))[..., 240:].astype(np.float32)
        out[key + '_xpts'] = decode(np.einsum('kpt,kd->ptd', lam['lam_pts'], z_all)).astype(np.float32)
    x = out[key + '_xsrc']
    print(f'{key} ({os.path.basename(run)} s{si}): d = {d}, pos = {pos}, latents vs the run\'s {err:.1e}, '
          f'min decoded {x.min():.3g}; Urban z {np.round(z_all[1], 2).tolist()} -> without coarse '
          f'{np.round(z_nc[1], 2).tolist()}')
np.savez(args.out, **out)
print('wrote', args.out)
