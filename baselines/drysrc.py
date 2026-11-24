#!/usr/bin/env python
"""drysrc: the three source populations without their water (2026-11-19, option 2).

The emission population is 74% water by mass (background 2%, urban 0.1%), and of the six
diagnostics only the optics see it. Option 2 drops the water from the three sources (every
particle's H2O mass set to 0) and keeps the trained decoders: the training library holds nearly
dry populations like the background and urban sources.

  --encode SEEDS   the five round-37 decoders' source latents from the dry samples
                   (data/affx/wrf3_jeff.npz 'm', the equal-weight samples the WRF runs' latents
                   come from, with H2O = 0), encoded here with each model's checkpoint
                   (data/models/<model>_dry.<seed>.pt). The same encoding of the wet samples is
                   first checked against data/wrfml_runs/zsrc_dry/zsrc_s<seed>.npz.
                   Writes data/wrfml_runs/zsrc_drysrc/zsrc_s<seed>.npz (pops and <model>_z; what
                   transport/run.py reads).
  --schemes        the schemes' source states from the dry sources (AEROMELD_DRY_SOURCES=1 is set
                   here): baselines/data/zsrc_baselines_drysrc.npz, the keys of zsrc_baselines.npz
                   for MODELS (and pmc_xsrc, the dry reference). The state layouts are unchanged:
                   the water components are 0. Prints how far each projection's other components
                   are from the wet one's.

  --merge          baselines/data/transport_terms_drysrc.json (read by paperfigs/comparison.py with
                   PAPERFIGS_DRYSRC=1), in transport_terms_dry.json's layout, from the per-case
                   scores transport_terms_<case>_drysrc.json (transport_terms.py with
                   AEROMELD_DRY_SOURCES=1 AEROMELD_SIZE_BASIS=dry BASELINES_ZSRC=zsrc_baselines_drysrc.npz
                   TT_TAG=_drysrc; WRF-AeroMELD also STATIC_RUN=drysrc<seed>_12h, the replay of
                   transport/rep12_array.sbatch with REP_PREFIX=drysrc ZSRC_DIR=zsrc_drysrc).
                   With dry sources the size basis does not matter: the wet diameter is the dry one.

Usage: python baselines/drysrc.py --encode 0 ; python baselines/drysrc.py --schemes ; ... --merge
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DECODERS = ('twin_k16', 'twin_k32', 'twin_pairs', 'twin_none', 'ax_rel1')
MODELS = ('sec8', 'tomas40', 'mam4l', 'matrix1')
SAMPLES = REPO / 'data' / 'affx' / 'wrf3_jeff.npz'
RUNS = REPO / 'data' / 'wrfml_runs'


def encoder(path, args):
    import torch
    import torch.nn.functional as F
    sd = torch.load(path, map_location='cpu', weights_only=False)
    W = [(sd['mlp.%d.weight' % i].double(), sd['mlp.%d.bias' % i].double()) for i in (0, 2, 4)]
    assert args.get('inputs') == 'sqrt' and args.get('pos') and not args.get('box') and not args.get('simplex')
    d_z = args['L'] - 1

    def enc(m):
        """(3, n, 15) kg, equal-weight particles -> (3, d_z): the particle mean of softplus(mlp(u))."""
        h = (torch.sqrt(torch.as_tensor(m, dtype=torch.float64)) - sd['u_shift'].double()) / sd['u_scale'].double()
        for k, (w, b) in enumerate(W):
            h = h @ w.T + b
            if k < 2:
                h = torch.relu(h)
        return F.softplus(h[..., :d_z]).mean(-2).numpy()
    return enc


def encode(seeds):
    sys.path.insert(0, str(HERE))
    import diag
    with np.load(SAMPLES) as f:
        m = f['m'].copy()
    dry = m.copy()
    dry[..., diag.I_H2O] = 0.0
    for si in seeds:
        with np.load(RUNS / 'zsrc_dry' / ('zsrc_s%d.npz' % si)) as f:
            ref = {k: f[k] for k in f.files}
        out = {'pops': ref['pops']}
        for n in DECODERS:
            key = '%s_dry.%d' % (n, si)
            args = json.loads((REPO / 'data' / 'models' / (key + '.json')).read_text())['args']
            enc = encoder(REPO / 'data' / 'models' / (key + '.pt'), args)
            dev = np.abs(enc(m) - ref[n + '_z']).max() / np.abs(ref[n + '_z']).max()
            if dev > 1e-9:
                raise SystemExit('%s: the wet encoding differs from zsrc_dry by %.1e' % (key, dev))
            out[n + '_z'] = enc(dry)
            print('%-16s wet check %.1e   dry - wet latents: max %.3f (scale %.3f)' % (
                key, dev, np.abs(out[n + '_z'] - ref[n + '_z']).max(), np.abs(ref[n + '_z']).max()))
        (RUNS / 'zsrc_drysrc').mkdir(exist_ok=True)
        np.savez(RUNS / 'zsrc_drysrc' / ('zsrc_s%d.npz' % si), **out)
        print('wrote', RUNS / 'zsrc_drysrc' / ('zsrc_s%d.npz' % si))


def schemes():
    os.environ['AEROMELD_DRY_SOURCES'] = '1'
    sys.path.insert(0, str(HERE))
    import common
    import diag
    assert diag.DRY_SOURCES
    m, w = diag.sources()
    assert not m[..., diag.I_H2O].any()
    with np.load(HERE / 'data' / 'zsrc_baselines.npz') as f:
        wet = {k: f[k] for k in f.files}
    layout = json.loads((HERE / 'data' / 'state_layout.json').read_text())['models']
    out = {'pops': wet['pops'], 'pmc_xsrc': diag.raw_vectors(m, w)}
    for name in MODELS:
        R = common.rep(name)
        x = R.project(m, w)
        out[name + '_z'] = x
        out[name + '_xsrc'] = common.particles_vectors(*R.particles(x), procs=1)
        if name.startswith('mam') or name.startswith('matrix'):
            out[name + '_raw_z'] = R.project(m, w, calcsize_dt=0)
        if name.startswith('tomas'):
            out[name + '_raw_z'] = R.project(m, w, mnfix_once=False)
        comps = layout[name]['components']
        water = np.array([c.get('species') in ('H2O', 'h2o', 'AW') for c in comps])
        assert x.shape == wet[name + '_z'].shape and len(comps) == x.shape[1]
        other = ~water
        dev = np.abs(x[:, other] - wet[name + '_z'][:, other]) / np.maximum(np.abs(wet[name + '_z'][:, other]), 1e-300)
        print('%-8s d = %3d (%d water components, now max %.1e)   other components: max relative change %.1e' % (
            name, x.shape[1], water.sum(), np.abs(x[:, water]).max() if water.any() else 0.0,
            dev[np.abs(wet[name + '_z'][:, other]) > 0].max()))
    np.savez(HERE / 'data' / 'zsrc_baselines_drysrc.npz', **out)
    print('wrote', HERE / 'data' / 'zsrc_baselines_drysrc.npz')


def merge():
    six = ['n_prthst', 'm_chmprthst', 'ccn_cdf', 'logfrznfrac_tmp', 'logqs_pop', 'logqa_pop']
    out = {'note': 'the three source populations without their water (baselines/drysrc.py); every scheme '
                   'scored from them at 12 h; WRF-AeroMELD the round-37 dry decoder twin_k16_dry seed 0 '
                   'replayed from the dry sources\' latents (drysrc0_12h)', 'six': six, 'cases': {}}
    for case in ('aeromeld',) + MODELS:
        p = HERE / 'data' / ('transport_terms_%s_drysrc.json' % case)
        if not p.exists():
            print('missing', p.name)
            continue
        r = json.loads(p.read_text())
        for t, v in r['12h'].items():
            assert abs(v['overall'] - np.mean([v[k] for k in six])) < 1e-12, (case, t)
        out['cases'][case] = {k: r[k] for k in ('12h', 'mixed_12h', 'run', 'key', 'model') if k in r}
    path = HERE / 'data' / 'transport_terms_drysrc.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path, sorted(out['cases']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--encode', default=None, help='seeds, e.g. 0 or 0-9')
    ap.add_argument('--schemes', action='store_true')
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    if a.encode:
        lo, _, hi = a.encode.partition('-')
        encode(range(int(lo), int(hi or lo) + 1))
    if a.schemes:
        schemes()
    if a.merge:
        merge()
