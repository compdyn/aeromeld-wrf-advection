#!/usr/bin/env python
"""baselines/data/zsrc_baselines.npz and baselines/data/state_layout.json: the three source
populations' per-particle state vectors in each baseline representation, for the transport
replay, and the layout of every representation's state.

zsrc_baselines.npz (the format of data/wrfml_runs/zsrc_of/zsrc_s<k>.npz, which transport/run.py
reads):
  pops            (3,) int, copied from zsrc_of/zsrc_s1.npz (pops // 25 = 561, 635, 812)
  <model>_z       (3, d) per-particle state x_s (fractions, kg per particle), float64
  <model>_xsrc    (3, 456) q_R(x_s), the raw per-particle diagnostics of each source
  pmc_xsrc        (3, 456) the particle-resolved sources' (baselines/diag.py)
  mam4_raw_z, mam4l_raw_z   the modal projections before the one calcsize call
models: sec4, sec8, sec20 (sectional), mam4, mam4l (modal, after calcsize once with dt = 20 s).

Usage: python baselines/make_sources.py              (all of common.MODELS; rewrites the files)
       python baselines/make_sources.py --add sec20w,mam4a   (adds keys, keeps the rest)
       AEROMELD_SIZE_BASIS=drydiam python baselines/make_sources.py --basis-file sec8,tomas40,...
           (zsrc_baselines_drydiam.npz: the same states, the source diagnostics in that basis; 2026-11-19)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import diag  # noqa: E402

OUT = HERE / 'data'
ZSRC_REF = diag.REPO / 'data' / 'wrfml_runs' / 'zsrc_of' / 'zsrc_s1.npz'


def add(models):
    """Add models' keys to the existing zsrc_baselines.npz and state_layout.json, leaving every
    existing key and layout entry unchanged (the transport runs use them)."""
    m, w = diag.sources()
    with np.load(OUT / 'zsrc_baselines.npz') as f:
        old = {k: f[k].copy() for k in f.files}
    with open(OUT / 'state_layout.json') as f:
        layout = json.load(f)
    out = dict(old)
    for name in models:
        R = common.rep(name)
        x = R.project(m, w)
        new = {name + '_z': x, name + '_xsrc': common.particles_vectors(*R.particles(x), procs=1)}
        if name.startswith('mam'):
            new[name + '_raw_z'] = R.project(m, w, calcsize_dt=0)
        if name.startswith('tomas'):                       # before the one mnfix (a no-op here)
            new[name + '_raw_z'] = R.project(m, w, mnfix_once=False)
        if name.startswith('matrix'):                      # before MATRIX's adjustment once (a no-op here)
            new[name + '_raw_z'] = R.project(m, w, calcsize_dt=0)
        for k, v in new.items():
            if k in old:
                raise ValueError('%s exists already; not overwriting' % k)
            out[k] = v
        L = R.layout()
        L['number_indices'] = [c['index'] for c in L['components'] if c['kind'] == 'number']
        nz = x[x > 0]
        L['source_x_range_nonzero'] = [float(nz.min()), float(nz.max())]
        if name == 'mam4a':
            L['calcsize'] = ('as mam4 (identical layout): baselines/modal.py calcsize(Q, dt, "mam4" or '
                             '"mam4a", xp)')
        if name == 'mam4la':
            L['calcsize'] = ('as mam4l (identical layout): baselines/modal.py calcsize(Q, dt, "mam4l" or '
                             '"mam4la", xp)')
        layout['models'][name] = L
        print('%-6s d = %3d   nonzero x in [%.2e, %.2e]' % (name, R.d, nz.min(), nz.max()))
    for k in old:
        assert np.array_equal(out[k], old[k])
    np.savez(OUT / 'zsrc_baselines.npz', **out)
    with open(OUT / 'state_layout.json', 'w') as f:
        json.dump(layout, f, indent=1)
    print('added', models, 'to', OUT / 'zsrc_baselines.npz', OUT / 'state_layout.json')


def basis_file(models):
    """zsrc_baselines_<basis>.npz for the size basis of the environment (diag.SIZE_BASIS, e.g. drydiam,
    2026-11-19): every model's source states (<model>_z, _raw_z) projected afresh, which must equal
    zsrc_baselines.npz's (the states do not depend on how the diagnostics bin), and the source
    diagnostics <model>_xsrc and pmc_xsrc, which do, computed in that basis."""
    assert diag.SIZE_BASIS != 'wet', 'basis_file is for a basis other than wet (AEROMELD_SIZE_BASIS)'
    m, w = diag.sources()
    with np.load(OUT / 'zsrc_baselines.npz') as f:
        old = {k: f[k].copy() for k in f.files}
    out = {'pops': old['pops'], 'pmc_xsrc': diag.raw_vectors(m, w)}
    for name in models:
        R = common.rep(name)
        x = R.project(m, w)
        out[name + '_z'] = x
        out[name + '_xsrc'] = common.particles_vectors(*R.particles(x), procs=1)
        if name.startswith('mam'):
            out[name + '_raw_z'] = R.project(m, w, calcsize_dt=0)
        if name.startswith('tomas'):
            out[name + '_raw_z'] = R.project(m, w, mnfix_once=False)
        if name.startswith('matrix'):
            out[name + '_raw_z'] = R.project(m, w, calcsize_dt=0)
        for k in (name + '_z', name + '_raw_z'):
            if k in out and k in old and not np.array_equal(out[k], old[k]):
                raise ValueError('%s differs from zsrc_baselines.npz (max |diff| %.2e): the states should not '
                                 'depend on the basis' % (k, np.abs(out[k] - old[k]).max()))
        dx = (np.abs(out[name + '_xsrc'] - old[name + '_xsrc']).max() / np.abs(old[name + '_xsrc']).max()
              if name + '_xsrc' in old else float('nan'))
        print('%-7s d = %3d   states equal to zsrc_baselines.npz; source diagnostics vs the wet ones: '
              'max rel. change %.2e' % (name, R.d, dx), flush=True)
    path = OUT / ('zsrc_baselines_%s.npz' % diag.SIZE_BASIS)
    np.savez(path, **out)
    print('wrote', path)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--add':
        return add(sys.argv[2].split(','))
    if len(sys.argv) > 2 and sys.argv[1] == '--basis-file':
        return basis_file(sys.argv[2].split(','))
    OUT.mkdir(exist_ok=True)
    m, w = diag.sources()
    with np.load(ZSRC_REF) as f:
        pops = f['pops'].copy()
    assert list(pops // 25) == [561, 635, 812]
    out = {'pops': pops, 'pmc_xsrc': diag.raw_vectors(m, w)}
    layout = {'about': 'Per-particle state vectors x of each baseline representation. A cell carries '
                       'Q = N x (N: number per kg of air), so Q[number components] is number per kg and '
                       'Q[mass components] kg per kg; mixing averages x in number. Sources: '
                       'zsrc_baselines.npz <model>_z (3, n_tracers), rows 561 (background), 635 (urban), '
                       '812 (emission).',
              'species_order': list(diag.SPECIES),
              'models': {}}
    for name in common.MODELS:
        R = common.rep(name)
        x = R.project(m, w)
        out[name + '_z'] = x
        out[name + '_xsrc'] = common.particles_vectors(*R.particles(x), procs=1)
        if name.startswith('mam'):
            out[name + '_raw_z'] = R.project(m, w, calcsize_dt=0)
        L = R.layout()
        L['number_indices'] = [c['index'] for c in L['components'] if c['kind'] == 'number']
        nz = x[x > 0]
        L['source_x_range_nonzero'] = [float(nz.min()), float(nz.max())]
        layout['models'][name] = L
        print('%-6s d = %3d   nonzero x in [%.2e, %.2e]' % (name, R.d, nz.min(), nz.max()))
    np.savez(OUT / 'zsrc_baselines.npz', **out)
    with open(OUT / 'state_layout.json', 'w') as f:
        json.dump(layout, f, indent=1)
    print('wrote', OUT / 'zsrc_baselines.npz', OUT / 'state_layout.json')


if __name__ == '__main__':
    main()
