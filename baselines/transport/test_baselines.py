#!/usr/bin/env python
"""Checks of run_baselines.py on a tiny synthetic case (numpy, about a minute).

Writes WRF-like inputs (a 1-minute flux file, a 15-minute "original" wrfout, the
WRF-PartMC IC/BC/emission control files) and uses the real source states
(baselines/data/zsrc_baselines.npz: sec8 128, mam4 64, mam4l 25 components) and the real
calcsize (baselines/modal.py), then checks:

  1. ChunkedRunner == schemes.Runner, bitwise, on transport/test_synthetic's domain;
  2. run_baselines.py reproduces the paper's run.py (N, oracle, sep:sec8) bitwise on the
     same inputs (run.py imported unchanged, its PMC path pointed at the test inputs);
  3. calcsize touches only its models: the _nocs copies equal a run without calcsize
     bitwise, whatever the chunking;
  4. the run with calcsize: finite, nonnegative, the extracts' times;
  5. check_run.py's checks pass on the output (species conserved by calcsize, modes moved
     toward their bounds, number totals, oracle and N equal to run.py's).

Usage: python test_baselines.py [--dir TMPDIR] [--steps 270]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent
TRANSPORT = HERE.parents[1] / 'transport'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRANSPORT))

import run_baselines as RB  # noqa: E402

ZSRC = HERE.parent / 'data' / 'zsrc_baselines.npz'
LAYOUT = HERE.parent / 'data' / 'state_layout.json'


def write_inputs(d, nz=6, ny=14, nx=24, steps=270, seed=0):
    rng = np.random.default_rng(seed)
    d = Path(d)
    (d / 'pmc').mkdir(parents=True, exist_ok=True)
    nfr = steps // 3 + 2
    mu = 9e4

    def times(ds, n, dt_s):
        ds.createDimension('Time', n)
        ds.createDimension('DateStrLen', 19)
        v = ds.createVariable('Times', 'S1', ('Time', 'DateStrLen'))
        for i in range(n):
            s = i * dt_s
            txt = '2010-06-17_%02d:%02d:%02d' % (s // 3600, (s % 3600) // 60, s % 60)
            v[i] = np.array(list(txt), 'S1')

    # 1-minute fluxes
    fx = netCDF4.Dataset(d / 'fluxes.nc', 'w')
    times(fx, nfr, 60)
    for name, n in (('bottom_top', nz), ('bottom_top_stag', nz + 1), ('south_north', ny),
                    ('south_north_stag', ny + 1), ('west_east', nx), ('west_east_stag', nx + 1)):
        fx.createDimension(name, n)
    fx.DX, fx.DY = 4000.0, 4000.0
    one = lambda name, dims, val: fx.createVariable(name, 'f4', dims).__setitem__(slice(None), val)
    one('MAPFAC_MX', ('Time', 'south_north', 'west_east'), np.ones((nfr, ny, nx)))
    one('MAPFAC_MY', ('Time', 'south_north', 'west_east'), np.ones((nfr, ny, nx)))
    one('RDNW', ('Time', 'bottom_top'), np.full((nfr, nz), -1 / 0.02))
    one('FNM', ('Time', 'bottom_top'), np.full((nfr, nz), 0.5))
    one('FNP', ('Time', 'bottom_top'), np.full((nfr, nz), 0.5))
    one('C1H', ('Time', 'bottom_top'), np.ones((nfr, nz)))
    one('C2H', ('Time', 'bottom_top'), np.zeros((nfr, nz)))
    one('MU', ('Time', 'south_north', 'west_east'), np.full((nfr, ny, nx), mu * 0.1))
    one('MUB', ('Time', 'south_north', 'west_east'), np.full((nfr, ny, nx), mu * 0.9))
    tt = np.arange(nfr)[:, None, None, None]
    ru = mu * (30.0 + 10 * np.sin(2 * np.pi * tt / 40)) * np.ones((1, nz, ny, nx + 1))
    rv = mu * 6.0 * np.cos(2 * np.pi * tt / 30) * np.ones((1, nz, ny + 1, nx))
    ww = np.zeros((nfr, nz + 1, ny, nx))
    ww[:, 1:-1] = (0.3 * np.sin(np.linspace(0, np.pi, nz - 1))[None, :, None, None]
                   * np.sin(np.linspace(0, 6, nx))[None, None, None, :] * np.cos(2 * np.pi * tt / 50))
    fx.createVariable('AVGFLX_RUM', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east_stag'))[:] = ru
    fx.createVariable('AVGFLX_RVM', 'f4', ('Time', 'bottom_top', 'south_north_stag', 'west_east'))[:] = rv
    fx.createVariable('AVGFLX_WWM', 'f4', ('Time', 'bottom_top_stag', 'south_north', 'west_east'))[:] = ww
    fx.close()

    # 15-minute "original" output (emission factor)
    no = steps // 45 + 2
    o = netCDF4.Dataset(d / 'orig.nc', 'w')
    times(o, no, 900)
    for name, n in (('bottom_top', nz), ('bottom_top_stag', nz + 1), ('south_north', ny), ('west_east', nx)):
        o.createDimension(name, n)
    m3 = ('Time', 'bottom_top', 'south_north', 'west_east')
    s3 = ('Time', 'bottom_top_stag', 'south_north', 'west_east')
    o.createVariable('T', 'f4', m3)[:] = rng.uniform(-2, 2, (no, nz, ny, nx))
    o.createVariable('QVAPOR', 'f4', m3)[:] = rng.uniform(0.005, 0.01, (no, nz, ny, nx))
    o.createVariable('P', 'f4', m3)[:] = rng.uniform(-500, 500, (no, nz, ny, nx))
    o.createVariable('PB', 'f4', m3)[:] = np.full((no, nz, ny, nx), 1e5)
    ph = np.cumsum(np.full((no, nz + 1, ny, nx), 9.81 * 50.0), axis=1) - 9.81 * 50.0
    o.createVariable('PH', 'f4', s3)[:] = rng.uniform(-5, 5, (no, nz + 1, ny, nx))
    o.createVariable('PHB', 'f4', s3)[:] = ph
    o.close()

    # WRF-PartMC control files
    ic = netCDF4.Dataset(d / 'pmc' / 'ic_control.nc', 'w')
    for name, n in (('n_source', 2), ('nx', nx), ('ny', ny), ('nz', nz)):
        ic.createDimension(name, n)
    c = np.zeros((2, nx, ny, nz))
    c[0] = 1e9
    c[0, 6:12, 3:9] = 0.0
    c[1, 6:12, 3:9] = 1.3e9
    ic.createVariable('num_conc', 'f8', ('n_source', 'nx', 'ny', 'nz'))[:] = c
    ic.close()
    bc = netCDF4.Dataset(d / 'pmc' / 'bc_control.nc', 'w')
    for name, n in (('n_source', 1), ('n_time', 1), ('nx', nx), ('ny', ny), ('nz', nz)):
        bc.createDimension(name, n)
    for name, along in (('num_conc_xs', 'ny'), ('num_conc_xe', 'ny'), ('num_conc_ys', 'nx'), ('num_conc_ye', 'nx')):
        n_al = ny if along == 'ny' else nx
        bc.createVariable(name, 'f8', ('n_source', 'n_time', along, 'nz'))[:] = \
            rng.uniform(0.8e9, 1.2e9, (1, 1, n_al, nz))
    bc.close()
    em = netCDF4.Dataset(d / 'pmc' / 'emis_control.nc', 'w')
    nt_em = steps * 20 // 3600 + 2
    for name, n in (('n_source', 2), ('n_time', nt_em), ('nx', nx), ('ny', ny), ('nk', 1)):
        em.createDimension(name, n)
    em.createVariable('emis_time', 'f8', ('n_time',))[:] = np.arange(nt_em) * 3600.0
    fl = np.zeros((2, nt_em, nx, ny, 1))
    fl[0, :, 10, 7, 0] = 2e7 * (1 + np.arange(nt_em))
    fl[1, :, 15, 3, 0] = 1e7
    em.createVariable('num_flux', 'f8', ('n_source', 'n_time', 'nx', 'ny', 'nk'))[:] = fl
    em.close()

    return d


def runner_equivalence():
    import schemes
    from test_synthetic import make
    R, S, W = RB.import_paper(TRANSPORT)
    g, F, M0 = make()
    nz, ny, nx = g.nz, g.ny, g.nx
    rng = np.random.default_rng(1)
    Z = {'a': rng.uniform(0.1, 2, (3, 13)), 'b': rng.uniform(0.1, 2, (3, 6))}
    ins = np.zeros((5, nz, ny, nx))
    ins[0] = 1e9
    ins[0, :, 4:10, 15:28] = 0
    ins[1, :, 4:10, 15:28] = 1.3e9
    bce = {'w': np.full((nz, ny), 1e9), 'e': np.full((nz, ny), 1.1e9), 's': np.full((nz, nx), 0.9e9),
           'n': np.full((nz, nx), 1e9)}
    C = RB.make_chunked_runner(S, W)
    runs = [schemes.Runner(g, Z, np, ins, bce, schemes=('oracle', 'sep'), sep_models=['a', 'b'])]
    runs += [C(g, Z, np, ins, bce, schemes=('oracle', 'sep'), sep_models=['a', 'b'], shB_models=[], chunk=c)
             for c in (1, 5, 1000)]
    Ms = [M0] * len(runs)
    for it in range(80):
        dn = np.zeros((2, ny, nx))
        dn[0, 7, 30] = 2e7
        dn[1, 3, 40] = 1e7
        for i, r in enumerate(runs):
            Ms[i] = r.step(F, Ms[i], 20.0)
            r.emit(dn)
    return all(np.array_equal(runs[0].q_mono, r.q_mono) for r in runs[1:]) and \
        all(np.array_equal(Ms[0], m) for m in Ms[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None)
    ap.add_argument('--steps', type=int, default=270)
    a = ap.parse_args()
    d = Path(a.dir or tempfile.mkdtemp(prefix='test_baselines_'))
    steps = a.steps
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print('%-70s %s %s' % (name, 'ok ' if cond else 'FAIL', detail), flush=True)
        ok &= bool(cond)

    check('1. ChunkedRunner == schemes.Runner (chunks 1, 5, all; bitwise)', runner_equivalence())

    write_inputs(d, steps=steps)
    common = ['--steps', str(steps), '--zsrc', str(ZSRC), '--fluxes', str(d / 'fluxes.nc'),
              '--orig', str(d / 'orig.nc')]
    # the paper's run.py, unchanged, its PMC pointed at the test inputs
    R, S, W = RB.import_paper(TRANSPORT)
    R.PMC = d / 'pmc'
    argv0 = sys.argv
    sys.argv = ['run.py', '--out', str(d / 'paper'), '--models', 'sec8', '--schemes', 'oracle,sep'] + common
    R.main()
    sys.argv = argv0
    base = common + ['--pmc', str(d / 'pmc'), '--transport-code', str(TRANSPORT)]
    RB.main(['--out', str(d / 'cs'), '--models', 'sec8,mam4,mam4l', '--nocs', 'mam4,mam4l',
             '--calcsize', 'mam4,mam4l', '--chunk', '7'] + base)
    RB.main(['--out', str(d / 'nocs'), '--models', 'mam4,mam4l', '--chunk', '1000'] + base)

    ld = lambda p: dict(np.load(p, allow_pickle=False))
    P, B, C = ld(d / 'paper' / 'surface.npz'), ld(d / 'cs' / 'surface.npz'), ld(d / 'nocs' / 'surface.npz')
    for k in ('time_s', 'N', 'oracle', 'sep:sec8', 'M_offline_over_wrf'):
        check('2. %s identical to the paper\'s run.py' % k, np.array_equal(P[k], B[k]),
              'max |diff| %.1e' % np.abs(P[k].astype(float) - B[k]).max())
    for m in ('mam4', 'mam4l'):
        check('3. sep:%s_nocs (calcsize run, chunk 7) == sep:%s (no calcsize)' % (m, m),
              np.array_equal(B['sep:%s_nocs' % m], C['sep:' + m]))
        check('3. calcsize changed sep:%s' % m, not np.array_equal(B['sep:' + m], C['sep:' + m]),
              'max rel change %.2f' % (np.abs(B['sep:' + m] - C['sep:' + m]).max() / np.abs(C['sep:' + m]).max()))

    st = json.loads(str(B['settings']))
    check('4. settings: calcsize every step, %d calls' % steps, st['calcsize_calls'] == steps)
    for k in [k for k in B if k.startswith('sep:') or k in ('N', 'oracle')]:
        v = B[k]
        check('4. %s finite and nonnegative (to 1e-10 of its max)' % k,
              np.isfinite(v).all() and v.min() >= -1e-10 * np.abs(v).max(), 'min %.2e' % v.min())
    D = ld(d / 'cs' / 'diag.npz')
    check('4. diag: whole 3-D state finite', D['n_nonfinite'].sum() == 0)
    rel = (D['min'] / np.maximum(np.abs(D['max']), 1e-300)).min()
    check('4. diag: whole 3-D state nonnegative (to 1e-10 of each max)', rel >= -1e-10, 'min/max %.1e' % rel)
    E, H = ld(d / 'cs' / 'surface_end.npz'), ld(d / 'cs' / 'surface_30min.npz')
    check('4. surface_30min: every 30 min', np.allclose(H['time_s'], np.arange(0, steps * 20 + 1, 1800)),
          str(H['time_s']))
    check('4. surface_end: the 0/4/12 h outputs present', set(E['time_s']) == {t for t in (0., 14400., 43200.)
                                                                              if t <= steps * 20})
    check('4. extracts = surface.npz at their times', all(
        np.array_equal(H[k], B[k][np.isin(B['time_s'], H['time_s'])]) for k in H if k not in ('settings', 'time_s')))

    import check_run
    res = check_run.check(d / 'cs', reference=d / 'paper' / 'surface.npz', layout=LAYOUT, zsrc=ZSRC,
                          modal_path=HERE.parent)
    print('\n'.join('    ' + m for m in res['messages']))
    check('5. check_run.py passes', res['ok'], '')
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    print('test files in', d)
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
