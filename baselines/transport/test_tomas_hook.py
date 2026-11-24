#!/usr/bin/env python
"""Checks of run_baselines.py's --mnfix hook (TOMAS-40) on test_baselines.py's tiny synthetic
case (numpy, about 20 s):

  1. without --mnfix nothing changes: sep:sec8 and the mam4 calcsize run equal a run of the same
     models made with --mnfix tomas40 added (bitwise), and sep:tomas40_nocs (--nocs copy) equals
     sep:tomas40 of a run without --mnfix (bitwise);
  2. with --mnfix: the 3-D state finite and nonnegative, every surface bin's mean TOMAS dry
     mass inside its bounds at every output, mnfix's events and domain-total changes recorded;
  3. check_run.py passes on the --mnfix run.

Usage: python test_tomas_hook.py [--dir TMPDIR] [--steps 90]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_baselines as RB  # noqa: E402
import test_baselines as TB  # noqa: E402
import tomas  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None)
    ap.add_argument('--steps', type=int, default=90)
    a = ap.parse_args()
    d = Path(a.dir or tempfile.mkdtemp(prefix='test_tomas_hook_'))
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print('%-74s %s %s' % (name, 'ok ' if cond else 'FAIL', detail), flush=True)
        ok &= bool(cond)

    TB.write_inputs(d, steps=a.steps)
    base = ['--steps', str(a.steps), '--zsrc', str(TB.ZSRC), '--fluxes', str(d / 'fluxes.nc'),
            '--orig', str(d / 'orig.nc'), '--pmc', str(d / 'pmc'), '--transport-code', str(TB.TRANSPORT)]
    RB.main(['--out', str(d / 'old'), '--models', 'sec8,mam4', '--nocs', 'mam4', '--calcsize', 'mam4',
             '--chunk', '40'] + base)
    RB.main(['--out', str(d / 'fix'), '--models', 'sec8,mam4,tomas40', '--nocs', 'mam4,tomas40',
             '--calcsize', 'mam4', '--mnfix', 'tomas40', '--chunk', '40'] + base)
    RB.main(['--out', str(d / 'nofix'), '--models', 'tomas40', '--chunk', '1000'] + base)
    ld = lambda p: dict(np.load(p, allow_pickle=False))
    O, F, N = ld(d / 'old' / 'surface.npz'), ld(d / 'fix' / 'surface.npz'), ld(d / 'nofix' / 'surface.npz')
    for k in ('N', 'oracle', 'sep:sec8', 'sep:mam4', 'sep:mam4_nocs'):
        check('1. %s unchanged by adding tomas40 with --mnfix (bitwise)' % k, np.array_equal(O[k], F[k]))
    check('1. sep:tomas40_nocs (mnfix run) == sep:tomas40 without --mnfix (bitwise)',
          np.array_equal(F['sep:tomas40_nocs'], N['sep:tomas40']))
    so, sf = json.loads(str(O['settings'])), json.loads(str(F['settings']))
    check('1. no mnfix keys in a run without --mnfix', 'mnfix' not in so)
    st = sf['mnfix_stats']['tomas40']
    print('    mnfix events (cell, bin) over %d calls: %s' % (sf['calcsize_calls'], st))
    D = ld(d / 'fix' / 'diag.npz')
    check('2. 3-D state finite', D['n_nonfinite'].sum() == 0)
    rel = (D['min'] / np.maximum(np.abs(D['max']), 1e-300)).min()
    check('2. 3-D state nonnegative (to 1e-10 of each max)', rel >= -1e-10, 'min/max %.1e' % rel)
    T = tomas.Tomas()
    for key in ('sep:tomas40', 'sep:tomas40_nocs'):
        q = F[key].astype(float)                                  # (t, d, y, x) float32 -> per particle
        x = np.moveaxis(q, 1, -1).reshape(-1, tomas.D)
        av = T.dry_mass_per_particle(x)
        lo, hi = tomas.XK[:-1], tomas.XK[1:]
        dry = x[:, :9 * tomas.NBINS].reshape(len(x), 9, -1)[:, tomas.DRY_J].sum(1)
        quirk = np.isclose(av, 1.2 * tomas.XK[0], rtol=1e-5)         # TOMAS's bottom case, see test_tomas
        out = ~np.isnan(av) & (dry > 0) & ~quirk & ((av < lo * (1 - 1e-5)) | (av > hi * (1 + 1e-5)))  # float32
        print('    %s: (cell, bin) means outside their bounds at the surface outputs: %d of %d'
              % (key, out.sum(), (~np.isnan(av)).sum()))
        if key == 'sep:tomas40':
            check('2. with mnfix every surface bin mean inside its bounds', out.sum() == 0)
    delta = D['mnfix_delta:tomas40'][-1]
    tot = D['total'][-1]
    names = list(D['tracer'])
    i0 = names.index('sep:tomas40:0')
    A = (delta[:9 * tomas.NBINS] / np.maximum(np.abs(tot[i0:i0 + 9 * tomas.NBINS]).reshape(9, -1).sum(1).repeat(
        tomas.NBINS), 1e-300)).reshape(9, tomas.NBINS)
    print('    mnfix cumulative domain change / domain total, per TOMAS species (summed over bins): %s'
          % dict(zip(tomas.TSPECIES, np.round(A.sum(1), 8))))
    import check_run
    res = check_run.check(d / 'fix', layout=TB.LAYOUT, zsrc=TB.ZSRC, modal_path=HERE.parent)
    print('\n'.join('    ' + m for m in res['messages']))
    check('3. check_run.py passes', res['ok'])
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    print('test files in', d)
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
