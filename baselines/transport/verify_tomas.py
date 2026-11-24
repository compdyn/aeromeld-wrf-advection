#!/usr/bin/env python
"""tomas.mnfix on the GPU: reproduce locally computed outputs with numpy and cupy.

  python verify_tomas.py --make tomas_test.npz           (locally, with the real diag.py)
  python verify_tomas.py MODAL_DIR tomas_test.npz [--gpu] (on Perlmutter, at the start of a job)

The test inputs are 600 perturbed mixtures of the real tomas40 source states (zsrc_baselines.npz),
some pushed far out of their bins so that every branch of MNFIX acts, scaled to per-kg values.
Prints the largest relative difference and whether the event counts agree.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def make(out, zsrc):
    sys.path.insert(0, str(HERE.parent))
    import tomas
    rng = np.random.default_rng(7)
    with np.load(zsrc) as f:
        z = f['tomas40_z']
    nb = tomas.NBINS
    x = rng.dirichlet(np.ones(3) * 0.5, 600) @ z
    x = x * rng.lognormal(0, 0.1, x.shape)
    x[:60, :nb] *= rng.lognormal(0, 1.5, (60, nb))
    a = x[:, :9 * nb].reshape(len(x), 9, nb)
    for c, target in ((60, 1e3 * tomas.XK[-1]), (61, 0.95 * tomas.XK[-1]), (62, 0.5 * tomas.XK[0]),
                      (63, 1.05 * tomas.XK[0])):
        k = 20 if target > 1e-15 else 9
        a[c, 1:8, k] *= target / (a[c, 1:8, k].sum() / a[c, 0, k])
    Q = (x * rng.lognormal(np.log(1e9), 1.0, (len(x), 1))).T.copy()
    st = {}
    ref = tomas.mnfix(Q, stats=st)
    np.savez(out, Q=Q, ref=ref, **{'stats_' + k: v for k, v in st.items()})
    print('wrote', out, 'events', st)


def verify(modal_dir, test, gpu):
    sys.path.insert(0, str(Path(modal_dir).resolve()))
    import tomas
    print('tomas', tomas.__file__, 'diag', tomas.diag.__file__)
    f = dict(np.load(test))
    Q, ref = f['Q'], f['ref']
    want = {k[6:]: int(v) for k, v in f.items() if k.startswith('stats_')}
    mods = [('numpy', np)]
    if gpu:
        import cupy
        mods.append(('cupy', cupy))
    worst, same = 0.0, True
    for name, xp in mods:
        st = {}
        out = tomas.mnfix(xp.asarray(Q), xp=xp, stats=st)
        out = out.get() if xp is not np else out
        rel = float(np.abs(out - ref).max() / np.abs(ref).max())
        worst = max(worst, rel)
        same &= st == want
        print('%-5s max |diff| / max %.2e; events %s (%s)' % (name, rel, st, 'same' if st == want else 'DIFFERENT'))
    ok = worst < 1e-12 and same
    print('OK' if ok else 'MISMATCH', 'worst %.2e' % worst)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('args', nargs='*')
    ap.add_argument('--make', default=None)
    ap.add_argument('--zsrc', default=str(HERE.parent / 'data' / 'zsrc_baselines.npz'))
    ap.add_argument('--gpu', action='store_true')
    a = ap.parse_args()
    if a.make:
        return make(a.make, a.zsrc) or True
    return verify(a.args[0], a.args[1], a.gpu)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
