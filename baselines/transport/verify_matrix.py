#!/usr/bin/env python
"""matrix.adjust on Perlmutter: reproduce locally computed outputs with numpy and cupy.

  python verify_matrix.py --make OUT.npz        (locally, with the real diag.py) writes test inputs: the
        real matrix1 sources' perturbed mixtures and test_matrix.py's hand-made sources mixed with them
        (so every operation acts), and adjust's outputs and event counts
  python verify_matrix.py MODAL_DIR OUT.npz [--gpu]   (on Perlmutter, with the constants-only diag.py)
        prints the largest relative difference per case and backend; exit 1 on a mismatch
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def make(out):
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE))
    import matrix
    import test_matrix
    with np.load(HERE.parent / 'data' / 'zsrc_baselines.npz') as f:
        z = f['matrix1_z']
    zh = test_matrix.stress_sources(Path(out).with_suffix('.stress.npz'))
    rng = np.random.default_rng(1)
    cases = {}
    for tag, zz in (('real', z), ('stress', np.vstack([zh, z]))):
        fr = rng.dirichlet(np.ones(len(zz)) * 0.5, 400)
        Q = (fr @ zz) * rng.lognormal(0, 0.05, (400, zz.shape[1])) * rng.lognormal(np.log(1e9), 1.0, (400, 1))
        cases[tag + '_Q'] = Q.T
        st = {}
        cases[tag + '_out'] = matrix.adjust(Q.T, 20.0, 'matrix1', xp=np, stats=st)
        for k, v in st.items():
            cases[tag + '_n_' + k.replace('->', '_to_')] = np.array(int(v))
        print(tag, {k: int(v) for k, v in st.items()})
    np.savez(out, **cases)
    print('wrote', out)


def verify(modal_dir, test, gpu):
    sys.path.insert(0, str(Path(modal_dir).resolve()))
    import matrix
    print('matrix', matrix.__file__, 'diag', matrix.diag.__file__)
    mods = [('numpy', np)]
    if gpu:
        import cupy
        mods.append(('cupy', cupy))
    worst = 0.0
    with np.load(test) as f:
        for tag in [k[:-2] for k in f.files if k.endswith('_Q')]:
            Q, ref = f[tag + '_Q'], f[tag + '_out']
            for name, xp in mods:
                st = {}
                out = matrix.adjust(xp.asarray(Q), 20.0, 'matrix1', xp=xp, stats=st)
                out = out.get() if xp is not np else out
                rel = float(np.abs(out - ref).max() / np.abs(ref).max())
                same = all(int(v) == int(f[tag + '_n_' + k.replace('->', '_to_')]) for k, v in st.items())
                worst = max(worst, rel, 0.0 if same else 1.0)
                print('%-7s %-5s max |diff| / max %.2e, event counts equal %s %s'
                      % (tag, name, rel, same, {k: int(v) for k, v in st.items()}))
    print('OK' if worst < 1e-12 else 'MISMATCH', 'worst %.2e' % worst)
    return worst < 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('args', nargs='+')
    ap.add_argument('--make', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    a = ap.parse_args()
    if a.make:
        return make(a.args[0]) or True
    return verify(a.args[0], a.args[1], a.gpu)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
