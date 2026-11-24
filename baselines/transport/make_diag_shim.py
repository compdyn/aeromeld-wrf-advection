#!/usr/bin/env python
"""For the Perlmutter copy of baselines/modal.py: a constants-only diag.py, and a calcsize test.

modal.py imports baselines/diag.py, which imports torch, the vendored PartMC diagnostics and
paperfigs. calcsize uses only diag's constants (SPECIES, NS, SP, RHO, KAPPA, DRY, ...). This script writes
every upper-case module-level constant of the real diag module as a literal in <outdir>/diag.py.
It also writes <outdir>/modal_test.npz: calcsize inputs (perturbed mixtures of the real source
states) and outputs computed locally with the real diag. verify_modal.py then reproduces those
outputs on Perlmutter (numpy and cupy).

Usage (from baselines/transport/): python make_diag_shim.py OUTDIR [--zsrc ../data/zsrc_baselines.npz]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import diag  # noqa: E402
import modal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--zsrc', default=str(HERE.parent / 'data' / 'zsrc_baselines.npz'))
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(diag.__file__)
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    lines = []
    for k, v in sorted(vars(diag).items()):                 # every module-level constant
        if k.startswith('_') or k.upper() != k:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in 'biuf' and v.size < 10000:
            lines.append('%s = np.array(%r, dtype=%r)' % (k, v.tolist(), str(v.dtype)))
        elif isinstance(v, (bool, int, float, str, tuple, list, dict)) and len(repr(v)) < 20000:
            if isinstance(v, float) or (isinstance(v, (tuple, list, dict)) and 'np.' in repr(v)):
                lines.append('%s = %s' % (k, repr(v).replace('np.float64', 'float')))
            else:
                lines.append('%s = %r' % (k, v))
    text = '''"""Constants-only stand-in for baselines/diag.py (md5 %s), written by
baselines/transport/make_diag_shim.py for the Perlmutter copy of modal.py: calcsize uses only
constants. diag.py's functions (the particle diagnostics) are deliberately absent."""
import numpy as np

%s
SOURCE_MD5 = %r
''' % (md5, '\n'.join(lines), md5)
    (out / 'diag.py').write_text(text)
    # calcsize test cases
    rng = np.random.default_rng(0)
    cases = {}
    with np.load(a.zsrc) as f:
        for m in [m for m in ('mam4', 'mam4l', 'mam4a') if m + '_z' in f.files]:
            z = f[m + '_z']
            fr = rng.dirichlet(np.ones(3) * 0.5, 300)                         # (P, 3)
            x = fr @ z
            x = x * rng.lognormal(0, 0.1, x.shape)                              # off the mixtures
            x[:50] = z[rng.integers(0, 3, 50)]                                  # pure sources
            Q = (x * rng.lognormal(np.log(1e9), 1.0, (len(x), 1))).T            # (d, P) per kg
            cases[m + '_Q'] = Q
            for dt in (20.0, 86400.0):
                cases['%s_out_%g' % (m, dt)] = modal.calcsize(Q, dt, model=m, xp=np)
    np.savez(out / 'modal_test.npz', **cases)
    print('wrote', out / 'diag.py', 'and', out / 'modal_test.npz')


if __name__ == '__main__':
    main()
