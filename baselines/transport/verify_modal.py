#!/usr/bin/env python
"""Reproduce the local calcsize test outputs (make_diag_shim.py) with this machine's modal.py and
diag.py, on numpy and (with --gpu) cupy. Prints the largest relative difference per case.

Usage: python verify_modal.py MODAL_DIR modal_test.npz [--gpu]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modal_dir')
    ap.add_argument('test')
    ap.add_argument('--gpu', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.modal_dir).resolve()))
    import modal
    print('modal', modal.__file__, 'diag', modal.diag.__file__)
    mods = [('numpy', np)]
    if a.gpu:
        import cupy
        mods.append(('cupy', cupy))
    worst = 0.0
    with np.load(a.test) as f:
        for m in [k[:-2] for k in f.files if k.endswith('_Q')]:
            Q = f[m + '_Q']
            for dt in (20.0, 86400.0):
                ref = f['%s_out_%g' % (m, dt)]
                for name, xp in mods:
                    out = modal.calcsize(xp.asarray(Q), dt, model=m, xp=xp)
                    out = out.get() if xp is not np else out
                    rel = float(np.abs(out - ref).max() / np.abs(ref).max())
                    worst = max(worst, rel)
                    print('%-6s dt %-7g %-5s max |diff| / max %.2e' % (m, dt, name, rel))
                if m == 'mam4a':                              # same calcsize as mam4
                    same = np.array_equal(modal.calcsize(Q, dt, model='mam4a', xp=np),
                                          modal.calcsize(Q, dt, model='mam4', xp=np))
                    worst = max(worst, 0.0 if same else 1.0)
                    print('mam4a dt %g: calcsize(mam4a) == calcsize(mam4) bitwise: %s' % (dt, same))
    print('OK' if worst < 1e-12 else 'MISMATCH', 'worst %.2e' % worst)
    return worst < 1e-12


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
