"""Shared helpers: the representations by name, and q_R(x) (representative particles through the
particle diagnostics) in parallel worker processes.

    from common import rep, evaluate
    R = rep('sec8')
    x = R.project(m, w)              # particles -> per-particle state
    q = evaluate(R, x)               # (P, 456) raw per-particle diagnostics
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag  # noqa: E402
import modal  # noqa: E402
import sectional  # noqa: E402

MODELS = ('sec4', 'sec8', 'sec20', 'mam4', 'mam4l')
#: Added 2026-11-14: each family's most favourable setup. sec20w = the coag paper's sectional
#: (wet diameter); mam4a = mam4 with fresh carbon above the pca dgnumhi sent to accumulation.
EXTRA = ('sec20w', 'mam4a')
#: Added 2026-11-14 (evening): TOMAS-40 (tomas.py) and MAM4's species with mam4a's projection.
EXTRA += ('tomas40', 'mam4la')
#: Added 2026-11-14 (evening): MATRIX mechanism 1 (matrix.py; GISS ModelE's aerosol microphysics).
EXTRA += ('matrix1',)
ALL = MODELS + EXTRA


def rep(name, **kw):
    if name.startswith('sec'):
        return sectional.Sectional(name, **kw)
    if name.startswith('tomas'):
        import tomas
        return tomas.Tomas(name, **kw)
    if name.startswith('matrix'):
        import matrix
        return matrix.Matrix(name, **kw)
    return modal.Modal(name, **kw)


def _init_worker():
    diag.set_threads(1)


def _work(args):
    m, w = args
    return diag.raw_vectors(m, w)


def _compact(m, w):
    """Drop particle slots with zero weight in every population of the batch, and put each row's
    live slots first (the diagnostics are independent of slot order)."""
    live = w > 0
    order = np.argsort(~live, axis=1, kind='stable')
    k = max(int(live.sum(1).max()), 1)
    idx = order[:, :k]
    return np.take_along_axis(m, idx[..., None], axis=1), np.take_along_axis(w, idx, axis=1)


_POOL = None


def pool(procs=None):
    global _POOL
    if _POOL is None:
        procs = procs or int(os.environ.get('BASELINES_PROCS', 12))
        _POOL = ProcessPoolExecutor(max_workers=procs, mp_context=mp.get_context('spawn'),
                                    initializer=_init_worker)
    return _POOL


def particles_vectors(m, w, chunk=50, procs=None):
    """raw_vectors of many populations (P, n, 15), (P, n) in worker processes."""
    m, w = _compact(np.asarray(m, float), np.asarray(w, float))
    jobs = [(m[a:a + chunk], w[a:a + chunk]) for a in range(0, len(m), chunk)]
    if len(jobs) == 1 or (procs == 1):
        diag.set_threads(1)
        return np.concatenate([_work(j) for j in jobs])
    return np.concatenate(list(pool(procs).map(_work, jobs)))


def evaluate(R, x, chunk=50, procs=None, block=3000):
    """q_R(x): raw per-particle 456-vectors (P, 456) of states x (P, d) of representation R."""
    x = np.atleast_2d(x)
    out = []
    for a in range(0, len(x), block):
        m, w = R.particles(x[a:a + block])
        out.append(particles_vectors(m, w, chunk=chunk, procs=procs))
    return np.concatenate(out)
