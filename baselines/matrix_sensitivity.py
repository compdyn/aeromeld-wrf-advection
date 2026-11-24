#!/usr/bin/env python
"""Static 12-h terms of MATRIX's other mechanisms and of one diagnosis choice, for notes_matrix.md only
(no transport runs; the paper shows mechanism 1).

Variants: matrix1..matrix8 (matrix.MECHANISMS; OCAR diagnosed as API1, the default), and matrix1 with OCAR
diagnosed as OC instead ('matrix1_oc'). Each source is projected afresh (matrix.Matrix(name).project, adjustment once), so
nothing is read from or written to zsrc_baselines.npz or the Q2 cache. The cells, fractions, numbers, the
reference and the error are static_terms.Case's (errors.decompose's), as for every model there:

  Q2 = q_R(N_PS, sum_s f_s x_s), Q1 = sum_s f_s q_R(x_s), Q0 = sum_s f_s q_PR,s at 12 h
  averaging e(Q2, Q1), reconstruction e(Q1, Q0), representation e(Q2, Q0); and each source's error.

Writes data/matrix_sensitivity.json (data/<basis>/ for AEROMELD_SIZE_BASIS other than wet). Perlmutter CPU (static_add.sbatch's layout):
  cd $S/repo_m/scripts && python ../baselines/matrix_sensitivity.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import diag  # noqa: E402
import matrix  # noqa: E402
import static_terms as ST  # noqa: E402
from paperfigs import config, errors  # noqa: E402

VARIANTS = ['matrix%d' % k for k in matrix.MECHANISMS] + ['matrix1_oc']


def rep(name):
    if name == 'matrix1_oc':
        return matrix.Matrix('matrix1', ocar_as='OC')
    return matrix.Matrix(name)


def main():
    case = ST.Case()
    m, w = diag.sources()
    f_O, n_O, mask, fm, nO = case.cells(12.0)
    n_src = np.array([config.POP_N_FULL[s] for s in config.POP_SCENARIOS])
    out = {'about': __doc__.split('\n\n')[0], 'mixed_12h': int(mask.sum()), 'variants': {}}
    q0 = fm @ case.x_pr
    for name in VARIANTS:
        t0 = time.perf_counter()
        R = rep(name)
        xs = R.project(m, w)
        qs = common.evaluate(R, xs)
        q2 = common.evaluate(R, fm @ xs)
        Q = {0: (q0, nO), 1: (fm @ qs, nO), 2: (q2, nO)}
        e = {t: errors.errors(*Q[a], *Q[b]) for t, (a, b) in ST.PAIRS.items()}
        es = errors.errors(qs, n_src, case.x_pr, n_src)
        pops = {R.pops[i]: [float(v) for v in xs[:, R.i_num[i]]] for i in range(R.np) if (xs[:, R.i_num[i]] > 0).any()}
        out['variants'][name] = {
            'n_tracers': R.d, 'paper_species': matrix.PAPER_SPECIES[R.mech], 'modelE212_ntmAMP': matrix.MODELE_NTMAMP[R.mech],
            'source_number_fractions': pops, '12h': ST.mean_terms(e),
            'sources': {str(s): dict({k: float(es[k][i]) for k in errors.SIX},
                                     overall=float(np.mean([es[k][i] for k in errors.SIX])))
                        for i, s in enumerate(config.POP_SCENARIOS)}}
        r = out['variants'][name]['12h']
        print('%-13s d %2d  averaging %.4f  reconstruction %.4f  representation %.4f  (%.0f s)  pops %s' % (
            name, R.d, r['mixture']['overall'], r['reconstruction']['overall'], r['representation']['overall'],
            time.perf_counter() - t0, sorted(pops)), flush=True)
    path = ST.OUT / 'matrix_sensitivity.json'          # data/<basis>/ for a basis other than wet
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
