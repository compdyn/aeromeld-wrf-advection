#!/usr/bin/env python
"""Static terms of the decomposition for each baseline representation (no new transport):

  Q2 = q_R(N_PS, sum_s f_s x_s)     the exact mixture, in the representation
  Q1 = sum_s f_s q_R(x_s)           each source represented alone, then mixed
  Q0 = sum_s f_s q_PR,s             the particle-resolved reference

  averaging error      e(Q2, Q1)    (errors.py key 'mixture')
  reconstruction error e(Q1, Q0)
  representation error e(Q2, Q0)    (AeroMELD: the decoder error)

with per-source transport's fractions f and number N of the paper's main replay
(transport_schemes.load('all1_12h'), 'oracle'), over the mixed cells (>= 2 sources with
>= analysis.MIX_MIN_FRAC), number per m3 = mixing ratio / ALT, exactly as errors.decompose.
The same code path is run for AeroMELD (twin_k16_of.1.0: q = its decoder, x = its latent) as a
check against data/decomposition.json.

Writes
  baselines/data/static_terms.json         means over the mixed cells: 12 h, and every 30 min;
                                           each source's own reconstruction error; sanity checks
  baselines/data/static_terms_12h.npz      per-cell errors at 12 h (mask, f, n, e[model][term][diag])
  baselines/cache/q2_<model>_<hours>h.npy  Q2 (float32) per mixed cell (regenerable cache)
  baselines/cache/simplex_<model>_K100.npy Q2 on the fraction simplex, step 0.01 (cache)

The time series (every 30 min) takes the baselines' Q2 from the simplex table by linear
interpolation (Q2 = q_R(sum_s f_s x_s) depends on the fractions alone); 12 h is exact per cell
and the interpolated 12 h means are stored beside it as a check.

Usage: python baselines/static_terms.py [--no-time]
       python baselines/static_terms.py --add-models sec20w,mam4a --with-12h --add-hours 0.5:1 --tag a
       python baselines/static_terms.py --merge-add
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import diag  # noqa: E402
from paperfigs import analysis, config, errors, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

#: The size basis of every number here (diag.SIZE_BASIS, env AEROMELD_SIZE_BASIS). Inputs (the schemes'
#: source states, the layouts) are in DATA; the outputs of a basis other than 'wet' go to DATA/<basis>/
#: (2026-11-19: 'drydiam', the paper's), so no basis overwrites another's results. The Q2 caches depend
#: on the basis too (CACHE/<basis>/).
BASIS = diag.SIZE_BASIS
DATA = HERE / 'data'
OUT = DATA if BASIS == 'wet' else DATA / BASIS
OUT.mkdir(exist_ok=True)
CACHE = (Path(os.environ['BASELINES_CACHE']) if os.environ.get('BASELINES_CACHE')
         else HERE / 'cache' if BASIS == 'wet' else HERE / 'cache' / BASIS)


def _paper_defaults():
    """The run and WRF-AeroMELD model the numbers of the paper's basis refer to: the representative
    seed of config.recipe('twin_k16') in data/decomposition.json, its replay config.REPLAY_12H. The
    wet basis keeps its own (round 33, seed 1)."""
    if BASIS == 'wet':
        return 'all1_12h', 'twin_k16_of.1.0'
    if BASIS != config.SIZE_BASIS:
        return None, None                              # an earlier basis: STATIC_RUN, STATIC_AEROMELD needed
    name = config.recipe('twin_k16')
    dec = json.loads((config.REPO / 'data' / 'decomposition.json').read_text())['models']
    if name not in dec:
        return None, None
    seed = int(dec[name]['representative_seed'])
    return config.REPLAY_12H % seed, '%s.%d.0' % (name, seed)


#: Overridable (2026-11-19, the dry-sources study): STATIC_RUN, STATIC_AEROMELD, BASELINES_ZSRC
#: (the schemes' source states, a file in data/; their source diagnostics depend on the basis),
#: BASELINES_CACHE (Q2 caches).
_RUN, _AEROMELD = _paper_defaults()
RUN = os.environ.get('STATIC_RUN', _RUN)
AEROMELD = os.environ.get('STATIC_AEROMELD', _AEROMELD)
if RUN is None or AEROMELD is None:
    raise SystemExit('static_terms: basis %s: set STATIC_RUN and STATIC_AEROMELD (no paper defaults)' % BASIS)
ZSRC_FILE = os.environ.get('BASELINES_ZSRC', 'zsrc_baselines.npz' if BASIS == 'wet' else 'zsrc_baselines_%s.npz' % BASIS)
MODELS = ('aeromeld',) + common.MODELS
PAIRS = {'mixture': (2, 1), 'reconstruction': (1, 0), 'representation': (2, 0)}
#: Time series: Q2 interpolated in a table on the fraction simplex, step 1/SIMPLEX_K (Q2 is a
#: function of the fractions alone); exact per cell at 12 h, where the two are compared.
SIMPLEX_K = 100


def mean_terms(e):
    return {t: dict({k: float(np.mean(v[k])) for k in errors.SIX},
                    overall=float(np.mean([np.mean(v[k]) for k in errors.SIX]))) for t, v in e.items()}


class Case:
    def __init__(self, run=RUN):
        self.s = T.load(run)
        self.pop = products.load('populations', model=AEROMELD)
        self.x_pr = errors.source_vectors(self.pop, 'pmc')                 # (3, 456)
        self.z_star = self.pop['z'].values.astype(float)
        self.name = AEROMELD.rsplit('.', 1)[0]
        if diag.DRY_SOURCES:
            # the sources without water: the reference from the dry particles (diag, the reference's
            # own computation), the latents re-encoded from the dry samples (baselines/drysrc.py)
            self.x_pr = diag.raw_vectors(*diag.sources())
            recipe, seed = self.name.rsplit('.', 1)
            from extract import affx
            with np.load(affx.RUNS_DIR / 'zsrc_drysrc' / ('zsrc_s%s.npz' % seed)) as f:
                self.z_star = f[affx.zsrc_key(recipe)].astype(float)
        with np.load(DATA / ZSRC_FILE) as f:
            self.zsrc = {k: f[k] for k in f.files}
        if not np.allclose(self.zsrc['pmc_xsrc'], self.x_pr, rtol=2e-6, atol=0):
            d = np.abs(self.zsrc['pmc_xsrc'] - self.x_pr).max()
            print('note: diag pmc_xsrc vs populations product: max |diff| %.2e' % d)
        self.reps = {m: common.rep(m) for m in common.ALL if m + '_z' in self.zsrc}

    def cells(self, hours):
        it = T.time_index(self.s, hours)
        f_O = T.fractions(self.s, 'oracle', it)
        n_O = T.number(self.s, 'oracle', it)
        mask = (f_O >= analysis.MIX_MIN_FRAC).sum(0) >= 2
        rho = 1.0 / errors.alt_surface(hours)[mask]
        return f_O, n_O, mask, f_O[:, mask].T, n_O[mask] * rho

    def source_q(self, model):
        if model == 'aeromeld':
            return errors.decode_cells(self.name, self.z_star)
        return self.zsrc[model + '_xsrc']

    def mixture_q(self, model, fm, hours=None):
        if model == 'aeromeld':
            return errors.decode_cells(self.name, fm @ self.z_star)
        cache = CACHE / ('q2_%s_%gh.npy' % (model, hours)) if hours is not None else None
        if cache is not None and cache.exists():
            q = np.load(cache)
            if len(q) == len(fm):
                return q.astype(float)
        q = common.evaluate(self.reps[model], fm @ self.zsrc[model + '_z'])
        if cache is not None:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(cache, q.astype(np.float32))
        return q

    def simplex_table(self, model, K=SIMPLEX_K):
        """Q2 on the simplex grid f = (i, j, K - i - j) / K (q_R(sum f_s x_s) depends on the
        fractions alone): (K + 1, K + 1, 456), NaN off the simplex; cached."""
        cache = CACHE / ('simplex_%s_K%d.npy' % (model, K))
        if cache.exists():
            return np.load(cache).astype(float)
        i, j = np.meshgrid(np.arange(K + 1), np.arange(K + 1), indexing='ij')
        ok = i + j <= K
        f = np.stack([i[ok], j[ok], K - i[ok] - j[ok]], 1) / K
        q = common.evaluate(self.reps[model], f @ self.zsrc[model + '_z'])
        tab = np.full((K + 1, K + 1, q.shape[1]), np.nan)
        tab[ok] = q
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, tab.astype(np.float32))
        return tab

    def mixture_q_interp(self, model, fm, K=SIMPLEX_K):
        """Q2 by linear (barycentric) interpolation in the simplex table."""
        tab = np.nan_to_num(self.simplex_table(model, K))   # off-simplex vertices get weight 0
        u, v = fm[:, 0] * K, fm[:, 1] * K
        i = np.clip(np.floor(u).astype(int), 0, K - 1)
        j = np.clip(np.floor(v).astype(int), 0, K - 1)
        a, b = u - i, v - j
        lower = a + b <= 1
        # lower triangle: (i,j) 1-a-b, (i+1,j) a, (i,j+1) b;
        # upper: (i+1,j+1) a+b-1, (i+1,j) 1-b, (i,j+1) 1-a  (inside the simplex when a+b > 1)
        i0 = np.where(lower, i, np.minimum(i + 1, K))
        j0 = np.where(lower, j, np.minimum(j + 1, K))
        w0 = np.where(lower, 1 - a - b, a + b - 1)
        w1 = np.where(lower, a, 1 - b)
        w2 = np.where(lower, b, 1 - a)
        return (w0[:, None] * tab[i0, j0] + w1[:, None] * tab[np.minimum(i + 1, K), j]
                + w2[:, None] * tab[i, np.minimum(j + 1, K)])

    def terms(self, model, hours, per_cell=False, interp=False):
        f_O, n_O, mask, fm, nO = self.cells(hours)
        if not mask.any():
            return None
        if model != 'aeromeld' and interp:
            q2 = self.mixture_q_interp(model, fm)
        else:
            q2 = self.mixture_q(model, fm, hours)
        Q = {0: (fm @ self.x_pr, nO), 1: (fm @ self.source_q(model), nO), 2: (q2, nO)}
        e = {t: errors.errors(*Q[a], *Q[b]) for t, (a, b) in PAIRS.items()}
        out = {'mixed': int(mask.sum()), 'terms': mean_terms(e)}
        if per_cell:
            out.update(cells=e, mask=mask, f=f_O, n=n_O)
        return out

    def source_errors(self, model):
        """Each source's own reconstruction error, at the full population's number (m-3)."""
        n = np.array([config.POP_N_FULL[s] for s in config.POP_SCENARIOS])
        e = errors.errors(self.source_q(model), n, self.x_pr, n)
        return {str(s): dict({k: float(e[k][i]) for k in errors.SIX},
                             overall=float(np.mean([e[k][i] for k in errors.SIX])))
                for i, s in enumerate(config.POP_SCENARIOS)}


def sanity_fine_sectional(case, nbins=(20, 50, 100, 200, 400), n_rep=20):
    """Each source's reconstruction error in sectional representations of increasing resolution
    (log-spaced 0.001-10 um), with bins and nodes in dry diameter (the baselines') and in wet
    diameter (the coag paper's), each source at the full population's number."""
    import sectional
    m, w = diag.sources()
    n = np.array([config.POP_N_FULL[s] for s in config.POP_SCENARIOS])
    out = {}
    for basis in ('dry', 'wet'):
        for nb in nbins:
            R = sectional.Sectional('fine', n_rep=n_rep, edges=np.logspace(-3, 1, nb + 1) * 1e-6, basis=basis)
            q = common.particles_vectors(*R.particles(R.project(m, w)), procs=1)
            e = errors.errors(q, n, case.x_pr, n)
            out['%s_%d_bins' % (basis, nb)] = {k: [float(v) for v in e[k]] for k in errors.SIX}
            print('  fine sectional (%s) %4d bins: ' % (basis, nb) +
                  '  '.join('%s %s' % (errors.SHORT[k], np.round(100 * e[k], 2)) for k in errors.SIX))
    return out


def time_series(case, hours, interp=False, models=MODELS):
    """Mean terms over the mixed cells at each of `hours` (exact Q2 unless interp)."""
    models = tuple(models)
    ts = {'hours': [], 'mixed': [], 'models': {m: [] for m in models},
          'note': ('Q2 of the baselines by linear interpolation in a table on the fraction simplex '
                   '(step 1/%d); AeroMELD exact' % SIMPLEX_K) if interp else 'exact per cell'}
    for h in hours:
        t0 = time.perf_counter()
        row = {m: case.terms(m, float(h), interp=interp) for m in models}
        if row[models[0]] is None:
            continue
        ts['hours'].append(float(h))
        ts['mixed'].append(row[models[0]]['mixed'])
        for m in models:
            ts['models'][m].append(row[m]['terms'])
        print('%5.1f h: %d mixed cells (%.0f s)' % (h, row[models[0]]['mixed'], time.perf_counter() - t0), flush=True)
    return ts


def merge():
    """static_terms.json's time series <- the exact parts (data/static_terms_time_*.json) and its
    own 12 h values."""
    with open(OUT / 'static_terms.json') as f:
        res = json.load(f)
    rows = []
    for p in sorted(OUT.glob('static_terms_time_*.json')):
        with open(p) as f:
            t = json.load(f)
        for k, h in enumerate(t['hours']):
            rows.append((h, t['mixed'][k], {m: t['models'][m][k] for m in MODELS}))
    rows.append((12.0, res['mixed_12h'], {m: res['12h'][m] for m in MODELS}))
    rows = sorted({r[0]: r for r in rows}.values())
    if 'time' in res and res['time'].get('note', '').startswith('Q2 of the baselines by linear'):
        res['time_interpolated'] = res.pop('time')
    res['time'] = {'hours': [r[0] for r in rows], 'mixed': [r[1] for r in rows],
                   'models': {m: [r[2][m] for r in rows] for m in MODELS}, 'note': 'exact per cell'}
    with open(OUT / 'static_terms.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('merged %d times into' % len(rows), OUT / 'static_terms.json')


def add_models(case, models, hours, tag, with_12h):
    """Static terms of models added after the main run (common.EXTRA): 12 h per cell and each
    source (with_12h), and the exact time series at `hours` -> data/static_terms_add_<tag>.json
    (+ data/static_terms_12h_add_<tag>.npz). Merge with --merge-add."""
    res = {'models': list(models)}
    if with_12h:
        res['12h'], npz = {}, {}
        for model in models:
            t0 = time.perf_counter()
            r = case.terms(model, 12.0, per_cell=True)
            res['12h'][model] = r['terms']
            res['mixed_12h'] = r['mixed']
            for t, e in r['cells'].items():
                for k, v in e.items():
                    npz['%s__%s__%s' % (model, t, k)] = np.asarray(v, np.float32)
            print('12 h %-9s (%d cells, %.0f s): ' % (model, r['mixed'], time.perf_counter() - t0) +
                  '  '.join('%s %.4f' % (t, v['overall']) for t, v in r['terms'].items()), flush=True)
        np.savez_compressed(OUT / ('static_terms_12h_add_%s.npz' % tag), **npz)
        res['sources'] = {model: case.source_errors(model) for model in models}
    if hours:
        res['time'] = time_series(case, hours, interp=False, models=models)
    out = OUT / ('static_terms_add_%s.json' % tag)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', out)


def merge_add():
    """Put data/static_terms_add_*.json (and *_12h_add_*.npz) into static_terms.json and
    static_terms_12h.npz: the added models' 12 h terms, source errors and exact time series
    (12 h appended to the series)."""
    with open(OUT / 'static_terms.json') as f:
        res = json.load(f)
    parts = []
    for p in sorted(OUT.glob('static_terms_add_*.json')):
        with open(p) as f:
            parts.append(json.load(f))
    added = sorted({m for q in parts for m in q['models']}, key=list(common.ALL).index)
    rows = {m: {} for m in added}
    for q in parts:
        for m in q['models']:
            if '12h' in q:
                if q['mixed_12h'] != res['mixed_12h']:
                    raise ValueError('mixed cells differ')
                res['12h'][m] = q['12h'][m]
                res['sources'][m] = q['sources'][m]
            if 'time' in q:
                for k, h in enumerate(q['time']['hours']):
                    rows[m][h] = q['time']['models'][m][k]
                    i = res['time']['hours'].index(h)
                    if res['time']['mixed'][i] != q['time']['mixed'][k]:
                        raise ValueError('mixed cells differ at %g h' % h)
    for m in added:
        rows[m][12.0] = res['12h'][m]
        missing = [h for h in res['time']['hours'] if h not in rows[m]]
        if missing:
            raise ValueError('%s: no time series at %s h' % (m, missing))
        res['time']['models'][m] = [rows[m][h] for h in res['time']['hours']]
        if m not in res['models']:
            res['models'].append(m)
    with open(OUT / 'static_terms.json', 'w') as f:
        json.dump(res, f, indent=1)
    with np.load(OUT / 'static_terms_12h.npz') as f:
        npz = {k: f[k] for k in f.files}
    for p in sorted(OUT.glob('static_terms_12h_add_*.npz')):
        with np.load(p) as f:
            npz.update({k: f[k] for k in f.files})
    np.savez_compressed(OUT / 'static_terms_12h.npz', **npz)
    print('merged', added, 'into', OUT / 'static_terms.json', 'and static_terms_12h.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-time', action='store_true')
    ap.add_argument('--every', type=float, default=0.5)
    ap.add_argument('--interp-time', action='store_true',
                    help='time series from the simplex table (fast, biased: see notes); default exact')
    ap.add_argument('--time-hours', help='only the exact time series at these hours (comma list) -> '
                                         'data/static_terms_time_<first>-<last>h.json')
    ap.add_argument('--merge', action='store_true',
                    help='put data/static_terms_time_*.json into static_terms.json as its time series')
    ap.add_argument('--add-models', help='comma list of added models (e.g. sec20w,mam4a): see add_models')
    ap.add_argument('--add-hours', default='', help='colon list of hours for --add-models\' time series')
    ap.add_argument('--tag', default='a')
    ap.add_argument('--with-12h', action='store_true')
    ap.add_argument('--merge-add', action='store_true')
    args = ap.parse_args()
    if args.merge:
        return merge()
    if args.merge_add:
        return merge_add()
    case = Case()
    if args.add_models:
        hours = [float(h) for h in args.add_hours.replace(',', ':').split(':') if h]
        return add_models(case, args.add_models.split(','), hours, args.tag, args.with_12h)
    if args.time_hours:
        hours = [float(h) for h in args.time_hours.split(',')]
        ts = time_series(case, hours, interp=False)
        out = OUT / ('static_terms_time_%g-%gh.json' % (hours[0], hours[-1]))
        with open(out, 'w') as f:
            json.dump(ts, f, indent=1)
        print('wrote', out)
        return
    res = {'run': RUN, 'aeromeld_model': AEROMELD, 'reference': 'per-source transport (oracle) x '
           'particle-resolved sources (errors.source_vectors of populations_%s)' % AEROMELD,
           'terms': {'mixture': 'averaging error e(Q2, Q1)', 'reconstruction': 'e(Q1, Q0)',
                     'representation': 'e(Q2, Q0)'},
           'six': list(errors.SIX), 'models': list(MODELS), 'n_rep_sectional': common.sectional.N_REP}

    # --- 12 h, per cell --------------------------------------------------------------------
    cells = {}
    res['12h'] = {}
    for model in MODELS:
        t0 = time.perf_counter()
        r = case.terms(model, 12.0, per_cell=True)
        res['12h'][model] = r['terms']
        res['mixed_12h'] = r['mixed']
        cells[model] = r
        print('12 h %-9s (%d cells, %.0f s): ' % (model, r['mixed'], time.perf_counter() - t0) +
              '  '.join('%s %.4f' % (t, v['overall']) for t, v in r['terms'].items()), flush=True)
    npz = {'mask': cells['aeromeld']['mask'], 'f': cells['aeromeld']['f'], 'n_mixing_ratio': cells['aeromeld']['n']}
    for model in MODELS:
        for t, e in cells[model]['cells'].items():
            for k, v in e.items():
                npz['%s__%s__%s' % (model, t, k)] = np.asarray(v, np.float32)
    np.savez_compressed(OUT / 'static_terms_12h.npz', **npz)

    # --- check: AeroMELD against data/decomposition.json ----------------------------------
    with open(diag.REPO / 'data' / 'decomposition.json') as f:
        _name, _seed, _ = AEROMELD.split('.')
        dec = json.load(f)['models'][_name]['seeds'][_seed]['oracle']
    chk = {}
    for t, key in (('mixture', 'mixture'), ('reconstruction', 'reconstruction'), ('representation', 'decoder')):
        a = res['12h']['aeromeld'][t]
        chk[t] = max(abs(a[k] - dec[key][k]) for k in list(errors.SIX) + ['overall'])
    res['check_aeromeld_vs_decomposition_json'] = {'max_abs_diff': chk}
    print('AeroMELD vs decomposition.json, max |diff|:', chk)

    # --- each source on its own -------------------------------------------------------------
    res['sources'] = {model: case.source_errors(model) for model in MODELS}

    # --- check: the simplex-table interpolation (used for the time series) at 12 h ----------
    chk = {}
    for model in common.MODELS:
        r = case.terms(model, 12.0, interp=True)['terms']
        chk[model] = {t: {'exact': res['12h'][model][t]['overall'], 'interpolated': r[t]['overall'],
                          'max_abs_diff_by_diag': max(abs(r[t][k] - res['12h'][model][t][k]) for k in errors.SIX)}
                      for t in PAIRS}
        print('  interpolation check %-6s ' % model + '  '.join(
            '%s %.4f/%.4f' % (t, v['exact'], v['interpolated']) for t, v in chk[model].items()))
    res['check_simplex_interpolation_12h'] = chk
    res['sanity_fine_sectional'] = sanity_fine_sectional(case)

    # --- every 30 minutes -------------------------------------------------------------------
    if not args.no_time:
        hours = [h for h in np.asarray(case.s['time_s']) / 3600.0
                 if h > 0 and np.isclose(h / args.every, round(h / args.every))]
        res['time'] = time_series(case, hours, interp=args.interp_time)
    with open(OUT / 'static_terms.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', OUT / 'static_terms.json')


if __name__ == '__main__':
    main()
