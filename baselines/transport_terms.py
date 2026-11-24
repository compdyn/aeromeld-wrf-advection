#!/usr/bin/env python
"""Transport terms of the baselines (and AeroMELD, as a check) from the 12-h replays: the full
decomposition of PLAN.md, per mixed cell, at 12 h and every 30 min.

  Q4 = q_R(x_S) on N_S          the transported state: x_S = Q / N_S, Q = sep:<model> (Q = N x)
  Q3 = q_R(x3)  on N_S          its nearest mixture: x3 = sum_s fbar_s x_s, fbar the implied
                                fractions clipped >= 0 and renormalized
  Q2, Q1, Q0                    as in static_terms.py (per-source transport's f_O, N_O)

  off-mixture e(Q4, Q3), fraction e(Q3, Q2), transport e(Q4, Q2), total e(Q4, Q0),
  averaging e(Q2, Q1), reconstruction e(Q1, Q0), representation e(Q2, Q0)

N_S, the number the representation carries: the sum of its number components (bins or modes)
in Q. For the modal runs with calcsize this differs from WRF's N, which calcsize does not
touch; for the others it equals N to 1e-5. AeroMELD (twin_k16_of.1.0, sep:twin_k16 of the
paper's all1_12h): x_S = Q / N, N_S = N, exactly as errors.decompose.

Implied fractions (baselines): weighted least squares of x_S on the plane through the three
source states (affine, sum f = 1, as mixing.implied_fractions does unweighted for latents).
Component k is weighted by 1 / max(spread_k, 1e-3 max_{k' in g} spread_k'), spread_k =
max_s x_sk - min_s x_sk over the sources and g its group (all number components; or all
bins/modes of one species), so every component counts on the scale of its variation across the
sources, and a bin that holds a negligible amount of a species does not get the weight of one
that holds most of it. Components that are equal in all sources (spread 0, e.g. the empty
coarse mode) get weight 0 when the whole group is zero, else the group floor. On an exact mixture
of the sources the fit returns its fractions whatever the weights.
AeroMELD: mixing.implied_fractions, unweighted, as the paper.

Cases (CASES): sec8, mam4, mam4_nocs, mam4l, mam4l_nocs, sec20w, mam4a, mam4a_nocs, aeromeld;
and two checks at 12 h only: 'oracle:<m>' (the per-source mixture as if transported: every
transport term must be 0) and 'donor:<m>' (the paper's donor-cell fractions applied to the
sources: an exact mixture, so the off-mixture error must be 0 and only the fraction error
remains).

Per case, writes data/transport_terms_<case>.json (12 h means, time series, stats; resumable:
times already done are skipped), and with --extras: _12h_<case>.npz (per-cell total and
transport error per diagnostic, 12 h), _maps_<case>.npz (implied fractions on the full surface,
and with --maps the 12-h surface maps of activated fraction, frozen fraction and optics),
_points_<case>.npz (q_R at the sampling points A, B, C). --merge combines them into
transport_terms.json / _12h.npz / _maps.npz / _points.npz.

Usage (Perlmutter, from scripts/; transport_terms.sbatch):
  python ../baselines/transport_terms.py --case sec8 --extras [--maps]
  python ../baselines/transport_terms.py --merge          (locally, parts in data/transport_terms_parts/)
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
import static_terms as ST  # noqa: E402
from paperfigs import config, errors, mixing  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

RUNS = Path(os.environ.get('BASELINES_TRANSPORT_RUNS', '/pscratch/sd/m/mwest/aeromeld/baselines/transport/runs'))
#: Outputs: static_terms.OUT (data/ for the wet basis, data/<basis>/ for the others, 2026-11-19).
DATA = ST.OUT
#: The per-case parts, which --merge reads (wet basis: downloaded there from Perlmutter's data/;
#: since 2026-11-19 every case writes its parts there directly).
PARTS = DATA / 'transport_terms_parts'
PARTS.mkdir(exist_ok=True)
#: case -> (run directory, Q key, representation)
CASES = {
    'sec8': ('sec8_12h', 'sep:sec8', 'sec8'),
    'mam4': ('mam4_12h', 'sep:mam4', 'mam4'),
    'mam4_nocs': ('mam4_12h', 'sep:mam4_nocs', 'mam4'),
    'mam4l': ('mam4l_12h', 'sep:mam4l', 'mam4l'),
    'mam4l_nocs': ('mam4l_12h', 'sep:mam4l_nocs', 'mam4l'),
    'sec20w': ('sec20w_12h', 'sep:sec20w', 'sec20w'),
    'mam4a': ('mam4a_12h', 'sep:mam4a', 'mam4a'),
    'mam4a_nocs': ('mam4a_12h', 'sep:mam4a_nocs', 'mam4a'),
    'aeromeld': (None, 'sep:twin_k16', 'aeromeld'),
}
#: Added 2026-11-14: MATRIX mechanism 1 (matrix.adjust after every step) and its no-adjustment copy.
CASES.update({'matrix1': ('matrix1_12h', 'sep:matrix1', 'matrix1'),
              'matrix1_nocs': ('matrix1_12h', 'sep:matrix1_nocs', 'matrix1')})
#: Added 2026-11-14: TOMAS-40 with mnfix after every step, and without it (a separate run, the same
#: numbers as a _nocs copy); mam4la (MAM4 species, large fresh carbon to accumulation) as mam4l.
CASES.update({'tomas40': ('tomas40_12h', 'sep:tomas40', 'tomas40'),
              'tomas40_nocs': ('tomas40nocs_12h', 'sep:tomas40', 'tomas40'),
              'mam4la': ('mam4la_12h', 'sep:mam4la', 'mam4la'),
              'mam4la_nocs': ('mam4la_12h', 'sep:mam4la_nocs', 'mam4la')})
#: Added 2026-11-16 (the dry rebuild): 20 bins in dry diameter, transported (baselines_array.sbatch task 8).
CASES.update({'sec20': ('sec20_12h', 'sep:sec20', 'sec20')})
#: The dry-sources study (2026-11-19): TT_RUN_TAG=drysrc12h reads the schemes' runs <name>_drysrc12h
#: instead of <name>_12h; TT_TAG=_drysrc appends to every output name (data/transport_terms_<case>_drysrc.json).
if os.environ.get('TT_RUN_TAG'):
    CASES = {k: ((r[:-len('12h')] + os.environ['TT_RUN_TAG']) if r and r.endswith('_12h') else r, q, m)
             for k, (r, q, m) in CASES.items()}
TT_TAG = os.environ.get('TT_TAG', '')
PAIRS = {'offmixture': (4, 3), 'fraction': (3, 2), 'transport': (4, 2), 'total': (4, 0),
         'mixture': (2, 1), 'reconstruction': (1, 0), 'representation': (2, 0)}
HOURS = [0.5 * k for k in range(1, 25)]
#: Map quantities (12 h, sec20w and mam4 and the reference): activated fraction at these
#: supersaturations (fraction), frozen fraction at these temperatures (C), optics (m).
MAP_SS = (0.001, 0.003, 0.006, 0.01)
MAP_T = (-40.0, -30.0, -20.0, -10.0)
MAP_WL = (400e-9, 1000e-9)
SPREAD_FLOOR = 1e-3


def number_index(R):
    if hasattr(R, 'i_num'):
        return np.asarray(R.i_num)
    return np.arange(R.nb) * (1 + diag.NS)


def species_groups(R):
    """Component groups for the weights: 'number', or the species name (diagnosed PartMC
    species for mam4l's lumped ones)."""
    g = []
    for k in range(R.d):
        if hasattr(R, 'comps'):
            i, kind, s = R.comps[k]
            g.append('number' if kind == 'number' else s)
        else:
            r = k % (1 + diag.NS)
            g.append('number' if r == 0 else diag.SPECIES[r - 1])
    return np.array(g)


def fit_weights(xs, groups):
    spread = xs.max(0) - xs.min(0)
    w = np.zeros(xs.shape[1])
    for g in np.unique(groups):
        sel = groups == g
        top = spread[sel].max()
        if top > 0:
            w[sel] = 1.0 / np.maximum(spread[sel], SPREAD_FLOOR * top)
    return w


def implied_fractions_weighted(x, xs, w):
    """x (n, d), sources xs (3, d), weights w (d) -> f (3, n) barycentric (sum 1) and the
    weighted distance from the plane (n)."""
    A = ((xs[1:] - xs[0]) * w).T                                  # (d, 2)
    b = ((x - xs[0]) * w).T                                        # (d, n)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    f = np.vstack([1 - coef.sum(0), coef])
    return f, np.linalg.norm(b - A @ coef, axis=0)


class Run:
    """One case's transported state, on the static case's cells."""

    def __init__(self, name, sc):
        self.name = name
        self.sc = sc
        check = name.split(':')[0] if ':' in name else None
        model = name.split(':')[1] if check else CASES[name][2]
        self.check, self.model = check, model
        self.aeromeld = model == 'aeromeld'
        if self.aeromeld:
            self.xs = sc.z_star
        else:
            self.R = sc.reps[model]
            self.xs = sc.zsrc[model + '_z']
            self.num = number_index(self.R)
            self.water = np.array([c.get('species') in ('H2O', 'h2o', 'AW') for c in self.R.layout()['components']])
            self.w = fit_weights(self.xs, species_groups(self.R))
        if check:
            self.time_s = sc.s['time_s']
            return
        run, key, _ = CASES[name]
        if self.aeromeld:
            self.time_s, self.Q, self.N = sc.s['time_s'], sc.s[key], sc.s['N']
        else:
            with np.load(RUNS / run / 'surface.npz') as f:
                self.time_s, self.Q, self.N = f['time_s'], f[key], f['N']
                it, jt = self.t_index(12.0), T.time_index(sc.s, 12.0)
                if not np.array_equal(f['oracle'][it], sc.s['oracle'][jt]):
                    raise ValueError('%s: oracle differs from the paper\'s all1_12h at 12 h' % run)

    def t_index(self, hours):
        return int(np.flatnonzero(np.isclose(self.time_s, hours * 3600.0))[0])

    def q(self, x):
        if self.aeromeld:
            return errors.decode_cells(self.sc.name, x)
        return common.evaluate(self.R, x)

    def fractions(self, x):
        if self.aeromeld:
            f, off = mixing.implied_fractions(x.T, self.xs)
            return f, off
        return implied_fractions_weighted(x, self.xs, self.w)

    def state(self, hours, sel):
        """x_S (n, d), N_S per kg (n), and WRF's N (n) at the cells sel (a mask or index)."""
        if self.check == 'oracle':
            it = T.time_index(self.sc.s, hours)
            f = T.fractions(self.sc.s, 'oracle', it)[:, sel].T
            n = T.number(self.sc.s, 'oracle', it)[sel]
            return f @ self.xs, n, n
        if self.check == 'donor':
            it = T.time_index(self.sc.s, hours)
            f = T.fractions(self.sc.s, 'donor', it)[:, sel].T
            n = T.number(self.sc.s, 'donor', it)[sel]
            return f @ self.xs, n, n
        it = self.t_index(hours)
        Q = self.Q[it][:, sel].astype(float).T
        if diag.DRY_SOURCES and not self.aeromeld:
            # the sources without water (2026-11-19): every other component of the projection is
            # unchanged (drysrc.py --schemes), each tracer is advected with its own limiter, and the
            # schemes' adjustments decide on dry quantities (MATRIX's water-dependent BC2 -> BC3 acted
            # in no cell), so the dry-source run is this run with its water components set to 0
            Q[:, self.water] = 0.0
        N = self.N[it][sel].astype(float)
        nS = N if self.aeromeld else Q[:, self.num].sum(1)
        return Q / nS[:, None], nS, N

    def terms(self, hours, per_cell=False):
        sc = self.sc
        f_O, n_O, mask, fm, nO = sc.cells(hours)
        rho = nO / n_O[mask]
        x, nS, N = self.state(hours, mask)
        f, off = self.fractions(x)
        fb = mixing.f_bar(f)
        x3 = fb.T @ self.xs
        Q = {0: (fm @ sc.x_pr, nO), 1: (fm @ sc.source_q(self.model), nO),
             2: (sc.mixture_q(self.model, fm, hours), nO),
             3: (self.q(x3), nS * rho), 4: (self.q(x), nS * rho)}
        e = {t: errors.errors(*Q[a], *Q[b]) for t, (a, b) in PAIRS.items()}
        stats = {'n_S_over_N': np.percentile(nS / N, [0, 1, 50, 99, 100]).tolist(),
                 'fraction_abs_err_vs_oracle': np.percentile(np.abs(fb.T - fm).max(1), [50, 99, 100]).tolist(),
                 'f_negative_min': float(f.min()), 'offplane_median': float(np.median(off))}
        out = {'mixed': int(mask.sum()), 'terms': ST.mean_terms(e), 'stats': stats}
        if per_cell:
            out.update(cells=e, mask=mask)
        return out


def extras(run, out_tag, maps):
    """12-h implied fractions on the full surface, the points A, B, C, and (maps) the surface maps."""
    sc = run.sc
    hours = 12.0
    it = T.time_index(sc.s, hours)
    f_O = T.fractions(sc.s, 'oracle', it)
    n_O = T.number(sc.s, 'oracle', it)
    rho = 1.0 / errors.alt_surface(hours)
    full = np.ones(n_O.shape, bool)
    x, nS, N = run.state(hours, full)
    f, off = run.fractions(x)
    res = {'f_raw': f.reshape(3, *n_O.shape).astype(np.float32),
           'f_bar': mixing.f_bar(f).reshape(3, *n_O.shape).astype(np.float32),
           'offplane': off.reshape(n_O.shape).astype(np.float32),
           'n_S_per_kg': nS.reshape(n_O.shape).astype(np.float32),
           'f_oracle': f_O.astype(np.float32), 'n_oracle_per_kg': n_O.astype(np.float32),
           'rho_kg_m3': rho.astype(np.float32)}
    # points A, B, C (0-based (j, i) from 1-based (i, j))
    pts = [(p[1] - 1, p[0] - 1) for p in config.LOCATIONS]
    jj, ii = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    flat = jj * n_O.shape[1] + ii
    xp = x[flat]
    points = {'q_S': run.q(xp), 'n_S_per_m3': nS[flat] * rho[jj, ii], 'x_S': xp,
              'f_raw': f[:, flat].T, 'f_oracle': f_O[:, jj, ii].T,
              'q_ref': f_O[:, jj, ii].T @ sc.x_pr, 'n_oracle_per_m3': n_O[jj, ii] * rho[jj, ii],
              'q_mixture': (sc.mixture_q(run.model, f_O[:, jj, ii].T) if run.aeromeld else
                            common.evaluate(run.R, f_O[:, jj, ii].T @ run.xs)),
              'point_j0': jj, 'point_i0': ii}
    np.savez_compressed(PARTS / ('transport_terms_points_%s.npz' % out_tag), **points)
    if maps:
        q = run.q(x)                                                 # (cells, 456), per particle
        res.update({'%s__%s' % (k, kind): v for kind, qq, n in
                    (('S', q, nS * rho.ravel()), ('ref', f_O.reshape(3, -1).T @ sc.x_pr, n_O.ravel() * rho.ravel()))
                    for k, v in map_quantities(qq, n, n_O.shape).items()})
    np.savez_compressed(PARTS / ('transport_terms_maps_%s.npz' % out_tag), **res)


def map_quantities(q, n_m3, shape):
    """Activated fraction at MAP_SS, frozen fraction at MAP_T, scattering and absorption (Mm-1)
    at MAP_WL, from raw per-particle vectors q (cells, 456) and number per m3."""
    eps = np.exp(np.linspace(np.log(1e-3), np.log(1e-1), 100))
    temp = np.linspace(-40.0, 0.0, 100)
    wl = np.array(diag.WAVELENGTHS_M)
    blk = lambda k: q[:, diag.OFFSET[k][0]:diag.OFFSET[k][0] + diag.OFFSET[k][1]]
    interp = lambda arr, grid, v: np.array([np.interp(v, grid, row) for row in arr])
    out = {'activated_fraction': np.stack([interp(blk('C'), np.log(eps), np.log(s)) for s in MAP_SS], -1),
           'frozen_fraction': np.stack([interp(blk('F'), temp, t) for t in MAP_T], -1)}
    for k, name in (('S', 'scattering_Mm-1'), ('A', 'absorption_Mm-1')):
        out[name] = np.stack([blk(k)[:, int(np.argmin(np.abs(wl - v)))] * n_m3 * 1e6 for v in MAP_WL], -1)
    return {k: v.reshape(*shape, -1).astype(np.float32) for k, v in out.items()}


def run_case(name, do_extras, maps, hours_list):
    sc = ST.Case()
    out_tag = name.replace(':', '_') + TT_TAG
    path = PARTS / ('transport_terms_%s.json' % out_tag)
    res = {'case': name, 'time': {}}
    if path.exists():
        with open(path) as f:
            res = json.load(f)
    t0 = time.perf_counter()
    run = Run(name, sc)
    print('%s: loaded in %.0f s' % (name, time.perf_counter() - t0), flush=True)
    if 'run' not in res and not run.check:
        res.update(run=CASES[name][0] or 'paper all1_12h', key=CASES[name][1], model=run.model)
    if not run.aeromeld:
        res['weights'] = {'groups': 'number; each species', 'floor': SPREAD_FLOOR,
                          'n_zero_weight': int((run.w == 0).sum()), 'd': int(run.R.d)}
    if '12h' not in res:
        t0 = time.perf_counter()
        r = run.terms(12.0, per_cell=True)
        res['12h'] = r['terms']
        res['mixed_12h'] = r['mixed']
        res['stats_12h'] = r['stats']
        npz = {'mask': r['mask']}
        for t in ('total', 'transport', 'offmixture'):
            for k, v in r['cells'][t].items():
                npz['%s__%s' % (t, k)] = np.asarray(v, np.float32)
        np.savez_compressed(PARTS / ('transport_terms_12h_%s.npz' % out_tag), **npz)
        print('12 h %s (%.0f s): ' % (name, time.perf_counter() - t0) +
              '  '.join('%s %.4f' % (t, v['overall']) for t, v in r['terms'].items()), flush=True)
        _write(path, res)
    if do_extras and not run.check:
        t0 = time.perf_counter()
        extras(run, out_tag, maps)
        print('extras %s (%.0f s)' % (name, time.perf_counter() - t0), flush=True)
    for h in ([] if run.check else hours_list):
        key = '%g' % h
        if key in res['time'] or h == 12.0:
            continue
        t0 = time.perf_counter()
        r = run.terms(h)
        if r is None:
            continue
        res['time'][key] = {'mixed': r['mixed'], 'terms': r['terms'], 'stats': r['stats']}
        _write(path, res)
        print('%5.1f h %s: %d cells (%.0f s) total %.4f transport %.4f' % (
            h, name, r['mixed'], time.perf_counter() - t0, r['terms']['total']['overall'],
            r['terms']['transport']['overall']), flush=True)
    print('done', name, flush=True)


def _write(path, res):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1)
    tmp.replace(path)


def merge():
    """transport_terms_<case>.json etc. -> transport_terms.json, _12h.npz, _maps.npz, _points.npz."""
    cases = {}
    for p in sorted(PARTS.glob('transport_terms_*.json')):
        with open(p) as f:
            r = json.load(f)
        cases[r['case']] = r
    out = {'pairs': {k: list(v) for k, v in PAIRS.items()}, 'six': list(errors.SIX),
           'cases': {}, 'checks': {}}
    for name, r in cases.items():
        if ':' in name:
            out['checks'][name] = {'12h': r['12h'], 'mixed_12h': r['mixed_12h'], 'stats_12h': r['stats_12h']}
            continue
        hrs = sorted(float(h) for h in r['time']) + [12.0]
        series = [r['time']['%g' % h] if h != 12.0 else {'mixed': r['mixed_12h'], 'terms': r['12h'],
                                                          'stats': r['stats_12h']} for h in hrs]
        out['cases'][name] = {k: r[k] for k in ('run', 'key', 'model', 'weights') if k in r}
        out['cases'][name].update({'12h': r['12h'], 'mixed_12h': r['mixed_12h'], 'stats_12h': r['stats_12h'],
                                   'time': {'hours': hrs, 'mixed': [s['mixed'] for s in series],
                                            'terms': [s['terms'] for s in series],
                                            'stats': [s['stats'] for s in series]}})
    with open(config.REPO / 'data' / 'decomposition.json') as f:
        _name, _seed, _ = ST.AEROMELD.split('.')
        decm = json.load(f)['models'][_name]
        dec = decm['seeds'][_seed]['sep']
    six = list(errors.SIX)
    if 'aeromeld' in out['cases']:
        a = out['cases']['aeromeld']['12h']
        diff = {t: max(abs(a[t][k] - dec[t if t != 'representation' else 'decoder'][k]) for k in six + ['overall'])
                for t in PAIRS}
        out['checks']['aeromeld_sep_vs_decomposition_json_max_abs_diff'] = diff
        # the same cells, reference and error as scripts/decomposition.py: equal to rounding
        assert max(diff.values()) < 1e-9, diff
        out['cases']['aeromeld'].update(seed=int(_seed), name=_name)
    # 2026-11-19: the median over the seeds (decomposition.json, separate limiters), as the round-37
    # merge (dry_variants.py --merge) gave it; its 'decoder' is the schemes' 'representation'
    per = [{('representation' if t == 'decoder' else t): v for t, v in s['sep'].items()} for s in decm['seeds'].values()]
    med = {t: {q: float(np.median([p[t][q] for p in per])) for q in six + ['overall']} for t in PAIRS}
    out['cases']['aeromeld_median'] = {
        '12h': med, 'name': _name, 'seeds': sorted(int(k) for k in decm['seeds']),
        'range_overall': {t: [min(p[t]['overall'] for p in per), max(p[t]['overall'] for p in per)]
                          for t in ('total', 'transport', 'representation')}}
    with open(DATA / 'transport_terms.json', 'w') as f:
        json.dump(out, f, indent=1)
    for kind in ('12h', 'maps', 'points'):
        npz = {}
        for p in sorted(PARTS.glob('transport_terms_%s_*.npz' % kind)):
            case = p.stem[len('transport_terms_%s_' % kind):]
            with np.load(p) as f:
                npz.update({'%s__%s' % (case, k): f[k] for k in f.files})
        if npz:
            np.savez_compressed(DATA / ('transport_terms_%s.npz' % kind), **npz)
    print('merged', sorted(out['cases']), sorted(out['checks']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', help='comma list of cases (CASES, oracle:<m>, donor:<m>)')
    ap.add_argument('--extras', action='store_true')
    ap.add_argument('--maps', action='store_true')
    ap.add_argument('--hours', default=':'.join('%g' % h for h in HOURS))
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    if args.merge:
        return merge()
    hours = [float(h) for h in args.hours.replace(',', ':').split(':') if h]
    for name in args.case.split(','):
        run_case(name, args.extras, args.maps, hours)


if __name__ == '__main__':
    main()
