#!/usr/bin/env python
"""Checks of one run_baselines.py output directory (numpy only; a minute at most).

Hard checks (FAIL):
  - every surface field (N, oracle, sep:*) and the whole 3-D state (diag.npz) finite and
    nonnegative to round-off (NEG_TOL of the field's maximum);
  - domain totals: the five oracle classes add up to N (within 1e-3);
  - models without calcsize: the number components of Q add up to N (domain totals, within
    2e-3: the sources' number fractions add up to 1 within 6e-4), and x = Q / N inside the
    sources' range (to 1% of it) at >= 99% of the surface points;
  - with --reference (the paper's 12-h replay surface.npz): the oracle and N equal the
    paper's at the common output times (same fluxes, same code) to TOL_REF of the maximum;
  - calcsize models: calcsize conserves every species' mass summed over the modes (its
    cumulative domain-total change).
Reported, and checked only in direction (calcsize moves modes toward their bounds, never away):
  - per mode, the number geometric mean diameter's excess outside [dgnumlo, dgnumhi]
    (|ln| distance) at the surface, with calcsize and without (the _nocs copy); MAM relaxes the
    number toward the bounds on a 1-day time scale, so modes are not expected inside them
    (e.g. the emission source's primary-carbon mode, dgn ~214 nm against dgnumhi 100 nm);
  - the mode numbers with / without calcsize (median and 99th percentile over the surface at
    the last output; domain totals), and the total number against N.

Usage: python check_run.py RUNDIR [--reference PAPER_SURFACE_NPZ] [--layout state_layout.json]
       [--zsrc zsrc.npz] [--modal-path DIR] [--file surface.npz]
Writes RUNDIR/check.json (or check_<file stem>.json for another --file).
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: Negative values allowed, relative to the field's maximum (Zalesak's limiter at round-off).
NEG_TOL = 1e-10
TOL_REF = 1e-5


def modal_report(s, st, modal_path, note, msgs):
    """dgn excess and mode numbers, calcsize against no calcsize, at every output."""
    sys.path.insert(0, str(Path(modal_path).resolve()))
    modal = importlib.import_module('modal')
    out = {}
    for m in st.get('calcsize', []):
        if 'sep:%s_nocs' % m not in s:
            msgs.append('    (%s: no _nocs copy, no comparison)' % m)
            continue
        M = modal.Modal(m)
        lo, hi = np.log(modal.DGNUMLO), np.log(modal.DGNUMHI)
        rows = []
        for it in range(len(s['time_s'])):
            row = {'t_h': float(s['time_s'][it] / 3600)}
            for tag, key in (('cs', 'sep:' + m), ('nocs', 'sep:%s_nocs' % m)):
                q = s[key][it].astype(float)                      # (d, y, x)
                x = q.reshape(q.shape[0], -1).T                   # (P, d)
                with np.errstate(invalid='ignore', divide='ignore'):
                    ld = np.log(M.dgn(x))                         # (P, 4), nan where empty
                exc = np.maximum(np.maximum(lo - ld, ld - hi), 0.0)
                nums = x[:, M.i_num]
                row[tag] = {'excess_mean': np.nanmean(exc, 0).tolist(),
                            'excess_p99': np.nanpercentile(exc, 99, 0).tolist(),
                            'frac_outside': np.nanmean(np.where(np.isnan(exc), np.nan, exc > 1e-9), 0).tolist(),
                            'dgn_median_nm': (np.exp(np.nanmedian(ld, 0)) * 1e9).tolist(),
                            'number_total': nums.sum(0).tolist()}
                row[tag + '_nums'] = nums
            nc, nn = row.pop('cs_nums'), row.pop('nocs_nums')
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.where(nn > 0, nc / nn, np.nan)
            row['number_ratio_median'] = np.nanmedian(ratio, 0).tolist()
            row['number_ratio_p99'] = np.nanpercentile(ratio, 99, 0).tolist()
            row['number_ratio_max'] = np.nanmax(ratio, 0).tolist()
            rows.append(row)
        last = rows[-1]
        for i, name in enumerate(modal.MODES):
            ec, en = last['cs']['excess_mean'][i], last['nocs']['excess_mean'][i]
            msgs.append('    %s %-14s t=%gh: dgn median %6.1f nm (no calcsize %6.1f), mean |ln| excess %.3f (%.3f), '
                        'outside %.2f (%.2f); number ratio median %.3f p99 %.3f max %.3f'
                        % (m, name, last['t_h'], last['cs']['dgn_median_nm'][i], last['nocs']['dgn_median_nm'][i],
                           ec, en, last['cs']['frac_outside'][i], last['nocs']['frac_outside'][i],
                           last['number_ratio_median'][i], last['number_ratio_p99'][i], last['number_ratio_max'][i]))
            if np.isfinite(ec) and np.isfinite(en):
                note('%s %s: calcsize does not move dgn away from the bounds (mean excess, last output)' % (m, name),
                     ec <= en * 1.01 + 1e-6, '%.4f vs %.4f' % (ec, en))
        out[m] = {'modes': list(modal.MODES), 'per_output': rows}
    return out


def tomas_report(s, st, rundir, modal_path, note, msgs):
    """--mnfix runs (TOMAS, added 2026-11-14): every surface bin's mean TOMAS dry mass inside its
    bounds at every output with mnfix (hard check), the share of occupied (cell, bin) pairs
    outside them without it (the _nocs copy), mnfix's event counts and its cumulative domain-total
    change per TOMAS species (summed over bins) relative to the domain total."""
    sys.path.insert(0, str(Path(modal_path).resolve()))
    tomas = importlib.import_module('tomas')
    out = {}
    dpath = Path(rundir) / 'diag.npz'
    D = dict(np.load(dpath, allow_pickle=False)) if dpath.exists() else {}
    for m in st.get('mnfix', []):
        T = tomas.Tomas(m)
        lo, hi = tomas.XK[:-1], tomas.XK[1:]
        rows = []
        for it in range(len(s['time_s'])):
            row = {'t_h': float(s['time_s'][it] / 3600)}
            for tag, key in (('mnfix', 'sep:' + m), ('nocs', 'sep:%s_nocs' % m)):
                if key not in s:
                    continue
                q = s[key][it].astype(float)
                x = q.reshape(q.shape[0], -1).T
                av = T.dry_mass_per_particle(x)
                # occupied: number and dry mass both > 0 in the float32 extract (a bin of ~1e-27
                # particles per kg has its dry mass underflow to 0 there)
                dry = x[:, :9 * tomas.NBINS].reshape(len(x), 9, -1)[:, tomas.DRY_J].sum(1)
                occ = ~np.isnan(av) & (dry > 0)
                # 1e-5: the surface extracts are float32. TOMAS's code leaves a bin whose mean is
                # below 1.1 Xk(1) with no bin below it at 1.2 Xk(1) (not a failure of the port)
                quirk = np.isclose(av, 1.2 * tomas.XK[0], rtol=1e-5)
                outside = occ & ~quirk & ((av < lo * (1 - 1e-5)) | (av > hi * (1 + 1e-5)))
                row[tag] = {'outside': int(outside.sum()), 'occupied': int(occ.sum()),
                            'cells_with_any_outside': int(outside.any(1).sum())}
            rows.append(row)
        worst = max(r['mnfix']['outside'] for r in rows)
        note('%s: with mnfix every surface bin\'s mean dry mass inside its bounds (all outputs)' % m, worst == 0,
             'worst output %d (cell, bin) outside' % worst)
        if 'nocs' in rows[-1]:
            msgs.append('    %s_nocs: (cell, bin) means outside their bounds at %gh: %d of %d occupied (%d cells); '
                        'max over outputs %d' % (m, rows[-1]['t_h'], rows[-1]['nocs']['outside'],
                                                 rows[-1]['nocs']['occupied'], rows[-1]['nocs']['cells_with_any_outside'],
                                                 max(r['nocs']['outside'] for r in rows)))
        res = {'per_output': rows, 'events': st.get('mnfix_stats', {}).get(m)}
        msgs.append('    %s: mnfix events (cell, bin) over the run: %s' % (m, res['events']))
        if 'mnfix_delta:' + m in D:
            names = list(D['tracer'])
            i0 = names.index('sep:%s:0' % m)
            nb = tomas.NBINS
            delta = D['mnfix_delta:' + m][-1][:9 * nb].reshape(9, nb).sum(1)
            tot = D['total'][-1][i0:i0 + 9 * nb].reshape(9, nb).sum(1)
            rel = dict(zip(tomas.TSPECIES, (delta / np.maximum(np.abs(tot), 1e-300)).tolist()))
            res['cumulative_domain_change_over_total'] = rel
            msgs.append('    %s: mnfix cumulative domain-total change / total at %gh: %s'
                        % (m, s['time_s'][-1] / 3600, {k: float('%.3g' % v) for k, v in rel.items()}))
        out[m] = res
    return out


def matrix_report(s, st, rundir, modal_path, note, msgs):
    """--matrix-adjust runs (MATRIX, added 2026-11-14): matrix.adjust conserves every MATRIX species
    summed over the populations and the bulk totals (cumulative domain totals, hard check); its event
    counts (cells x calls where each operation acted); the number change per population; and the
    largest difference of the adjusted model from its _nocs copy at the surface (every output)."""
    sys.path.insert(0, str(Path(modal_path).resolve()))
    matrix = importlib.import_module('matrix')
    out = {}
    dpath = Path(rundir) / 'diag.npz'
    D = dict(np.load(dpath, allow_pickle=False)) if dpath.exists() else {}
    for m in st.get('matrix_adjust', []):
        M = matrix.Matrix(m)
        res = {'events': st.get('matrix_adjust_stats', {}).get(m)}
        msgs.append('    %s: matrix.adjust events (cell x call) over the run: %s' % (m, res['events']))
        if 'matrix_adjust_delta:' + m in D:
            names = list(D['tracer'])
            i0 = names.index('sep:%s:0' % m)
            delta = D['matrix_adjust_delta:' + m][-1]
            tot = np.abs(D['total'][-1][i0:i0 + M.d])
            worst = 0.0
            for sp in matrix.MSPC + matrix.BULK:
                idx = [k for k, c in enumerate(M.comps) if c[1] == 'mass' and c[2] == sp]
                moved = np.abs(delta[idx]).sum()
                if moved > 0:
                    worst = max(worst, abs(delta[idx].sum()) / moved)
            note('%s: matrix.adjust conserves each MATRIX species summed over populations (domain totals)' % m,
                 worst < 1e-9, 'worst |net| / |moved| %.1e' % worst)
            dn = delta[M.i_num] / np.maximum(tot[M.i_num].sum(), 1e-300)
            res['number_change_over_total_number'] = dict(zip(M.pops, dn.tolist()))
            res['max_abs_change_over_total'] = float((np.abs(delta) / np.maximum(tot, 1e-300)).max())
            msgs.append('    %s: cumulative domain number change per population / total number: %s'
                        % (m, {p: float('%.3g' % v) for p, v in res['number_change_over_total_number'].items() if v}))
        if 'sep:%s_nocs' % m in s:
            a, b = s['sep:' + m].astype(float), s['sep:%s_nocs' % m].astype(float)
            res['max_abs_diff_vs_nocs_over_max'] = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
            res['identical_to_nocs'] = bool(np.array_equal(a, b))
            msgs.append('    %s vs %s_nocs (surface, all outputs): max |diff| / max %.2e, identical %s'
                        % (m, m, res['max_abs_diff_vs_nocs_over_max'], res['identical_to_nocs']))
        out[m] = res
    return out


def check(rundir, reference=None, layout=None, file='surface.npz', zsrc=None, modal_path=None):
    rundir = Path(rundir)
    msgs, ok = [], True

    def note(name, cond, detail=''):
        nonlocal ok
        msgs.append('%-78s %s %s' % (name, 'ok ' if cond else 'FAIL', detail))
        ok &= bool(cond)

    with np.load(rundir / file, allow_pickle=False) as f:
        s = {k: f[k] for k in f.files}
    st = json.loads(str(s['settings']))
    t = s['time_s']
    res = {'settings': st, 'file': file, 'times_h': (t / 3600).tolist()}
    for k in [k for k in s if k == 'N' or k == 'oracle' or k.startswith('sep:')]:
        v = s[k]
        note('%s finite, nonnegative to round-off (surface)' % k,
             bool(np.isfinite(v).all() and v.min() >= -NEG_TOL * np.abs(v).max()),
             'min %.2e (max %.2e)' % (float(v.min()), float(np.abs(v).max())))
    lay = json.loads(Path(layout).read_text())['models'] if layout else {}
    dpath = rundir / 'diag.npz'
    if dpath.exists():
        with np.load(dpath, allow_pickle=False) as f:
            D = {k: f[k] for k in f.files}
        names = list(D['tracer'])
        note('3-D state finite at every output', int(D['n_nonfinite'].sum()) == 0, str(int(D['n_nonfinite'].sum())))
        mn, mx = D['min'].min(axis=0), np.abs(D['max']).max(axis=0)
        neg = [(names[i], float(mn[i] / mx[i])) for i in np.flatnonzero(mn < -NEG_TOL * mx)]
        note('3-D state nonnegative to round-off (%g of max) at every output' % NEG_TOL, not neg, str(neg[:4]))
        res['min_over_max'] = {names[i]: float(mn[i] / mx[i]) for i in np.flatnonzero(mn < 0)}
        tot = D['total']
        iN = names.index('N')
        io = [i for i, n in enumerate(names) if n.startswith('oracle:')]
        if io:
            r = tot[:, io].sum(1) / tot[:, iN]
            note('domain total: sum of oracle classes / N within 1e-3 of 1', bool(np.abs(r - 1).max() < 1e-3),
                 'range %.6f .. %.6f' % (r.min(), r.max()))
        res['N_total'] = tot[:, iN].tolist()
        res['air_total'] = D['air_total'].tolist()
        res['number_over_N'] = {}
        for m in st['sep_models']:
            base = m[:-5] if m.endswith('_nocs') else m
            if base not in lay:
                continue
            i0 = names.index('sep:%s:0' % m)
            r = tot[:, [i0 + j for j in lay[base]['number_indices']]].sum(1) / tot[:, iN]
            res['number_over_N'][m] = r.tolist()
            if m in st.get('calcsize', []) or m in st.get('matrix_adjust', []):
                msgs.append('    %s: domain number / N (calcsize changes number): %.4f at 0 h -> %.4f at %g h'
                            % (m, r[0], r[-1], t[-1] / 3600))
            else:
                note('%s: domain total of the number components / N within 2e-3 of 1' % m,
                     bool(np.abs(r - 1).max() < 2e-3), 'range %.5f .. %.5f' % (r.min(), r.max()))
        for k in [k for k in D if k.startswith('calcsize_delta:')]:
            m = k.split(':', 1)[1]
            i0 = names.index('sep:%s:0' % m)
            delta = D[k][-1]
            d = delta.shape[0]
            scale = np.abs(tot[:, i0:i0 + d]).max(0)
            res['calcsize_rel_change:' + m] = (np.abs(delta) / np.maximum(scale, 1e-300)).tolist()
            if modal_path:
                sys.path.insert(0, str(Path(modal_path).resolve()))
                modal = importlib.import_module('modal')
                Mo = modal.Modal(m)
                species = sorted({c[2] for c in Mo.comps if c[1] == 'mass'})
                worst = 0.0
                for sp in species:
                    idx = [j for j, c in enumerate(Mo.comps) if c[1] == 'mass' and c[2] == sp]
                    moved = np.abs(delta[idx]).sum()
                    if moved > 0:
                        worst = max(worst, abs(delta[idx].sum()) / moved)
                note('%s: calcsize conserves each species summed over modes (domain totals)' % m, worst < 1e-9,
                     'worst |net| / |moved| %.1e' % worst)
                numc = [j for j, c in enumerate(Mo.comps) if c[1] == 'number']
                msgs.append('    %s: calcsize cumulative domain number change per mode / mode total: %s'
                            % (m, np.array2string(delta[numc] / np.maximum(np.abs(tot[-1, [i0 + j for j in numc]]),
                                                                          1e-300), precision=4)))
    # x = Q / N within the sources' range (models without calcsize)
    if zsrc is None:
        zsrc = st.get('zsrc')
    zs = {}
    if zsrc and Path(zsrc).exists():
        with np.load(zsrc) as f:
            zs = {m: f[m + '_z'] for m in {k[:-5] if k.endswith('_nocs') else k for k in st['sep_models']}
                  if m + '_z' in f.files}
    N = s['N'].astype(float)
    inner = (slice(None), slice(None), slice(2, -2), slice(2, -2))
    for k in [k for k in s if k.startswith('sep:')]:
        m = k[4:]
        base = m[:-5] if m.endswith('_nocs') else m
        if m in st.get('calcsize', []) or m in st.get('matrix_adjust', []) or m in st.get('mnfix', []) \
                or base not in zs:
            continue
        z = zs[base]
        with np.errstate(invalid='ignore', divide='ignore'):
            x = s[k].astype(float) / np.where(N > 0, N, np.nan)[:, None]
        lo, hi = z.min(0)[None, :, None, None], z.max(0)[None, :, None, None]
        span = np.maximum(hi - lo, 1e-300)
        out = (np.maximum(lo - x, x - hi) / span)[inner]
        frac = float(np.nanmean(out > 0.01))
        note('%s: x = Q/N within the sources\' range (to 1%% of it) at >= 99%% of points' % k, frac < 0.01,
             'outside %.2e, worst %.2e' % (frac, float(np.nanmax(out))))
        res['outside_source_range:' + m] = frac
    if reference:
        with np.load(reference, allow_pickle=False) as f:
            tr, orc, Nr = f['time_s'], f['oracle'], f['N']
        common = np.intersect1d(t, tr)
        a = s['oracle'][np.isin(t, common)].astype(float)
        b = orc[np.isin(tr, common)].astype(float)
        rel = np.abs(a - b) / np.maximum(np.abs(b).max(), 1e-30)
        note('oracle == the paper\'s at %d common times (max |diff| / max)' % common.size,
             bool(rel.max() < TOL_REF), '%.2e (identical: %s)' % (rel.max(), bool(np.array_equal(a, b))))
        relN = np.abs(s['N'][np.isin(t, common)].astype(float) - Nr[np.isin(tr, common)]) / Nr.max()
        note('N == the paper\'s (max |diff| / max)', bool(relN.max() < TOL_REF), '%.2e' % relN.max())
        res['oracle_vs_paper'] = float(rel.max())
        res['oracle_vs_paper_identical'] = bool(np.array_equal(a, b))
        res['N_vs_paper'] = float(relN.max())
        res['reference'] = str(reference)
    if modal_path and st.get('calcsize'):
        res['modal'] = modal_report(s, st, modal_path, note, msgs)
    if modal_path and st.get('matrix_adjust'):
        res['matrix'] = matrix_report(s, st, rundir, modal_path, note, msgs)
    if modal_path and st.get('mnfix'):
        res['tomas'] = tomas_report(s, st, rundir, modal_path, note, msgs)
    res['ok'] = bool(ok)
    res['messages'] = msgs
    name = 'check.json' if file == 'surface.npz' else 'check_%s.json' % Path(file).stem
    (rundir / name).write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rundir')
    ap.add_argument('--reference', default=None)
    ap.add_argument('--layout', default=None)
    ap.add_argument('--zsrc', default=None)
    ap.add_argument('--modal-path', default=str(HERE.parent), help='directory holding modal.py')
    ap.add_argument('--file', default='surface.npz')
    a = ap.parse_args()
    r = check(a.rundir, a.reference, a.layout, a.file, a.zsrc, a.modal_path)
    print('\n'.join(r['messages']))
    print('ALL OK' if r['ok'] else 'SOME CHECKS FAILED')


if __name__ == '__main__':
    main()
