#!/usr/bin/env python
"""Two checks of the paper's size-distribution errors (notes_dist_checks.md): is the wet-diameter
L2 comparison unfair to the bin and mode schemes, and would a Wasserstein metric tell another
story? At 12 h over the 5965 mixed cells of per-source transport (static_terms.Case.cells), for
the paper's four schemes (MOSAIC 8 `sec8`, TOMAS-40 `tomas40`, MAM4 `mam4l`, MATRIX `matrix1`)
and WRF-AeroMELD (twin_k16_of, representative seed 1; ten seeds for the median).

1. Dry diameter (schemes only). The number and speciated-mass size distributions of the
   reference and of each scheme binned by DRY diameter (H2O excluded from the diameter and from
   the mass: 14 species) in the paper's 15 bins (diag.EDGES[:16], 0.25 decade from 10^-8.75 m,
   [lo, hi), out of range: no bin), scored with the paper's e = ||yhat - y|| / (||yhat|| + ||y||)
   (training/paper_errors.relerr on paper_transform's units: clipped at 0, number per cm3 and
   ug m-3 per decade; both jointly over species and bins). WRF-AeroMELD's decoder outputs only
   the wet distributions, so it is not scored in dry diameter.
2. Wasserstein. W1 in log10 D between the normalized distributions (decades):
   number:          W1 = 0.25 sum_b |F_hat(b) - F(b)|, F the cumulative fractions over the 15 bins;
   speciated mass:  each species' W1 (its mass size distribution, normalized), weighted by the
                    species' reference mass fraction, over the species both carry; the reference
                    mass fraction in species the prediction leaves empty (lumped schemes) is
                    reported as `missing`.
   Lumping-free companions (the total mass size distribution, species summed; isolates where the
   mass is in size from which species it is reported as): e_mt (e, as paper_errors' m_prthst
   row) and w1_mt (W1).
   W1 ignores amount, so the totals' relative errors over the 15 bins are reported beside it:
   tot_n = |N_hat - N| / N; tot_m = sum_k |M_hat_k - M_k| / sum_k M_k (species totals, i.e. the
   reference-mass-weighted mean of the species' relative errors). Wet diameter for all five,
   dry diameter for the schemes.

States, each against Q0 = sum_s f_PS,s q_PR,s on N_PS (the paper's reference):
  representation  Q2 = q_R(sum_s f_PS,s x_s) on N_PS  (static_terms.py; WRF-AeroMELD: the decoder,
                  its 'decoder' term)
  transported     Q4 = q_R(x_S) on N_S, the 12-h state of the scheme's own replay with separate
                  limiters (transport_terms.py's case of the same name: MAM4 with calcsize, TOMAS
                  with MNFIX, MATRIX with its adjustments; read from the run's surface_end.npz);
                  WRF-AeroMELD: sep:twin_k16 of all<seed>_12h, as errors.decompose ('total').
The schemes' distributions come from their representative particles (R.particles, the code path
of common.evaluate) binned here; wet, this is diag.size_blocks exactly, and the wet e reproduces
static_terms.json / transport_terms.json (checked per cell in --merge). The reference's dry
distributions come from the source particle files (diag.sources), mixed in number with f_PS; its
wet ones are the paper's (errors.source_vectors of the populations product), checked against the
particle files' wet binning.

Writes (means and medians over the mixed cells; per-cell float32 values in the _cells npz)
  data/dist_checks_schemes.json, data/dist_checks_cells_schemes.npz    (--part schemes; Perlmutter)
  data/dist_checks_aeromeld.json, data/dist_checks_cells_aeromeld.npz  (--part aeromeld; seconds, locally)
  data/dist_checks.json                                                (--merge, with the checks)

Usage (from scripts/, like static_terms.py):
  python ../baselines/dist_checks.py --part schemes [--cells 200] [--models sec8,tomas40]
  python ../baselines/dist_checks.py --part aeromeld [--seeds 0-9]
  python ../baselines/dist_checks.py --merge
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag  # noqa: E402
import static_terms as ST  # noqa: E402
import transport_terms as TT  # noqa: E402
from paperfigs import analysis, errors, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402
from paper_errors import relerr  # noqa: E402  (on sys.path through paperfigs.errors)

#: Outputs in static_terms.OUT (data/ for the wet basis, data/<basis>/ otherwise; 2026-11-19).
DATA = ST.OUT
SCHEMES = ('sec8', 'tomas40', 'mam4l', 'matrix1')
LABELS = {'aeromeld': 'WRF-AeroMELD', 'sec8': 'MOSAIC 8', 'tomas40': 'TOMAS-40', 'mam4l': 'MAM4',
          'matrix1': 'MATRIX'}
AEROMELD = 'twin_k16_of'
#: --part aeromeld --current: the paper's current models and replays instead (config.recipe, config.REPLAY_12H;
#: since 2026-11-16 the dry-trained decoders, whose N and M blocks are binned by dry diameter, as their
#: populations product's reference): written to dist_checks_aeromeld_<SIZE_BASIS>.json, key SIZE_BASIS.
AM_RUN, AM_BASIS, AM_OUT = 'all%d_12h', 'wet', 'dist_checks_aeromeld.json'
#: What --merge reads for WRF-AeroMELD: the wet-era file, or the --current file of the run's basis.
AM_OUT_MERGE = 'dist_checks_aeromeld.json' if diag.SIZE_BASIS == 'wet' else 'dist_checks_aeromeld_%s.json' % diag.SIZE_BASIS
#: The bases scored (diag.BASES): wet, and the run's own (env AEROMELD_SIZE_BASIS; 2026-11-19 'drydiam').
#: The reference's distributions in the run's basis are the paper's (the populations product's, x_pr).
BASES = ('wet', 'dry') if diag.SIZE_BASIS == 'wet' else ('wet', diag.SIZE_BASIS)
STATES = ('representation', 'transported')
METRICS = ('e_n', 'e_m', 'e_mt', 'w1_n', 'w1_m', 'w1_mt', 'tot_n', 'tot_m', 'missing')
NB = diag.N_BINS                                        # 15 bins up to 10 um
LOG10_BIN = errors.LOG10_BIN                            # 0.25 decade
#: paper_transform's unit factors (number per cm3 and ug m-3 per decade); only relerr's 1e-30 sees them
SCALE_N, SCALE_M = 1e-6 / LOG10_BIN, 1e9 / LOG10_BIN
I_M, I_N = slice(0, 225), slice(225, 240)         # diag.OFFSET's M and N blocks
DRY_SPECIES = [s for s in diag.SPECIES if s != 'H2O']


# --- distributions ---------------------------------------------------------------------------

def size_dists(m, w, basis):
    """Particles m (P, n, 15) kg, weights w (P, n) -> per particle of the population: N (P, 15)
    the fraction of particles in each bin and M (P, C, 15) kg per bin, bins of `basis` diameter:
    'wet' (all 15 species, C = 15), 'drydiam' (dry diameter, all 15 species, C = 15; both
    diag.size_blocks) or 'dry' (H2O excluded from the diameter and from M, C = 14). The weights are
    cleaned and normalized as diag.raw_vectors does."""
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    if m.ndim == 2:
        m, w = m[None], w[None]
    m, w = diag._clean(m, w)
    tot = w.sum(-1, keepdims=True)
    wn = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 0.0)
    if basis != 'dry':
        M, N = diag.size_blocks(m, wn, basis)
        return N, M
    P = len(m)
    D = diag.dry_diameter(m)
    b = np.searchsorted(diag.EDGES, D, side='right') - 1            # as diag.size_blocks
    b = np.where(D == diag.EDGES[-1], diag.N_BINS_ALL - 1, b)
    ok = (b >= 0) & (b < NB) & (wn > 0)
    idx = (np.arange(P)[:, None] * NB + np.where(ok, b, 0)).ravel()
    wv = np.where(ok, wn, 0.0)
    N = np.bincount(idx, weights=wv.ravel(), minlength=P * NB).reshape(P, NB)
    dry = np.flatnonzero(diag.DRY)
    M = np.empty((P, len(dry), NB))
    for j, k in enumerate(dry):
        M[:, j] = np.bincount(idx, weights=(wv * m[..., k]).ravel(), minlength=P * NB).reshape(P, NB)
    return N, M


def blocks_of(x):
    """Raw 456-vectors (P, 456) -> N (P, 15), M (P, 15, 15) (the wet distributions)."""
    return x[:, I_N], x[:, I_M].reshape(len(x), diag.NS, NB)


def rep_dists(R, x, bases=BASES, block=500):
    """A representation's states x (P, d) -> {basis: (N, M)} from its representative particles
    (the modal schemes' node grid aligned with the bins of each basis: modal.GRID_BASIS)."""
    import modal
    out = {b: ([], []) for b in bases}
    keep = modal.GRID_BASIS
    try:
        for b in bases:
            modal.GRID_BASIS = b
            for a in range(0, len(x), block):
                m, w = R.particles(x[a:a + block])
                N, M = size_dists(m, w, b)
                out[b][0].append(N)
                out[b][1].append(M)
    finally:
        modal.GRID_BASIS = keep
    return {b: (np.concatenate(v[0]), np.concatenate(v[1])) for b, v in out.items()}


# --- scores ----------------------------------------------------------------------------------

def w1(p, q):
    """1-Wasserstein distance in log10 D (decades) between the normalized distributions p, q
    (..., 15) on bins of LOG10_BIN; nan where either is empty."""
    sp, sq = p.sum(-1, keepdims=True), q.sum(-1, keepdims=True)
    ok = (sp[..., 0] > 0) & (sq[..., 0] > 0)
    F = np.cumsum(p / np.where(sp > 0, sp, 1.0), -1)
    G = np.cumsum(q / np.where(sq > 0, sq, 1.0), -1)
    return np.where(ok, LOG10_BIN * np.abs(F - G).sum(-1), np.nan)


def scores(Nh, Mh, nh, Nr, Mr, nr):
    """Per cell: prediction (N (n, 15), M (n, C, 15) per particle, number nh per m3) against the
    reference -> {metric: (n,)} (METRICS; see the module docstring)."""
    Nh, Mh, Nr, Mr = (np.clip(a, 0, None) for a in (Nh, Mh, Nr, Mr))
    An, Bn = Nh * (nh * SCALE_N)[:, None], Nr * (nr * SCALE_N)[:, None]
    Am, Bm = Mh * (nh * SCALE_M)[:, None, None], Mr * (nr * SCALE_M)[:, None, None]
    out = {'e_n': relerr(Bn[:, None], An[:, None]), 'e_m': relerr(Bm, Am),
           'e_mt': relerr(Bm.sum(1, keepdims=True), Am.sum(1, keepdims=True)),
           'w1_n': w1(Nh, Nr), 'w1_mt': w1(Mh.sum(1), Mr.sum(1))}
    Wk = w1(Mh, Mr)                                                  # (n, C)
    th, tr = Am.sum(-1), Bm.sum(-1)                                  # species totals (n, C)
    wt = tr / tr.sum(-1, keepdims=True)
    valid = (th > 0) & (tr > 0)
    out['w1_m'] = (np.where(valid, wt * np.nan_to_num(Wk), 0.0).sum(-1)
                   / np.where(valid, wt, 0.0).sum(-1))
    out['missing'] = np.where((tr > 0) & ~(th > 0), wt, 0.0).sum(-1)
    out['tot_n'] = np.abs(An.sum(-1) - Bn.sum(-1)) / Bn.sum(-1)
    out['tot_m'] = np.abs(th - tr).sum(-1) / tr.sum(-1)
    return out


def summary(sc):
    return {k: {'mean': float(np.nanmean(v)), 'median': float(np.nanmedian(v)),
                'n_nan': int(np.isnan(v).sum())} for k, v in sc.items()}


# --- the reference -----------------------------------------------------------------------------

class Reference:
    """Q0's distributions on the mixed cells: in the run's basis (diag.SIZE_BASIS) the paper's
    (populations product, x_pr), in the others the source particle files binned here; all mixed
    in number with f_PS."""

    def __init__(self, x_pr):
        m, w = diag.sources()
        self.src = {b: size_dists(m, w, b) for b in BASES}
        N, M = blocks_of(x_pr)
        Nw, Mw = self.src[diag.SIZE_BASIS]
        self.check_wet = {'basis': diag.SIZE_BASIS, 'N_max_rel_diff': float(np.abs(Nw - N).max() / np.abs(N).max()),
                          'M_max_rel_diff_by_species': float(max(
                              np.abs(Mw[:, k] - M[:, k]).max() / max(np.abs(M[:, k]).max(), 1e-300)
                              for k in range(diag.NS) if np.abs(M[:, k]).max() > 0))}
        self.src[diag.SIZE_BASIS] = (N, M)                          # the paper's

    def mixed(self, fm, basis):
        N, M = self.src[basis]
        return fm @ N, np.einsum('ns,skb->nkb', fm, M)


# --- schemes ---------------------------------------------------------------------------------

def surface_end_path(run):
    """A scheme run's surface_end.npz (0, 4, 12 h): Perlmutter's runs directory, else the local copy."""
    p = TT.RUNS / run / 'surface_end.npz'
    return p if p.exists() else HERE / 'data' / 'transport' / run / 'surface_end.npz'


def transported_state(sc, model, mask):
    """transport_terms.Run.state at 12 h from the run's surface_end.npz: x_S = Q / N_S (n, d) and
    N_S per kg (n), N_S the sum of the number components; the oracle checked against all1_12h's."""
    run, key, rep = TT.CASES[model]
    p = surface_end_path(run)
    with np.load(p) as f:
        it = int(np.flatnonzero(np.isclose(f['time_s'], 12 * 3600.0))[0])
        if not np.array_equal(f['oracle'][it], sc.s['oracle'][T.time_index(sc.s, 12.0)]):
            raise ValueError('%s: oracle differs from the paper\'s all1_12h at 12 h' % run)
        Q = f[key][it][:, mask].astype(float).T
    nS = Q[:, TT.number_index(sc.reps[rep])].sum(1)
    return Q / nS[:, None], nS, str(p)


def part_schemes(models, ncells=None):
    sc = ST.Case()
    f_O, n_O, mask, fm, nO = sc.cells(12.0)
    rho = nO / n_O[mask]                                             # kg m-3
    sel = slice(None) if not ncells else slice(0, ncells)
    ref = Reference(sc.x_pr)
    res = {'mixed_12h': int(mask.sum()), 'cells_used': int(len(fm[sel])),
           'check_reference_wet_vs_particle_files': ref.check_wet, 'models': {}}
    cells = {}
    for model in models:
        R = sc.reps[model]
        r = res['models'][model] = {'label': LABELS[model]}
        for state in STATES:
            t0 = time.perf_counter()
            if state == 'representation':
                x, nh = fm[sel] @ sc.zsrc[model + '_z'], nO[sel]
            else:
                xs, nS, path = transported_state(sc, model, mask)
                x, nh = xs[sel], (nS * rho)[sel]
                r['transported_file'] = path
            d = rep_dists(R, x)
            for b in BASES:
                s = scores(*d[b], nh, *ref.mixed(fm[sel], b), nO[sel])
                r.setdefault(state, {})[b] = summary(s)
                for k, v in s.items():
                    cells['%s__%s__%s__%s' % (model, state, b, k)] = v.astype(np.float32)
            print('%-8s %-14s (%.0f s): ' % (model, state, time.perf_counter() - t0) + '  '.join(
                '%s e %.2f/%.2f W1 %.3f/%.3f' % (b, 100 * r[state][b]['e_n']['mean'], 100 * r[state][b]['e_m']['mean'],
                                                 r[state][b]['w1_n']['mean'], r[state][b]['w1_m']['mean'])
                for b in BASES), flush=True)
    tag = '' if not ncells else '_test%d' % ncells
    _write(DATA / ('dist_checks_schemes%s.json' % tag), res)
    np.savez_compressed(DATA / ('dist_checks_cells_schemes%s.npz' % tag), mask=mask, **cells)
    print('wrote', DATA / ('dist_checks_schemes%s.json' % tag))


# --- WRF-AeroMELD ------------------------------------------------------------------------------

def aeromeld_seed(seed, key='sep'):
    """One seed's wet scores, representation (decoder) and transported (sep), with the paper's e
    from errors.errors on the same vectors (the reproduction of decomposition.json)."""
    model = '%s.%d.0' % (AEROMELD, seed)
    run = AM_RUN % seed
    s = T.load(run)
    it = T.time_index(s, 12.0)
    rm = T.run_model(model)
    name = model.rsplit('.', 1)[0]
    pop = products.load('populations', model=model)
    z_star = pop['z'].values.astype(float)
    f_O = T.fractions(s, 'oracle', it)
    n_O = T.number(s, 'oracle', it)
    mask = (f_O >= analysis.MIX_MIN_FRAC).sum(0) >= 2
    rho = 1.0 / errors.alt_surface(12.0)[mask]
    fm, nO = f_O[:, mask].T, n_O[mask] * rho
    x0 = fm @ errors.source_vectors(pop, 'pmc')
    skey = '%s:%s' % (key, rm)
    z = T.latent(s, skey, z_star, rm, it)[:, mask].T
    nS = T.number(s, skey, it)[mask] * rho
    xq = {'representation': (errors.decode_cells(name, fm @ z_star), nO),
          'transported': (errors.decode_cells(name, z), nS)}
    out, cells = {'model': model, 'run': run, 'mixed_12h': int(mask.sum())}, {}
    for state, (x, n) in xq.items():
        sco = scores(*blocks_of(x), n, *blocks_of(x0), nO)
        e = errors.errors(x, n, x0, nO)
        out[state] = {AM_BASIS: summary(sco),
                      'paper_e_errors_py': {k: float(np.mean(e[k])) for k in ('n_prthst', 'm_chmprthst')},
                      'max_abs_diff_e_vs_errors_py': float(max(np.abs(sco['e_n'] - e['n_prthst']).max(),
                                                               np.abs(sco['e_m'] - e['m_chmprthst']).max()))}
        cells.update({'%s__%s__%s' % (state, AM_BASIS, k): v.astype(np.float32) for k, v in sco.items()})
    return out, cells, mask


def part_aeromeld(seeds):
    with open(diag.REPO / 'data' / 'decomposition.json') as f:
        dec = json.load(f)['models'][AEROMELD]
    res = {'recipe': AEROMELD, 'representative_seed': int(dec['representative_seed']), 'seeds': {}}
    cells_rep = None
    for k in seeds:
        t0 = time.perf_counter()
        out, cells, mask = aeromeld_seed(k)
        d = dec['seeds'][str(k)]
        out['decomposition_json'] = {
            'representation': {q: d['oracle']['decoder'][q] for q in ('n_prthst', 'm_chmprthst')},
            'transported': {q: d['sep']['total'][q] for q in ('n_prthst', 'm_chmprthst')}}
        res['seeds'][str(k)] = out
        if k == res['representative_seed']:
            cells_rep = dict(cells, mask=mask)
        print('seed %d (%.1f s): ' % (k, time.perf_counter() - t0) + '  '.join(
            '%s e %.2f/%.2f (json %.2f/%.2f) W1 %.3f/%.3f' % (
                st, 100 * out[st][AM_BASIS]['e_n']['mean'], 100 * out[st][AM_BASIS]['e_m']['mean'],
                100 * out['decomposition_json'][st]['n_prthst'], 100 * out['decomposition_json'][st]['m_chmprthst'],
                out[st][AM_BASIS]['w1_n']['mean'], out[st][AM_BASIS]['w1_m']['mean']) for st in STATES), flush=True)
    if len(seeds) > 1:
        res['median_over_seeds'] = {st: {m: {a: float(np.median([res['seeds'][str(k)][st][AM_BASIS][m][a] for k in seeds]))
                                             for a in ('mean', 'median')} for m in METRICS} for st in STATES}
    _write(DATA / AM_OUT, res)
    if cells_rep is not None and AM_OUT in ('dist_checks_aeromeld.json', AM_OUT_MERGE):
        np.savez_compressed(DATA / 'dist_checks_cells_aeromeld.npz', **cells_rep)
    print('wrote', DATA / AM_OUT)


# --- merge -------------------------------------------------------------------------------------

def merge():
    """dist_checks_{schemes,aeromeld}.json -> dist_checks.json, with the reproduction checks:
    the run basis's e (B, diag.SIZE_BASIS) against the published means (transport_terms.json
    representation and total of the same basis; decomposition.json for WRF-AeroMELD) and per cell
    (transport_terms_12h.npz, the transported state), the ranking under each metric, and per-cell
    win fractions. (Before 2026-11-19, wet only, with static_terms.json for the representation.)"""
    B = diag.SIZE_BASIS
    with open(DATA / 'dist_checks_schemes.json') as f:
        sch = json.load(f)
    with open(DATA / AM_OUT_MERGE) as f:
        am = json.load(f)
    with open(DATA / 'transport_terms.json') as f:
        tt = json.load(f)['cases']
    cs = np.load(DATA / 'dist_checks_cells_schemes.npz')
    ca = np.load(DATA / 'dist_checks_cells_aeromeld.npz')
    ptt = np.load(DATA / 'transport_terms_12h.npz')
    pmask = ptt['%s__mask' % SCHEMES[0]]
    if not (np.array_equal(cs['mask'], pmask) and np.array_equal(ca['mask'], pmask)):
        raise ValueError('mixed cells differ')
    rep = str(am['representative_seed'])
    out = {'hours': 12.0, 'mixed_12h': sch['mixed_12h'], 'metrics': {
        'e_n, e_m': "the paper's e (paper_errors.relerr) of the number / speciated-mass size distribution",
        'w1_n': 'W1 in log10 D (decades) of the normalized number distributions, 0.25 sum_b |F_hat - F|',
        'w1_m': "each species' W1 weighted by its reference mass fraction (species both carry)",
        'e_mt, w1_mt': 'e and W1 of the total mass size distribution (species summed; lumping-free)',
        'tot_n': '|N_hat - N| / N over the 15 bins', 'tot_m': 'sum_k |M_hat_k - M_k| / sum_k M_k over the 15 bins',
        'missing': 'reference mass fraction in species the prediction leaves empty (excluded from w1_m)'},
        'bins': '15 bins of 0.25 decade from 10^-8.75 m (diag.EDGES[:16]); wet = wet diameter, all species incl. H2O; '
                'dry = H2O excluded from the diameter and the mass; drydiam = dry diameter, all species incl. H2O',
        'basis': B,
        'states': {'representation': 'Q2 vs Q0 (transport_terms representation term; AeroMELD decoder term)',
                   'transported': 'Q4 vs Q0 (transport_terms total; AeroMELD sep total)'},
        'check_reference_wet_vs_particle_files': sch['check_reference_wet_vs_particle_files'],
        'models': {}, 'checks': {}}
    out['models']['aeromeld'] = {'label': LABELS['aeromeld'], 'seed': int(rep),
                                 **{s: am['seeds'][rep][s] for s in STATES}}
    if 'median_over_seeds' in am:
        out['models']['aeromeld']['median_over_seeds'] = am['median_over_seeds']
        out['models']['aeromeld']['seeds_used'] = sorted(int(k) for k in am['seeds'])
    for m in SCHEMES:
        out['models'][m] = sch['models'][m]
    # reproduction of the published wet e
    chk = {}                                       # {who: {state: {diagnostic: {...}}}}
    for m in SCHEMES:
        chk[m] = {}
        for state, pub in (('representation', tt[m]['12h']['representation']), ('transported', tt[m]['12h']['total'])):
            c = chk[m][state] = {}
            for q, k in (('n_prthst', 'e_n'), ('m_chmprthst', 'e_m')):
                mine = cs['%s__%s__%s__%s' % (m, state, B, k)]
                c[q] = {'published_mean': pub[q], 'this_mean': float(np.mean(mine.astype(float)))}
                if state == 'transported':
                    c[q]['max_abs_diff_per_cell'] = float(np.abs(mine - ptt['%s__total__%s' % (m, q)]).max())
    for k, v in am['seeds'].items():
        chk['aeromeld seed %s' % k] = {st_: {q: {'published_mean': v['decomposition_json'][st_][q],
                                                 'this_mean': v[st_][B]['e_n' if q == 'n_prthst' else 'e_m']['mean']}
                                             for q in ('n_prthst', 'm_chmprthst')} for st_ in STATES}
    out['checks']['%s_e_vs_published' % B] = chk
    out['checks']['max_abs_diff_of_means'] = float(max(abs(q['published_mean'] - q['this_mean'])
                                                       for c in chk.values() for d in c.values() for q in d.values()))
    out['checks']['max_abs_diff_per_cell_schemes'] = float(max(q['max_abs_diff_per_cell'] for m in SCHEMES
                                                                for d in chk[m].values() for q in d.values()
                                                                if 'max_abs_diff_per_cell' in q))
    # rankings (best first) and per-cell fractions where WRF-AeroMELD beats each scheme
    rank = {}
    for state in STATES:
        for b in BASES:
            for k in ('e_n', 'e_m', 'e_mt', 'w1_n', 'w1_m', 'w1_mt', 'tot_n', 'tot_m'):
                names = [m for m in ('aeromeld',) + SCHEMES if b in out['models'][m][state]]
                for a in ('mean', 'median'):
                    v = {m: out['models'][m][state][b][k][a] for m in names}
                    rank['%s %s %s %s' % (state, b, k, a)] = sorted(v, key=v.get)
    out['rankings_best_first'] = rank
    wins = {}
    for state in STATES:
        for k in ('e_n', 'e_m', 'e_mt', 'w1_n', 'w1_m', 'w1_mt'):
            a = ca['%s__%s__%s' % (state, B, k)]
            wins['%s %s %s' % (state, B, k)] = {m: float(np.mean(a < cs['%s__%s__%s__%s' % (m, state, B, k)])) for m in SCHEMES}
    out['aeromeld_better_fraction_of_cells'] = wins
    _write(DATA / 'dist_checks.json', out)
    print('wrote', DATA / 'dist_checks.json')
    print('reproduction: max |diff| of means %.2e, per cell (schemes) %.2e' % (
        out['checks']['max_abs_diff_of_means'], out['checks']['max_abs_diff_per_cell_schemes']))
    table(out)


def table(out):
    """The notes' tables (%, decades), means over the cells (medians in brackets for W1)."""
    def g(m, state, b, k, a='mean'):
        try:
            return out['models'][m][state][b][k][a]
        except KeyError:
            return None
    pc = lambda v: '-' if v is None else '%.1f' % (100 * v)
    D = BASES[1]                                       # the dry-diameter basis of these numbers
    for state in STATES:
        print('\n%s (means; W1 in decades, [median])' % state)
        print('| | wet e N | wet e M | %s e N | %s e M | wet W1 N | wet W1 M | %s W1 N | %s W1 M '
              '| wet tot N | wet tot M | %s tot M | missing wet/%s |' % ((D,) * 6))
        print('|---' * 13 + '|')
        for m in ('aeromeld',) + SCHEMES:
            w = lambda b, k: '-' if g(m, state, b, k) is None else '%.3f [%.3f]' % (g(m, state, b, k), g(m, state, b, k, 'median'))
            print('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s/%s |' % (
                LABELS[m], pc(g(m, state, 'wet', 'e_n')), pc(g(m, state, 'wet', 'e_m')),
                pc(g(m, state, D, 'e_n')), pc(g(m, state, D, 'e_m')),
                w('wet', 'w1_n'), w('wet', 'w1_m'), w(D, 'w1_n'), w(D, 'w1_m'),
                pc(g(m, state, 'wet', 'tot_n')), pc(g(m, state, 'wet', 'tot_m')), pc(g(m, state, D, 'tot_m')),
                pc(g(m, state, 'wet', 'missing')), pc(g(m, state, D, 'missing'))))
        print('\n%s, total mass size distribution (species summed): e (%%) and W1 (decades)' % state)
        print('| | wet e | %s e | wet W1 | %s W1 |\n|---|---|---|---|---|' % (D, D))
        for m in ('aeromeld',) + SCHEMES:
            w = lambda b: '-' if g(m, state, b, 'w1_mt') is None else '%.3f' % g(m, state, b, 'w1_mt')
            print('| %s | %s | %s | %s | %s |' % (LABELS[m], pc(g(m, state, 'wet', 'e_mt')), pc(g(m, state, D, 'e_mt')),
                                                w('wet'), w(D)))


def _write(path, res):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('schemes', 'aeromeld'))
    ap.add_argument('--models', default=','.join(SCHEMES))
    ap.add_argument('--cells', type=int, default=0, help='schemes: only the first N mixed cells (a test)')
    ap.add_argument('--seeds', default='0-9', help='aeromeld: seeds a-b')
    ap.add_argument('--current', action='store_true', help='aeromeld: the current models (config.recipe)')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--table', action='store_true', help='print the table of data/dist_checks.json')
    args = ap.parse_args()
    if args.merge:
        return merge()
    if args.table:
        with open(DATA / 'dist_checks.json') as f:
            return table(json.load(f))
    if args.part == 'schemes':
        diag.set_threads(1)
        return part_schemes(args.models.split(','), args.cells or None)
    if args.part == 'aeromeld':
        if args.current:
            from paperfigs import config
            global AEROMELD, AM_RUN, AM_BASIS, AM_OUT
            AEROMELD, AM_RUN, AM_BASIS = config.recipe('twin_k16'), config.REPLAY_12H, config.SIZE_BASIS
            AM_OUT = 'dist_checks_aeromeld_%s.json' % AM_BASIS
        lo, hi = (int(v) for v in args.seeds.split('-')) if '-' in args.seeds else (int(args.seeds),) * 2
        return part_aeromeld(list(range(lo, hi + 1)))
    ap.error('--part, --merge or --table')


if __name__ == '__main__':
    main()
