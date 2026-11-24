r"""One error measure for the whole paper, and the four-term error decomposition.

The measure is the test set's (training/paper_errors.py, imported, not re-implemented), in the
variant logratio_metrics: for each population (here each grid cell) and diagnostic, on the test
set's transformed diagnostics (activated fraction at s = 0.1-0.6 %, frozen fraction at -25 to
-10 C, optics in Mm-1 clamped at 0.1, number and speciated mass size distributions up to 10 um),
e = ||y - yhat|| / (||y|| + ||yhat||) for the three linear diagnostics, and half the RMS log
ratio for the three logarithmic ones (frozen fraction, scattering, absorption); both read as
about half the relative error. Six diagnostics (SIX); the overall error is their mean.

A cell's diagnostics are computed exactly as the test set's: per-particle decoder outputs
(the 456 raw values of the model's blocks) times the number concentration per m3, which is the
number mixing ratio divided by the dry-air ALT (only the log optics depend on it).

The decomposition of a transport scheme S's error against the particle-resolved reference,
through five stages, each per particle mixed in number fractions f (per-source transport's f_PS):

  Q4 = D(z_S)          N_S     the scheme's state, decoded
  Q3 = D(zbar_S)       N_S     its nearest mixture (fractions clipped >= 0, renormalized)
  Q2 = D(z_PS)         N_PS    the exact mixture, decoded           z_PS = sum_s f_PS,s z*_s
  Q1 = sum f_PS,s D(z*_s)      the decoded source populations, mixed
  Q0 = sum f_PS,s q_PR,s       the particle-resolved reference

  off-mixture error    e(Q4, Q3)   \
  fraction error       e(Q3, Q2)    > transport error e(Q4, Q2)
  averaging error      e(Q2, Q1)   \   (key 'mixture'; the paper's name since 2026-11-13)
  reconstruction error e(Q1, Q0)    > decoder error   e(Q2, Q0)
  total error          e(Q4, Q0)

The decoder's terms use f_PS, so they are the same for every scheme. The terms do not add, but
their sum bounds the stage's error exactly: e = ||a - b|| / (||a|| + ||b||) is a metric, and half
the RMS log ratio is a norm of a difference, so each obeys the triangle inequality,
e(Q4, Q2) <= e(Q4, Q3) + e(Q3, Q2), and so do their means over cells and diagnostics. Likewise
the decoder error is at most the averaging plus the reconstruction error, and the total error
at most the sum of all four terms.
"""
import sys
import types
from pathlib import Path

import numpy as np

from . import analysis, config, mixing, products

sys.modules.setdefault('h5py', types.ModuleType('h5py'))   # paper_errors imports it for its CLI only
sys.path.insert(0, str(config.REPO / 'training'))
from paper_errors import logratio_metrics, ROWS  # noqa: E402

#: The six diagnostics (test-set names), in the paper's order, and their labels.
SIX = ('n_prthst', 'm_chmprthst', 'ccn_cdf', 'logfrznfrac_tmp', 'logqs_pop', 'logqa_pop')
LABELS = {'n_prthst': 'Number distribution', 'm_chmprthst': 'Speciated mass distribution',
          'ccn_cdf': 'CCN spectrum', 'logfrznfrac_tmp': 'Frozen fraction',
          'logqs_pop': 'Scattering', 'logqa_pop': 'Absorption'}
SHORT = {'n_prthst': 'Number', 'm_chmprthst': 'Speciated mass', 'ccn_cdf': 'CCN',
         'logfrznfrac_tmp': 'Frozen', 'logqs_pop': 'Scattering', 'logqa_pop': 'Absorption'}
TERMS = ('offmixture', 'fraction', 'mixture', 'reconstruction')
GROUPS = ('transport', 'decoder', 'total')
TERM_LABELS = {'offmixture': 'Off-mixture error', 'fraction': 'Fraction error', 'mixture': 'Averaging error',
               'reconstruction': 'Reconstruction error', 'transport': 'Transport error',
               'decoder': 'Representation error', 'total': 'Overall error'}
_PAIRS = {'offmixture': (4, 3), 'fraction': (3, 2), 'mixture': (2, 1), 'reconstruction': (1, 0),
          'transport': (4, 2), 'decoder': (2, 0), 'total': (4, 0)}

#: The decoder's raw per-particle blocks (training/affx_mixk.py BLOCKS): name, size, shape, extensive.
BLOCKS = (('m_chmprthst', 225, (15, 15), True), ('n_prthst', 15, (1, 15), True),
          ('ccn_cdf', 100, (1, 100), False), ('qs_pop', 8, (1, 8), True),
          ('qa_pop', 8, (1, 8), True), ('frznfrac_tmp', 100, (1, 100), False))
LOG10_BIN = 0.25
ALT_12H = config.REPO / 'data' / 'wrfml_runs' / 'alt_surface_12h.npz'


def units(x, n):
    """Raw per-particle vectors x (..., 456) and number per m3 n (...) -> the test set's
    diagnostic arrays {name: (..., C, L)} (training/affx_mixk.py units)."""
    out, o = {}, 0
    for name, size, shape, ext in BLOCKS:
        v = x[..., o:o + size].reshape(*x.shape[:-1], *shape)
        out[name] = v * n[..., None, None] if ext else v
        o += size
    return out


def errors(xa, na, xb, nb):
    """Per-population errors {six name: (...)} between two sets of raw vectors."""
    e = logratio_metrics(units(xa, na), units(xb, nb))
    return {ROWS[k]: v for k, v in e.items() if ROWS[k] in SIX}


def source_vectors(pop, kind):
    """The three source populations' raw per-particle vectors (3, 456) from the populations
    product: kind 'pmc' (particle-resolved) or 'ml' (decoded by the product's model)."""
    nt = pop[kind + '_ntot'].values.astype(float)[:, None]
    per = lambda v: pop['%s_%s' % (kind, v)].values.astype(float)
    m = per('dMdlogDp') / nt[:, :, None] * LOG10_BIN                 # (3, species, bin)
    parts = dict(m_chmprthst=m.reshape(3, -1), n_prthst=per('dNdlogDp') / nt * LOG10_BIN,
                 ccn_cdf=per('ccn_cdf'), qs_pop=per('scattering') / nt, qa_pop=per('absorption') / nt,
                 frznfrac_tmp=per('frozen_fraction'))
    return np.concatenate([parts[name] for name, *_ in BLOCKS], axis=1)


def alt_surface(hours):
    """The lowest level's ALT (m3 kg-1) at `hours` of the 12-hour WRF run (y, x)."""
    with np.load(ALT_12H) as f:
        alt, times = f['alt'], f['times']
    h = [(np.datetime64(str(t).replace('_', 'T')) - np.datetime64(str(times[0])[:10] + 'T00:00:00'))
         / np.timedelta64(1, 'h') for t in times]
    return alt[int(np.flatnonzero(np.isclose(h, hours))[0])].astype(float)


def decode_cells(name, z, model_cache={}):
    """Raw per-particle vectors (n, 456) for latents z (n, d), model <recipe>.<seed>."""
    from extract import affx
    if name not in model_cache:
        model_cache[name] = affx.model(name)
    return model_cache[name].decode(np.asarray(z, dtype=float))


def decompose(model_key, run, hours=12.0, keys=None, per_cell=False):
    """The four-term decomposition of every scheme in replay `run` at `hours`, for one model
    (registry key, e.g. twin_k16_of.6.0), over the mixed cells of per-source transport.

    Returns {'mixed': n, 'schemes': {key: {term or group: {six name: mean e, 'overall': mean}}}};
    the decoder's terms are under every scheme alike. per_cell=True adds the per-cell errors
    ('cells': {key: {term: {name: (n_mixed,)}}}) and the mask ('mask': (y, x))."""
    from . import transport_schemes as T
    s = T.load(run)
    it = T.time_index(s, hours)
    rm = T.run_model(model_key)
    name = model_key.rsplit('.', 1)[0]
    pop = products.load('populations', model=model_key)
    z_star = pop['z'].values.astype(float)                             # (3, d)
    f_O = T.fractions(s, 'oracle', it)                                 # (3, y, x)
    n_O = T.number(s, 'oracle', it)
    mask = (f_O >= analysis.MIX_MIN_FRAC).sum(0) >= 2
    rho = 1.0 / alt_surface(hours)[mask]                               # kg m-3
    fm = f_O[:, mask].T                                                # (n, 3)
    nO = n_O[mask] * rho
    x_pr, x_ml = source_vectors(pop, 'pmc'), decode_cells(name, z_star)
    Q = {0: (fm @ x_pr, nO), 1: (fm @ x_ml, nO), 2: (decode_cells(name, fm @ z_star), nO)}
    dec = {t: errors(*Q[a], *Q[b]) for t, (a, b) in _PAIRS.items() if t in ('mixture', 'reconstruction', 'decoder')}
    out, cells = {}, {}
    for key in keys or T.scheme_keys(s, rm):
        z = T.latent(s, key, z_star, rm, it)[:, mask].T                # (n, d)
        nS = T.number(s, key, it)[mask] * rho
        f, _ = mixing.implied_fractions(z.T[:, :, None], z_star)
        zb = mixing.f_bar(f)[:, :, 0].T @ z_star
        Qs = {**Q, 3: (decode_cells(name, zb), nS), 4: (decode_cells(name, z), nS)}
        e = dict(dec)
        for t, (a, b) in _PAIRS.items():
            if t not in e:
                e[t] = errors(*Qs[a], *Qs[b])
        out[key] = {t: dict({k: float(np.mean(v[k])) for k in SIX},
                            overall=float(np.mean([np.mean(v[k]) for k in SIX]))) for t, v in e.items()}
        if per_cell:
            cells[key] = e
    res = {'model': model_key, 'run': run, 'hours': hours, 'mixed': int(mask.sum()), 'schemes': out}
    if per_cell:
        res.update(cells=cells, mask=mask)
    return res
