"""The offline latent-transport schemes (transport/), as the paper's analysis sees them.

transport/run.py replays the CARES case's aerosol transport offline on WRF's own
1-minute averaged mass fluxes, for every scheme at once, and writes the lowest
level every 15 minutes (data/transport/<run>/surface.npz). Here each scheme's
04 UTC state becomes a latent field and a number field, is decoded with a
model, and is attributed like the WRF runs (mixing.attribute), with the
decoding term split into "off the mixtures" (the latent is not a mixture of
the sources) and "decoding mixtures".

Scheme keys (transport/schemes.py): 'oracle' (each WRF-PartMC source its own
tracer and limiter), 'donor' (WRF-AeroMELD's), 'shN', 'shB:<model>',
'shB:frac', 'sep:<model>'.
"""
import json

import numpy as np

from . import analysis, config, mixing, products

DATA = config.REPO / 'data' / 'transport'
#: The oracle's five classes -> the three populations (IC + BC background, urban, emissions 1 + 2).
ORACLE_GROUPS = ((0, 2), (1,), (3, 4))
#: Model registry key -> the run's model key in transport/run.py (source latents of zsrc_of).
def run_model(model_key):
    """A registry model key (e.g. twin_k16_dry.3.0) -> its name in the replay (twin_k16)."""
    return config.base_name(model_key)


#: The mixing section's models, in its order.
MODELS = tuple(config.recipe(b) + '.0.0' for b in config.BASES)
RUN_MODEL = {k: run_model(k) for k in MODELS}


def label(key):
    base = key.split(':')[0]
    return {'oracle': 'Per source (needs source tags)',
            'donor': 'Donor-cell (WRF-AeroMELD)',
            'shN': 'Shared limiter from N',
            'shB': 'Shared limiter, bounded latent' if not key.endswith(':frac')
            else 'Shared limiter, bounded fractions (needs source tags)',
            'sep': 'Separate limiters'}[base]


def load(run='r1'):
    with np.load(DATA / run / 'surface.npz', allow_pickle=False) as f:
        s = {k: f[k] for k in f.files}
    s['settings'] = json.loads(str(s['settings']))
    return s


def time_index(s, hours):
    return int(np.flatnonzero(np.isclose(s['time_s'], hours * 3600.0))[0])


def fractions(s, key, it=-1):
    """Source fractions (3, y, x) of a group-carrying scheme (not sep)."""
    if key == 'oracle':
        n = s['oracle'][it].astype(float)
        n = np.stack([n[list(g)].sum(0) for g in ORACLE_GROUPS])
    else:
        n = s[key][it].astype(float)
    return n / n.sum(0)


def number(s, key, it=-1):
    """Number mixing ratio (y, x) the scheme carries."""
    if key == 'oracle':
        return s['oracle'][it].astype(float).sum(0)
    if key.startswith('shB'):
        return s[key][it].astype(float).sum(0)
    return s['N'][it].astype(float)            # donor, shN and sep share WRF's monotonic N


def latent(s, key, z_star, run_model, it=-1):
    """Latent field (d, y, x): the sources' latents mixed by the scheme's
    fractions, or Q / N for sep."""
    if key.startswith('sep:'):
        if key != 'sep:' + run_model:
            raise ValueError('%s was run for another model than %s' % (key, run_model))
        return s[key][it].astype(float) / s['N'][it].astype(float)[None]
    return np.tensordot(np.asarray(z_star, float).T, fractions(s, key, it), axes=1)


def scheme_keys(s, run_model):
    keys = [k for k in ('oracle', 'donor', 'shN') if k in s]
    keys += [k for k in ('shB:' + run_model, 'shB:frac', 'sep:' + run_model) if k in s]
    return keys


def evaluate(model_key, run='r1', reference='partmc', keys=None):
    """{scheme: mixing.attribute(...) result + its latent and number fields},
    for one model, at config.TIMESTAMP_FINAL. reference 'oracle' measures
    transport against the offline per-source scheme instead of WRF-PartMC."""
    from extract import affx
    s = load(run)
    it = time_index(s, 4)
    rm = run_model(model_key)
    pop = products.load('populations', model=model_key)
    pmc = products.load('partmc_diag_domain')
    fields = products.load('partmc_fields')
    ts = config.TIMESTAMP_FINAL
    z_star = pop['z'].values
    f_ref = fractions(s, 'oracle', it) if reference == 'oracle' else None
    name = model_key.rsplit('.', 1)[0]
    out = {}
    for key in keys or scheme_keys(s, rm):
        z = latent(s, key, z_star, rm, it)
        n = number(s, key, it)
        f, off = mixing.implied_fractions(z, z_star)
        zb = np.tensordot(z_star.T, mixing.f_bar(f), axes=1)
        attrs = {'timestamp': ts, 'scheme': key, 'model': model_key}
        ml = affx.decode_domain(name, z, n, attrs)
        ml_bar = affx.decode_domain(name, zb, n, attrs)
        m = mixing.attribute(ml, pmc, fields, z, pop, ts, ml_bar=ml_bar, f_ref=f_ref)
        m.update(z=z, n=n, f_raw=f, out_of_plane_raw=off, ml=ml)
        out[key] = m
    return out


def exact_reference(pop, f, n, ts):
    """The exact answer for this passive case: the three source populations'
    particle-resolved diagnostics (pop's pmc_*), mixed in the per-source
    scheme's fractions f (3, y, x), on its number n (y, x). An
    aeromeld_domain-like dataset (optics per kg, as the products)."""
    import xarray as xr
    per = lambda v: pop['pmc_' + v].values.astype(float)               # (3, P)
    mixf = lambda a: np.einsum('sp,syx->yxp', a, f)
    n_src = pop['pmc_ntot'].values.astype(float)[:, None]
    data = {'ccn_cdf': (('south_north', 'west_east', 'eps'), mixf(per('ccn_cdf'))),
            'frozen_fraction': (('south_north', 'west_east', 'temperature'), mixf(per('frozen_fraction'))),
            'scattering': (('south_north', 'west_east', 'wavelength'), n[..., None] * mixf(per('scattering') / n_src)),
            'absorption': (('south_north', 'west_east', 'wavelength'), n[..., None] * mixf(per('absorption') / n_src)),
            'ntot': (('south_north', 'west_east'), n)}
    return xr.Dataset(data, coords={c: pop[c] for c in ('eps', 'temperature', 'wavelength')},
                      attrs={'timestamp': ts, 'reference': 'per-source transport x particle-resolved sources'})


def evaluate_exact(model_key, run, keys=None, hours=4.0):
    """Like evaluate(), against the exact reference at `hours` (any output
    time): transport against the per-source scheme, no sampling term, the
    mixed cells from the per-source fractions."""
    from extract import affx
    s = load(run)
    it = time_index(s, hours)
    rm = run_model(model_key)
    pop = products.load('populations', model=model_key)
    z_star = pop['z'].values
    f_ref = fractions(s, 'oracle', it)
    ts = 't+%gh' % hours
    ref = exact_reference(pop, f_ref, number(s, 'oracle', it), ts)
    mixed = (f_ref >= analysis.MIX_MIN_FRAC).sum(0) >= 2
    name = model_key.rsplit('.', 1)[0]
    out = {}
    for key in keys or scheme_keys(s, rm):
        z = latent(s, key, z_star, rm, it)
        n = number(s, key, it)
        f, off = mixing.implied_fractions(z, z_star)
        zb = np.tensordot(z_star.T, mixing.f_bar(f), axes=1)
        attrs = {'timestamp': ts, 'scheme': key, 'model': model_key}
        ml = affx.decode_domain(name, z, n, attrs)
        ml_bar = affx.decode_domain(name, zb, n, attrs)
        m = mixing.attribute(ml, ref, None, z, pop, ts, ml_bar=ml_bar, f_ref=f_ref, mixed=mixed)
        m.update(z=z, n=n, f_raw=f, out_of_plane_raw=off)
        out[key] = m
    return out


#: The error budget against the exact answer: one term per control, adding up exactly.
BUDGET_TERMS = ('scheme', 'mixing', 'endmember')
BUDGET_LABELS = {'scheme': 'Transport scheme', 'mixing': 'Decoding mixtures', 'endmember': 'End-member',
                 'total': 'Total'}


def budget(model_key, run, keys=None, hours=4.0):
    """The error budget of each scheme against the exact answer, at one output:
    per point (log10 ratios, extensive optics per kg),

      scheme     qML(z_S, N_S) / qML(z_O, N_O)     the scheme's latent and number
                                                    against per-source transport's
      mixing     qML(z_O) / sum_s f_O,s q*ML_s     decoding the exact mixture
                                                    against mixing the decoded sources
      endmember  sum_s f_O,s q*ML_s / q_exact      decoded against particle-resolved sources

    with z_O = sum_s f_O,s z*_s, so scheme + mixing + endmember = total exactly.
    Each cell's term is the mean over the spectrum's sample points (points where
    any term is undefined are left out for all). Returns {scheme: {group:
    {'error': mean spectrum error e, 'rms': {term: RMS over mixed cells},
    'mean': {...}}}}."""
    from extract import affx
    s = load(run)
    it = time_index(s, hours)
    rm = run_model(model_key)
    pop = products.load('populations', model=model_key)
    z_star = pop['z'].values.astype(float)
    f_O = fractions(s, 'oracle', it)
    n_O = number(s, 'oracle', it)
    ts = 't+%gh' % hours
    ref = exact_reference(pop, f_O, n_O, ts)
    mixed = (f_O >= analysis.MIX_MIN_FRAC).sum(0) >= 2
    name = model_key.rsplit('.', 1)[0]
    attrs = {'timestamp': ts}
    spectra_of = lambda z, n: {sp['group']: sp for sp in analysis.spectra(affx.decode_domain(name, z, n, attrs), ref)}
    sp_O = spectra_of(np.tensordot(z_star.T, f_O, axes=1), n_O)
    src = {sp['group']: sp for sp in analysis.spectra(*mixing._source_spectra(pop, ts))}
    lin = {}
    for g, sp in sp_O.items():
        j = [int(np.flatnonzero(np.isclose(src[g]['axis'], a, rtol=1e-6))[0]) for a in sp['axis']]
        s_ml = src[g]['ml'][:, j]
        if g in mixing.EXTENSIVE:                              # per particle -> per kg on N_O
            s_ml = s_ml / pop['ml_ntot'].values.astype(float)[:, None]
            lin[g] = n_O[..., None] * np.einsum('sp,syx->yxp', s_ml, f_O)
        else:
            lin[g] = np.einsum('sp,syx->yxp', s_ml, f_O)
    lg = lambda a, b: np.where((a > 0) & (b > 0), np.log10(a / b), np.nan)
    out = {}
    for key in keys or scheme_keys(s, rm):
        sp_S = spectra_of(latent(s, key, z_star, rm, it), number(s, key, it))
        res = {}
        for g, sp in sp_S.items():
            q_S, q_O, q_ref = sp['ml'], sp_O[g]['ml'], sp['pmc']
            pts = dict(scheme=lg(q_S, q_O), mixing=lg(q_O, lin[g]), endmember=lg(lin[g], q_ref))
            ok = np.all([np.isfinite(v) for v in pts.values()], axis=0)
            n_ok = ok.sum(-1)
            with np.errstate(invalid='ignore'):
                cell = {t: np.where(n_ok > 0, np.where(ok, v, 0).sum(-1) / np.maximum(n_ok, 1), np.nan)
                        for t, v in pts.items()}
            cell['total'] = sum(cell[t] for t in BUDGET_TERMS)
            res[g] = {'error': float(np.nanmean(analysis.spectrum_error(q_S, q_ref)[mixed])),
                      'rms': {t: float(np.sqrt(np.nanmean(v[mixed] ** 2))) for t, v in cell.items()},
                      'mean': {t: float(np.nanmean(v[mixed])) for t, v in cell.items()}}
        out[key] = res
    return out


def summary(m):
    """{group: {'error': mean spectrum error e over the mixed cells,
                'rms': {term: RMS over the mixed cells}, 'mean': {...}}}."""
    mixed = m['mixed']
    out = {}
    for sp in m['spectra']:
        d = {'error': float(np.nanmean(sp['error'][mixed]))}
        for stat, fn in (('rms', lambda a: float(np.sqrt(np.nanmean(a ** 2)))), ('mean', lambda a: float(np.nanmean(a)))):
            d[stat] = {t: fn(v[mixed]) for t, v in sp['terms'].items()}
            d[stat]['total'] = fn(sp['total'][mixed])
        out[sp['group']] = d
    return out


def transport_stats(s, key, f_ref, mixed, z_star, run_model, it=-1):
    """Fraction-space measures over the mixed cells: RMS(f - f_ref) per source,
    the lowest fraction, the mean background-fraction gradient, and the
    distance of the latent from the sources' plane (/ edge)."""
    z = latent(s, key, z_star, run_model, it)
    f, off = mixing.implied_fractions(z, z_star)
    d = f - f_ref
    gy, gx = np.gradient(np.nan_to_num(f[0]))
    return dict(rms_df=[float(np.sqrt(np.nanmean(d[k][mixed] ** 2))) for k in range(3)],
                min_f=float(np.nanmin(f[:, mixed])),
                grad_bg=float(np.hypot(gx, gy)[mixed].mean()),
                off_plane=float(np.nanmax(off[mixed]) / mixing.edge_length(z_star)))


def partmc_fractions(ts=None):
    fields = products.load('partmc_fields')
    ts = ts or config.TIMESTAMP_FINAL
    return mixing.source_fractions(fields, ts), analysis.mixed_cells(fields, ts)[0]
