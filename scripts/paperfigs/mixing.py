"""Source mixing in latent space, and the attribution of WRF-AeroMELD's error.

Transport-only physics makes every surface cell a mixture of the three source
populations (config.POP_SCENARIOS). Each model gives its own source fractions
(the paper's f^PR and f^ML):

  f_pr  WRF-PartMC's: number fractions by population, from its source tags
        (partmc_fields).
  f_ml  WRF-AeroMELD's: the barycentric coordinates of the advected latent
        vector z_i in the triangle spanned by the three source latents z*_s
        (populations_<model>), by least squares on the triangle's plane.

Latent transport mixes exactly: z_i lies in that plane (out_of_plane <= 1.3e-6
of the mean edge length for the main run, i.e. float32 roundoff) and
0 <= f_ml <= 1 in every cell.

Attribution, per spectrum (analysis.spectra: CCN/CN, scattering, absorption,
frozen fraction, each at its shared sample points). For a per-particle value q
(optical coefficients divided by the number), write qML(f) = sum_s f_s q*ML_s
and qPR(f) = sum_s f_s q*PR_s for the sources mixed in proportions f (q*ML_s
decoded sources, q*PR_s their particle values; populations_<model>). Changing
one ingredient at a time,

  log10(qML / qPR) =  log10(qML / qML(f_ml))           decoding mixtures
                    + log10(qML(f_ml) / qML(f_pr))     transport (f_ml vs f_pr)
                    + log10(qML(f_pr) / qPR(f_pr))     end-member reconstruction
                    + log10(qPR(f_pr) / qPR)           benchmark sampling

exactly, at every sample point; the per-kg optics add log10(NML / NPR), the
number. A cell's spectrum term is the mean of its point terms over the
spectrum, so the terms still add up exactly. Log terms are NaN where a value
is not positive (affine decoders decode negative values); such points are left
out of that cell's mean for every term alike. No further decoding is needed.
"""
import numpy as np

from . import analysis, common, config, products

TERMS = ('mixing', 'transport', 'endmember', 'sampling', 'number')
TERM_LABELS = {
    'offmanifold': 'Off the mixtures',
    'mixing': 'Decoding mixtures',
    'transport': 'Transport',
    'endmember': 'End-member reconstruction',
    'sampling': 'Benchmark sampling',
    'number': 'Number',
    'total': 'Total',
}
#: Spectra whose values are per kg of dry air in the domain products.
EXTENSIVE = ('scattering', 'absorption')


def source_fractions(fields, timestamp):
    """f_pr: (3, south_north, west_east) WRF-PartMC number fractions by
    population, in config.POP_SCENARIOS order."""
    src = fields['n_source'].sel(instant=timestamp).values
    scen = fields['source_scenario'].values
    n = np.stack([src[scen == s].sum(0) for s in config.POP_SCENARIOS]).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return n / n.sum(0)


def implied_fractions(z, z_star):
    """(f_ml (3, ...), out_of_plane (...)): barycentric coordinates of the
    latent field z (d, ...) on the plane through the three source latents
    z_star (3, d), and each cell's distance from that plane."""
    shape = z.shape[1:]
    zf = z.reshape(z.shape[0], -1).astype(float)
    zs = np.asarray(z_star, dtype=float)
    A = np.stack([zs[1] - zs[0], zs[2] - zs[0]], axis=1)            # (d, 2)
    coef, *_ = np.linalg.lstsq(A, zf - zs[0][:, None], rcond=None)   # (2, cells)
    f_ml = np.vstack([1 - coef.sum(0), coef])
    out = np.linalg.norm(zf - (zs[0][:, None] + A @ coef), axis=0)
    return f_ml.reshape(3, *shape), out.reshape(shape)


def f_bar(f):
    """Fractions clipped to >= 0 and renormalized: the nearest mixture's."""
    c = np.clip(f, 0, None)
    return c / c.sum(0)


def edge_length(z_star):
    """Mean distance between the three source latents."""
    zs = np.asarray(z_star, dtype=float)
    return float(np.mean([np.linalg.norm(zs[a] - zs[b]) for a, b in ((0, 1), (0, 2), (1, 2))]))


def _source_spectra(pop, timestamp):
    """analysis.spectra() of the three source populations, per particle:
    {group: (ml (3, P), pmc (3, P))}."""
    import xarray as xr

    def side(prefix):
        ds = xr.Dataset(
            {'ccn_cdf': (('scenario', 'eps'), pop[prefix + '_ccn_cdf'].values),
             'frozen_fraction': (('scenario', 'temperature'), pop[prefix + '_frozen_fraction'].values),
             'scattering': (('scenario', 'wavelength'), pop[prefix + '_scattering'].values),
             'absorption': (('scenario', 'wavelength'), pop[prefix + '_absorption'].values)},
            coords={'eps': pop['eps'], 'temperature': pop['temperature'],
                    'wavelength': pop['wavelength']})
        ds.attrs['timestamp'] = timestamp
        return ds
    return side('ml'), side('pmc')


def compute(run):
    """Everything the attribution figures need, for one registry run:

    dict(f_pr, f_ml, out_of_plane (/ edge), mixed, z_star, edge, timestamp,
         spectra=[{group, name, span, axis,
                   error (per-cell spectrum error, mean e),
                   terms={term: per-cell spectrum mean of the log10 term},
                   total=per-cell spectrum mean log10 ratio}, ...])"""
    ml = products.load('aeromeld_domain', run=run.key)
    pmc = products.load('partmc_diag_domain')
    fields = products.load('partmc_fields')
    wf = products.load('wrfml_fields', run=run.key)
    pop = products.load('populations', model=run.model.key)
    ts = common.same_instant(ml, pmc)
    if wf.attrs.get('timestamp', ts) != ts:
        raise ValueError('wrfml_fields at %s, diagnostics at %s' % (wf.attrs['timestamp'], ts))
    return attribute(ml, pmc, fields, wf['z'].values, pop, ts)


def attribute(ml, pmc, fields, z, pop, ts, ml_bar=None, f_ref=None, mixed=None):
    """compute() for any latent field z (d, y, x) and its decoding ml (an
    aeromeld_domain-like dataset), e.g. an offline transport scheme's.

    ml_bar: the decoding of z_bar = sum_s f_bar_s z*_s, z's fractions clipped to
    >= 0 and renormalized (its nearest mixture). When given, the decoding term
    is split into 'offmanifold' (log10 qML(z) / qML(z_bar)) and 'mixing'
    (log10 qML(z_bar) / qML(f_bar)), and transport compares f_bar.
    f_ref: the reference fractions transport is measured against (default
    WRF-PartMC's f_pr); 'sampling' then holds log10 qPR(f_ref) / qPR.
    fields may be None when f_ref and mixed (the cells to report) are given."""
    if [int(s) for s in pop['scenario'].values] != list(config.POP_SCENARIOS):
        raise ValueError('populations product scenarios are not config.POP_SCENARIOS')
    f_pr = source_fractions(fields, ts) if fields is not None else None
    f_ref = f_pr if f_ref is None else f_ref
    z_star = pop['z'].values
    f_ml, out = implied_fractions(z, z_star)
    if ml_bar is not None:
        f_ml = f_bar(f_ml)
    edge = edge_length(z_star)
    if mixed is None:
        mixed, _ = analysis.mixed_cells(fields, ts)
    n_ml = ml['ntot'].values.astype(float)[..., None]
    n_pmc = pmc['ntot'].values.astype(float)[..., None]
    src = {s['group']: s for s in analysis.spectra(*_source_spectra(pop, ts))}
    n_src_ml = pop['ml_ntot'].values.astype(float)[:, None]
    n_src_pmc = pop['pmc_ntot'].values.astype(float)[:, None]

    bar = {s['group']: s['ml'] for s in analysis.spectra(ml_bar, pmc)} if ml_bar is not None else {}
    out_spectra = []
    for sp in analysis.spectra(ml, pmc):
        g = sp['group']
        q_ml, q_pr = sp['ml'], sp['pmc']                         # (y, x, P)
        q_bar = bar.get(g)
        # The sources' own grids can be finer (all 8 wavelengths on both sides):
        # take them at the domain's sample points, exactly.
        j = [int(np.flatnonzero(np.isclose(src[g]['axis'], a, rtol=1e-6))[0]) for a in sp['axis']]
        s_ml, s_pr = src[g]['ml'][:, j], src[g]['pmc'][:, j]     # (3, P)
        error = analysis.spectrum_error(q_ml, q_pr)
        number = np.zeros(q_ml.shape[:2])
        if g in EXTENSIVE:                                       # per kg -> per particle
            q_ml, q_pr = q_ml / n_ml, q_pr / n_pmc
            q_bar = None if q_bar is None else q_bar / n_ml
            s_ml, s_pr = s_ml / n_src_ml, s_pr / n_src_pmc
            with np.errstate(divide='ignore', invalid='ignore'):
                number = np.log10(n_ml[..., 0] / n_pmc[..., 0])
        ml_of_f_ml = np.einsum('sp,syx->yxp', s_ml, f_ml)      # qML(f_ml)
        ml_of_f_pr = np.einsum('sp,syx->yxp', s_ml, f_ref)     # qML(f_pr)
        pr_of_f_pr = np.einsum('sp,syx->yxp', s_pr, f_ref)     # qPR(f_pr)
        with np.errstate(divide='ignore', invalid='ignore'):
            lg = lambda a, b: np.where((a > 0) & (b > 0), np.log10(a / b), np.nan)
            pts = dict(mixing=lg(q_ml, ml_of_f_ml), transport=lg(ml_of_f_ml, ml_of_f_pr),
                       endmember=lg(ml_of_f_pr, pr_of_f_pr), sampling=lg(pr_of_f_pr, q_pr))
            if q_bar is not None:
                pts['offmanifold'] = lg(q_ml, q_bar)
                pts['mixing'] = lg(q_bar, ml_of_f_ml)
        ok = np.all([np.isfinite(v) for v in pts.values()], axis=0)          # (y, x, P)
        with np.errstate(invalid='ignore'):
            n_ok = ok.sum(-1)
            mean = lambda v: np.where(n_ok > 0, np.where(ok, v, 0).sum(-1) / n_ok, np.nan)
            terms = {t: mean(v) for t, v in pts.items()}
        terms['number'] = number
        total = sum(terms[t] for t in terms)
        out_spectra.append(dict(group=g, name=sp['name'], span=sp['span'], axis=sp['axis'],
                                error=error, terms=terms, total=total,
                                frac_points_kept=float(ok[mixed].mean())))
    return dict(f_pr=f_pr, f_ml=f_ml, out_of_plane=out / edge, mixed=mixed, z_star=z_star, edge=edge,
                spectra=out_spectra, timestamp=ts)
