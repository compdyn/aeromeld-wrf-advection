"""AeroMELD-vs-reference error statistics, shared by the violin figure and the
metrics table so the two can never disagree. numpy only.

The reference is the exact answer of the passive case (paperfigs.reference,
since 2026-11-12); before that it was WRF-PartMC. The second argument of the
functions below, and the 'pmc' key of their results, are the reference
whatever it is (the key name is historical).

The error measure is the symmetric relative error

    e = |ML - PMC| / ((|ML| + |PMC|) / 2)          in [0, 2]

For positive values it depends only on the ratio r = ML / PMC,
e = 2 |r - 1| / (r + 1), so it reads as a factor: 0.01 ~ 1%, 0.1 ~ 10%,
0.67 = 2x, 1 = 3x, 1.64 = 10x. It is bounded and treats over- and undershoots
alike, which a plain relative error does not (frozen fraction spans eight
decades between -40 and 0 C). It saturates toward 2 and drops the sign, so
order-of-magnitude misses are ranked rather than measured: quote the ratio
statistics alongside it (they are in every row).

SMAPE is the mean of e over the cells, in percent.

Everything here is numpy, except compute(), which loads the three products it
needs (numpy + xarray; never torch).
"""
import numpy as np

from . import common, products

#: Conditions, left to right: CCN/CN at four s_env (fractions), optics at two
#: wavelengths present on BOTH grids exactly (nm), frozen fraction at four
#: temperatures (deg C, snapped to the 0.4 C grid).
SMAPE_SAT = (0.001, 0.003, 0.006, 0.01)
SMAPE_WL_NM = (400, 1000)
SMAPE_TEMP_C = (-40, -30, -20, -10)

#: log10 needs e > 0; anything below is clipped to this (and counted).
E_FLOOR = 1e-6

#: A cell is MIXED when at least two populations (scenarios, not sources) each
#: hold this number fraction of its PartMC number. Populations, not sources: the
#: BCs feed in the same scenario 561 as the IC background, so a cell holding only
#: those two is not a mixture.
MIX_MIN_FRAC = 0.05

GROUPS = (
    ('ccn', 'CCN/CN fraction\n' r'at $s_{\rm env}$ (%)', 'CCN/CN fraction at s_env (%)'),
    ('scattering', 'Scattering\n' r'at $\lambda$ (nm)', 'Scattering at lambda (nm)'),
    ('absorption', 'Absorption\n' r'at $\lambda$ (nm)', 'Absorption at lambda (nm)'),
    ('frozen_fraction', 'Frozen fraction\n' r'at $T$ ($^{\circ}$C)', 'Frozen fraction at T (deg C)'),
)


def sym_rel_err(ml, pmc):
    """Per-cell symmetric relative error. 0 where both sides are exactly 0 (a
    match, not 0/0); NaN propagates, so an uncovered cell drops out."""
    den = 0.5 * (np.abs(ml) + np.abs(pmc))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den == 0, 0.0, np.abs(ml - pmc) / den)


def mixed_cells(fields, timestamp):
    """(mask (south_north, west_east), populations) from partmc_fields."""
    src = fields['n_source'].sel(instant=timestamp).values
    scen = fields['source_scenario'].values
    pops = np.unique(scen)
    n_pop = np.stack([src[scen == p].sum(0) for p in pops])
    with np.errstate(invalid='ignore', divide='ignore'):
        mixed = (n_pop / n_pop.sum(0) >= MIX_MIN_FRAC).sum(0) >= 2
    return mixed, pops


def conditions(ml, pmc):
    """[{group, label, label_plain, tick, condition, ml, pmc}, ...] -- one entry
    per compared quantity, as (south_north, west_east) float64 fields, from an
    aeromeld_domain and a partmc_diag_domain product at the same instant."""
    if ml.attrs['timestamp'] != pmc.attrs['timestamp']:
        raise ValueError('AeroMELD at %s, PartMC at %s'
                         % (ml.attrs['timestamp'], pmc.attrs['timestamp']))
    f64 = lambda a: np.asarray(a, dtype=float)
    temp = pmc['temperature'].values
    if not np.allclose(temp, ml['temperature'].values, atol=1e-4):
        raise ValueError('the two temperature grids differ')
    out = []

    def add(g, tick, cond, a, b):
        key, label, plain = GROUPS[g]
        out.append(dict(group=key, label=label, label_plain=plain, tick=tick,
                        condition=cond, ml=f64(a), pmc=f64(b)))

    for sat in SMAPE_SAT:
        add(0, '%.1f' % (100 * sat), 100 * sat,
            common.interp_along_last_axis(f64(ml['ccn_cdf']), ml['eps'].values, sat),
            common.interp_along_last_axis(f64(pmc['ccn_cdf']), pmc['eps'].values, sat))
    for g, var in ((1, 'scattering'), (2, 'absorption')):
        for wl in SMAPE_WL_NM:
            i_ml = common.exact_index(ml['wavelength'].values * 1e9, wl, 'wavelength (nm)')
            i_pmc = common.exact_index(pmc['wavelength'].values * 1e9, wl, 'wavelength (nm)')
            add(g, '%d' % wl, wl, ml[var].values[..., i_ml], pmc[var].values[..., i_pmc])
    for T in SMAPE_TEMP_C:
        i_T = int(np.argmin(np.abs(temp - T)))
        add(3, '$%d$' % round(float(temp[i_T])), round(float(temp[i_T]), 2),
            ml['frozen_fraction'].values[..., i_T], pmc['frozen_fraction'].values[..., i_T])
    return out


#: Spectrum errors: one error per diagnostic, averaged over every grid point
#: the two models share within the per-point conditions' ranges: 50
#: log-spaced s_env from 0.1 to 1 %, 75 temperatures from -40 to -10 C (the
#: 0.4 C grid), and the 4 wavelengths on both grids (300, 400, 600, 1000 nm).
SPECTRUM_SAT = (0.001, 0.01)
SPECTRUM_TEMP_C = (-40, -10)
#: (group, name, span): matplotlib labels, so plain text plus mathtext.
SPECTRUM_GROUPS = (
    ('ccn', 'CCN/CN', '0.1–1%'),
    ('scattering', 'Scattering', '300–1000 nm'),
    ('absorption', 'Absorption', '300–1000 nm'),
    ('frozen_fraction', 'Frozen fraction', '−40 to −10 °C'),
)


def spectra(ml, pmc):
    """[{group, name, span, axis, ml, pmc}, ...] -- one entry per diagnostic, its
    values at the shared sample points as (..., points) float64 arrays, from an
    aeromeld_domain and a partmc_diag_domain product (or any pair of datasets
    with the same variables and grids; the populations product's sources work
    too). Optics stay on the products' own basis."""
    if ml.attrs.get('timestamp') != pmc.attrs.get('timestamp'):
        raise ValueError('AeroMELD at %s, PartMC at %s'
                         % (ml.attrs.get('timestamp'), pmc.attrs.get('timestamp')))
    f64 = lambda a: np.asarray(a, dtype=float)
    eps = ml['eps'].values
    if not np.allclose(eps, pmc['eps'].values, rtol=1e-6):
        raise ValueError('the two supersaturation grids differ')
    temp = pmc['temperature'].values
    if not np.allclose(temp, ml['temperature'].values, atol=1e-4):
        raise ValueError('the two temperature grids differ')
    i_eps = np.flatnonzero((eps >= SPECTRUM_SAT[0] * (1 - 1e-6)) & (eps <= SPECTRUM_SAT[1] * (1 + 1e-6)))
    i_T = np.flatnonzero((temp >= SPECTRUM_TEMP_C[0] - 1e-3) & (temp <= SPECTRUM_TEMP_C[1] + 1e-3))
    wl_ml, wl_pmc = ml['wavelength'].values * 1e9, pmc['wavelength'].values * 1e9
    shared = [w for w in wl_pmc if np.any(np.isclose(wl_ml, w))]
    j_ml = [common.exact_index(wl_ml, w, 'wavelength (nm)') for w in shared]
    j_pmc = [common.exact_index(wl_pmc, w, 'wavelength (nm)') for w in shared]
    out = []
    for group, name, span in SPECTRUM_GROUPS:
        if group == 'ccn':
            axis, a, b = 100 * eps[i_eps], ml['ccn_cdf'].values[..., i_eps], pmc['ccn_cdf'].values[..., i_eps]
        elif group == 'frozen_fraction':
            axis = temp[i_T]
            a, b = ml['frozen_fraction'].values[..., i_T], pmc['frozen_fraction'].values[..., i_T]
        else:
            axis, a, b = np.asarray(shared), ml[group].values[..., j_ml], pmc[group].values[..., j_pmc]
        out.append(dict(group=group, name=name, span=span, axis=axis, ml=f64(a), pmc=f64(b)))
    return out


def spectrum_error(ml_values, pmc_values):
    """Each cell's spectrum error: the mean over the sample points (last axis)
    of the symmetric relative error e."""
    return sym_rel_err(ml_values, pmc_values).mean(-1)


def layout(conds):
    """Violin x positions with a one-slot gap between groups, and each group's
    (label, first x, last x) span."""
    positions, spans, x, prev = [], [], 0, None
    for c in conds:
        if prev is not None and c['group'] != prev:
            x += 1
        if c['group'] != prev:
            spans.append([c['label'], x, x])
            prev = c['group']
        positions.append(x)
        spans[-1][2] = x
        x += 1
    return positions, [tuple(s) for s in spans]


def condition_stats(c, mixed, plotted):
    """Statistics of one condition. `plotted` selects the cells the violin,
    SMAPE and the ratio statistics describe (the mixed cells, or every cell);
    the unmixed and every-cell SMAPEs, and the *_all ratio statistics, are
    reported regardless.

    The ratio r = ML / PMC is taken wherever PartMC is positive, and KEEPS its
    sign: a cell where AeroMELD decodes to zero or below (the affine decoders
    in physical units do, in up to half the mixed cells) ranks lowest instead of
    dropping out, so the percentiles cannot improve by losing the worst cells.
    n_nonpositive counts those cells, n_pmc_nonpositive the cells with no ratio.
    (Until 2026-11-08 the ratio statistics covered every cell whatever
    `plotted` said, and dropped non-positive AeroMELD values.)"""
    e_all = sym_rel_err(c['ml'], c['pmc'])
    fin = np.isfinite(e_all)
    e = e_all[plotted & fin]
    ok = np.isfinite(c['ml']) & np.isfinite(c['pmc'])
    has_r = ok & (c['pmc'] > 0)
    r_all = c['ml'][has_r] / c['pmc'][has_r]
    r = c['ml'][has_r & plotted] / c['pmc'][has_r & plotted]
    return dict(
        n_cells=int(e.size),
        smape_pct=100 * float(e.mean()),
        e_p05=float(np.percentile(e, 5)), e_median=float(np.percentile(e, 50)),
        e_p95=float(np.percentile(e, 95)),
        n_below_floor=int((e < E_FLOOR).sum()),
        smape_unmixed_pct=100 * float(e_all[~mixed & fin].mean()),
        smape_all_pct=100 * float(e_all[fin].mean()),
        ratio_cells=int(r.size),
        ratio_p05=float(np.percentile(r, 5)), ratio_median=float(np.percentile(r, 50)),
        ratio_p95=float(np.percentile(r, 95)),
        pct_cells_ml_high=100 * float(np.mean(r > 1)),
        n_nonpositive=int((r <= 0).sum()),
        n_pmc_nonpositive=int((ok & plotted & ~has_r).sum()),
        ratio_median_all=float(np.percentile(r_all, 50)),
        pct_cells_ml_high_all=100 * float(np.mean(r_all > 1)),
        log10_e=np.log10(np.maximum(e, E_FLOOR)),
    )


def compute(run, mixed_only=True):
    """(conditions, stats per condition, mixed mask, populations) for a registry
    Run against the exact answer (paperfigs.reference): the one computation
    behind both the violin figure and data/metrics.csv. `mixed_only=False`
    makes the violins and SMAPE cover every cell."""
    from . import reference
    ml = products.load('aeromeld_domain', run=run.key)
    ref = reference.domain(run, ml.attrs['timestamp'])
    ts = common.same_instant(ml, ref)
    conds = conditions(ml, ref)
    mixed, pops = reference.mixed(run, ts), np.array(reference.SCENARIOS)
    plotted = mixed if mixed_only else np.ones_like(mixed)
    stats = [condition_stats(c, mixed, plotted) for c in conds]
    return conds, stats, mixed, pops
