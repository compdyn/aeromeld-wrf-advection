"""The exact answer of the passive case, the paper's reference (since 2026-11-12).

In this case every cell holds the three source populations, mixed in the
proportions that per-source transport gives, so its diagnostics are the
sources' particle-resolved diagnostics mixed in those proportions
(transport_schemes.exact_reference; paper Eq. eq:exact). The sources are the
three 1000-particle files (the populations product's pmc_*), and the
proportions and number come from the per-source ("oracle") tracers of the
same offline replay that carried the model (data/transport/<replay>), so the
reference and the model see the same winds.

This module returns that reference in the layout of the products it replaces
(partmc_diag_domain, partmc_diag_points, partmc_fields' source fractions), so
the figure scripts and analysis.compute only change where they load it.
WRF-PartMC is no longer used: in a passive case it is a Monte Carlo estimate of
this reference (notes/paper_outline.md, 2026-11-12).

Only replay runs have a reference (their wrf_dir is .../transport/runs/<name>).
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import analysis, common, config, products
from . import transport_schemes as T

#: The three populations, in the populations product's order (background,
#: urban, emission), as per-source transport groups its classes (ORACLE_GROUPS).
SCENARIOS = (561, 635, 812)
START = datetime(2010, 6, 17, 0, 0, 0)


def replay(run):
    """The replay of a registry Run: its name under data/transport, loaded."""
    p = Path(run.wrf_dir)
    if p.parent.name != 'runs' or p.parent.parent.name != 'transport':
        raise ValueError('%s is not an offline replay (wrf_dir %s); only replays have the exact reference'
                         % (run.key, run.wrf_dir))
    return T.load(p.name)


def timestamps(s):
    return [(START + timedelta(seconds=float(t))).strftime('%Y-%m-%d_%H:%M:%S') for t in s['time_s']]


def _index(s, timestamp):
    ts = timestamps(s)
    if timestamp not in ts:
        raise KeyError('%s is not an output time of the replay' % timestamp)
    return ts.index(timestamp)


def _sources(run):
    pop = products.load('populations', model=run.model.key)
    if tuple(int(x) for x in pop['scenario'].values) != SCENARIOS:
        raise ValueError('populations product scenarios %s, expected %s' % (pop['scenario'].values, SCENARIOS))
    return pop


def fractions(run, timestamp=config.TIMESTAMP_FINAL, s=None):
    """Per-source number fractions (3, south_north, west_east)."""
    s = replay(run) if s is None else s
    return T.fractions(s, 'oracle', _index(s, timestamp))


def mixed(run, timestamp=config.TIMESTAMP_FINAL, s=None):
    """The mixed cells: at least two populations each hold analysis.MIX_MIN_FRAC
    of the number (per-source transport)."""
    return (fractions(run, timestamp, s) >= analysis.MIX_MIN_FRAC).sum(0) >= 2


def number(run, timestamp=config.TIMESTAMP_FINAL, s=None):
    """Total number mixing ratio (kg^-1) of per-source transport."""
    s = replay(run) if s is None else s
    return T.number(s, 'oracle', _index(s, timestamp))


def domain(run, timestamp=config.TIMESTAMP_FINAL):
    """The exact answer on the lowest level, as a partmc_diag_domain-like
    dataset (ccn_cdf, frozen_fraction, scattering, absorption per kg, ntot)."""
    s = replay(run)
    pop = _sources(run)
    ds = T.exact_reference(pop, fractions(run, timestamp, s), number(run, timestamp, s), timestamp)
    ds.attrs['reference'] = 'exact: per-source transport (replay %s) x the sources\' particle diagnostics' \
        % Path(run.wrf_dir).name
    return ds


def points(run):
    """The exact answer at the sampling points at every output time, as a
    partmc_diag_points-like dataset: ccn_cdf, frozen_fraction, scattering and
    absorption (per kg), dNdlogDp and dMdlogDp (per kg, wet diameter, the
    decoder's bins and species), ntot, and the per-source fractions f_source."""
    import xarray as xr
    s = replay(run)
    pop = _sources(run)
    ts = timestamps(s)
    pts = [common.point_index(p) for p in config.LOCATIONS]           # 0-based (j, i)
    n_src = pop['pmc_ntot'].values.astype(float)
    per = lambda v: pop['pmc_' + v].values.astype(float)            # (3, ...)
    f = np.stack([[T.fractions(s, 'oracle', it)[:, j, i] for (j, i) in pts] for it in range(len(ts))], 1)  # (P, T, 3)
    n = np.stack([[T.number(s, 'oracle', it)[j, i] for (j, i) in pts] for it in range(len(ts))], 1)       # (P, T)
    mix = lambda a: np.einsum('pts,s...->pt...', f, a)
    data = {
        'ccn_cdf': (('point', 'time', 'eps'), mix(per('ccn_cdf'))),
        'frozen_fraction': (('point', 'time', 'temperature'), mix(per('frozen_fraction'))),
        'scattering': (('point', 'time', 'wavelength'), n[..., None] * mix(per('scattering') / n_src[:, None])),
        'absorption': (('point', 'time', 'wavelength'), n[..., None] * mix(per('absorption') / n_src[:, None])),
        'dNdlogDp': (('point', 'time', 'bin'), n[..., None] * mix(per('dNdlogDp') / n_src[:, None])),
        'dMdlogDp': (('point', 'time', 'species', 'bin'),
                     n[..., None, None] * mix(per('dMdlogDp') / n_src[:, None, None])),
        'ntot': (('point', 'time'), n),
        'f_source': (('point', 'time', 'source'), f),
    }
    coords = {c: pop[c] for c in ('eps', 'temperature', 'wavelength', 'species', 'log10_d_lo', 'log10_d_hi')}
    coords.update(point=('point', list(config.LOC_LABELS)), time=('time', ts),
                  hours=('time', np.asarray(s['time_s'], float) / 3600.0),
                  point_j=('point', [p[0] for p in pts]), point_i=('point', [p[1] for p in pts]),
                  source=('source', list(SCENARIOS)))
    return xr.Dataset(data, coords=coords, attrs={'reference': 'exact', 'replay': Path(run.wrf_dir).name})
