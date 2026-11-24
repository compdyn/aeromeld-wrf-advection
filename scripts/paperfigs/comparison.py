"""The comparison of WRF-AeroMELD with sectional and modal schemes (paper Section 4.1): which
schemes, their names, styles and tracer counts, and the loaders of their numbers.

The numbers come from the baseline study (baselines/, REPORT.md there), in the size basis's
directory (BASIS below; the paper's: baselines/data/drydiam/):
  transport_terms.json   case -> '12h' -> term -> diagnostic: mean error over the mixed cells at
                         12 h (fractions), for the 12-h transport runs (separate limiters; the
                         schemes' state-driven adjustments after every step); 'aeromeld' is
                         WRF-AeroMELD at its representative seed.
  kcurves_baselines.json the same test mixtures as the decoders' K-curves (config.MIXK), in their
                         layout (rep -> 'K' -> K -> [per split seed]).
  baselines/style.py TRACERS            advected tracers per grid cell.
"""
import json
import os
import sys

import numpy as np

from . import config

BASELINES = config.REPO / 'baselines'
#: The numbers are those of the paper's size basis (config.SIZE_BASIS; baselines/diag.BASES): since
#: 2026-11-19 'drydiam', the two size distributions binned by dry diameter with every species' mass,
#: water included, for WRF-AeroMELD (round 39) and the schemes alike (static_terms.py,
#: transport_terms.py, kcurves.py with AEROMELD_SIZE_BASIS=drydiam). PAPERFIGS_BASIS=wet or dry reads
#: an earlier basis's numbers instead (wet: the round-33 models; dry: round 37, water left out of the
#: mass); the figures then get a _<basis> suffix.
BASIS = os.environ.get('PAPERFIGS_BASIS', config.SIZE_BASIS)
#: wet and dry: the files of those rebuilds (data/*.json, data/*_dry.json); drydiam and later: the
#: basis's own directory (static_terms.OUT: data/drydiam/), written by transport_terms.py --merge
#: and kcurves.py merge.
TRANSPORT_TERMS, KCURVES = {
    'wet': (BASELINES / 'data' / 'transport_terms.json', BASELINES / 'data' / 'kcurves_baselines.json'),
    'dry': (BASELINES / 'data' / 'transport_terms_dry.json', BASELINES / 'data' / 'kcurves_baselines_dry.json'),
    'drydiam': (BASELINES / 'data' / 'drydiam' / 'transport_terms.json',
                BASELINES / 'data' / 'drydiam' / 'kcurves_baselines.json')}[BASIS]
SUFFIX = '' if BASIS == config.SIZE_BASIS else '_' + BASIS
#: PAPERFIGS_DRYSRC=1 (2026-11-19, the dry-sources study; baselines/drysrc.py): the three source
#: populations without their water, every scheme scored from them (transport_terms_drysrc.json); the
#: sectional and modal schemes then advect no water tracers. The figures get a _drysrc suffix.
DRYSRC = os.environ.get('PAPERFIGS_DRYSRC', '0') == '1'
if DRYSRC:
    # the study scored the round-37 decoders and the dry basis (water left out of the mass)
    assert BASIS == 'dry', 'PAPERFIGS_DRYSRC=1 needs PAPERFIGS_BASIS=dry (the study was of round 37)'
    TRANSPORT_TERMS = BASELINES / 'data' / 'transport_terms_drysrc.json'
    SUFFIX += '_drysrc'
#: The water tracers of each scheme (MOSAIC 8: one per bin; TOMAS-40: AW per bin; MAM4: one per mode;
#: MATRIX: total H2O), left out of the tracer counts when the sources are dry.
WATER_TRACERS = {'sec8': 8, 'tomas40': 40, 'mam4l': 4, 'matrix1': 1}

#: The schemes in the paper's order: key -> (name, family, colour, marker, line style).
#: Sectional browns, modal purples, WRF-AeroMELD its decoder's green (AFFX_STYLES).
SCHEMES = {
    'aeromeld': ('WRF-AeroMELD', None, '#238b45', '^', '-'),
    'sec8': ('MOSAIC 8', 'Sectional', '#bf812d', 's', '--'),
    'tomas40': ('TOMAS-40', 'Sectional', '#8c510a', 'p', '--'),
    'mam4l': ('MAM4', 'Modal', '#542788', 'D', '-.'),
    'matrix1': ('MATRIX', 'Modal', '#8073ac', 'h', '-.'),
}
#: The overall error without the frozen fraction: the mean of the other five diagnostics.
NO_FROZEN = 'no_frozen'


def tracers():
    sys.path.insert(0, str(BASELINES))
    from style import TRACERS
    if DRYSRC:
        return {k: v - WATER_TRACERS.get(k, 0) for k, v in TRACERS.items()}
    return TRACERS


def transport_terms():
    return json.loads(TRANSPORT_TERMS.read_text())['cases']


def present(keys=tuple(SCHEMES)):
    """The schemes with a transport run, in order (the later ones may not exist yet)."""
    tt = transport_terms()
    return [k for k in keys if k in tt]


def error(case, key='overall', term='total'):
    """Mean error over the mixed cells at 12 h (fraction) of a case: a diagnostic, 'overall' or
    NO_FROZEN, of a term ('total', 'transport', 'representation', ...)."""
    from .errors import SIX
    d = transport_terms()[case]['12h'][term]
    if key == NO_FROZEN:
        return float(np.mean([d[k] for k in SIX if k != 'logfrznfrac_tmp']))
    return float(d[key])
