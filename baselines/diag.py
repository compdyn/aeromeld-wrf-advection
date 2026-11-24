"""Particle diagnostics: per-particle species masses and number weights -> the paper's raw
456-vector per population, per particle (the basis of `paperfigs.errors`).

This is the reference's own computation (`scripts/extract/populations.py`, with the vendored
`vendor/vae_partmc_utils.py` and `vendor/extract_diagnostics*.py`), batched over populations so
that thousands of grid cells can be evaluated at once:

  M  (225)  per-species mass in 15 WET-diameter bins (0.25 decade from 10^-8.75 m to 10 um),
            15 species in config.ML_SPECIES order, species-major; kg per particle.
            Wet diameter = (6/pi sum_k m_k / rho_k)^(1/3) over all 15 species incl. H2O, with
            the particle files' densities (= extract_diagnostics.DEFAULT_CHEM_DENSITY). The bins
            are vae_partmc_utils.bin_sum's (np.histogram on np.logspace edges, [lo, hi)); a
            particle outside 10^-8.75 m .. 10 um is in no bin but still counts in the total.
  N  (15)   fraction of particles in each bin.
  C  (100)  CCN/CN at the 100 supersaturations 1e-3..1e-1: vendor calc_crh + calc_ccncdf
            per particle (kappa-Koehler, 287 K, the files' kappas = DEFAULT_CHEM_KAPPA). This is
            what the twin_k16_of populations product holds (data/affx/wrf3_jeff.npz, the training
            code's values, identical to the vendor code at 287 K: notes/pmc_diagnostics_check.md).
  S  (8)    scattering cross section per particle (m^2) at 300..1000 nm: vendor get_bulkoptical
            per particle (core = BC + OIN, shell = the rest, full-diameter cross-section).
  A  (8)    absorption cross section per particle (m^2), same.
  F  (100)  frozen fraction at -40..0 C: vendor get_bulkfrznfrac per particle.

Weights are normalized per population (any scale in, per particle out). The vendor functions are
called directly (CCN, frozen fraction, and the Mie routine get_coreshell, which is elementwise),
so the numbers are the reference's; only the loop structure differs: the optics flatten every
(population, wavelength, particle) into chunks of one get_coreshell call instead of one call per
(population, wavelength) row.

Usage:
    from diag import raw_vectors
    x = raw_vectors(m, w)          # m (P, n, 15) kg, w (P, n) -> (P, 456)
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'vendor'))

import torch  # noqa: E402
import extract_diagnostics as ED  # noqa: E402

from paperfigs import config  # noqa: E402

SPECIES = tuple(config.ML_SPECIES)
NS = len(SPECIES)
SP = {s: i for i, s in enumerate(SPECIES)}
#: Densities (kg m-3): the particle files' (checked equal to ED.DEFAULT_CHEM_DENSITY).
RHO = np.array([ED.DEFAULT_CHEM_DENSITY[s] for s in SPECIES])
KAPPA = np.array([ED.DEFAULT_CHEM_KAPPA[s] for s in SPECIES])
I_H2O = SP['H2O']
DRY = np.array([s != 'H2O' for s in SPECIES])

# grids (vendor/vae_partmc_utils)
DIAM_LOW, DIAM_HIGH, N_BINS_ALL = 10 ** (-8.75), 1e-4, 19
N_BINS = 15                                   # the decoder's (PM10)
EDGES = np.logspace(np.log10(DIAM_LOW), np.log10(DIAM_HIGH), N_BINS_ALL + 1)
WAVELENGTHS_M = (300e-9, 400e-9, 500e-9, 600e-9, 700e-9, 800e-9, 900e-9, 1000e-9)
CCN_T_K = 287.0
N_EPS, N_TEMP = 100, 100
BLOCKS = (('M', 225), ('N', 15), ('C', 100), ('S', 8), ('A', 8), ('F', 100))
OFFSET = {}
_o = 0
for _k, _n in BLOCKS:
    OFFSET[_k] = (_o, _n)
    _o += _n
NX = _o                                        # 456

DT = torch.float64
CDT = torch.complex128


def set_threads(n=None):
    """torch threads for the Mie loop (default: env BASELINES_THREADS or 8)."""
    torch.set_num_threads(int(n or os.environ.get('BASELINES_THREADS', 8)))


def wet_diameter(m):
    """(..., 15) kg -> wet diameter (m), all species incl. H2O."""
    return (6.0 / np.pi * (m / RHO).sum(-1)) ** (1.0 / 3.0)


def dry_diameter(m):
    """(..., 15) kg -> dry diameter (m), H2O excluded."""
    return (6.0 / np.pi * (m[..., DRY] / RHO[DRY]).sum(-1)) ** (1.0 / 3.0)


def _clean(m, w):
    """vendor clean_prtcls: negative weights/masses to 0, masses of zero-weight particles to 0,
    weights of massless particles to 0."""
    w = np.where(w > 0, w, 0.0)
    m = np.where(w[..., None] > 0, m, 0.0)
    m = np.where(m > 0, m, 0.0)
    w = np.where(m.sum(-1) == 0.0, 0.0, w)
    return m, w


#: The binning of the size distributions (env AEROMELD_SIZE_BASIS), as the decoders' training targets
#: (training/dataset/dry_cache.py):
#:   'wet' (the default): wet diameter, every species' mass in M (round 33);
#:   'dry' (2026-11-16): dry diameter, H2O left out of M (its row 0) (round 37);
#:   'drydiam' (2026-11-19, the final choice): dry diameter, every species' mass in M, water
#:   included (round 39; the paper's: paperfigs/config.SIZE_BASIS).
SIZE_BASIS = os.environ.get('AEROMELD_SIZE_BASIS', 'wet')
BASES = ('wet', 'dry', 'drydiam')
assert SIZE_BASIS in BASES, SIZE_BASIS


def dry_binned(basis=None):
    """True when the size distributions of `basis` (default SIZE_BASIS) are binned by dry diameter."""
    return (basis or SIZE_BASIS) != 'wet'
#: AEROMELD_DRY_SOURCES=1 (2026-11-19, option 2: the source populations without their water):
#: read_source sets every particle's H2O mass to 0, so every consumer of sources() (the reference, the
#: schemes' source states) sees dry particles. The particle files themselves are unchanged.
DRY_SOURCES = os.environ.get('AEROMELD_DRY_SOURCES', '0') == '1'


def size_blocks(m, w, basis=None):
    """M (P, 15 species, 15 bins) and N (P, 15): weighted sums on the size bins of `basis` (default
    SIZE_BASIS): wet diameter ('wet') or dry diameter ('dry', H2O's row then 0; 'drydiam', every
    species' mass), per particle of the population (w normalized)."""
    basis = basis or SIZE_BASIS
    assert basis in BASES, basis
    P, n, _ = m.shape
    D = dry_diameter(m) if dry_binned(basis) else wet_diameter(m)
    b = np.searchsorted(EDGES, D, side='right') - 1           # np.histogram's [lo, hi)
    b = np.where(D == EDGES[-1], N_BINS_ALL - 1, b)
    ok = (b >= 0) & (b < N_BINS) & (w > 0)
    idx = (np.arange(P)[:, None] * N_BINS + np.where(ok, b, 0)).ravel()
    wv = np.where(ok, w, 0.0).ravel()
    N = np.bincount(idx, weights=wv, minlength=P * N_BINS).reshape(P, N_BINS)
    M = np.zeros((P, NS, N_BINS))
    for k in range(NS):
        if basis == 'dry' and not DRY[k]:
            continue
        M[:, k] = np.bincount(idx, weights=(np.where(ok, w, 0.0) * m[..., k]).ravel(),
                              minlength=P * N_BINS).reshape(P, N_BINS)
    return M, N


@torch.no_grad()
def ccn_block(m, w):
    """CCN/CN (P, 100): vendor calc_crh + calc_ccncdf, 287 K."""
    P, n, _ = m.shape
    mt = torch.as_tensor(m, dtype=DT)
    wt = torch.as_tensor(w, dtype=DT)
    crh = ED.calc_crh(mt, wt, torch.full((P,), CCN_T_K, dtype=DT), torch.as_tensor(RHO, dtype=DT),
                      torch.as_tensor(KAPPA, dtype=DT), list(SPECIES), P, n, NS, 0.0, 100, 1e-12)['crh_prt']
    r = ED.calc_ccncdf(crh, wt, P, n, ED.CCN_EPS_MIN, ED.CCN_EPS_MAX, N_EPS, torch.device('cpu'), DT)
    return r['ccn_cdf'].numpy()


@torch.no_grad()
def frozen_block(m, w):
    """Frozen fraction (P, 100): vendor get_bulkfrznfrac per particle."""
    P, n, _ = m.shape
    r = ED.get_bulkfrznfrac(torch.as_tensor(m[None], dtype=DT), torch.as_tensor(w[None, ..., None], dtype=DT),
                            -40.0, 0.0, N_TEMP, torch.as_tensor(RHO, dtype=DT), list(SPECIES),
                            1, P, NS, n, input_type='prt')
    return r['frznfrac_tmp'].reshape(P, N_TEMP).numpy()


_REFR = np.array([ED.DEFAULT_CHEM_REFRACTIVE.get(s, 1.5 + 0j) for s in SPECIES])
_CORE = np.array([s in ('BC', 'OIN') for s in SPECIES])


def _vwa(v, sel, void=1.5 + 0j):
    """vendor get_vwakappa on a species subset: volume-weighted refractive index."""
    vs = v[..., sel]
    vt = vs.sum(-1, keepdims=True)
    r = ((vs / (vt + 1e-50)) * _REFR[sel]).sum(-1)
    return np.where(vt[..., 0] > 0, r, void)


@torch.no_grad()
def optics_blocks(m, w, chunk=60000):
    """Scattering and absorption cross sections per particle (P, 8) each: vendor get_bulkoptical's
    per-particle quantities, every (particle, wavelength) through one get_coreshell per chunk.
    Particles with zero weight are skipped (they contribute nothing)."""
    P, n, _ = m.shape
    v = m / RHO
    d = (6 * v.sum(-1) / math.pi) ** (1.0 / 3.0)                # (P, n)
    dc = (6 * v[..., _CORE].sum(-1) / math.pi) ** (1.0 / 3.0)
    rc, rs = _vwa(v, _CORE), _vwa(v, ~_CORE)                    # wavelength-independent indices
    live = np.flatnonzero((w > 0).ravel())
    wl = np.asarray(WAVELENGTHS_M)
    nw = len(wl)
    qs = np.zeros((P * n, nw))
    qe = np.zeros((P * n, nw))
    dflat, dcflat, rcf, rsf = d.ravel()[live], dc.ravel()[live], rc.ravel()[live], rs.ravel()[live]
    xs = dflat[:, None] * (math.pi / wl[None, :])               # (L, nw)
    xc = dcflat[:, None] * (math.pi / wl[None, :])
    mc = np.broadcast_to(rcf[:, None], xs.shape)
    ms = np.broadcast_to(rsf[:, None], xs.shape)
    L = xs.size
    out_e = np.empty(L)
    out_s = np.empty(L)
    xs_f, xc_f, mc_f, ms_f = xs.ravel(), xc.ravel(), mc.ravel(), ms.ravel()
    for a in range(0, L, chunk):
        sl = slice(a, min(a + chunk, L))
        e, s, _ = ED.get_coreshell(torch.as_tensor(mc_f[sl], dtype=CDT), torch.as_tensor(ms_f[sl], dtype=CDT),
                                   torch.as_tensor(xc_f[sl], dtype=DT), torch.as_tensor(xs_f[sl], dtype=DT),
                                   100, torch.device('cpu'), DT, CDT)
        out_e[sl] = e.numpy()
        out_s[sl] = s.numpy()
    qe[live] = out_e.reshape(-1, nw)
    qs[live] = out_s.reshape(-1, nw)
    qe, qs = qe.reshape(P, n, nw), qs.reshape(P, n, nw)
    cross = d ** 2 * (math.pi / 4)
    S = np.einsum('pn,pnw->pw', cross * w, qs)
    A = np.einsum('pn,pnw->pw', cross * w, qe - qs)
    return S, A


def raw_vectors(m, w, chunk=60000, timing=None):
    """Per-particle species masses m (P, n, 15) kg (config.ML_SPECIES order) and number weights
    w (P, n) -> the raw 456-vector per population (P, 456), per particle of the population.
    A population with no weight gets zeros."""
    import time
    m = np.asarray(m, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if m.ndim == 2:
        m, w = m[None], w[None]
    m, w = _clean(m, w)
    tot = w.sum(-1, keepdims=True)
    wn = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 0.0)
    t = {}
    t0 = time.perf_counter()
    M, N = size_blocks(m, wn)
    t['size'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    C = ccn_block(m, wn)
    t['ccn'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    F = frozen_block(m, wn)
    t['frozen'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    S, A = optics_blocks(m, wn, chunk=chunk)
    t['optics'] = time.perf_counter() - t0
    x = np.concatenate([M.reshape(len(m), -1), N, C, S, A, F], axis=1)
    x[tot[:, 0] <= 0] = 0.0
    if timing is not None:
        timing.update(t)
    return x


def raw_vectors_batched(m, w, batch=500, **kw):
    """raw_vectors over populations in batches of `batch` (memory: frozen fraction holds
    (batch, n, 100) floats)."""
    out = np.empty((len(m), NX))
    for a in range(0, len(m), batch):
        out[a:a + batch] = raw_vectors(m[a:a + batch], w[a:a + batch], **kw)
    return out


# --- the three source populations ------------------------------------------------------------

SOURCE_DIR = REPO / 'data' / 'sources'


def read_source(scenario):
    """One source's particles: masses (n, 15) kg in ML_SPECIES order and weights (n,) m-3.
    Checks that the file's other species are empty and its densities and kappas are the
    diagnostics'."""
    from netCDF4 import Dataset
    with Dataset(SOURCE_DIR / ('scenario_%d_filtered_0001_00000013.nc' % scenario)) as d:
        names = d.variables['aero_species'].names.split(',')
        mm = np.asarray(d['aero_particle_mass'][:], np.float64)          # (species, particle)
        keep = [names.index(s) for s in SPECIES]
        other = [i for i in range(len(names)) if i not in keep]
        if mm[other].any():
            raise ValueError('scenario %d has mass in species outside ML_SPECIES' % scenario)
        if not np.allclose(np.asarray(d['aero_density'][keep]), RHO):
            raise ValueError('densities differ')
        if not np.allclose(np.asarray(d['aero_kappa'][keep]), KAPPA):
            raise ValueError('kappas differ')
        w = np.asarray(d['aero_num_conc'][:], np.float64)
    m = mm[keep].T.copy()
    if DRY_SOURCES:
        m[:, I_H2O] = 0.0
    return m, w


def sources():
    """(m (3, 1000, 15), w (3, 1000)) of config.POP_SCENARIOS."""
    ms, ws = zip(*[read_source(s) for s in config.POP_SCENARIOS])
    return np.stack(ms), np.stack(ws)


if __name__ == '__main__':
    set_threads()
    m, w = sources()
    tm = {}
    x = raw_vectors(m, w, timing=tm)
    print(x.shape, tm)
