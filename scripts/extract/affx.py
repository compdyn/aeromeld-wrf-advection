"""The paper's own models (training/, "affx"): numpy decoders, and the run extracts.

The models are the nonlinear twins and the affine AX-rel1 decoder trained by
training/affx.py (training/README.md). Each model seed is one numpy-only file,
data/models/<name>.<seed>.npz (converted from the torch checkpoint by
training/wrf/pt_to_npz.py), so decoding needs neither torch nor partnn.

Decoder output: 456 values per particle, in physical units, in the dataset's
order (training/affx.py):

    M  225  per-species mass size distribution, 15 species x 15 bins, kg per particle
    N   15  number size distribution, fraction of particles per bin
    C  100  CCN/CN fraction at the dataset's 100 supersaturations
    S    8  scattering cross section per particle, m^2, at 8 wavelengths
    A    8  absorption cross section per particle, m^2
    F  100  frozen fraction at the dataset's 100 temperatures, -40 to 0 C

Extensive blocks (M, N, S, A) times the number concentration give the
population's values; C and F are fractions. The grids are those of the
paper's partnn decoders (05/06/11 models: PM10, 15 bins), so the products have
the same coordinates either way.

Run extracts. The WRF-AeroMELD runs of these models were made on Perlmutter
(training/wrf/); their wrfout files (2.8 GB each) stay there. Each run's
extract, data/wrfml_runs/<run>.npz (training/wrf/extract_runs.py), holds what
the products need: the surface latents and ntot at 00 and 04 UTC, the three
sampling points at every output time, and the three source latents.
"""
import json
from pathlib import Path

import numpy as np

from paperfigs import config

MODELS_DIR = config.REPO / 'data' / 'models'
RUNS_DIR = config.REPO / 'data' / 'wrfml_runs'

#: (offset, size) of each block in the 456 decoder outputs.
BLOCKS = dict(M=(0, 225), N=(225, 15), C=(240, 100), S=(340, 8), A=(348, 8), F=(356, 100))
N_SPECIES, N_BINS = 15, 15


def block(x, k):
    o, n = BLOCKS[k]
    return x[..., o:o + n]


def _softplus(v):
    return np.logaddexp(0.0, v)


class Model:
    """One model seed: decode(z (n, d)) -> (n, 456), physical per-particle units."""

    def __init__(self, key):
        self.key = key
        self.path = MODELS_DIR / ('%s.npz' % key)
        with np.load(self.path) as f:
            self.meta = json.loads(str(f['__meta__']))
            self.w = {k: f[k].astype(np.float64) for k in f.files if k != '__meta__'}
        self.kind = self.meta.get('dec', 'affine')
        self.s = self.w['s']
        if self.kind == 'mlp':
            self.layers = sorted({int(k.split('.')[1]) for k in self.w
                                  if k.startswith('dmlp.') and k.endswith('.weight')})
            self.d_z = self.w['dmlp.%d.weight' % self.layers[0]].shape[1]
        else:
            self.A, self.b = self.w['A_eff'], self.w['b_eff']
            self.d_z = self.A.shape[1]

    def decode(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.kind != 'mlp':
            return (z @ self.A.T + self.b) * self.s
        if self.meta.get('bounded'):
            raise NotImplementedError('bounded (softmax-head) twins are not among the paper models')
        h = z
        for i, li in enumerate(self.layers):
            h = h @ self.w['dmlp.%d.weight' % li].T + self.w['dmlp.%d.bias' % li]
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        v = _softplus(h)
        parts = []
        for k in ('M', 'N', 'C', 'S', 'A', 'F'):
            vk = block(v, k)
            if self.meta.get('mono') and k in ('C', 'F'):
                # monotone heads: cumulative sums of nonnegative increments (C = 1 - U)
                vk = vk @ self.w['G' + k].T + (self.w['cC'] if k == 'C' else 0.0)
            parts.append(vk)
        return np.concatenate(parts, axis=-1) * self.s


def model(key):
    return Model(key)


def extract_path(run_key):
    return RUNS_DIR / ('%s.npz' % run_key)


def read_extract(run_key):
    """The run extract as a dict of arrays (see the module docstring)."""
    with np.load(extract_path(run_key), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# --- grids (vendor/vae_partmc_utils, without importing torch) -----------------

EPS_HISTBINS = np.exp(np.linspace(np.log(1e-3), np.log(1e-1), 100))
TEMPERATURE_C = np.linspace(-40.0, 0.0, 100)
WAVELENGTHS_M = np.array((300e-9, 400e-9, 500e-9, 600e-9, 700e-9, 800e-9, 900e-9, 1000e-9))
DIAM_LOW = 10 ** (-8.75)
LOG10_BIN_WIDTH = 0.25
_EDGES = np.log10(DIAM_LOW) + LOG10_BIN_WIDTH * np.arange(N_BINS + 1)


def _grid_coords():
    return {'eps': ('eps', EPS_HISTBINS.astype(np.float32),
                    {'units': '1', 'long_name': 'supersaturation s_env (fraction)'}),
            'temperature': ('temperature', TEMPERATURE_C.astype(np.float32), {'units': 'degC'}),
            'wavelength': ('wavelength', WAVELENGTHS_M.astype(np.float32), {'units': 'm'})}


def _times(e):
    return [str(t) for t in e['times']]


def _attrs(r, **more):
    return {'run': r.key, 'wrf_dir': r.wrf_dir, 'model': r.model.key, 'seed_idx': r.model.seed_idx,
            'n_latent': r.model.n_latent, 'level': config.LEVEL, 'device': 'numpy',
            'source': 'run extract %s (training/wrf/extract_runs.py)' % r.extract.name, **more}


def _sources(r):
    from paperfigs import products
    return [r.extract, r.model.checkpoint, products.path('grid')]


def _check_points(e):
    got = [tuple(int(v) for v in ij) for ij in e['point_ij']]
    if got != [tuple(p) for p in config.LOCATIONS]:
        raise ValueError('extract points %s are not config.LOCATIONS %s' % (got, config.LOCATIONS))


# --- product builders for affx runs (called by extract.aeromeld / wrfml / populations) ---

def build_domain(r):
    """aeromeld_domain from the run extract (same variables and coordinates as
    extract.aeromeld.build_domain)."""
    e = read_extract(r.key)
    if str(e['dom_times'][1]) != config.TIMESTAMP_FINAL.replace(' ', '_'):
        raise ValueError('extract %s ends at %s, not %s' % (r.extract, e['dom_times'][1], config.TIMESTAMP_FINAL))
    z, ntot = e['z_dom'][1].astype(np.float64), e['ntot_dom'][1].astype(np.float64)   # (d, y, x), (y, x)
    ds = decode_domain(r.model.key.rsplit('.', 1)[0], z, ntot, _attrs(r, timestamp=config.TIMESTAMP_FINAL))
    return ds, _sources(r)


def decode_domain(name, z, ntot, attrs):
    """An aeromeld_domain-like dataset for any latent field z (d, y, x) and
    number mixing ratio ntot (y, x), decoded with model `name` (<recipe>.<seed>)."""
    import xarray as xr
    n_y, n_x = ntot.shape
    x = model(name).decode(np.moveaxis(z, 0, -1).reshape(-1, z.shape[0]))
    x = x.reshape(n_y, n_x, -1)
    nd = ntot[..., None]
    data = {}
    for var, dim, v, long_name in (('ccn_cdf', 'eps', block(x, 'C'), 'CCN/CN fraction activated at s_env'),
                                   ('frozen_fraction', 'temperature', block(x, 'F'), 'frozen fraction'),
                                   ('scattering', 'wavelength', block(x, 'S') * nd, 'bulk scattering coefficient'),
                                   ('absorption', 'wavelength', block(x, 'A') * nd, 'bulk absorption coefficient')):
        va = {'long_name': long_name, 'decoder_node': var}
        if dim == 'wavelength':
            va.update(units='m-1 m3 kg-1', basis='per kg dry air (decoded from WRF-ML ntot)')
        else:
            va['units'] = '1'
        data[var] = (('south_north', 'west_east', dim), v.astype(np.float32), va)
    data['ntot'] = (('south_north', 'west_east'), ntot.astype(np.float32),
                    {'units': 'kg-1', 'basis': 'per kg dry air',
                     'long_name': 'WRF-ML total number mixing ratio (decoder input)'})
    return xr.Dataset(data, coords=_grid_coords(), attrs=attrs)


def build_points(r):
    """aeromeld_points from the run extract (as extract.aeromeld.build_points)."""
    import xarray as xr
    from paperfigs import common
    e = read_extract(r.key)
    _check_points(e)
    times = _times(e)
    z = np.moveaxis(e['z_pts'].astype(np.float64), 1, -1)            # (point, time, d)
    ntot = e['ntot_pts'].astype(np.float64)                          # (point, time)
    n_pt, n_t = ntot.shape
    x = model(r.model.key.rsplit('.', 1)[0]).decode(z.reshape(-1, z.shape[-1])).reshape(n_pt, n_t, -1)
    nd = ntot[..., None]
    w = LOG10_BIN_WIDTH
    vals = {
        'ccn_cdf': (('eps',), block(x, 'C'), '1', 'CCN/CN fraction activated at s_env'),
        'frozen_fraction': (('temperature',), block(x, 'F'), '1', 'frozen fraction'),
        'scattering': (('wavelength',), block(x, 'S') * nd, 'm-1 m3 kg-1', 'bulk scattering coefficient'),
        'absorption': (('wavelength',), block(x, 'A') * nd, 'm-1 m3 kg-1', 'bulk absorption coefficient'),
        'dNdlogDp': (('bin',), block(x, 'N') * nd / w, 'kg-1',
                     'number size distribution dN/dlog10(D) (a density: n_prthst / bin width)'),
        'dMdlogDp': (('species', 'bin'), block(x, 'M').reshape(n_pt, n_t, N_SPECIES, N_BINS) * nd[..., None] / w,
                     'kg kg-1', 'per-species mass size distribution dM/dlog10(D) (a density: '
                     'm_chmprthst / bin width)'),
    }
    data = {}
    for name, (dims, v, units, long_name) in vals.items():
        attrs = {'units': units, 'long_name': long_name, 'decoder_node': name}
        if units != '1':
            attrs['basis'] = 'per kg dry air (decoded from WRF-ML ntot)'
        data[name] = (('point', 'time') + dims, v.astype(np.float32), attrs)
    data['ntot'] = (('point', 'time'), ntot.astype(np.float32),
                    {'units': 'kg-1', 'basis': 'per kg dry air',
                     'long_name': 'WRF-ML total number mixing ratio (decoder input)'})
    coords = dict(_grid_coords())
    coords.update({
        'point': ('point', list(config.LOC_LABELS)),
        'point_i': ('point', np.array([p[0] for p in config.LOCATIONS], np.int32),
                    {'long_name': '1-based west_east index'}),
        'point_j': ('point', np.array([p[1] for p in config.LOCATIONS], np.int32),
                    {'long_name': '1-based south_north index'}),
        'time': ('time', times),
        'hours': ('time', common.hours_since(times).astype(np.float32),
                  {'units': 'h', 'long_name': 'hours since %s' % config.TIMESTAMP_START}),
        'species': ('species', list(config.ML_SPECIES), {'long_name': 'species, in the VAE order'}),
        'log10_d_lo': ('bin', _EDGES[:-1], {'units': 'log10(m)', 'long_name': 'bin lower edge, log10 diameter'}),
        'log10_d_hi': ('bin', _EDGES[1:], {'units': 'log10(m)', 'long_name': 'bin upper edge, log10 diameter'}),
    })
    ds = xr.Dataset(data, coords=coords, attrs=_attrs(r))
    return ds, _sources(r)


def build_fields(r):
    """wrfml_fields from the run extract (as extract.wrfml.build_fields)."""
    import xarray as xr
    from paperfigs import common
    e = read_extract(r.key)
    _check_points(e)
    times = _times(e)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    per_kg = {'units': 'kg-1', 'basis': 'per kg dry air'}
    n_lat = e['z_dom'].shape[1]
    out = xr.Dataset(
        {
            'z': (('latent', 'south_north', 'west_east'), f32(e['z_dom'][1]),
                  {'units': '1', 'long_name': 'latent variable z_n (ltnt_var<n>), intensive '
                                              '(particle mean of the encoder features, >= 0)'}),
            'ntot': (('south_north', 'west_east'), f32(e['ntot_dom'][1]),
                     dict(per_kg, long_name='WRF-ML total number mixing ratio')),
            'z_points': (('latent', 'point', 'time'), f32(np.moveaxis(e['z_pts'], 1, 0)),
                         {'units': '1', 'long_name': 'latent at the sampling points'}),
            'ntot_points': (('point', 'time'), f32(e['ntot_pts']),
                            dict(per_kg, long_name='ntot at the sampling points')),
        },
        coords={
            'latent': ('latent', np.arange(1, n_lat + 1, dtype=np.int32),
                       {'long_name': 'n of z_n, 1-based as in ltnt_var<n>'}),
            'point': ('point', list(config.LOC_LABELS)),
            'point_i': ('point', np.array([p[0] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based west_east index'}),
            'point_j': ('point', np.array([p[1] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based south_north index'}),
            'time': ('time', times),
            'hours': ('time', common.hours_since(times).astype(np.float32),
                      {'units': 'h', 'long_name': 'hours since %s' % config.TIMESTAMP_START}),
        },
        attrs={'run': r.key, 'wrf_dir': r.wrf_dir, 'model': r.model.key, 'n_latent': n_lat,
               'timestamp': config.TIMESTAMP_FINAL, 'level': config.LEVEL,
               'source': 'run extract %s' % r.extract.name},
    )
    from paperfigs import products
    return out, [r.extract, products.path('grid')]


#: Model name -> its key in the source-latent files (training/wrf/encode_sources.sh).
ZSRC_KEYS = dict(twin_k16='k16', twin_k32='k32', twin_pairs='pairs', twin_none='none', twin_mix0='mix0',
                 ax_rel1='ax', ax_k16='axk16')
#: The PartMC side of the populations product comes from this model's product: it
#: describes the same three particle files, whatever model encodes them.
PMC_REFERENCE_MODEL = '05_nonlinvlong.0.0'


#: Models retrained on the optics-corrected targets (round 33, notes/pmc_diagnostics_check.md)
#: are named <model>_of; their source latents are in zsrc_of/, keyed <model>_z. The dry-diameter
#: rebuild (round 37, 2026-11-16) is named <model>_dry, source latents in zsrc_dry/. The final
#: rebuild (round 39, 2026-11-19: dry diameter, every mass including water) is <model>_dd, in zsrc_dd/.
OF = '_of'
DRY = '_dry'
DD = '_dd'
#: The size basis of each family's targets (baselines/diag.BASES); the reference is binned to match.
SUFFIX_BASIS = {OF: 'wet', DRY: 'dry', DD: 'drydiam'}


def zsrc_path(seed, name=''):
    sub = 'zsrc' + next((k for k in SUFFIX_BASIS if name.endswith(k)), '')
    return RUNS_DIR / sub / ('zsrc_s%d.npz' % seed)


def zsrc_key(name):
    for s in (OF, DRY, DD):
        if name.endswith(s):
            return name[:-len(s)] + '_z'
    return ZSRC_KEYS[name] + '_z'


def size_distributions():
    """The particle-resolved size distributions of the three source files in each size basis
    (baselines/diag.BASES): 'wet' (wet diameter), 'dry' (dry diameter, H2O excluded from the
    diameter and from the mass; its row is 0) and 'drydiam' (dry diameter, every species' mass,
    water included), on the populations product's basis: dN/dlog10 D (3, bins) and dM/dlog10 D
    (3, species, bins) per m3 of the file's own weights. Checked: the same code by wet diameter
    reproduces the product's pmc_ (wet) values."""
    sys_path = str(config.REPO / 'baselines')
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import diag                                               # baselines/diag.py: the reference's code
    m, w = diag.sources()                                     # (3, 1000, 15) kg, (3, 1000) m-3
    out = {}
    for basis in diag.BASES:
        D = diag.dry_diameter(m) if diag.dry_binned(basis) else diag.wet_diameter(m)
        b = np.searchsorted(diag.EDGES, D, side='right') - 1
        ok = (b >= 0) & (b < N_BINS) & (w > 0)
        N = np.zeros((3, N_BINS))
        M = np.zeros((3, N_SPECIES, N_BINS))
        for p in range(3):
            np.add.at(N[p], b[p][ok[p]], w[p][ok[p]])
            for k in range(N_SPECIES):
                if basis == 'dry' and not diag.DRY[k]:
                    continue
                np.add.at(M[p, k], b[p][ok[p]], (w[p] * m[p][:, k])[ok[p]])
        out[basis] = (N / LOG10_BIN_WIDTH, M / LOG10_BIN_WIDTH)
    return out


#: PartMC CCN/CN of the three particle files at 287 K, the temperature of the training
#: data and of the WRF-PartMC reference extractor (vendor/extract_diagnostics.py
#: CCN_TEMPERATURE_K). The reference product's pmc_ccn_cdf is PyPartMC's at 290 K
#: (vae_partmc_utils.ccn_env_state), up to 0.013 different for Urban; these are
#: code_partnn's kappa-Koehler on the same particles (identical to the vendor code's at
#: 287 K, scripts/checks/pmc_diagnostics_compare.py).
PMC_CCN_287 = config.DATA_DIR / 'affx' / 'wrf3_jeff.npz'


def build_populations(m):
    """populations for an affx model: the three source latents (encode_sources.sh:
    the model's encoder on the 1000-particle files, checked there against the
    run's own latents), decoded per particle and put on the full populations'
    number concentration (config.POP_N_FULL, as ml_ntot). The PartMC side
    (pmc_*) is the paper's own, from PMC_REFERENCE_MODEL's product."""
    import xarray as xr
    from paperfigs import products
    name, seed, _ = m.key.split('.')
    with np.load(zsrc_path(int(seed), name)) as f:
        pops = (f['pops'] // 25).tolist()
        z = f[zsrc_key(name)].astype(np.float64)
    if pops != list(config.POP_SCENARIOS):
        raise ValueError('source latents are for scenarios %s, not %s' % (pops, config.POP_SCENARIOS))
    ref = products.load('populations', model=PMC_REFERENCE_MODEL)
    x = model('%s.%s' % (name, seed)).decode(z)                    # (3, 456) per particle
    n = np.array([config.POP_N_FULL[s] for s in config.POP_SCENARIOS])[:, None]   # m^-3
    w = LOG10_BIN_WIDTH
    ml = {'ccn_cdf': block(x, 'C'), 'frozen_fraction': block(x, 'F'),
          'scattering': block(x, 'S') * n, 'absorption': block(x, 'A') * n,
          'dNdlogDp': block(x, 'N') * n / w,
          'dMdlogDp': block(x, 'M').reshape(3, N_SPECIES, N_BINS) * n[..., None] / w}
    ds = ref.copy()
    for k, v in ml.items():
        var = ds['ml_' + k]
        ds['ml_' + k] = (var.dims, v.astype(np.float32),
                         {**var.attrs, 'basis': 'decoded per particle x ml_ntot (the full population, '
                                                'config.POP_N_FULL), per m^3'})
    basis = next(b for k, b in SUFFIX_BASIS.items() if name.endswith(k)) if name.endswith(tuple(SUFFIX_BASIS)) else 'wet'
    if basis != 'wet':
        # the reference's size distributions by dry diameter, as these models' targets
        sd = size_distributions()
        for k, i in (('dNdlogDp', 0), ('dMdlogDp', 1)):
            var = ds['pmc_' + k]
            dev = np.abs(sd['wet'][i] - var.values).max() / np.abs(var.values).max()
            if dev > 1e-5:
                raise ValueError('pmc_%s recomputed by wet diameter differs from the product by %.1e' % (k, dev))
            ds['pmc_' + k] = (var.dims, sd[basis][i].astype(np.float32),
                              {**var.attrs, 'long_name': var.attrs.get('long_name', '').replace('wet', 'dry'),
                               'basis': basis,
                               'note': '%s (extract/affx.size_distributions); wet recomputed to %.1e of the product'
                                       % ({'dry': 'dry diameter, H2O excluded',
                                           'drydiam': 'dry diameter, every species including H2O'}[basis], dev)})
        for k in ('ml_dNdlogDp', 'ml_dMdlogDp'):
            ds[k].attrs['long_name'] = ds[k].attrs.get('long_name', '').replace('wet', 'dry')
    with np.load(PMC_CCN_287) as f:
        ccn287 = f['X'][:, 240:340]
    var = ds['pmc_ccn_cdf']
    ds['pmc_ccn_cdf'] = (var.dims, ccn287.astype(np.float32),
                         {**var.attrs, 'note': 'kappa-Koehler at 287 K (training data and reference '
                                               'convention), not the reference product\'s 290 K'})
    ds['ml_ntot'] = (('scenario',), n[:, 0].astype(np.float32),
                     {'units': 'm-3', 'long_name': 'number concentration of the full population '
                                                   '(config.POP_N_FULL); the decoder is per particle'})
    ds = ds.drop_vars('z').drop_dims('latent', errors='ignore')
    ds['z'] = (('scenario', 'latent'), z.astype(np.float32),
               {'units': '1', 'long_name': 'encoder mean latent vector (WRF ltnt_var1..n)'})
    ds = ds.assign_coords(latent=('latent', np.arange(1, z.shape[1] + 1, dtype=np.int32),
                                  {'long_name': '1-based latent index (WRF ltnt_var<n>)'}))
    ds.attrs = {'model': m.key, 'seed_idx': m.seed_idx, 'n_latent': m.n_latent,
                'particle_time': config.PARTICLE_TIME, 'n_particles': ref.attrs.get('n_particles'),
                'device': 'numpy', 'pmc_from': 'populations_%s' % PMC_REFERENCE_MODEL,
                'z_from': str(zsrc_path(int(seed), name).relative_to(RUNS_DIR)), 'pmc_ccn_from': PMC_CCN_287.name}
    return ds, [zsrc_path(int(seed), name), m.checkpoint, PMC_CCN_287,
                products.path('populations', model=PMC_REFERENCE_MODEL)]
