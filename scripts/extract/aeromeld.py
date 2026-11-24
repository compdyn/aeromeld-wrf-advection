"""AeroMELD products: WRF-ML latents decoded with the run's own model.

`aeromeld_domain` holds the decoded fields over the surface at
config.TIMESTAMP_FINAL; `aeromeld_points` the sampling points at every wrfout
time, with the size distributions added. Each run is decoded with the model it
was encoded with (its registry entry), checked against the run's files first
(registry.verify).

Basis. WRF-ML's ntot is a number mixing ratio (# per kg dry air) and the decoder
is handed it as is, so the extensive outputs (optics, size distributions) come
out per kg of dry air -- the basis the PartMC products are converted to.
CCN/CN and frozen fraction are intensive.

Size distributions. n_prthst / m_chmprthst are per-BIN amounts (aerometrics
scatter-adds particles into bins and never divides by the width), on the
decoder's own grid: 0.25-decade bins from 10^-8.75 m, 19 of them for 02_mlpprt
and the first 15 (PM10) for the 05/06/11 models. PartMC's grid is 20 bins of
0.2 decade from 10^-9 m. Per-bin amounts on different grids cannot be compared,
so the product stores densities, dN/dlog10(D) = n_prthst / 0.25, and carries
its own bin edges, taken from the decoded shape.
"""
import sys
import time

import numpy as np
import xarray as xr

from paperfigs import common, config, products, registry

#: Rows of the domain decoded per call; any size works, this bounds GPU memory.
ROWS_PER_CHUNK = 20

#: Decoder output node -> (product variable, dimension, long name).
DOMAIN_NODES = {
    'ccn_cdf': ('ccn_cdf', 'eps', 'CCN/CN fraction activated at s_env'),
    'frznfrac_tmp': ('frozen_fraction', 'temperature', 'frozen fraction'),
    'qs_pop': ('scattering', 'wavelength', 'bulk scattering coefficient'),
    'qa_pop': ('absorption', 'wavelength', 'bulk absorption coefficient'),
}


def vae_utils():
    """vae_partmc_utils, the repository's vendored copy (imports torch, partnn, PyPartMC)."""
    if str(config.VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(config.VENDOR_DIR))
    import vae_partmc_utils as vpu
    return vpu


def load_model(model):
    """(vae_partmc_utils, components) for a registry Model."""
    import torch
    torch.set_num_threads(1)          # intra-op threads spin against the CPU quota
    vpu = vae_utils()
    comp = vpu.load_vae_components(fpidx=model.key, resdir=config.MODEL_RESDIR,
                                   device_name=vpu.default_device(), dtype_name='float32',
                                   seed_idx=model.seed_idx, verbose=False)
    if comp['ltnt_dim'] != model.n_latent + 1:
        raise registry.RegistryError('model %s decodes %d latents (z + ntot), registry says %d + 1'
                                     % (model.key, comp['ltnt_dim'], model.n_latent))
    return vpu, comp


def read_latents(run, it, sl=np.s_[:, :]):
    """(latents (..., n_latent), ntot (...)) from the run's wrfout at step `it`, level LEVEL.

    Slicing before .values makes xarray read only the slab; ds[v].values[...]
    would load the whole variable (~70 MB each, ~3.5 GB for 49 latents)."""
    k = config.LEVEL
    with xr.open_dataset(run.wrfout) as ds:
        ltnt = np.stack([ds['ltnt_var%d' % i][(it, k) + sl].values
                         for i in range(1, run.model.n_latent + 1)], axis=-1)
        ntot = ds['ntot'][(it, k) + sl].values
    return ltnt, ntot


def decode(vpu, comp, ltnt, ntot, nodes, channels=False):
    """{node: (n, len)} for latent rows (n, n_latent) and physical ntot (n,);
    with channels=True, {node: (n, n_channels, len)} (m_chmprthst has one
    channel per species, every other node one)."""
    lv = np.concatenate([ltnt, vpu.ntot_to_latent(ntot.reshape(-1, 1), comp)], axis=1)
    x_recon = vpu.decode_latents(comp, lv)['x_recon']
    out = {node: x_recon[node][0].cpu().numpy() for node in nodes}
    return out if channels else {node: v[:, 0, :] for node, v in out.items()}


def check_grid(run):
    grid = products.load('grid')
    with xr.open_dataset(run.wrfout) as ds:
        xlat = ds['XLAT'].isel(Time=0).values
        xlong = ds['XLONG'].isel(Time=0).values
    if xlat.shape != grid['XLAT'].shape or not (
            np.allclose(xlat, grid['XLAT'], atol=1e-4) and np.allclose(xlong, grid['XLONG'], atol=1e-4)):
        raise ValueError('%s is not on the PartMC grid' % run.wrfout)
    return grid


def build_domain(run):
    r = registry.load().run(run)
    if r.model.backend == 'affx':                 # the paper's own models (extract/affx.py)
        from extract import affx
        registry.verify(r)
        return affx.build_domain(r)
    registry.verify(r)
    check_grid(r)
    with xr.open_dataset(r.wrfout) as ds:
        it = common.time_index(common.times_of(ds), config.TIMESTAMP_FINAL)
    ltnt, ntot = read_latents(r, it)
    if not (np.isfinite(ltnt).all() and np.isfinite(ntot).all()):
        raise ValueError('non-finite latents or ntot in %s at %s' % (r.wrfout, config.TIMESTAMP_FINAL))
    n_y, n_x = ntot.shape

    vpu, comp = load_model(r.model)
    t0 = time.perf_counter()
    out = {}
    for j0 in range(0, n_y, ROWS_PER_CHUNK):
        j1 = min(j0 + ROWS_PER_CHUNK, n_y)
        got = decode(vpu, comp, ltnt[j0:j1].reshape(-1, r.model.n_latent),
                     ntot[j0:j1].reshape(-1), DOMAIN_NODES)
        for node, v in got.items():
            if node not in out:
                out[node] = np.zeros((n_y, n_x, v.shape[-1]), dtype=np.float32)
            out[node][j0:j1] = v.reshape(j1 - j0, n_x, -1)
    print('  decoded %d cells with %s on %s in %.1f s'
          % (n_y * n_x, r.model.key, comp['device'], time.perf_counter() - t0))
    if not np.isfinite(out['ccn_cdf']).all():
        raise ValueError('non-finite decoded ccn_cdf')

    grids = {'eps': np.asarray(vpu.EPS_HISTBINS), 'temperature': np.asarray(vpu.TEMPERATURE_C),
             'wavelength': np.asarray(vpu.WAVELENGTHS_M)}
    data = {}
    for node, (name, dim, long_name) in DOMAIN_NODES.items():
        if out[node].shape[-1] != grids[dim].size:
            raise ValueError('%s has %d values along %s, the grid has %d -- is the decoding '
                             'config the one vae_partmc_utils describes?'
                             % (node, out[node].shape[-1], dim, grids[dim].size))
        attrs = {'long_name': long_name, 'decoder_node': node}
        if dim == 'wavelength':
            attrs.update(units='m-1 m3 kg-1', basis='per kg dry air (decoded from WRF-ML ntot)')
        else:
            attrs['units'] = '1'
        data[name] = (('south_north', 'west_east', dim), out[node], attrs)
    data['ntot'] = (('south_north', 'west_east'), ntot.astype(np.float32),
                    {'units': 'kg-1', 'basis': 'per kg dry air',
                     'long_name': 'WRF-ML total number mixing ratio (decoder input)'})

    ds = xr.Dataset(
        data,
        coords={'eps': ('eps', grids['eps'].astype(np.float32),
                        {'units': '1', 'long_name': 'supersaturation s_env (fraction)'}),
                'temperature': ('temperature', grids['temperature'].astype(np.float32), {'units': 'degC'}),
                'wavelength': ('wavelength', grids['wavelength'].astype(np.float32), {'units': 'm'})},
        attrs={'run': r.key, 'wrf_dir': r.wrf_dir, 'model': r.model.key,
               'seed_idx': r.model.seed_idx, 'n_latent': r.model.n_latent,
               'timestamp': config.TIMESTAMP_FINAL, 'level': config.LEVEL,
               'device': str(comp['device'])},
    )
    return ds, [r.wrfout, r.model.checkpoint, products.path('grid')]


#: Decoder output node -> (product variable, dimensions after (point, time),
#: units, long name). n_prthst and m_chmprthst become densities (module docstring).
POINT_NODES = {
    'ccn_cdf': ('ccn_cdf', ('eps',), '1', 'CCN/CN fraction activated at s_env'),
    'frznfrac_tmp': ('frozen_fraction', ('temperature',), '1', 'frozen fraction'),
    'qs_pop': ('scattering', ('wavelength',), 'm-1 m3 kg-1', 'bulk scattering coefficient'),
    'qa_pop': ('absorption', ('wavelength',), 'm-1 m3 kg-1', 'bulk absorption coefficient'),
    'n_prthst': ('dNdlogDp', ('bin',), 'kg-1',
                 'number size distribution dN/dlog10(D) (a density: n_prthst / bin width)'),
    'm_chmprthst': ('dMdlogDp', ('species', 'bin'), 'kg kg-1',
                    'per-species mass size distribution dM/dlog10(D) (a density: '
                    'm_chmprthst / bin width)'),
}


def build_points(run):
    r = registry.load().run(run)
    if r.model.backend == 'affx':                 # the paper's own models (extract/affx.py)
        from extract import affx
        registry.verify(r)
        return affx.build_points(r)
    registry.verify(r)
    check_grid(r)
    k, n_lat = config.LEVEL, r.model.n_latent
    pts = [common.point_index(p) for p in config.LOCATIONS]
    with xr.open_dataset(r.wrfout) as ds:
        times = common.times_of(ds)
        ltnt = np.stack([np.stack([ds['ltnt_var%d' % i][:, k, j, ii].values
                                   for i in range(1, n_lat + 1)], axis=-1)
                         for j, ii in pts])                         # (point, time, n_latent)
        ntot = np.stack([ds['ntot'][:, k, j, ii].values for j, ii in pts])   # (point, time)
    for ts in (config.TIMESTAMP_FINAL,) + tuple(config.TIMESTAMPS_SPECTRA):
        common.time_index(times, ts)                    # every instant a figure uses
    if not (np.isfinite(ltnt).all() and np.isfinite(ntot).all()):
        raise ValueError('non-finite latents or ntot at the points in %s' % r.wrfout)
    n_pt, n_t = ntot.shape

    vpu, comp = load_model(r.model)
    if tuple(vpu.CHEM_SPECIES) != tuple(config.ML_SPECIES):
        raise ValueError('vae_partmc_utils.CHEM_SPECIES %s is not config.ML_SPECIES'
                         % (vpu.CHEM_SPECIES,))
    t0 = time.perf_counter()
    got = decode(vpu, comp, ltnt.reshape(-1, n_lat), ntot.reshape(-1), POINT_NODES, channels=True)
    print('  decoded %d point-times with %s on %s in %.1f s'
          % (n_pt * n_t, r.model.key, comp['device'], time.perf_counter() - t0))

    n_bins = got['n_prthst'].shape[-1]
    if got['m_chmprthst'].shape[1:] != (len(config.ML_SPECIES), n_bins) or n_bins > vpu.N_BINS:
        raise ValueError('unexpected size-distribution shapes: n_prthst %s, m_chmprthst %s'
                         % (got['n_prthst'].shape, got['m_chmprthst'].shape))
    width = vpu.LOG10_BIN_WIDTH
    edges = np.log10(vpu.DIAM_LOW) + width * np.arange(n_bins + 1)
    grids = {'eps': np.asarray(vpu.EPS_HISTBINS), 'temperature': np.asarray(vpu.TEMPERATURE_C),
             'wavelength': np.asarray(vpu.WAVELENGTHS_M), 'species': list(config.ML_SPECIES),
             'bin': edges[:-1]}

    data = {}
    for node, (name, dims, units, long_name) in POINT_NODES.items():
        v = got[node]
        v = v[:, 0, :] if len(dims) == 1 else v                  # one channel, or species
        v = v.reshape((n_pt, n_t) + v.shape[1:])
        for d, size in zip(dims, v.shape[2:]):
            if len(grids[d]) != size:
                raise ValueError('%s has %d values along %s, the grid has %d'
                                 % (node, size, d, len(grids[d])))
        if node in ('n_prthst', 'm_chmprthst'):
            v = v / width
        attrs = {'units': units, 'long_name': long_name, 'decoder_node': node}
        if units != '1':
            attrs['basis'] = 'per kg dry air (decoded from WRF-ML ntot)'
        data[name] = (('point', 'time') + dims, v.astype(np.float32), attrs)
    data['ntot'] = (('point', 'time'), ntot.astype(np.float32),
                    {'units': 'kg-1', 'basis': 'per kg dry air',
                     'long_name': 'WRF-ML total number mixing ratio (decoder input)'})

    ds = xr.Dataset(
        data,
        coords={
            'point': ('point', list(config.LOC_LABELS)),
            'point_i': ('point', np.array([p[0] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based west_east index'}),
            'point_j': ('point', np.array([p[1] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based south_north index'}),
            'time': ('time', times),
            'hours': ('time', common.hours_since(times).astype(np.float32),
                      {'units': 'h', 'long_name': 'hours since %s' % config.TIMESTAMP_START}),
            'eps': ('eps', grids['eps'].astype(np.float32),
                    {'units': '1', 'long_name': 'supersaturation s_env (fraction)'}),
            'temperature': ('temperature', grids['temperature'].astype(np.float32), {'units': 'degC'}),
            'wavelength': ('wavelength', grids['wavelength'].astype(np.float32), {'units': 'm'}),
            'species': ('species', grids['species'], {'long_name': 'species, in the VAE order'}),
            'log10_d_lo': ('bin', edges[:-1], {'units': 'log10(m)',
                                               'long_name': 'bin lower edge, log10 diameter'}),
            'log10_d_hi': ('bin', edges[1:], {'units': 'log10(m)',
                                              'long_name': 'bin upper edge, log10 diameter'}),
        },
        attrs={'run': r.key, 'wrf_dir': r.wrf_dir, 'model': r.model.key,
               'seed_idx': r.model.seed_idx, 'n_latent': n_lat, 'level': config.LEVEL,
               'device': str(comp['device'])},
    )
    return ds, [r.wrfout, r.model.checkpoint, products.path('grid')]
