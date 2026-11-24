"""The `wrfml_fields` product: WRF-ML's own fields, not decoded.

The latent variables and ntot over the surface at config.TIMESTAMP_FINAL, and
at the sampling points at every wrfout time. The latents are stored as WRF-ML
carries them: intensive (per particle) and standardized, so they are unitless
and need no basis conversion. ntot is a number mixing ratio (# per kg dry air).

Reading needs only xarray; nothing here imports torch.
"""
import numpy as np
import xarray as xr

from paperfigs import common, config, products, registry

from .aeromeld import check_grid


def build_fields(run):
    r = registry.load().run(run)
    if r.model.backend == 'affx':                 # the paper's own models (extract/affx.py)
        from extract import affx
        registry.verify(r)
        return affx.build_fields(r)
    registry.verify(r)
    check_grid(r)
    k, n_lat = config.LEVEL, r.model.n_latent
    names = ['ltnt_var%d' % i for i in range(1, n_lat + 1)]
    pts = [common.point_index(p) for p in config.LOCATIONS]
    with xr.open_dataset(r.wrfout) as ds:
        times = common.times_of(ds)
        it = common.time_index(times, config.TIMESTAMP_FINAL)
        # Slice before .values so only the slab is read (see aeromeld.read_latents).
        ltnt = np.stack([ds[v][it, k].values for v in names])          # (latent, y, x)
        ntot = ds['ntot'][it, k].values
        ltnt_pts = np.stack([np.stack([ds[v][:, k, j, i].values for j, i in pts])
                             for v in names])                          # (latent, point, time)
        ntot_pts = np.stack([ds['ntot'][:, k, j, i].values for j, i in pts])
    if not (np.isfinite(ltnt).all() and np.isfinite(ltnt_pts).all()):
        raise ValueError('non-finite latents in %s' % r.wrfout)

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    per_kg = {'units': 'kg-1', 'basis': 'per kg dry air'}
    out = xr.Dataset(
        {
            'z': (('latent', 'south_north', 'west_east'), f32(ltnt),
                       {'units': '1', 'long_name': 'latent variable z_n (ltnt_var<n>), '
                                                   'intensive and standardized'}),
            'ntot': (('south_north', 'west_east'), f32(ntot),
                     dict(per_kg, long_name='WRF-ML total number mixing ratio')),
            'z_points': (('latent', 'point', 'time'), f32(ltnt_pts),
                              {'units': '1', 'long_name': 'latent at the sampling points'}),
            'ntot_points': (('point', 'time'), f32(ntot_pts),
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
        attrs={'run': r.key, 'wrf_dir': r.wrf_dir, 'model': r.model.key,
               'n_latent': n_lat, 'timestamp': config.TIMESTAMP_FINAL, 'level': config.LEVEL},
    )
    return out, [r.wrfout, products.path('grid')]
