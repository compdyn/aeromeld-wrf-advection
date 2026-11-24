"""The `grid` product: horizontal grid, map projection, emission source boxes.

The grid is read from the WRF-PartMC aerosols file. Every WRF-ML run is on the
same grid; the WRF-ML extractors check their XLAT/XLONG against this product.
"""
import numpy as np
import xarray as xr
from netCDF4 import Dataset

from paperfigs import config

#: Global attributes that define the WRF Lambert conformal projection.
PROJ_ATTRS = ('MAP_PROJ', 'STAND_LON', 'TRUELAT1', 'TRUELAT2', 'MOAD_CEN_LAT', 'CEN_LAT',
              'CEN_LON', 'DX', 'DY')


def build():
    with Dataset(config.PARTMC_AEROSOLS) as f:
        xlat = np.asarray(f['XLAT'][0], dtype=np.float32)
        xlong = np.asarray(f['XLONG'][0], dtype=np.float32)
        proj = {k: f.getncattr(k).item() for k in PROJ_ATTRS}
    if proj['MAP_PROJ'] != 1:
        raise ValueError('MAP_PROJ = %s; only Lambert conformal (1) is supported' % proj['MAP_PROJ'])
    n_y, n_x = xlat.shape

    # Emission source boxes: the cells where a source ever emits (num_flux > 0),
    # as 0-based inclusive index ranges -- what the domain figure outlines.
    with Dataset(config.EMIS_CONTROL) as f:
        flux = np.asarray(f['num_flux'][:])        # (source, time, west_east, south_north, z)
        dims = f['num_flux'].dimensions
    if dims[2:4] != ('west_east', 'south_north'):
        raise ValueError('num_flux dims %s: expected (source, time, west_east, south_north, z)' % (dims,))
    extent = flux.max(axis=1)[..., 0] > 0           # (source, west_east, south_north)
    boxes = np.full((extent.shape[0], 4), -1, dtype=np.int32)
    for s in range(extent.shape[0]):
        i_pos, j_pos = np.nonzero(extent[s])
        if i_pos.size:
            boxes[s] = (i_pos.min(), i_pos.max(), j_pos.min(), j_pos.max())

    ds = xr.Dataset(
        {
            'XLAT': (('south_north', 'west_east'), xlat,
                     {'units': 'degree_north', 'long_name': 'latitude of mass points'}),
            'XLONG': (('south_north', 'west_east'), xlong,
                      {'units': 'degree_east', 'long_name': 'longitude of mass points'}),
            'source_box': (('source', 'corner'), boxes,
                           {'long_name': 'cells where the emission source ever emits '
                                         '(num_flux > 0), 0-based inclusive; -1 = never emits'}),
        },
        coords={'corner': ['i_min', 'i_max', 'j_min', 'j_max']},
        attrs={'proj_' + k: v for k, v in proj.items()} | {
            'grid_shape': '%d x %d (south_north x west_east)' % (n_y, n_x)},
    )
    return ds, [config.PARTMC_AEROSOLS, config.EMIS_CONTROL]
