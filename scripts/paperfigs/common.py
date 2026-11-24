"""Small helpers shared by extractors and plot scripts. numpy only."""
import numpy as np

from . import config


def times_of(ds):
    """WRF 'Times' as ['YYYY-MM-DD_HH:MM:SS', ...] (xarray or netCDF4 dataset)."""
    raw = ds['Times'][:]
    raw = getattr(raw, 'values', raw)
    out = []
    for t in raw:
        if isinstance(t, bytes):
            out.append(t.decode())
        elif isinstance(t, np.ndarray) and t.dtype.kind == 'S':      # netCDF4 char array
            out.append(b''.join(t.tolist()).decode())
        else:
            out.append(str(t))
    return out


def time_index(times, timestamp):
    """Index of an exact timestamp in a list of timestamps -- never the nearest.

    The sources write on different cadences, so snapping to the nearest step
    silently pairs different instants (this hid a 3.5 h offset once)."""
    try:
        return list(times).index(timestamp)
    except ValueError:
        raise KeyError('%s is not an output time; have %s .. %s (%d steps)'
                       % (timestamp, times[0], times[-1], len(times))) from None


def hours_since(timestamps, origin=config.TIMESTAMP_START):
    """Hours from `origin` to each 'YYYY-MM-DD_HH:MM:SS' timestamp."""
    t0 = np.datetime64(origin.replace('_', 'T'))
    return np.array([(np.datetime64(t.replace('_', 'T')) - t0) / np.timedelta64(1, 'h')
                     for t in timestamps])


def point_index(loc):
    """0-based (j, i) array index of a 1-based (i, j) sampling point.

    Use as ``field[..., j, i]``. config.LOCATIONS is 1-based; indexing an array
    with it directly reads the cell one step north-east of the point."""
    i, j = loc
    return j - 1, i - 1


def interp_along_last_axis(arr, grid, value):
    """Linear interpolation of `arr` along its last axis at `value`, on the
    increasing `grid` (e.g. a supersaturation slice of a CCN spectrum)."""
    grid = np.asarray(grid)
    i_hi = np.clip(np.searchsorted(grid, value), 1, len(grid) - 1)
    i_lo = i_hi - 1
    w = (value - grid[i_lo]) / (grid[i_hi] - grid[i_lo])
    return arr[..., i_lo] * (1 - w) + arr[..., i_hi] * w


def at_time(ds, timestamp):
    """A points product at one exact timestamp (isel on its `time` axis)."""
    return ds.isel(time=time_index([str(t) for t in ds['time'].values], timestamp))


def log10_edges(ds):
    """Bin edges in log10(m) of a product's size distributions (n_bin + 1)."""
    lo, hi = np.asarray(ds['log10_d_lo']), np.asarray(ds['log10_d_hi'])
    if not np.allclose(lo[1:], hi[:-1]):
        raise ValueError('size bins are not contiguous')
    return np.append(lo, hi[-1])


def latents_to_show(z, n=9):
    """0-based indices of the latents the 3x3 latent figures show, from a
    (latent, south_north, west_east) field: all of them for a 9-latent run;
    for a larger latent space, the `n` with the largest spatial spread (many
    dimensions are collapsed and nearly uniform), in index order."""
    z = np.asarray(z)
    if z.shape[0] < n:
        raise ValueError('%d latents, the figures need %d' % (z.shape[0], n))
    if z.shape[0] == n:
        return list(range(n))
    spread = z.reshape(z.shape[0], -1).std(axis=1)
    return sorted(int(k) for k in np.argsort(spread)[::-1][:n])


def same_instant(*datasets):
    """The one timestamp every product describes; raise if they differ."""
    stamps = {d.attrs['timestamp'] for d in datasets}
    if len(stamps) != 1:
        raise ValueError('products at different instants: %s' % ', '.join(sorted(stamps)))
    return stamps.pop()


def exact_index(grid, value, what='value'):
    """Index of `value` on `grid`, exact to rounding -- for grids that differ
    between the two models (wavelengths), where the nearest point on each side
    could be a different wavelength."""
    hit = np.flatnonzero(np.isclose(np.asarray(grid), value))
    if hit.size != 1:
        raise ValueError('%s %g is not on the grid %s' % (what, value, np.asarray(grid)))
    return int(hit[0])
