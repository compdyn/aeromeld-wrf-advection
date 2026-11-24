"""PartMC products: `partmc_fields` (the aerosols file) and `partmc_diag_domain`
(the per-cell domain diagnostics).

Basis. WRF-ML carries number as a mixing ratio (# per kg of dry air). The
aerosols file and the diagnostics are per m^3 of air. Every extensive PartMC
quantity in a product is multiplied by PartMC's own ALT (1/rho_dry, m^3 kg^-1)
at the same cell and instant, so both models sit on the per-kg basis and a plot
can compare them directly. Without this the comparison carries 1/rho_dry
(median 0.865 at the surface at 04:00, following the terrain). Intensive
quantities (CCN/CN fraction, frozen fraction) are left alone.
"""
import glob
import re

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from paperfigs import common, config, products

#: Domain instants partmc_fields keeps: the domain figure shows the start, the
#: comparisons the end.
FIELD_INSTANTS = (config.TIMESTAMP_START, config.TIMESTAMP_FINAL)

_DIAG_RE = re.compile(r'diagnostics_batch(\d+)_z(\d+)_t(\d+)\.nc$')


# =============================================================================
# partmc_fields
# =============================================================================

def source_scenarios():
    """Scenario number of each PartMC source, in num_conc_source_NNN order (ICs,
    then BCs, then emissions), from the files each control file lists."""
    scen = []
    for f in (config.IC_CONTROL, config.BC_CONTROL, config.EMIS_CONTROL):
        with Dataset(f) as d:
            scen += [int(re.search(r'scenario_(\d+)_', s).group(1))
                     for s in d.source_files.split(',') if s]
    return np.array(scen, dtype=np.int32)


def build_fields():
    k = config.LEVEL
    scen = source_scenarios()
    n_src = scen.size
    with xr.open_dataset(config.PARTMC_AEROSOLS) as ds:
        times = common.times_of(ds)
        it_dom = [common.time_index(times, t) for t in FIELD_INSTANTS]
        # The file allocates more num_conc_source_NNN slots than the run has
        # sources (150 vs 5); the ones past the control files' list must be empty.
        extra = 'num_conc_source_%03d' % n_src
        if extra in ds.variables and np.any(ds[extra].isel(Time=it_dom[-1], bottom_top=k).values):
            raise ValueError('%s is nonzero, but the control files list only %d sources'
                             % (extra, n_src))
        pts = [common.point_index(p) for p in config.LOCATIONS]

        def domain(var):          # (instant, south_north, west_east)
            return np.stack([ds[var].isel(Time=it, bottom_top=k).values for it in it_dom])

        def points(var):          # (point, time)
            return np.stack([ds[var][:, k, j, i].values for j, i in pts])

        alt_dom, alt_pts = domain('ALT'), points('ALT')
        ntot_dom = domain('TOT_NUM_CONC') * alt_dom
        ntot_pts = points('TOT_NUM_CONC') * alt_pts
        src_dom = np.stack([domain('num_conc_source_%03d' % s) * alt_dom for s in range(n_src)])
        src_pts = np.stack([points('num_conc_source_%03d' % s) * alt_pts for s in range(n_src)])

        # Near-surface wind, destaggered to mass points and averaged over the
        # whole run: the median speed doubles (about 7 to 14 m/s) by 04 UTC while
        # each cell's direction holds steady, so the mean is the honest summary.
        u_stag = ds['U'].isel(bottom_top=k).values      # (Time, south_north, west_east_stag)
        v_stag = ds['V'].isel(bottom_top=k).values      # (Time, south_north_stag, west_east)
    u = (0.5 * (u_stag[..., :-1] + u_stag[..., 1:])).mean(axis=0)
    v = (0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])).mean(axis=0)

    per_kg = {'units': 'kg-1', 'basis': 'per kg dry air (per m^3 x ALT)'}
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    ds_out = xr.Dataset(
        {
            'ntot': (('instant', 'south_north', 'west_east'), f32(ntot_dom),
                     dict(per_kg, long_name='total number mixing ratio (TOT_NUM_CONC x ALT)')),
            'n_source': (('source', 'instant', 'south_north', 'west_east'), f32(src_dom),
                         dict(per_kg, long_name='number mixing ratio per PartMC source')),
            'ALT': (('instant', 'south_north', 'west_east'), f32(alt_dom),
                    {'units': 'm3 kg-1', 'long_name': 'inverse dry-air density'}),
            'ntot_points': (('point', 'time'), f32(ntot_pts), dict(per_kg, long_name='ntot at the sampling points')),
            'n_source_points': (('source', 'point', 'time'), f32(src_pts),
                                dict(per_kg, long_name='n_source at the sampling points')),
            'ALT_points': (('point', 'time'), f32(alt_pts),
                           {'units': 'm3 kg-1', 'long_name': 'ALT at the sampling points'}),
            'u_mean': (('south_north', 'west_east'), f32(u),
                       {'units': 'm s-1', 'long_name': 'grid-relative x wind, destaggered, run mean'}),
            'v_mean': (('south_north', 'west_east'), f32(v),
                       {'units': 'm s-1', 'long_name': 'grid-relative y wind, destaggered, run mean'}),
        },
        coords={
            'instant': ('instant', list(FIELD_INSTANTS)),
            'time': ('time', times),
            'hours': ('time', common.hours_since(times).astype(np.float32),
                      {'units': 'h', 'long_name': 'hours since %s' % config.TIMESTAMP_START}),
            'point': ('point', list(config.LOC_LABELS)),
            'point_i': ('point', np.array([p[0] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based west_east index'}),
            'point_j': ('point', np.array([p[1] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based south_north index'}),
            'source': ('source', np.arange(n_src, dtype=np.int32),
                       {'long_name': 'PartMC source NNN of num_conc_source_NNN (ICs, BCs, emissions)'}),
            'source_scenario': ('source', scen, {'long_name': 'scenario number of the source population'}),
        },
        attrs={'level': config.LEVEL, 'wind_level': config.LEVEL},
    )
    return ds_out, [config.PARTMC_AEROSOLS, config.IC_CONTROL, config.BC_CONTROL,
                    config.EMIS_CONTROL]


def alt_at(fields, timestamp):
    """(south_north, west_east) ALT at a domain instant of partmc_fields."""
    instants = list(fields['instant'].values)
    return fields['ALT'].values[common.time_index(instants, timestamp)]


def alt_points_at(fields, timestamps):
    """((point, len(timestamps)) ALT at the sampling points, [timestamps that
    had to be interpolated]).

    The aerosols file is on a 10-min cadence (a mistake, since fixed; the rerun
    is pending) and the diagnostics on 15 min, so :15 and :45 fall between two
    outputs and are interpolated linearly in time. Checked 2026-11-05 on the
    surface: across a 20-min gap that misses by 0.4% at p99 and 2.6% at worst,
    under the ~5% cell-to-cell number noise. After the 15-min rerun every
    instant is an exact match; then make interpolation an error (PLAN.md step 8).
    """
    t = np.array([np.datetime64(s.replace('_', 'T')) for s in fields['time'].values])
    alt = fields['ALT_points'].values.astype(float)            # (point, time)
    out = np.empty((alt.shape[0], len(timestamps)))
    interpolated = []
    for n, ts in enumerate(timestamps):
        target = np.datetime64(ts.replace('_', 'T'))
        if not t[0] <= target <= t[-1]:
            raise KeyError('%s is outside the aerosols file, %s .. %s' % (ts, t[0], t[-1]))
        k = int(np.searchsorted(t, target))
        if t[k] == target:
            out[:, n] = alt[:, k]
        else:
            w = (target - t[k - 1]) / (t[k] - t[k - 1])
            out[:, n] = (1 - w) * alt[:, k - 1] + w * alt[:, k]
            interpolated.append(ts)
    return out, interpolated


# =============================================================================
# partmc_diag_domain
# =============================================================================

def diag_dir():
    return config.DIAG_ROOT / config.DIAG_RUN


def diag_files(tag):
    """Batch files of one time tag, in batch order. The batch count is not fixed
    and the last batch is ragged, so the files are discovered, never assumed."""
    files = glob.glob('%s/diagnostics_batch*_z%d_t%s.nc' % (diag_dir(), config.DIAG_Z, tag))
    if not files:
        raise FileNotFoundError('no z%d batches for t%s in %s' % (config.DIAG_Z, tag, diag_dir()))
    return sorted(files, key=lambda f: int(_DIAG_RE.search(f).group(1)))


def diag_times(origin):
    """{'YYYY-MM-DD_HH:MM:SS': tag} for the times the diagnostics run holds.

    The files carry seconds since the start; the wall clock comes from the
    PartMC run's own first output time, `origin`."""
    tags = sorted({_DIAG_RE.search(f).group(3) for f in
                   glob.glob('%s/diagnostics_batch*_z%d_t*.nc' % (diag_dir(), config.DIAG_Z))})
    if not tags:
        raise FileNotFoundError('no diagnostics in %s' % diag_dir())
    t0 = np.datetime64(origin.replace('_', 'T'))
    table = {}
    for tag in tags:
        with Dataset(diag_files(tag)[0]) as d:
            sec = int(round(float(np.unique(d['time'][:])[0])))
        table[str(t0 + np.timedelta64(sec, 's')).replace('T', '_')] = tag
    return table


def read_diag_domain(tag, variables, n_y, n_x):
    """({name: (n_y, n_x, ...)}, {axis name: values}, [files]) for one time tag.

    Cells are placed by each file's own x_index/y_index: batches are packed by
    MPI rank, so a batch number says nothing about which cells it holds."""
    files = diag_files(tag)
    fields, axes = {}, {}
    for path in files:
        with Dataset(path) as d:
            iy = d['y_index'][:].astype(int) - 1
            ix = d['x_index'][:].astype(int) - 1
            if np.any(d['z_index'][:] != config.DIAG_Z):
                raise ValueError('%s holds levels other than z=%d' % (path, config.DIAG_Z))
            for name in variables:
                v = np.asarray(d[name][:])
                if name not in fields:
                    fields[name] = np.full((n_y, n_x) + v.shape[1:], np.nan)
                fields[name][iy, ix] = v
            for name in ('eps_histbins', 'temperature_C', 'wavelength_m'):
                axes.setdefault(name, np.asarray(d[name][:]))
    return fields, axes, files


def build_diag_domain():
    fields = products.load('partmc_fields')
    times = list(fields['time'].values)
    if times[0] != config.TIMESTAMP_START:
        raise ValueError('PartMC starts at %s, config says %s' % (times[0], config.TIMESTAMP_START))
    table = diag_times(times[0])
    ts = config.TIMESTAMP_FINAL
    if ts not in table:
        raise KeyError('%s is not in diagnostics run %s; have %s'
                       % (ts, config.DIAG_RUN, ', '.join(sorted(table))))
    n_y, n_x = fields.sizes['south_north'], fields.sizes['west_east']
    got, axes, files = read_diag_domain(
        table[ts], ('ccn_cdf', 'frozen_fraction', 'optical_scattering', 'optical_absorption',
                    'num_conc_total'), n_y, n_x)

    covered = np.isfinite(got['num_conc_total'])
    alt = alt_at(fields, ts)
    # The aerosols file and the diagnostics are set up independently: check they
    # are the same run, instant and level, and that no source is missing.
    n_src = (fields['n_source'].values[:, list(fields['instant'].values).index(ts)].sum(0)
             / alt)                                          # back to m^-3
    if not np.allclose(n_src[covered], got['num_conc_total'][covered], rtol=1e-5):
        raise ValueError('aerosols-file sources do not add up to diagnostics run %s '
                         'num_conc_total at %s' % (config.DIAG_RUN, ts))

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    per_kg = 'per kg dry air (per m^3 x ALT)'
    ds = xr.Dataset(
        {
            'ccn_cdf': (('south_north', 'west_east', 'eps'), f32(got['ccn_cdf']),
                        {'units': '1', 'long_name': 'CCN/CN fraction activated at s_env'}),
            'frozen_fraction': (('south_north', 'west_east', 'temperature'),
                                f32(got['frozen_fraction']),
                                {'units': '1', 'long_name': 'frozen fraction'}),
            'scattering': (('south_north', 'west_east', 'wavelength'),
                           f32(got['optical_scattering'] * alt[:, :, None]),
                           {'units': 'm-1 m3 kg-1', 'basis': per_kg,
                            'long_name': 'bulk scattering coefficient'}),
            'absorption': (('south_north', 'west_east', 'wavelength'),
                           f32(got['optical_absorption'] * alt[:, :, None]),
                           {'units': 'm-1 m3 kg-1', 'basis': per_kg,
                            'long_name': 'bulk absorption coefficient'}),
            'ntot': (('south_north', 'west_east'), f32(got['num_conc_total'] * alt),
                     {'units': 'kg-1', 'basis': per_kg,
                      'long_name': 'number of the diagnosed particles (num_conc_total x ALT)'}),
        },
        coords={
            'eps': ('eps', axes['eps_histbins'], {'units': '1', 'long_name': 'supersaturation s_env (fraction)'}),
            'temperature': ('temperature', axes['temperature_C'], {'units': 'degC'}),
            'wavelength': ('wavelength', axes['wavelength_m'], {'units': 'm'}),
        },
        attrs={'timestamp': ts, 'diag_run': config.DIAG_RUN, 'diag_tag': table[ts],
               'diag_z': config.DIAG_Z,
               'n_uncovered': int((~covered).sum()),
               'note': 'NaN = cell not covered by the diagnostics'},
    )
    return ds, [products.path('partmc_fields')] + files


# =============================================================================
# partmc_diag_points
# =============================================================================

DIAG_POINT_VARS = ('ccn_cdf', 'frozen_fraction', 'optical_scattering', 'optical_absorption',
                   'dNdlogDp', 'dMdlogDp', 'num_conc_total')
DIAG_AXES = ('eps_histbins', 'temperature_C', 'wavelength_m', 'logDp_bins_lb', 'logDp_bins_rb')


def read_diag_points(tag, points, variables):
    """({name: (n_point, ...)}, {axis: values}, [files used]) for 1-based (i, j)
    `points` at one time tag. Every batch file of the tag is searched, since
    batches are packed by MPI rank; a point found twice is an error, as in
    DiagnosticsIndex."""
    want = {tuple(p): n for n, p in enumerate(points)}
    got, axes, used, seen = {}, {}, [], set()
    for path in diag_files(tag):
        with Dataset(path) as d:
            cells = zip(d['x_index'][:].tolist(), d['y_index'][:].tolist())
            hits = [(s, want[c]) for s, c in enumerate(cells) if c in want]
            if not hits:
                continue
            used.append(path)
            for s, n in hits:
                if n in seen:
                    raise ValueError('point %s found twice at t%s (second time in %s)'
                                     % (points[n], tag, path))
                seen.add(n)
                if int(d['z_index'][s]) != config.DIAG_Z:
                    raise ValueError('%s sample %d is not level z=%d' % (path, s, config.DIAG_Z))
                for name in variables:
                    v = np.asarray(np.ma.filled(d[name][s], np.nan), dtype=float)
                    if name not in got:
                        got[name] = np.full((len(points),) + v.shape, np.nan)
                    got[name][n] = v
            if not axes:
                axes = {name: np.asarray(d[name][:], dtype=float) for name in DIAG_AXES}
                axes['species'] = [s.strip() for s in d['species'].getncattr('names').split(',')]
    missing = [points[n] for n in range(len(points)) if n not in seen]
    if missing:
        raise ValueError('points %s are not in any z%d batch for t%s' % (missing, config.DIAG_Z, tag))
    return got, axes, used


def build_diag_points():
    fields = products.load('partmc_fields')
    times_pmc = list(fields['time'].values)
    table = diag_times(times_pmc[0])
    stamps = sorted(table)
    per = {name: [] for name in DIAG_POINT_VARS}
    axes, files = None, []
    for ts in stamps:
        got, ax, used = read_diag_points(table[ts], config.LOCATIONS, DIAG_POINT_VARS)
        if axes is None:
            axes = ax
        elif any(not np.array_equal(ax[k], axes[k]) for k in DIAG_AXES) or ax['species'] != axes['species']:
            raise ValueError('the diagnostics grids change between times (at %s)' % ts)
        for name in DIAG_POINT_VARS:
            per[name].append(got[name])
        files += used
    v = {name: np.stack(a, axis=1) for name, a in per.items()}        # (point, time, ...)

    # Same check as the domain product, at every instant both files hold.
    n_checked = 0
    for n, ts in enumerate(stamps):
        if ts in times_pmc:
            it = times_pmc.index(ts)
            n_src = (fields['n_source_points'].values[:, :, it].sum(0)
                     / fields['ALT_points'].values[:, it])               # back to m^-3
            if not np.allclose(n_src, v['num_conc_total'][:, n], rtol=1e-5):
                raise ValueError('aerosols-file sources do not add up to diagnostics run %s '
                                 'num_conc_total at the points at %s' % (config.DIAG_RUN, ts))
            n_checked += 1
    alt, interpolated = alt_points_at(fields, stamps)

    lo, hi = axes['logDp_bins_lb'], axes['logDp_bins_rb']
    if not np.allclose(lo[1:], hi[:-1]):
        raise ValueError('PartMC logDp bins are not contiguous')
    try:
        perm = [axes['species'].index(s) for s in config.ML_SPECIES]
    except ValueError as e:
        raise KeyError('a VAE species is missing from the diagnostics: %s' % e) from None

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    per_kg = 'per kg dry air (per m^3 x ALT)'
    a3, a4 = alt[:, :, None], alt[:, :, None, None]
    pt = ('point', 'time')
    ds = xr.Dataset(
        {
            'ccn_cdf': (pt + ('eps',), f32(v['ccn_cdf']),
                        {'units': '1', 'long_name': 'CCN/CN fraction activated at s_env'}),
            'frozen_fraction': (pt + ('temperature',), f32(v['frozen_fraction']),
                                {'units': '1', 'long_name': 'frozen fraction'}),
            'scattering': (pt + ('wavelength',), f32(v['optical_scattering'] * a3),
                           {'units': 'm-1 m3 kg-1', 'basis': per_kg,
                            'long_name': 'bulk scattering coefficient'}),
            'absorption': (pt + ('wavelength',), f32(v['optical_absorption'] * a3),
                           {'units': 'm-1 m3 kg-1', 'basis': per_kg,
                            'long_name': 'bulk absorption coefficient'}),
            'dNdlogDp': (pt + ('bin',), f32(v['dNdlogDp'] * a3),
                         {'units': 'kg-1', 'basis': per_kg,
                          'long_name': 'number size distribution dN/dlog10(D) (a density)'}),
            'dMdlogDp': (pt + ('species', 'bin'), f32(v['dMdlogDp'][:, :, perm, :] * a4),
                         {'units': 'kg kg-1', 'basis': per_kg,
                          'long_name': 'per-species mass size distribution dM/dlog10(D) (a density)'}),
            'ntot': (pt, f32(v['num_conc_total'] * alt),
                     {'units': 'kg-1', 'basis': per_kg,
                      'long_name': 'number of the diagnosed particles (num_conc_total x ALT)'}),
            'ALT': (pt, f32(alt), {'units': 'm3 kg-1', 'long_name': 'inverse dry-air density '
                                   '(interpolated in time where alt_interpolated says)'}),
        },
        coords={
            'point': ('point', list(config.LOC_LABELS)),
            'point_i': ('point', np.array([p[0] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based west_east index'}),
            'point_j': ('point', np.array([p[1] for p in config.LOCATIONS], np.int32),
                        {'long_name': '1-based south_north index'}),
            'time': ('time', stamps),
            'hours': ('time', common.hours_since(stamps).astype(np.float32),
                      {'units': 'h', 'long_name': 'hours since %s' % config.TIMESTAMP_START}),
            'eps': ('eps', axes['eps_histbins'], {'units': '1', 'long_name': 'supersaturation s_env (fraction)'}),
            'temperature': ('temperature', axes['temperature_C'], {'units': 'degC'}),
            'wavelength': ('wavelength', axes['wavelength_m'], {'units': 'm'}),
            'species': ('species', list(config.ML_SPECIES), {'long_name': 'species, in the VAE order'}),
            'log10_d_lo': ('bin', lo, {'units': 'log10(m)', 'long_name': 'bin lower edge, log10 diameter'}),
            'log10_d_hi': ('bin', hi, {'units': 'log10(m)', 'long_name': 'bin upper edge, log10 diameter'}),
        },
        attrs={'diag_run': config.DIAG_RUN, 'diag_z': config.DIAG_Z,
               'alt_interpolated': ','.join(interpolated),
               'n_times_checked_against_aerosols_file': n_checked},
    )
    return ds, [products.path('partmc_fields')] + files
