#!/usr/bin/env python
"""
Small extracts of WRF-AeroMELD runs, for decoding elsewhere (needs xarray only):
per run, the surface latents z (ltnt_var1..9) and ntot at the final instant
(04 UTC) and at the start, the three sampling points at every wrfout time, and
the three source latents the run was built from. Same slicing as the paper's
wrfml_fields product (level 0; points are WRF's 1-based (i, j)).

With --full FPIDX[,FPIDX...], those runs also keep the surface latents and ntot at
every output time (z_surf: (time, latent, y, x), ntot_surf: (time, y, x); ~18 MB).

Usage: python extract_runs.py WORK OUTDIR [--full FPIDX,...] [RUN_DIR ...]
       (default: every SUCCESS run in WORK/runs)
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

argv = sys.argv[1:]
full = set()
if '--full' in argv:
    i = argv.index('--full'); full = set(argv[i + 1].split(',')); del argv[i:i + 2]
work, outdir = Path(argv[0]), Path(argv[1])
outdir.mkdir(parents=True, exist_ok=True)
runs = [Path(r) for r in argv[2:]] or sorted((work / 'runs').glob('latent_advection_cares_v3match_*'))
# Runs with a LATENTS_EXTENSIVE marker (per-tracer latent advection, to_extensive.py)
# store ltnt_var = ntot * z; their latents are divided by ntot here.
POINTS = {'A': (64, 54), 'B': (95, 24), 'C': (104, 14)}   # = scripts/paperfigs/config.LOCATIONS (2026-11-19, by mass with water)
SCEN = (561, 635, 812)
for run in runs:
    rsl = run / 'rsl.error.0000'
    if not (rsl.exists() and 'SUCCESS COMPLETE WRF' in rsl.read_text(errors='ignore')):
        print('skip (not SUCCESS)', run.name); continue
    name = run.name.replace('latent_advection_cares_v3match_', '')
    out = outdir / ('%s.npz' % name)
    if out.exists():
        print('exists', out.name); continue
    fpidx = name.rsplit('_', 1)[0]
    lat = work / 'latents' / fpidx
    zsrc = np.array([np.loadtxt(lat / ('latent_scenario_%d_0001_%08d.txt' % (s, 13)))[:-1] for s in SCEN])
    extensive = (run / 'LATENTS_EXTENSIVE').exists()     # per-tracer runs store ntot * z (to_extensive.py)
    with xr.open_dataset(run / 'wrfout_d01_2010-06-17_00:00:00') as ds:
        if extensive:
            ds = ds.copy()
            for v in [v for v in ds.data_vars if v.startswith('ltnt_var')]:
                ds[v] = ds[v] / ds['ntot']
        n = len([v for v in ds.data_vars if v.startswith('ltnt_var')])
        names = ['ltnt_var%d' % i for i in range(1, n + 1)]
        times = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in ds['Times'].values.astype(str)])
        z_dom = np.stack([ds[v][[0, -1], 0].values for v in names], 1).astype(np.float32)  # (2, latent, y, x)
        ntot_dom = ds['ntot'][[0, -1], 0].values.astype(np.float32)
        z_pts = np.stack([np.stack([ds[v][:, 0, j - 1, i - 1].values for (i, j) in POINTS.values()])
                          for v in names], 1).astype(np.float32)                       # (point, latent, time)
        ntot_pts = np.stack([ds['ntot'][:, 0, j - 1, i - 1].values for (i, j) in POINTS.values()]).astype(np.float32)
        extra = {}
        if fpidx in full:
            extra = dict(z_surf=np.stack([ds[v][:, 0].values for v in names], 1).astype(np.float32),
                         ntot_surf=ds['ntot'][:, 0].values.astype(np.float32))
    assert np.isfinite(z_dom).all() and np.isfinite(z_pts).all(), run
    np.savez_compressed(out, z_dom=z_dom, ntot_dom=ntot_dom, dom_times=times[[0, -1]], z_pts=z_pts,
                        ntot_pts=ntot_pts, times=times, points=list(POINTS), point_ij=np.array(list(POINTS.values())),
                        zsrc=zsrc, scenarios=np.array(SCEN), fpidx=fpidx, **extra)
    print('wrote', out.name, 'latents', n, 'times', len(times), times[-1], 'full' if extra else '')
