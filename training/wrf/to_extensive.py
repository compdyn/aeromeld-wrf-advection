#!/usr/bin/env python
"""A WRF-AeroMELD run directory for per-tracer latent advection, from an existing one.

Per-tracer latent advection (WRF branch latent_per_tracer, training/wrf/
latent_per_tracer.patch) stores the latents extensively, ltnt_var = ntot * z (per
kg), and advects them with WRF's scalar advection like any chem species. The
input builder writes them intensively (z), so this copies a run directory and
converts:

  wrfinput_d01  ltnt_var<k> -> ntot * ltnt_var<k>
  wrfbdy_d01    ltnt_var<k>_B{XS,XE,YS,YE} -> ntot_B... * ltnt_var<k>_B...; the
                tendencies _BT... recomputed from the converted values at the two
                ends of each boundary interval
  wrfchemi_*    unchanged: E_LTNT_VAR<k> is the emitted population's latent, and
                the emission code adds dN * E_LTNT_VAR<k>

Everything else is linked. Writes LATENTS_EXTENSIVE in the new directory, so the
extractor (extract_runs.py) divides by ntot, and batch.sh pointing at EXE.

Usage (Perlmutter): python to_extensive.py SRC_RUN_DIR DST_RUN_DIR EXE
"""
import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np

src, dst, exe = (Path(a).resolve() for a in sys.argv[1:4])
dst.mkdir(parents=True, exist_ok=False)
skip = {'wrfinput_d01', 'wrfbdy_d01', 'namelist.output', 'batch.sh', 'wrf.exe'}
for f in src.iterdir():
    if f.name in skip or f.name.startswith(('rsl.', 'slurm-', 'wrfout_')):
        continue
    (dst / f.name).symlink_to(f.resolve() if f.is_symlink() else f)
(dst / 'wrf.exe').symlink_to(exe)
for name in ('wrfinput_d01', 'wrfbdy_d01'):
    shutil.copyfile(src / name, dst / name)

with netCDF4.Dataset(dst / 'wrfinput_d01', 'r+') as ds:
    n = ds['ntot'][:].astype(np.float64)
    lv = sorted(v for v in ds.variables if v.startswith('ltnt_var'))
    for v in lv:
        ds[v][:] = (ds[v][:].astype(np.float64) * n).astype(np.float32)
    print('wrfinput: %d latents x ntot' % len(lv))

with netCDF4.Dataset(dst / 'wrfbdy_d01', 'r+') as ds:
    dt = float(ds.getncattr('INTERVAL_SECONDS')) if 'INTERVAL_SECONDS' in ds.ncattrs() else 21600.0
    lv = sorted({v.rsplit('_', 1)[0] for v in ds.variables if v.startswith('ltnt_var') and v.endswith('_BXS')})
    for v in lv:
        for e in ('XS', 'XE', 'YS', 'YE'):
            z, zt = ds['%s_B%s' % (v, e)][:].astype(np.float64), ds['%s_BT%s' % (v, e)][:].astype(np.float64)
            n, nt = ds['ntot_B%s' % e][:].astype(np.float64), ds['ntot_BT%s' % e][:].astype(np.float64)
            q0 = z * n
            q1 = (z + zt * dt) * (n + nt * dt)
            ds['%s_B%s' % (v, e)][:] = q0.astype(np.float32)
            ds['%s_BT%s' % (v, e)][:] = ((q1 - q0) / dt).astype(np.float32)
    print('wrfbdy: %d latents x ntot (interval %g s)' % (len(lv), dt))

(dst / 'LATENTS_EXTENSIVE').write_text('ltnt_var<k> = ntot * z (per kg); converted from %s\n' % src)
batch = (src / 'batch.sh').read_text().replace(str(src), str(dst))
batch = batch.replace('#SBATCH -J ', '#SBATCH -J pt_', 1)            # job name: pt_<run>, per tracer
(dst / 'batch.sh').write_text(batch)
print('wrote', dst)
