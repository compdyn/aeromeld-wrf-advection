#!/usr/bin/env python
"""
The full transport record of one WRF-AeroMELD run, compactly: at every output
time and level, each cell's latent is a mixture of the three source latents, so
store the source fractions f^ML (barycentric weights in the triangle of the
source latents, least squares) and ntot, plus the off-plane residual. With
these, any model's latent field anywhere in the run is sum_s f_s z*_s (the
fractions do not depend on the model: they agree to 6e-5 across all 54 runs).

Output (npz, float32): f (3, time, level, y, x), ntot (time, level, y, x),
offplane_max (time,) relative to the longest triangle edge, times, zsrc.

Usage: python extract_transport.py RUN_DIR LATENT_DIR OUT.npz
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

run, lat, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SCEN = (561, 635, 812)
zs = np.array([np.loadtxt(lat / ('latent_scenario_%d_0001_%08d.txt' % (s, 13)))[:-1] for s in SCEN])
E = np.stack([zs[1] - zs[0], zs[2] - zs[0]], 1)
edge = np.linalg.norm(E, axis=0).max()
P = np.linalg.pinv(E)                                                  # (2, d)
with xr.open_dataset(run / 'wrfout_d01_2010-06-17_00:00:00') as ds:
    n = len([v for v in ds.data_vars if v.startswith('ltnt_var')])
    times = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in ds['Times'].values.astype(str)])
    nt, nz, ny, nx = ds['ntot'].shape
    f = np.zeros((3, nt, nz, ny, nx), np.float32)
    ntot = ds['ntot'].values.astype(np.float32)
    off = np.zeros(nt)
    for it in range(nt):
        z = np.stack([ds['ltnt_var%d' % i][it].values for i in range(1, n + 1)]).astype(np.float64)   # (d, z, y, x)
        R = z.reshape(n, -1) - zs[0][:, None]
        l12 = P @ R
        off[it] = np.linalg.norm(R - E @ l12, axis=0).max() / edge
        f[1:, it] = l12.reshape(2, nz, ny, nx)
        f[0, it] = 1 - f[1, it] - f[2, it]
        print(times[it], 'off-plane max %.2g' % off[it], 'f range %.3g %.3g' % (f[:, it].min(), f[:, it].max()), flush=True)
np.savez_compressed(out, f=f, ntot=ntot, offplane_max=off, times=times, zsrc=zs.astype(np.float32))
print('wrote', out)
