#!/bin/bash
# The lowest level's inverse dry-air density ALT (m3 kg-1) at every output of the 12-hour
# WRF-AeroMELD run (15-minute outputs), for converting per-kg diagnostics to per-m3 ones
# (the test set's units: optics in Mm-1). Writes extracts_full/alt_surface_12h.npz (~5 MB).
#   sbatch training/wrf/alt12h.sbatch
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J aeromeld_alt12h
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
W=/pscratch/sd/m/mwest/aeromeld/wrfml
source $W/venv/bin/activate
cd $W
timeout 15m python - <<'EOF'
import netCDF4
import numpy as np
f = 'work/runs/cares_v3match_twin_k16_of.6.0_pertracer_12h/wrfout_d01_2010-06-17_00:00:00'
with netCDF4.Dataset(f) as ds:     # no ALT in this output: WRF's equation of state, as transport/run.py
    times = np.array([b''.join(t).decode() for t in ds['Times'][:]])
    r_d, r_v, p0 = 287.0, 461.6, 1e5
    cvpm = -(3.5 * r_d - r_d) / (3.5 * r_d)
    th = np.asarray(ds['T'][:, 0], np.float64) + 300.0
    qv = np.asarray(ds['QVAPOR'][:, 0], np.float64)
    p = np.asarray(ds['P'][:, 0] + ds['PB'][:, 0], np.float64)
    alt = ((r_d / p0) * th * (1 + r_v / r_d * qv) * (p / p0) ** cvpm).astype(np.float32)
np.savez_compressed('extracts_full/alt_surface_12h.npz', alt=alt, times=times)
print('alt', alt.shape, float(alt.min()), float(alt.max()), times[0], times[-1])
EOF
echo "exit $? $(date)"
