#!/bin/bash
# 12-hour offline replays for every training seed (job array index = seed): all five
# decoders' source latents, schemes oracle, donor, sep and shB, on the 12-hour WRF fluxes.
# Each run also writes two small extracts for download:
#   surface_end.npz    the outputs at 0, 4 and 12 h (all seeds: trade-offs, bars, seeds)
#   surface_30min.npz  every 30 minutes (the representative seeds' time series)
# Seed 6 repeats rep6_12h (twin_k16 alone), a check that the runs reproduce.
# The dry rebuild (2026-11-16): REP_PREFIX=dry ZSRC_DIR=zsrc_dry sbatch -J aeromeld_rep12dry rep12_array.sbatch
# (runs dry<seed>_12h with the round-37 source latents).
#   cd $SCRATCH/aeromeld/transport/code && sbatch rep12_array.sbatch
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH --array=0-9
#SBATCH -J aeromeld_rep12
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%A_%a.out
set -euo pipefail
cd /pscratch/sd/m/mwest/aeromeld/transport/code
k=$SLURM_ARRAY_TASK_ID
out=../runs/${REP_PREFIX:-all}${k}_12h
PY=/pscratch/sd/m/mwest/aeromeld/wrfml/venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
timeout 55m $PY run.py --out $out --gpu \
  --models twin_none,twin_pairs,twin_k16,twin_k32,ax_rel1 --schemes oracle,donor,sep,shB --steps 2160 \
  --zsrc ../${ZSRC_DIR:-zsrc}/zsrc_s${k}.npz --fluxes ../wrf_fluxes12/wrfout_d01_2010-06-17_00:00:00 \
  --orig /pscratch/sd/m/mwest/aeromeld/wrfml/work/runs/cares_v3match_twin_k16_of.6.0_pertracer_12h/wrfout_d01_2010-06-17_00:00:00
timeout 10m $PY - "$out" <<'EOF'
import sys
import numpy as np
out = sys.argv[1]
with np.load(out + '/surface.npz', allow_pickle=False) as f:
    s = {k: f[k] for k in f.files}
t = s['time_s']
for name, keep in (('surface_end', np.isin(t, [0.0, 4 * 3600.0, 12 * 3600.0])),
                   ('surface_30min', np.isclose(t % 1800.0, 0.0))):
    sub = {k: (v[keep] if v.ndim and v.shape[0] == t.size else v) for k, v in s.items()}
    np.savez_compressed('%s/%s.npz' % (out, name), **sub)
    print(name, int(keep.sum()), 'outputs')
EOF
ls -la $out
