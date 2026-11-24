#!/bin/bash
# Exact time series of the static terms at some hours (baselines/static_terms.py --time-hours),
# on a Perlmutter CPU node; layout as static_terms.sbatch. Usage:
#   sbatch --export=ALL,HOURS=0.5:1:1.5 static_time.sbatch   (colons: commas would split --export)
# then locally: python baselines/static_terms.py --merge (after downloading the parts).
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 64
#SBATCH -t 00:45:00
#SBATCH --job-name aeromeld_base_static
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/static_time_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd $S/repo/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINES_PROCS=32
timeout 43m $S/venv/bin/python ../baselines/static_terms.py --time-hours "${HOURS//:/,}"
echo "exit $?"
