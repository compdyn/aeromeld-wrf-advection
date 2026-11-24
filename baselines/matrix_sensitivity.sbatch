#!/bin/bash
# MATRIX's other mechanisms and the OCAR diagnosis choice, static 12-h terms (matrix_sensitivity.py) on a
# Perlmutter CPU node; layout as static_add.sbatch, in the repo copy REPO (default $S/repo_m).
#   cd $S/repo_m/baselines && sbatch matrix_sensitivity.sbatch
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --job-name aeromeld_base_matrix_sens
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/matrix_sens_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd ${REPO:-$S/repo_m}/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINES_PROCS=120
timeout 18m $S/venv/bin/python ../baselines/matrix_sensitivity.py
echo "exit $?"
