#!/bin/bash
# Analysis commands (affx_mixk.py, affx_wrf3_npz.py loops, ...) in a batch job instead of on a login node:
#   sbatch [-t HH:MM:SS] alcf/affx/affx_ana.sbatch 'COMMAND ...'
# runs COMMAND with bash in $SCRATCH/aeromeld/affx, with the pytorch module.
#SBATCH -A gen0019_g
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -J affx_ana
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
cd /pscratch/sd/m/mwest/aeromeld/affx
module load pytorch/2.8.0
echo "== $(hostname) $(date) $1"
bash -c "$1" 2>&1 | grep -v "UserWarning\|warnings.warn"
echo "== done $(date)"
