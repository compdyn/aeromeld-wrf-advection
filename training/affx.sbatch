#!/bin/bash
# Run the affx.py experiments listed in a task file (one argument string per
# line, blank lines and # comments skipped), in order, on one shared GPU.
#   sbatch [-t HH:MM:SS] training/affx.sbatch TASKFILE [OUTDIR]   (AFFX=path/to/affx.py to override)
#SBATCH -A gen0019_g
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -J affx
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
TASKS=${1:?usage: affx.sbatch TASKFILE [OUTDIR]}
OUT=${2:-${M}/affx/runs}
module load pytorch/2.8.0
source ${M}/code_partnn/venv/bin/activate
export PYTHONPATH=${M}/code_partnn
cd ${M}/code_partnn
echo "== $(hostname) $(date) tasks=${TASKS}"
nvidia-smi -L
grep -v '^\s*#' "${TASKS}" | grep -v '^\s*$' | while read -r LINE; do
  echo "== $(date +%T) ${LINE}"
  eval python ${AFFX:-${M}/wrfml/paper/training/affx.py} --out ${OUT} ${LINE} 2>&1 | grep -v "UserWarning\|warnings.warn"
done
echo "== done $(date)"
