#!/bin/bash
# Rounds 36 and 37 (2026-11-16): the lines of a task file (affx.py options, as affx.sbatch) side by
# side on one full GPU node, two or three per GPU. Round 36: the dry-diameter decoders with a frozen
# encoder (tasks/r36/r36_dry_k16.txt); round 37: the clean dry rebuild, five decoders x two seeds per
# job (tasks/r37/r37_dry_s<ab>.txt, with -J aeromeld_r37_s<ab>).
#   CODE=<dir with affx.py, paper_errors.py> sbatch [-J NAME] dry_decoders.sbatch TASKFILE [OUTDIR]
# Logs: the job's own log, plus logs/<job name>_<jobid>_<name>.out per line.
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 02:00:00
#SBATCH -J aeromeld_dry_k16
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
L=$M/logs
TASKS=${1:?usage: dry_decoders.sbatch TASKFILE [OUTDIR]}
OUT=${2:-$M/affx/r36}
CODE=${CODE:-$M/affx/r36/code}
module load pytorch/2.8.0
source ${M}/code_partnn/venv/bin/activate
export PYTHONPATH=${M}/code_partnn OMP_NUM_THREADS=4
cd ${M}/code_partnn
echo "== $(hostname) $(date) tasks=${TASKS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pids=()
i=0
while read -r LINE; do
  name=$(echo "$LINE" | awk '{print $2}')
  gpu=$((i % 4))
  echo "$name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu eval python $CODE/affx.py --out $OUT $LINE > $L/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$name.out 2>&1 &
  pids+=($!)
  i=$((i + 1))
done < <(grep -v '^\s*#' "$TASKS" | grep -v '^\s*$')
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "== all finished, rc $rc $(date)"
exit $rc
