#!/bin/bash
# Test-set K-curves of the sectional and modal baselines (baselines/kcurves.py run), one split
# seed per array task, on the CPU account. Code in $B/code (baselines/, training/paper_errors.py
# and affx_mixk.py, scripts/paperfigs/{__init__,config}.py, scripts/vendor/extract_diagnostics.py,
# copied from the repo), results in $B/data/kcurves_s<seed>.json (resumable).
#   sbatch --array=0-9 kcurves.sbatch                       # all ten splits
#   sbatch --array=0 -t 00:30:00 kcurves.sbatch --K 1 2 16 5000 --nmix 100 --out-tag test   # a test
# Resumable: a rerun of a seed skips the (representation, K) pairs already in its JSON.
#   KC_CODE=$B/code2 sbatch --array=0-9 kcurves.sbatch --reps prdiag,sec20w,mam4a --out-tag b   # added models
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -J aeromeld_base_kcurve
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%A_%a.out
set -uo pipefail
B=/pscratch/sd/m/mwest/aeromeld/baselines/kcurves
si=${SLURM_ARRAY_TASK_ID:-0}
tag=""
args=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--out-tag" ]; then tag="_$2"; shift 2; else args+=("$1"); shift; fi
done
module load pytorch/2.8.0
# one worker per physical core (-c counts hyperthreads); each diag.py worker peaks at ~3 GB (the
# Mie chunk of 60000 particle-wavelengths), and -c 32 in shared gets ~64 GB: 32 workers ran out
export BASELINES_PROCS=$(( SLURM_CPUS_PER_TASK / 2 )) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $B/data
cd ${KC_CODE:-$B/code}/baselines        # KC_CODE: another copy of the code (e.g. $B/code2)
echo "== $(hostname) $(date) seed $si ${args[*]:-}"
python -c "import torch, numpy, h5py; print('torch', torch.__version__, 'numpy', numpy.__version__, 'h5py', h5py.__version__)"
timeout ${KC_TIMEOUT:-115m} \
  python kcurves.py run --seed $si --out $B/data/kcurves${tag}_s$si.json "${args[@]}" 2>&1 | grep -v "UserWarning\|warnings.warn"
rc=${PIPESTATUS[0]}
echo "== done $(date) exit $rc"
exit $rc
