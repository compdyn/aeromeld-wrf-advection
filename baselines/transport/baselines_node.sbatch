#!/bin/bash
# Array tasks of baselines_array.sbatch run side by side on one full GPU node (regular QOS),
# one task per GPU; same run directories, extracts and checks as the shared-QOS array.
# Used for tasks 3 (sec20w) and 4 (mam4a + mam4a_nocs) after the shared array waited ~2.5 h.
# Environment: TASKS (default "3 4"), and baselines_array.sbatch's STEPS, TAG, LIMIT, CODE, INP.
# ALT_JOB (added 2026-11-14): the id of a shared-QOS array doing the same tasks; at start, if any of
# its tasks runs this job exits, else the array is cancelled (so the two never write the same runs).
#   cd $B/code2 && sbatch baselines_node.sbatch
# Logs: the job's own log, plus logs/aeromeld_base_rep12c_<jobid>_<task>.out per task.
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 01:30:00
#SBATCH -J aeromeld_base_rep12c
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
B=/pscratch/sd/m/mwest/aeromeld/baselines/transport
L=/pscratch/sd/m/mwest/aeromeld/logs
export STEPS=${STEPS:-2160} TAG=${TAG:-12h} LIMIT=${LIMIT:-80m}
export CODE=${CODE:-$B/code2} INP=${INP:-$B/inputs2}
TASKS=${TASKS:-"3 4"}
if [ -n "${ALT_JOB:-}" ]; then
  if squeue -h -j "$ALT_JOB" -t RUNNING,COMPLETING 2>/dev/null | grep -q .; then
    echo "ALT_JOB $ALT_JOB is running: exiting"; exit 0
  fi
  scancel "$ALT_JOB" 2>/dev/null; echo "cancelled ALT_JOB $ALT_JOB (pending)"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pids=()
gpu=0
for k in $TASKS; do
  echo "task $k on GPU $gpu -> $L/aeromeld_base_rep12c_${SLURM_JOB_ID}_$k.out"
  CUDA_VISIBLE_DEVICES=$gpu SLURM_ARRAY_TASK_ID=$k bash $CODE/baselines_array.sbatch \
    > $L/aeromeld_base_rep12c_${SLURM_JOB_ID}_$k.out 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "all tasks finished, rc $rc"
exit $rc
