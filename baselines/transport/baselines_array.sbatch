#!/bin/bash
# 12-hour offline replays of the sectional and modal baselines (run_baselines.py) on the
# paper's 12-h WRF fluxes, IC/BC/emissions and transport code, one GPU per task:
#   0  sec8                     oracle + sep:sec8
#   1  mam4  + mam4_nocs        oracle + sep, MAM calcsize every CS_EVERY steps on mam4 only
#   2  mam4l + mam4l_nocs       the same with MAM4's species
#   3  sec20w                   oracle + sep:sec20w (20 bins in wet diameter, 320 tracers)
#   4  mam4a + mam4a_nocs       as 1, large fresh carbon projected to accumulation
#   5  tomas40                  TOMAS-40 (362 tracers) with TOMAS's mnfix every CS_EVERY steps
#   6  tomas40 (no mnfix)       the same without mnfix, in runs/tomas40nocs_<TAG>
#   7  mam4la + mam4la_nocs     MAM4 species (as 2), large fresh carbon projected to accumulation
#   8  sec20                    oracle + sep:sec20 (20 bins in dry diameter, 320 tracers; 2026-11-16, the dry rebuild)
#   (5-7 added 2026-11-14 with CODE=$B/code_t INP=$B/inputs_t; verify_tomas.py checks mnfix on the GPU)
# Before: verify_modal.py (modal.py + the constants-only diag.py reproduce the local calcsize
# outputs, numpy and cupy). After: check_run.py (finite, nonnegative, totals, oracle and N
# against the paper's all1_12h, calcsize conservation and direction).
# Environment: STEPS (default 2160), TAG (run-name suffix, default 12h), CS_EVERY (default 1),
# LIMIT (python timeout, default 85m), CODE (default $B/code), INP (default $B/inputs), RUNS (default $B/runs). Smoke run:
#   cd $B/code && sbatch --export=ALL,STEPS=45,TAG=smoke,LIMIT=12m -t 00:15:00 -J aeromeld_base_smoke baselines_array.sbatch
# Full run:
#   cd $B/code && sbatch baselines_array.sbatch
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH --array=0-2
#SBATCH -J aeromeld_base_rep12
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%A_%a.out
set -euo pipefail
B=/pscratch/sd/m/mwest/aeromeld/baselines/transport
T=/pscratch/sd/m/mwest/aeromeld/transport
ORIG=/pscratch/sd/m/mwest/aeromeld/wrfml/work/runs/cares_v3match_twin_k16_of.6.0_pertracer_12h/wrfout_d01_2010-06-17_00:00:00
PY=/pscratch/sd/m/mwest/aeromeld/wrfml/venv/bin/python
STEPS=${STEPS:-2160}
TAG=${TAG:-12h}
CS_EVERY=${CS_EVERY:-1}
LIMIT=${LIMIT:-85m}
CODE=${CODE:-$B/code}
INP=${INP:-$B/inputs}
RUNS=${RUNS:-$B/runs}
cd $CODE
case $SLURM_ARRAY_TASK_ID in
  0) name=sec8;  opts="--models sec8" ;;
  1) name=mam4;  opts="--models mam4 --nocs mam4 --calcsize mam4" ;;
  2) name=mam4l; opts="--models mam4l --nocs mam4l --calcsize mam4l" ;;
  3) name=sec20w; opts="--models sec20w" ;;
  4) name=mam4a; opts="--models mam4a --nocs mam4a --calcsize mam4a" ;;
  5) name=tomas40; opts="--models tomas40 --mnfix tomas40" ;;
  6) name=tomas40nocs; opts="--models tomas40" ;;
  7) name=mam4la; opts="--models mam4la --nocs mam4la --calcsize mam4la" ;;
  8) name=sec20; opts="--models sec20" ;;
  *) echo "no task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
out=$RUNS/${name}_${TAG}
echo "task $SLURM_ARRAY_TASK_ID: $name -> $out (steps $STEPS, calcsize every $CS_EVERY, code $CODE, inputs $INP)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum run_baselines.py check_run.py verify_modal.py modal.py diag.py $T/code/run.py $T/code/schemes.py $T/code/wrfadv.py $INP/* || true
timeout 5m $PY verify_modal.py $CODE $CODE/modal_test.npz --gpu
if [ -f $CODE/tomas_test.npz ]; then timeout 5m $PY verify_tomas.py $CODE $CODE/tomas_test.npz --gpu; fi
timeout $LIMIT $PY run_baselines.py --out $out --gpu $opts --steps $STEPS --calcsize-every $CS_EVERY \
  --zsrc $INP/zsrc_baselines.npz \
  --fluxes $T/wrf_fluxes12/wrfout_d01_2010-06-17_00:00:00 --orig $ORIG --pmc $T/partmc_inputs \
  --transport-code $T/code --modal-path $CODE --chunk 72
timeout 10m $PY check_run.py $out --reference $T/runs/all1_12h/surface.npz \
  --layout $INP/state_layout.json --zsrc $INP/zsrc_baselines.npz --modal-path $CODE || true
ls -la $out
