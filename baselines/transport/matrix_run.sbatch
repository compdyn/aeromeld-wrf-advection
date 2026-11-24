#!/bin/bash
# 12-hour offline replay of MATRIX mechanism 1 (baselines/matrix.py) on the paper's 12-h WRF fluxes,
# IC/BC/emissions and transport code, one shared GPU (as baselines_array.sbatch's tasks):
#   oracle + sep:matrix1 (matrix.adjust after every step, --calcsize-every 1) + sep:matrix1_nocs
# Before: verify_matrix.py (matrix.py + the constants-only diag.py reproduce the local adjust outputs and
# event counts, numpy and cupy). After: check_run.py (finite, nonnegative, totals, oracle and N against the
# paper's all1_12h, matrix.adjust's species conservation, events, and the difference from _nocs).
# Environment: STEPS (default 2160), TAG (run-name suffix, default 12h), CS_EVERY (default 1),
# LIMIT (python timeout, default 50m), CODE (default $B/code_m), INP (default $B/inputs_m), RUNS (default $B/runs).
#   cd $B/code_m && sbatch matrix_run.sbatch
#   cd $B/code_m && sbatch --export=ALL,STEPS=45,TAG=smoke,LIMIT=10m -t 00:15:00 -J aeromeld_base_matrix_smoke matrix_run.sbatch
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:55:00
#SBATCH -J aeromeld_base_matrix
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -euo pipefail
B=/pscratch/sd/m/mwest/aeromeld/baselines/transport
T=/pscratch/sd/m/mwest/aeromeld/transport
ORIG=/pscratch/sd/m/mwest/aeromeld/wrfml/work/runs/cares_v3match_twin_k16_of.6.0_pertracer_12h/wrfout_d01_2010-06-17_00:00:00
PY=/pscratch/sd/m/mwest/aeromeld/wrfml/venv/bin/python
STEPS=${STEPS:-2160}
TAG=${TAG:-12h}
CS_EVERY=${CS_EVERY:-1}
LIMIT=${LIMIT:-50m}
CODE=${CODE:-$B/code_m}
INP=${INP:-$B/inputs_m}
RUNS=${RUNS:-$B/runs}
cd $CODE
out=$RUNS/matrix1_${TAG}
echo "matrix1 -> $out (steps $STEPS, adjust every $CS_EVERY, code $CODE, inputs $INP)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum run_baselines.py check_run.py verify_matrix.py matrix.py modal.py diag.py $T/code/run.py $T/code/schemes.py $T/code/wrfadv.py $INP/* || true
timeout 5m $PY verify_matrix.py $CODE $CODE/matrix_test.npz --gpu
timeout $LIMIT $PY run_baselines.py --out $out --gpu --models matrix1 --nocs matrix1 --matrix-adjust matrix1 \
  --steps $STEPS --calcsize-every $CS_EVERY --zsrc $INP/zsrc_baselines.npz \
  --fluxes $T/wrf_fluxes12/wrfout_d01_2010-06-17_00:00:00 --orig $ORIG --pmc $T/partmc_inputs \
  --transport-code $T/code --modal-path $CODE --chunk 72
timeout 10m $PY check_run.py $out --reference $T/runs/all1_12h/surface.npz \
  --layout $INP/state_layout.json --zsrc $INP/zsrc_baselines.npz --modal-path $CODE || true
ls -la $out
