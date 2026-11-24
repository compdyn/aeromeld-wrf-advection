#!/bin/bash
# Round 34 on one interactive GPU node (4 GPUs): the ten seeds of twin_mix0_of, four at a time, as
# affx.sbatch would run them one per GPU. Usage, from a login node:
#   salloc -A gen0019_g -C gpu -q interactive -N 1 -G 4 -c 128 -t 01:30:00 bash training/wrf/run_r34_interactive.sh
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
TASKS=$M/wrfml/paper/training/tasks/r34
OUT=$M/affx/r34
module load pytorch/2.8.0
source $M/code_partnn/venv/bin/activate
export PYTHONPATH=$M/code_partnn
cd $M/code_partnn
run() {   # seed
  local line; line=$(grep -v '^\s*#' $TASKS/r34_twin_mix0_s$1.txt | grep -v '^\s*$')
  # srun gives each step its own GPU (--exact: concurrent steps get disjoint resources)
  srun -n 1 --gpus 1 --exact -c 32 bash -c "python $M/wrfml/paper/training/affx.py --out $OUT $line" \
    > $M/logs/r34_twin_mix0_s$1.log 2>&1
}
for batch in "0 1 2 3" "4 5 6 7" "8 9"; do
  for s in $batch; do run $s & done; wait
  echo "batch $batch done $(date)"
done
grep -h "test geo" $M/logs/r34_twin_mix0_s*.log | cut -c1-120
