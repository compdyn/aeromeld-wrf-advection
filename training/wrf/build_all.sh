#!/bin/bash
# Build (and optionally submit) WRF-AeroMELD runs on Perlmutter with
# scripts/inputs/build_run.py, from latents written by write_latents.py.
#   bash build_all.sh [--submit] FPIDX [FPIDX ...]
set -uo pipefail
W=/pscratch/sd/m/mwest/aeromeld/wrfml
export PAPERFIGS_INPUT_BUNDLE=$W/bundle PAPERFIGS_WRF_DIR=$W/WRF AEROMELD_WORK=$W/work HDF5_USE_FILE_LOCKING=FALSE
module load python >/dev/null 2>&1
. $W/venv/bin/activate
submit=0; if [ "${1:-}" = "--submit" ]; then submit=1; shift; fi
cd $W/paper/scripts
for f in "$@"; do
  run=$AEROMELD_WORK/runs/latent_advection_cares_v3match_${f}_5e7
  if [ ! -d $run ]; then
    python inputs/build_run.py $f --batch-template $W/paper/training/wrf/batch_perlmutter.sh > $W/logs/build_$f.log 2>&1
    echo "$f build exit $? : $(grep -E 'Built|FAILED|ok \(|Error' $W/logs/build_$f.log | tr '\n' ' ')"
  fi
  if [ $submit = 1 ] && [ -f $run/batch.sh ] && ! ls $run/slurm-*.out >/dev/null 2>&1; then
    (cd $run && sbatch --parsable batch.sh | sed "s/^/$f submitted /")
  fi
done
