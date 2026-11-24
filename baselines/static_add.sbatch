#!/bin/bash
# Static terms of the added models (static_terms.py --add-models) on a Perlmutter CPU node;
# layout as static_terms.sbatch. Usage (colons in HOURS; commas would split --export):
#   sbatch --export=ALL,MODELS=sec20w:mam4a,HOURS=0.5:1:1.5,TAG=a,W12=--with-12h static_add.sbatch
#   REPO=$S/repo_m: another copy of the repo (default $S/repo; added 2026-11-14 for MATRIX)
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH --job-name aeromeld_base_static
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/static_add_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd ${REPO:-$S/repo}/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINES_PROCS=120
timeout 23m $S/venv/bin/python ../baselines/static_terms.py --add-models "${MODELS//:/,}" \
    --add-hours "$HOURS" --tag "$TAG" $W12
echo "exit $?"
