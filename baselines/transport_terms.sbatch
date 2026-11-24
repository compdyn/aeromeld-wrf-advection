#!/bin/bash
# Transport terms of the baselines (baselines/transport_terms.py) on Perlmutter CPU nodes, one
# group of cases per node; layout as static_terms.sbatch (repo copy and venv in
# /pscratch/sd/m/mwest/aeromeld/baselines/static/). Usage (groups separated by ';', cases by
# '+', options per group after '|'; no commas: sbatch --export splits on them):
#   sbatch -N 3 --export=ALL,TT_GROUPS='sec8+oracle:sec8+donor:sec8|--extras;mam4|--extras --maps;...' transport_terms.sbatch
#   REPO=$S/repo_m: another copy of the repo (default $S/repo; added 2026-11-14 for MATRIX)
#   TT_TIMEOUT: per-case python timeout (default 28m; added 2026-11-14 for TOMAS-40, with -q regular -t ...)
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --job-name aeromeld_base_tterms
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/tterms_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd ${REPO:-$S/repo}/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINES_PROCS=120
IFS=';' read -ra G <<< "$TT_GROUPS"
for g in "${G[@]}"; do
    cases=${g%%|*}; opts=${g#*|}; [ "$opts" = "$g" ] && opts=""
    for c in ${cases//+/ }; do
        echo "group: $c $opts"
    done
    srun -N1 -n1 -c 256 --cpu-bind=none bash -c "
        for c in ${cases//+/ }; do
            timeout ${TT_TIMEOUT:-28m} $S/venv/bin/python ../baselines/transport_terms.py --case \$c $opts
            echo \"exit \$? \$c\"
        done" &
done
wait
echo "all done"
