#!/bin/bash
# Dry-diameter and Wasserstein checks of the schemes (baselines/dist_checks.py --part schemes) on
# a Perlmutter CPU core (one process, a few minutes). Layout as static_terms.sbatch, in its own
# copy $S/repo_d: data/ copied from $S/repo_m (sources, populations, ALT, all1_12h -> the replay),
# the current baselines/*.py, scripts/{paperfigs,extract,vendor}, training/paper_errors.py and
# baselines/data/zsrc_baselines.npz uploaded from the repo; the schemes' 12-h states from
# $S/../transport/runs/<run>/surface_end.npz. WRF-AeroMELD's part (--part aeromeld) and --merge
# run locally (seconds).
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --job-name aeromeld_dist_checks
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/dist_checks_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd ${REPO:-$S/repo_d}/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
md5sum ../data/transport/all1_12h/surface.npz ../baselines/data/zsrc_baselines.npz
timeout 18m $S/venv/bin/python ../baselines/dist_checks.py --part schemes
echo "exit $?"
