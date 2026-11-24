#!/bin/bash
# The paper comparison with dry-diameter size distributions, the schemes' part
# (dry_comparison.py --part schemes, 2026-11-16), on a Perlmutter CPU core, a few minutes.
# Same layout as dist_checks.sbatch, in its copy $S/repo_d (data/ from $S/repo_m; the current
# baselines/*.py, scripts/{paperfigs,extract,vendor}, training/paper_errors.py and
# baselines/data/zsrc_baselines.npz uploaded from the repo; the schemes' 12-h states from
# $S/../transport/runs/<run>/surface_end.npz). WRF-AeroMELD's part (--part aeromeld) and
# --merge run locally (seconds).
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --job-name aeromeld_dry_comparison
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/dry_comparison_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd ${REPO:-$S/repo_d}/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
md5sum ../data/transport/all1_12h/surface.npz ../baselines/data/zsrc_baselines.npz
timeout 18m $S/venv/bin/python ../baselines/dry_comparison.py --part schemes
echo "exit $?"
