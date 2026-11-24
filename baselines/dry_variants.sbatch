#!/bin/bash
# Every baseline number the paper quotes, with the two size distributions binned by dry diameter
# (baselines/dry_variants.py, 2026-11-16, the clean dry rebuild), on a Perlmutter CPU core, a few
# minutes. Same layout as dist_checks.sbatch, in its copy $S/repo_d (data/ from $S/repo_m; the
# current baselines/*.py, scripts/{paperfigs,extract,vendor}, training/paper_errors.py and
# baselines/data/*.json, zsrc_baselines.npz uploaded from the repo; the schemes' 12-h states from
# $S/../transport/runs/<run>/surface_end.npz).
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH --job-name aeromeld_dry_variants
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/baselines/static/dry_variants_%j.log
S=/pscratch/sd/m/mwest/aeromeld/baselines/static
cd ${REPO:-$S/repo_d}/scripts || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
md5sum ../data/transport/all1_12h/surface.npz ../baselines/data/zsrc_baselines.npz
timeout 35m $S/venv/bin/python ../baselines/dry_variants.py "$@"
echo "exit $?"
