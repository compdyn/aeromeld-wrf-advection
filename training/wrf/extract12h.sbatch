#!/bin/bash
# Extract the 12-hour online run of the main model with separate limiters (per-tracer advection
# inside WRF) for the online/offline check against the 12-hour replay rep6_12h
# (checks/pertracer_online.py). One shared CPU node slice; the extraction is one Python process.
#   sbatch training/wrf/extract12h.sbatch
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J aeromeld_extract12h
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
W=/pscratch/sd/m/mwest/aeromeld/wrfml
RUN=latent_advection_cares_v3match_twin_k16_of.6.0_pertracer12h
cd $W/work/runs && ln -sfn cares_v3match_twin_k16_of.6.0_pertracer_12h $RUN
source $W/venv/bin/activate
cd $W
rm -f extracts_full/twin_k16_of.6.0_pertracer12h.npz   # extract_runs.py skips existing outputs
timeout 55m python paper/training/wrf/extract_runs.py work extracts_full --full twin_k16_of.6.0 \
  work/runs/$RUN
echo "exit $? $(date)"
ls -la extracts_full/ | grep pertracer12h
