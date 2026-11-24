#!/bin/bash
# The dry-diameter training cache (dry_cache.py, 2026-11-16) on a shared CPU node, a few minutes.
#   CODE=<dir with dry_cache.py> sbatch dry_cache.sbatch [--basis drydiam]   (round 39, 2026-11-19: -c 32 for the second cache in memory)
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J aeromeld_dry_cache
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
module load pytorch/2.8.0
source ${M}/code_partnn/venv/bin/activate
echo "== $(hostname) $(date)"
python ${CODE:-$M/affx/r36/code}/dry_cache.py "$@"
echo "exit $? $(date)"
