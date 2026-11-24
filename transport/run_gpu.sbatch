#!/bin/bash
# Offline replay of the CARES aerosol transport with every scheme (run.py), one GPU.
# Usage (from $SCRATCH/aeromeld/transport/code): sbatch run_gpu.sbatch <run name> [run.py options]
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -J transport_schemes
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
set -e
cd /pscratch/sd/m/mwest/aeromeld/transport/code
name=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/pscratch/sd/m/mwest/aeromeld/wrfml/venv/bin/python run.py --out ../runs/$name --gpu "$@"
