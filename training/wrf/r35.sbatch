#!/bin/bash
# Round 35 (twin_k16kl0_of, ten seeds, four at a time) and its test-set K-curves, on one GPU node.
#   sbatch training/wrf/r35.sbatch
#SBATCH -A gen0019_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 01:40:00
#SBATCH -J r35_twin_k16kl0
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/logs/%x_%j.out
T=/pscratch/sd/m/mwest/aeromeld/wrfml/paper/training
bash $T/wrf/run_r35_interactive.sh
srun -n 1 --gpus 1 -c 32 bash $T/wrf/after_r35.sh
