#!/bin/bash
# build_run.py --batch-template for Perlmutter CPU: one node, 128 MPI ranks.
# {name}, {run} and {mail} are filled in by build_run.py.
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -J {name}
#SBATCH -o {run}/slurm-%j.out
{mail}
module load cpu PrgEnv-gnu cray-hdf5 cray-netcdf cray-parallel-netcdf
cd {run}
export OMP_NUM_THREADS=1
srun -n 128 --cpu-bind=cores ./wrf.exe
grep -q "SUCCESS COMPLETE WRF" rsl.error.0000 && rm -f rsl.out.0* && echo "WRF SUCCESS"
