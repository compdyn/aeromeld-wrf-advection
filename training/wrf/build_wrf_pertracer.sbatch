#!/bin/bash
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -J wrf_pertracer_build
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/wrfml/build_pertracer_%j.log
# Incremental build of WRF-AeroMELD with per-tracer latent advection: a copy of the
# built latent_advection tree (838cfd48) on the local branch latent_per_tracer
# (training/wrf/latent_per_tracer.patch). Same environment as build_wrf.sbatch; no
# clean or configure, so only the changed files and their dependents recompile.
set -x
module load cpu PrgEnv-gnu cray-hdf5 cray-netcdf cray-parallel-netcdf
cd /pscratch/sd/m/mwest/aeromeld/wrfml/WRF_pertracer
git log --oneline -1; git branch --show-current; git diff --stat
export WRF_EM_CORE=1 WRF_CHEM=1 WRF_KPP=0 NETCDF_classic=1 WRFIO_NCD_LARGE_FILE_SUPPORT=1
unset PNETCDF_QUILT
export HDF5=$HDF5_DIR NETCDF=$NETCDF_DIR PNETCDF=$PNETCDF_DIR
export PATH=$NETCDF_DIR/bin:$HDF5_DIR/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF_DIR/lib:/usr/lib64:$LD_LIBRARY_PATH
export J="-j 16"
./compile em_real > compile_pertracer.log 2>&1
ls -l main/*.exe
grep -c -i " error" compile_pertracer.log
tail -20 compile_pertracer.log
