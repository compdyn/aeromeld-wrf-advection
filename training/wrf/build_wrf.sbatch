#!/bin/bash
#SBATCH -A gen0019
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH -t 02:30:00
#SBATCH -J wrfml_build
#SBATCH -o /pscratch/sd/m/mwest/aeromeld/wrfml/build_%j.log
# Build WRF-AeroMELD (compdyn/WRF, release wrf-aeromeld-v1.0) on Perlmutter CPU:
# GNU dmpar (configure option 34, basic nesting), Cray MPI wrappers (NERSC recipe).
set -x
module load cpu PrgEnv-gnu cray-hdf5 cray-netcdf cray-parallel-netcdf
module -t list
cd /pscratch/sd/m/mwest/aeromeld/wrfml/WRF
git log --oneline -1
export WRF_EM_CORE=1 WRF_CHEM=1 WRF_KPP=0 NETCDF_classic=1 WRFIO_NCD_LARGE_FILE_SUPPORT=1
unset PNETCDF_QUILT
export HDF5=$HDF5_DIR NETCDF=$NETCDF_DIR PNETCDF=$PNETCDF_DIR
export PATH=$NETCDF_DIR/bin:$HDF5_DIR/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF_DIR/lib:/usr/lib64:$LD_LIBRARY_PATH
./clean -a >/dev/null 2>&1
printf "34\n1\n" | ./configure
sed -i "s/-cc=\$(SCC)/ /; s/mpif90/ftn/; s/mpicc/cc/" configure.wrf
grep -E "^(SFC|SCC|CCOMP|DM_FC|DM_CC)\b" configure.wrf
export J="-j 16"
./compile em_real > compile.log 2>&1
ls -l main/*.exe
grep -c -i " error" compile.log
tail -20 compile.log
