#!/bin/bash
#SBATCH --ntasks-per-node=100 ###number of cores per CPU
#SBATCH --contiguous       # Ensure that all of the cores are on the same Infiniband network, adjust if necessary
#SBATCH --time=12:00:00
#SBATCH --partition=gpu ###partition name
#SBATCH --account=bd1083 ###your account name
#SBATCH --nodes=1
#SBATCH --mem=0
####SBATCH --constraint=512G


module load gcc #load GCC compiler
export I_MPI_PMI=pmi2 #mpi with PMI-2 API, adjust if neeeded
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so #mpi with PMI-2 API, adjust if neeeded
srun --mpi=pmi2 -n $SLURM_NTASKS python -u run_confidence_interval_allseeds.py
echo "done!"