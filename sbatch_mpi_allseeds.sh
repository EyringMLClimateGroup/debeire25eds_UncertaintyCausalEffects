#!/bin/bash
#SBATCH --ntasks-per-node=100 # number of cores per CPU /!\ NEED to be EQUAL to number of bootstrap realizations
#SBATCH --contiguous # Ensure that all of the cores are on the same Infiniband network, adjust if necessary
#SBATCH --time=12:00:00 #adjust
#SBATCH --partition=xxx ###partition name to adjust
#SBATCH --account=xxx ###your account name to adjust
#SBATCH --nodes=1
#SBATCH --mem=0 #allocate all available memory


module load gcc #load GCC compiler
export I_MPI_PMI=pmi2 #mpi with PMI-2 API, adjust if neeeded
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so #mpi with PMI-2 API, adjust if neeeded

srun --mpi=pmi2 -n $SLURM_NTASKS python -u run_confidence_interval_allseeds.py
echo "done!"