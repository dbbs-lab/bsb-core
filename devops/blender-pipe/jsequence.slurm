#!/bin/bash -l
#SBATCH --job-name="Blender seauence"
#SBATCH --mail-type=ALL
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source load_blender.sh
srun blender -b $1 -E CYCLES -P sequence.py -- $2
