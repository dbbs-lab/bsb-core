#!/bin/bash -l
#SBATCH --job-name="Blender render"
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --nodes=25
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc&startx
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source load_blender.sh
srun python render.py $1 $2
