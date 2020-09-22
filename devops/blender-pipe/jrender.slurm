#!/bin/bash -l
#SBATCH --job-name="Blender"
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --nodes=25
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module swap gcc/8.3.0 gcc/9.3.0
module load daint-mc
module load cray-python/3.8.2.1
module load PyExtensions/python3-CrayGNU-20.08
module load h5py/2.10.0-CrayGNU-20.08-python3-parallel

srun python render.py $1 $2
