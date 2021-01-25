#!/bin/bash -l
#SBATCH --job-name="Render Blender"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.deschepper93@gmail.com
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/load_blender.sh
srun blender-2.91.0-cacd57b67a15-linux64/blender -b $1 -E CYCLES -P sequence.py -- $2
