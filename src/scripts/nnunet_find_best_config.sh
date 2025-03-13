#!/bin/bash
#SBATCH --job-name=nnunet_find_best_config
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request 3 GPUs as 3 GPUs per node

#SBATCH --ntasks-per-node=1 # Number of tasks per node - advice is to experiment with this value
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G   # memory per cpu-core

#SBATCH --time=48:00:00

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

# Load necessary modules
module load cuda

export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
module load miniforge
conda activate pred-knee-replacement-oai

# Initial planning and preprocessing
nnUNetv2_find_best_configuration 014 -c ['2d', '3d_fullres'] 