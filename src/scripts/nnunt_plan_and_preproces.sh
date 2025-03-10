#!/bin/bash
#SBATCH --job-name=train_nnunet
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request 3 GPUs as 3 GPUs per node

#SBATCH --ntasks-per-node=1 # Number of tasks per node - advice is to experiment with this value
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G   # memory per cpu-core

#SBATCH --time=48:00:00

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

#SBATCH --array=1-5 # Run as task array

# Load necessary modules
module load cuda

export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
module load miniforge
conda activate pred-knee-replacement-oai

# Run the training script with the selected input file

# Initial planning and preprocessing
nnUNetv2_plan_and_preprocess -d 014 --verify_dataset_integrity

