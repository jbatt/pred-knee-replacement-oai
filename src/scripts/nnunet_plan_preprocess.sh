#!/bin/bash
#SBATCH --job-name=nnunet_plan_preprocess    
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


# Activate conda environment
module load miniforge
conda activate pred-knee-replacement-oai

# Run the training script with the selected input file

# Initial planning and preprocessing
nnUNetv2_plan_and_preprocess -d 014 --verify_dataset_integrity


# nnUnet model training
# Train the 3d full res model for each fold
# for FOLD in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 014 3d_fullres $FOLD --npz -device 'cuda'
# done