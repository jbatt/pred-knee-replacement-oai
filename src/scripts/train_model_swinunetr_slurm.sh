#!/bin/bash
#SBATCH --job-name=train_swinunetr
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:3        # Request 3 GPUs as 3 GPUs per node

#SBATCH --ntasks-per-node=1 # Number of tasks per node - advice is to experiment with this value
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G   # memory per cpu-core

#SBATCH --time=48:00:00

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

# Load necessary modules
module load cuda

module load miniforge
conda activate pred-knee-replacement-oai

# Adjust pytorch garbage collection threshold to manage memory
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.2,expandable_segments:True

# Run the training script with the selected input file
python train_distributed_patch_monai.py --model swin_unetr --hpc-flag 1 < config/config_swinunetr_9.json

