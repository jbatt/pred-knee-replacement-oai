#!/bin/bash
#SBATCH --job-name=train_seg_model
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request 3 GPUs as 3 GPUs per node

#SBATCH --ntasks-per-node=1 # Number of tasks per node - advice is to experiment with this value
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G   # memory per cpu-core

#SBATCH --time=48:00:00

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

# Load necessary modules
module load cuda

# Load environment
module load miniforge
conda activate pred-knee-replacement-oai

# Run the training script with the selected input file
python train.py --model manet --hpc-flag 1 < config/config_manet_4.json

