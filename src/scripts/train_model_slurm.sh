#!/bin/bash
#SBATCH --job-name=train_seg_model
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:3        # Request 3 GPUs as 3 GPUs per node

#SBATCH --time=48:00:00

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

# Load necessary modules
module load cuda


# Run the training script with the selected input file
python train.py --model unet --hpc-flag 1 < config/config_unet_1.json

