#!/bin/bash
#SBATCH --job-name=train_seg_model
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request a single GPU

#SBATCH --time=6:00:00
#SBATCH --array=1-2 # Set up a task arrya with two tasks 

# Load necessary modules
module load cuda


# Run the training script with the selected input file
python train.py --model unet --hpc-flag 1 < $infile

