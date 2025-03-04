#!/bin/bash
#SBATCH --job-name=train_seg_model
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:3        # Request 3 GPUs as 3 GPUs per node

#SBATCH --time=48:00:00
#SBATCH --array=1-2 # Set up a task arrya with two tasks 

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

# Load necessary modules
module load cuda


# Use the task ID to select the required config file from those specified in 
# the input file (config/config_unet.txt) file
infile=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" config/config_unet.txt)

# Run the training script with the selected input file
python train.py --model unet --hpc-flag 1 < $infile

