#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# partition
#SBATCH --partition=small

# set max wallclock time
#SBATCH --time=04:00:00

# set name of job
#SBATCH --job-name=unet_train

# set number of GPUs
#SBATCH --gres=gpu:2

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scjb@leeds.ac.uk

# run the application
module load python/anaconda3
module load cuda
module load pytorch

source activate pred-knee-replacement-oai

# cd pred-knee-replacement-oai/scripts/
# export WANDB_DIR=/jmain02/home/J2AD014/mtc13/jjb87-mtc13/pred-knee-replacement-oai/src/wandb

python train_UNet.py
