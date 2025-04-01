#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request GPUs - 3 GPUs per node

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

# Run prediction, ensembling and postprocessing

nnUNetv2_predict -d Dataset014_OAISubset -i /mnt/scratch/scjb/nnUNet_raw/Dataset014_OAISubset/imagesTs -o /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/2d -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities
nnUNetv2_predict -d Dataset014_OAISubset -i /mnt/scratch/scjb/nnUNet_raw/Dataset014_OAISubset/imagesTs -o /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/3d_fullres -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans --save_probabilities

nnUNetv2_ensemble -i /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/2d /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/3d_fullres -o /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/ensemble -np 8

nnUNetv2_apply_postprocessing -i /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/ensemble -o /mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/postprocesing -pp_pkl_file /mnt/scratch/scjb/nnUNet_results/Dataset014_OAISubset/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /mnt/scratch/scjb/nnUNet_results/Dataset014_OAISubset/ensembles/ensemble___nnUNetTrainer__nnUNetPlans__2d___nnUNetTrainer__nnUNetPlans__3d_fullres___0_1_2_3_4/plans.json