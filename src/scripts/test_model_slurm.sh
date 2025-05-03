#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request 3 GPUs as 3 GPUs per node

#SBATCH --ntasks-per-node=1 # Number of tasks per node - advice is to experiment with this value
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G   # memory per cpu-core

#SBATCH --time=4:00:00

#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

module load miniforge
conda activate pred-knee-replacement-oai

# Run the visualization script
# python test.py --model nnunet --config /users/scjb/pred-knee-replacement-oai/src/config/config_swinunetr_8.json
# python test.py --config /users/scjb/pred-knee-replacement-oai/src/config/config_swinunetr_8.json --model swin_unetr --model_weights /mnt/scratch/scjb/models/checkpoints/2025-04-25-14_19_27093743_swin_unetr_multiclass_eternal-sweep-1_early_stop_E200.pth  --inference
# python test.py --config /users/scjb/pred-knee-replacement-oai/src/config/config_swinunetr_9.json --model swin_unetr --model_weights /mnt/scratch/scjb/models/checkpoints/2025-04-28-10_37_47962559_swin_unetr_multiclass_sparkling-sweep-1_early_stop_E800.pth  --inference


python test.py --config /users/scjb/pred-knee-replacement-oai/src/config/config_segformer3d_11.json --model segformer3d --model_weights /mnt/scratch/scjb/models/checkpoints/2025-04-28-10_37_55865479_segformer3d_multiclass_northern-sweep-1_early_stop_E800.pth  --inference