#!/bin/bash
#SBATCH --job-name=visualize      # Job name
#SBATCH --nodes=1                 # Run on one node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=1        # Use 1 CPU core
#SBATCH --mem=16G                 # Request 16GB of memory
#SBATCH --time=48:00:00          # Time limit hrs:min:sec
#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

module load miniforge
conda activate pred-knee-replacement-oai

# Run the visualization script
# python visualization/visualize.py --model nnunet --run_start_time 2025-04-01_11-39-52 --pred_masks_dir /mnt/scratch/scjb/data/processed/ --results_dir /mnt/scratch/scjb/results/

python visualization/visualize.py --model_1 nnunet --run_start_time_1 2025-05-06_16-19-24 --model_2 swin_unetr --run_start_time_2 2025-05-06_16-20-19 --model_3 segformer3d --run_start_time_3 2025-05-06_16-24-43 --results_dir /mnt/scratch/scjb/results/

# End of script