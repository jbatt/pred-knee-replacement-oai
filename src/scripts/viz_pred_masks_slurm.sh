#!/bin/bash
#SBATCH --job-name=visualize      # Job name
#SBATCH --nodes=1                 # Run on one node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=1        # Use 1 CPU core
#SBATCH --mem=16G                 # Request 16GB of memory
#SBATCH --time=48:00:00          # Time limit hrs:min:sec
#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

# Run the visualization script
python visualization/visualize.py --project_name oai_subset_knee_cart_seg --model unet --pred_masks_dir /mnt/scratch/scjb/data/processed/ --results_dir /mnt/scratch/scjb/results/
# End of script

