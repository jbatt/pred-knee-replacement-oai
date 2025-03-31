#!/bin/bash
#SBATCH --job-name=visualize      # Job name
#SBATCH --nodes=1                 # Run on one node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=1        # Use 1 CPU core
#SBATCH --mem=16G                 # Request 16GB of memory
#SBATCH --time=4:00:00          # Time limit hrs:min:sec
#SBATCH --mail-user=scjb@leeds.ac.uk # Email address for notifications
#SBATCH --mail-type=BEGIN,END

module load miniforge
conda activate pred-knee-replacement-oai

# Run the visualization script
python test.py --model nnunet

