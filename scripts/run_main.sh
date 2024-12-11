#!/bin/bash
#SBATCH --job-name=dp_sgd_main      # Job name
#SBATCH --output=dp_sgd_main.out     # Output file
#SBATCH --error=dp_sgd_main.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)

python ../mia/main.py \
--dname "MNIST" \
--assign_budget "even"
 
