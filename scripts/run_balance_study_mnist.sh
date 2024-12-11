#!/bin/bash
#SBATCH --job-name=dp_sgd_cifar10_mia     # Job name
#SBATCH --output=dp_sgd_cifar10_mia.out     # Output file
#SBATCH --error=dp_sgd_cifar10_mia.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)

python ../idp_sgd/dpsgd_algos/advantage_study_mnist.py