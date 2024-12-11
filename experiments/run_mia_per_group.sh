#!/bin/bash
#SBATCH --job-name=mia_per_group     # Job name
#SBATCH --output=mia_per_group.out     # Output file
#SBATCH --error=mia_per_group.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)

export base_folder="../cifar_results/sampling/CIFAR10/epochs_60_batch_1024_lr_0.1_max_grad_norm_1.8_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"
export dname="MNIST"
export seed=0

nvidia-smi
python ../mia/mia_per_group.py \
    --basefolder $base_folder \
    --dname $dname\
    --seed $seed \

# MNIST A - Sampling
# export base_folder="../mnist_results/sampling/MNIST/epochs_80_batch_512_lr_0.6_max_grad_norm_0.2_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"

    