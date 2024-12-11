#!/bin/bash
#SBATCH --job-name=dp_sgd_cifar10_scoring     # Job name
#SBATCH --output=dp_sgd_cifar10_scoring.out     # Output file
#SBATCH --error=dp_sgd_cifar10_scoring.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)



python ../mia/scoring.py \
    --basefolder "../results1/sampling/MNIST/epochs_5_batch_1024_lr_0.7_max_grad_norm_0.4_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0" \
    --dname "MNIST"\
    --num_shadow_models 4 \
    --seed 0 \
    --individualize "sampling" \
    --score_method "logits"

    