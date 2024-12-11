#!/bin/bash
#SBATCH --job-name=dp_sgd_cifar10_mia     # Job name
#SBATCH --output=dp_sgd_cifar10_mia.out     # Output file
#SBATCH --error=dp_sgd_cifar10_mia.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)


nvidia-smi
python ../mia/inference.py \
    --basefolder "../mnist_results/sampling/MNIST//local/scratch/manwa22/idp-sgd/mnist_results/sampling/MNIST/epochs_1_batch_1024_lr_0.2_max_grad_norm_1.0_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0" \
    --target_model_name "None" \
    --num_shadow_models 1  \
    --dname "MNIST"\
    --seed 42 \
    --individualize "sampling"

    