#!/bin/bash
#SBATCH --job-name=cifar_B_clip      # Job name
#SBATCH --output=cifar_B_clip.out     # Output file
#SBATCH --error=cifar_B_clip.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)


# DATASET
export dname="CIFAR10"
# Architecture
export architecture="CIFAR10_CNN"
# Individualize
export individualize="clipping"
# Privacy Budgets
export budgets="1.0 2.0 3.0"
# Ratios
export ratios="0.54 0.37 0.09"
# Learning Rate
export lr=0.1
# Batch Size
export batch_size=1024
# Epochs
export epochs=60
# Max Grad Norm
export max_grad_norm=1.8
# Noise Multiplier
export noise_multiplier=3.17792
# Number of data points for MIA
export mia_ndata=50000
# Mode
export mode="mia"
# Save Directory
export save_dir="../cifar1_results/"

# Run the script
python ../idp_sgd/dpsgd_algos/individual_dp_sgd.py \
    --save_path $save_dir \
    --dname $dname \
    --architecture $architecture \
    --individualize $individualize \
    --lr $lr \
    --epochs $epochs \
    --batch_size $batch_size \
    --budgets $budgets \
    --ratios  $ratios \
    --max_grad_norm ${max_grad_norm} \
    --mode $mode \
    --mia_count 0 \
    --mia_ndata $mia_ndata \
    --noise_multiplier $noise_multiplier \
    --max_physical_batch_size 1024

