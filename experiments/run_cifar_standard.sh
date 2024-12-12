#!/bin/bash
#SBATCH --job-name=cifar_standard      # Job name
#SBATCH --output=cifar_standard.out     # Output file
#SBATCH --error=cifar_standard.err      # Error file
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
export individualize=None
# Learning Rate
export lr=0.7
# Batch Size
export batch_size=1024
# Epochs
export epochs=30
# Max Grad Norm
export max_grad_norm=0.4
# Noise Multiplier
export noise_multiplier=3.29346
# Number of data points for MIA
export mia_ndata=50000
# Mode
export mode="mia"
# Save Directory
export save_dir="../cifar_results/"

# Run the script
python ../idp_sgd/dpsgd_algos/individual_dp_sgd.py \
    --save_path $save_dir \
    --dname $dname \
    --architecture $architecture \
    --individualize $individualize \
    --lr $lr \
    --epochs $epochs \
    --batch_size $batch_size \
    --max_grad_norm ${max_grad_norm} \
    --mode $mode \
    --noise_multiplier $noise_multiplier\
    --mia_ndata $mia_ndata \


