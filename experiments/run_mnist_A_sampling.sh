#!/bin/bash
#SBATCH --job-name=mnist_A_sample      # Job name
#SBATCH --output=mnist_A_sample.out     # Output file
#SBATCH --error=mnist_A_sample.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)


# DATASET
export dname="MNIST"
# Architecture
export architecture="MNIST_CNN"
# Individualize
export individualize="sampling"
# Privacy Budgets
export budgets="1.0 2.0 3.0"
# Ratios
export ratios="0.34 0.43 0.23"
# Learning Rate
export lr=0.6
# Batch Size
export batch_size=512
# Epochs
export epochs=80
# Max Grad Norm
export max_grad_norm=0.2
# Noise Multiplier
export noise_multiplier=3.42529
# Number of data points for MIA
export mia_ndata=60000
# Mode
export mode="mia"
# Save Directory
export save_dir="../mnist_results/"

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
