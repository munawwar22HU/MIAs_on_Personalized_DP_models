#!/bin/bash
#SBATCH --job-name=dp_sgd_cifar10      # Job name
#SBATCH --output=dp_sgd_cifar10.out     # Output file
#SBATCH --error=dp_sgd_cifar10.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)



export SAVE_DIR="./results/"  # Replace with your save directory
export LEARNING_RATE=0.7               # Replace with your learning rate
export seed=42                          # Replace with your desired seed
export max_grad_norm=0.4               # Replace with your desired max grad norm

# Run the script
python individual_dp_sgd.py \
    --save_path $SAVE_DIR \
    --seeds ${seed} \
    --dname "CIFAR10" \
    --architecture "CIFAR10_CNN" \
    --individualize "None" \
    --lr $LEARNING_RATE \
    --epochs 30 \
    --batch_size 1024 \
    --budgets 1.0 2.0 3.0 \
    --ratios 0.54 0.37 0.09 \
    --max_grad_norm ${max_grad_norm} \
    --mia_ndata 73257 \
    --mode 'mia'
