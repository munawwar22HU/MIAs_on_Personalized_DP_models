#!/bin/bash
#SBATCH --job-name=dp_sgd_svhn_debug      # Job name
#SBATCH --output=dp_sgd_svhn_debug_1.out     # Output file
#SBATCH --error=dp_sgd_svhn_debug_1.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)



export SAVE_DIR="../svhn1_results/"  # Replace with your save directory
export LEARNING_RATE=0.2               # Replace with your learning rate
export seed=42                          # Replace with your desired seed
export max_grad_norm=0.9              # Replace with your desired max grad norm

# Run the script
python ../idp_sgd/dpsgd_algos/individual_dp_sgd.py \
    --save_path $SAVE_DIR \
    --seeds ${seed} \
    --dname "SVHN" \
    --individualize "sampling" \
    --architecture "CIFAR10_CNN" \
    --lr $LEARNING_RATE \
    --epochs 80 \
    --batch_size 1024 \
    --budgets 1.0 2.0 3.0 \
    --max_grad_norm ${max_grad_norm} \
    --mia_ndata 73257 \
    --mode 'mia' \
    --noise_multiplier 2.53261

