#!/bin/bash
#SBATCH --job-name=mia_per_group     # Job name
#SBATCH --output=mia_per_group.out     # Output file
#SBATCH --error=mia_per_group.err      # Error file
#SBATCH --ntasks=1                     # Number of tasks (one task for one GPU job)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=1G                      # Memory allocation
#SBATCH --time=2:00:00                # Maximum runtime (12 hours)

export base_folder="../svhn_results/standard/SVHN/epochs_30_batch_1024_lr_0.2_max_grad_norm_0.9_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"
export dname="SVHN"
export seed=0

nvidia-smi
python ../mia/mia_per_group.py \
    --basefolder $base_folder \
    --dname $dname\
    --seed $seed \

export base_folder1="../cifar_results/clipping/CIFAR10/epochs_60_batch_1024_lr_0.1_max_grad_norm_1.8_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"
export base_folder2="../cifar_results/clipping/CIFAR10/epochs_70_batch_1024_lr_0.2_max_grad_norm_1.1_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder3="../cifar_results/sampling/CIFAR10/epochs_60_batch_1024_lr_0.1_max_grad_norm_1.8_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"
export base_folder4="../cifar_results/sampling/CIFAR10/epochs_60_batch_1024_lr_0.2_max_grad_norm_1.0_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder5="../cifar_results/standard/CIFAR10/epochs_30_batch_1024_lr_0.7_max_grad_norm_0.4_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"

export base_folder6="../mnist_results/clipping/MNIST/epochs_80_batch_512_lr_0.6_max_grad_norm_0.2_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder7="../mnist_results/clipping/MNIST/epochs_80_batch_512_lr_0.6_max_grad_norm_0.2_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"
export base_folder8="../mnist_results/sampling/MNIST/epochs_80_batch_512_lr_0.6_max_grad_norm_0.2_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder9="../mnist_results/sampling/MNIST/epochs_80_batch_512_lr_0.6_max_grad_norm_0.2_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"
export base_folder10="../mnist_results/standard/MNIST/epochs_80_batch_512_lr_0.6_max_grad_norm_0.2_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0"

export base_folder11="../svhn_results/clipping/SVHN/epochs_50_batch_1024_lr_0.1_max_grad_norm_1.6_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder12="../svhn_results/clipping/SVHN/epochs_50_batch_1024_lr_0.1_max_grad_norm_2.0_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder13="../svhn_results/sampling/SVHN/epochs_50_batch_1024_lr_0.1_max_grad_norm_0.6_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_42"
export base_folder14="../svhn_results/sampling/SVHN/epochs_80_batch_1024_lr_0.2_max_grad_norm_0.6_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"
export base_folder15="../svhn_results/standard/SVHN/epochs_30_batch_1024_lr_0.2_max_grad_norm_0.9_budgets_1.0_2.0_3.0_ratios_0.34_0.43_0.23_seeds_0"