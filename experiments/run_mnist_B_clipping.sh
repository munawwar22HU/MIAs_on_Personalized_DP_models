#!/bin/bash




# DATASET
export dname="MNIST"
# Architecture
export architecture="MNIST_CNN"
# Individualize
export individualize="clipping"
# Privacy Budgets
export budgets="1.0 2.0 3.0"
# Ratios
export ratios="0.54 0.37 0.09"
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
    --noise_multiplier $noise_multiplier
    

