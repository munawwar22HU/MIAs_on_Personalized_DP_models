# Membership Inference Attack on Personalized Differential Private Models

# Group Members
1. Muhammad Munawwwar Anwar
2. Sai Prasad Gudari

# Background
This repository contains the code for the final project of the course CS 573: Data Privacy and Security at Emory University. This project builds up on the paper "Have it your way: Individualized Privacy Assignment for DP-SGD" by Franziska Boenisch, Christopher Mühl, Adam Dziedzic, and Roy Rinberg. The paper presented a variant of the DP-SGD which supports individualized privacy budgets according to the user preferences. This project builds on the paper by empirically evaluating the privacy risk of the individualized DP-SGD models using membership inference attacks. We perform black-box membership inference attacks on the models trained using the individualized DP-SGD and the compare the results for each group of users with different privacy budgets by computing the roc-auc scores.

# Library Installation
The code is written in Python 3.9. To install the libraries, run the following command:
```bash
conda env create -f environment.yml
```
which will create a conda environment named `idp` with all the required libraries.

# Experiments
We conducted a total of 18 experiments, 6 for each dataset (CIFAR-10, CIFAR-100, and SVHN). For each dataset, we trained models with three different privacy budgets (1.0, 2.0, and 3.0) and two ratios of the individualized privacy budgets (0.54,0.37,0.09) and
(0.34,0.43,0.23)

# Code Structure
-- experiments: Contains the bash scripts to train the models, perform model inference, and then perform membership inference attacks. To run the experiments, execute the following commands, go to the experiments directory by running the bash script. An example is shown below:
```bash
sbatch run_cifar_A.sh
```
-- idp_sgd: Contains the implementation of the individualized DP-SGD algorithm. 

-- mia: Contains the implementation of the membership inference attack.

-- results: Contains the results of the experiments.

-- opacus: Contains the changes to the Opacus library to support individualized privacy budgets.



