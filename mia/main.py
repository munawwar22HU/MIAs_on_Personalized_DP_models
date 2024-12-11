from collections import OrderedDict
import argparse
import numpy as np
import os
import torch
import sys
sys.path.append('../opacus')
sys.path.append('../idp_sgd/dpsgd_algos')
sys.path.append('../idp_sgd/training_utils')
sys.path.append('../idp_sgd/')
from training_utils.datasets import get_dataset, assign_budgets,get_indexed_dataset

assign_budget = 'even'
individualize = 'clipping'

# Define the parser
parser = argparse.ArgumentParser(description='MIA Parser')
parser.add_argument('--dname', type=str,
                    default='CIFAR10',
                    help='dataset to be learned',
                    )

parser.add_argument(
    '--assign_budget',
    type=str,
    default=assign_budget,
    choices=['random', 'even', 'per-class'],
    help='The type of budget assignment.',
)
parser.add_argument(
    '--individualize',
    type=str,
    default=individualize,
    help='(i)DP-SGD method ("None", "clipping", "sampling")',
)




# Load MNIST dataset
def load_data(args):
    dataset = get_indexed_dataset("MNIST",args,False)
    return dataset

# PRINT the pp_budget of each data point
def print_pp_budgets(args, dataset):
    print(dataset)
    budgets = assign_budgets(dataset, args)
    print(budgets)
    print(budgets.shape)

# Run the main function
def main(args):
    print(args)
    dataset = load_data(args)
    print_pp_budgets(args, dataset)

# Run the main function
if __name__ == '__main__':
    main(parser.parse_args())

