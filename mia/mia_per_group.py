
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sklearn.metrics
import torch
from typing import Tuple
import sys
sys.path.append('../idp_sgd/dpsgd_algos')
sys.path.append('../idp_sgd/training_utils')
sys.path.append('../idp_sgd/')
from dpsgd_algos.individual_dp_sgd import initialize_training

individualize = 'sampling'
NUM_POINTS = 10_000


parser = argparse.ArgumentParser(description='MIA Parser')
parser.add_argument('--basefolder', type=str,
                    default='',
                    help='location of the run folders',
                    )
parser.add_argument('--dname', type=str,
                    default='CIFAR10',
                    help='dataset to be learned',
                    )
parser.add_argument('--score_method', type=str,
                    default='logits',
                    choices=['probs', 'logits'],
                    help='the method used to do the scoring',
                    )
parser.add_argument('--seed', type=int,
                    default=0,
                    help='keys for reproducible pseudo-randomness',
)
parser.add_argument('--individualize', type=str,
                    default=individualize,
                    help='(i)DP-SGD method ("None", "clipping", "sampling")',
)
parser.add_argument(
    '--num_points', type=int,
    default=NUM_POINTS,
    help="The number of member and non-member samples to use for the Membership Inference Attack"
)
def load_logit_scores(args) -> Tuple[np.ndarray, np.ndarray]:
    non_target_logits_file=os.path.join(
        args.basefolder, 'non_target_logits.npy')
    non_target_logits=np.load(file=non_target_logits_file)

    target_logits_file=os.path.join(args.basefolder, 'target_logits.npy')
    target_logits=np.load(file=target_logits_file)
    return non_target_logits, target_logits

def load_pp_budgets(args) -> np.ndarray:

    pp_budget_file=os.path.join(args.basefolder, 'run0/pp_budgets.npy')
    pp_budgets=np.load(file=pp_budget_file)
    return pp_budgets




def load_assignments_target(args) -> np.ndarray:
    assignment_file=os.path.join(args.basefolder, f'run0/assignment.npy')
    assignment=np.load(file=assignment_file)
    print("assignment shape: ", assignment.shape)
    return assignment



def get_labels(args):
    device, train_loader, test_loader=initialize_training(dataset_name=args.dname,
                                                  cuda=True,
                                                  epochs=-1,
                                                  n_workers=6,
                                                  batch_size=1000,
                                                  seed=args.seed,
                                                  shuffle=False,
                                                  args=args)
    train_labels, test_labels=[], []
    for _, label in train_loader:
        train_labels.extend(list(label.cpu().numpy()))
    for _, label in test_loader:
        test_labels.extend(list(label.cpu().numpy()))
    train_labels, test_labels=np.array(train_labels), np.array(test_labels)
    return train_labels, test_labels





def main(args):
    
    non_target_logits, target_logits=load_logit_scores(args=args)
    pp_budgets=load_pp_budgets(args=args)
    assignment=load_assignments_target(args=args)
    train_labels, test_labels=get_labels(args=args)
    
    # Get the indices of the groups
    group_1 = np.where(pp_budgets == 1)[0]
    group_2 = np.where(pp_budgets == 10)[0]
    group_3 = np.where(pp_budgets == 100)[0]

    
    tr_target_logits = target_logits[0][assignment]
    
    tr_target_logits = tr_target_logits.reshape(1,tr_target_logits.shape[0], tr_target_logits.shape[1])
    tr_train_labels = train_labels[assignment]

    
    

    train_target_class_logits = np.take_along_axis(tr_target_logits, tr_train_labels[np.newaxis, :, np.newaxis], axis=2)
    test_target_class_logits = np.take_along_axis(non_target_logits, test_labels[np.newaxis, :, np.newaxis], axis=2)

    

    group_1_logits = train_target_class_logits[0][group_1]
    group_2_logits = train_target_class_logits[0][group_2]
    group_3_logits = train_target_class_logits[0][group_3]

    print("group_1_logits shape: ", group_1_logits.shape)
    print("group_2_logits shape: ", group_2_logits.shape)
    print("group_3_logits shape: ", group_3_logits.shape)

    group_1_sample = np.random.choice(group_1_logits.shape[0], size=args.num_points, replace=False)
    group_2_sample = np.random.choice(group_2_logits.shape[0], size=args.num_points, replace=False)
    group_3_sample = np.random.choice(group_3_logits.shape[0], size=args.num_points, replace=False)

    test_sample = np.random.choice(test_target_class_logits.shape[1], size=args.num_points, replace=False)

    print("Poop")
    
    group_1_sample_logits = group_1_logits[group_1_sample]
    group_2_sample_logits = group_2_logits[group_2_sample]
    group_3_sample_logits = group_3_logits[group_3_sample]
    test_sample_logits = test_target_class_logits[0][test_sample]
   

    y_member = np.ones(args.num_points)
    y_non_member = np.zeros(args.num_points)
    y_true = np.concatenate((y_member, y_non_member))

    likelihood_scores_1 = np.concatenate((group_1_sample_logits.flatten(), test_sample_logits.flatten()))
    likelihood_scores_2 = np.concatenate((group_2_sample_logits.flatten(), test_sample_logits.flatten()))
    likelihood_scores_3 = np.concatenate((group_3_sample_logits.flatten(), test_sample_logits.flatten()))
    
   

    fpr_1, tpr_1, _ = sklearn.metrics.roc_curve(y_true, likelihood_scores_1, pos_label=1)
    fpr_2, tpr_2, _ = sklearn.metrics.roc_curve(y_true, likelihood_scores_2, pos_label=1)
    fpr_3, tpr_3, _ = sklearn.metrics.roc_curve(y_true, likelihood_scores_3, pos_label=1)

    auc1 = sklearn.metrics.auc(fpr_1, tpr_1)
    auc2 = sklearn.metrics.auc(fpr_2, tpr_2)
    auc3 = sklearn.metrics.auc(fpr_3, tpr_3)

    print(auc1,auc2,auc3)

    precision = 3
    auc1 = "{:.3f}".format(round(auc1, precision))
    auc2 = "{:.3f}".format(round(auc2, precision))
    auc3 = "{:.3f}".format(round(auc3, precision))

    plt.plot(fpr_1, tpr_1, label=f'Group 1 {auc1}')
    plt.plot(fpr_2, tpr_2, label=f'Group 2 {auc2}')
    plt.plot(fpr_3, tpr_3, label=f'Group 3 {auc3}')
    plt.legend()


    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    save_path = os.path.join(args.basefolder, f"roc_curve_per_privacy_group.pdf")
    plt.savefig(save_path)
    print(f"ROC curve saved at {save_path}")
    print("Hello World")
    
if __name__ == "__main__":
    args=parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args=args)
