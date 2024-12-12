
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
def load_logit_scores(args) -> Tuple[np.ndarray, np.ndarray]:
    non_target_logits_file=os.path.join(
        args.basefolder, 'non_target_logits.npy')
    non_target_logits=np.load(file=non_target_logits_file)

    target_logits_file=os.path.join(args.basefolder, 'target_logits.npy')
    target_logits=np.load(file=target_logits_file)

    print("non_target_logits shape: ", non_target_logits.shape)
    print("target_logits shape: ", target_logits.shape)

    return non_target_logits, target_logits

def load_pp_budgets(args) -> np.ndarray:

    pp_budget_file=os.path.join(args.basefolder, 'run0/pp_budgets.npy')
    pp_budgets=np.load(file=pp_budget_file)
    print("pp_budgets shape: ", pp_budgets.shape)
    return pp_budgets




def load_assignments_target(args) -> np.ndarray:
    assignment_file=os.path.join(args.basefolder, f'run0/assignment.npy')
    assignment=np.load(file=assignment_file)
    print("assignment shape: ", assignment.shape)
    return assignment


def convert_logit_to_prob(logit: np.ndarray, axis=1) -> np.ndarray:
    # """Converts logits to probability vectors.
    # Args:
    #   logit: n by c array where n is the number of samples and c is the number of
    #     classes.
    # Returns:
    #   The probability vectors as n by c array
    # """
    # prob = logit - np.max(logit, axis=axis, keepdims=True)
    # prob = np.array(np.exp(prob), dtype=np.float64)
    # prob = prob / np.sum(prob, axis=axis, keepdims=True)
    # return prob
    pass


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
    print("train_labels shape: ", train_labels.shape)
    print("test_labels shape: ", test_labels.shape)
    return train_labels, test_labels




def compute_likelihood(args):
    pass
    # assignments = load_assignments_shadows(args=args)
    # shadow_logits, target_logits = load_logit_scores(args=args)

    # if args.score_method == 'probs':
    #     shadow_logits = convert_logit_to_prob(logit=shadow_logits, axis=2)
    #     target_logits = convert_logit_to_prob(logit=target_logits, axis=2)

    # labels = get_labels(args=args)
    # num_points = len(labels)

    # print("target model train accuracy: ", np.sum(np.argmax(target_logits[0], axis=1) == labels) / num_points *100)

    # # extract logits for the correct labels
    # shadow_logits = np.take_along_axis(shadow_logits, labels[np.newaxis, :, np.newaxis], axis=2)
    # target_logits = np.take_along_axis(target_logits, labels[np.newaxis, :, np.newaxis], axis=2)

    # shadow_logits = shadow_logits.reshape(shadow_logits.shape[0], -1)
    # target_logits = target_logits.reshape(target_logits.shape[0], -1)


    # # split to member and non-member lists
    # mean_members = np.zeros(num_points)  # logits of members (data points)
    # std_members = np.zeros(num_points)
    # mean_nonmembers = np.zeros(num_points)  # logits of non-members
    # std_nonmembers = np.zeros(num_points)

    # all_indices = np.arange(num_points)

    # for data_idx in all_indices:
    #     # print('data_idx: ', data_idx)
    #     member_list = []
    #     nonmember_list = []
    #     for model_idx in range(args.num_shadow_models):
    #         member_indices = assignments[model_idx]
    #         if data_idx in member_indices:
    #             member_list.append(shadow_logits[model_idx][data_idx])
    #         else:
    #             nonmember_list.append(shadow_logits[model_idx][data_idx])
    #     mean_member = np.median(member_list)
    #     std_member = np.std(member_list)
    #     mean_nonmember = np.median(nonmember_list)
    #     std_nonmember = np.std(nonmember_list)

    #     mean_members[data_idx] = mean_member
    #     std_members[data_idx] = std_member
    #     mean_nonmembers[data_idx] = mean_nonmember
    #     std_nonmembers[data_idx] = std_nonmember
    # mean_member_file = os.path.join(args.basefolder, f"mean_members_{args.score_method}.npy")
    # np.save(arr=mean_members, file=mean_member_file)
    # std_member_file = os.path.join(args.basefolder, f"std_members_{args.score_method}.npy")
    # np.save(arr=std_members, file=std_member_file)

    # mean_nonmember_file = os.path.join(args.basefolder, f"mean_nonmembers_{args.score_method}.npy")
    # np.save(arr=mean_nonmembers, file=mean_nonmember_file)
    # std_nonmember_file = os.path.join(args.basefolder, f"std_nonmembers_{args.score_method}.npy")
    # np.save(arr=std_nonmembers, file=std_nonmember_file)

    # mean_member_default = np.nanmean(mean_members)
    # mean_members = np.nan_to_num(mean_members, nan=mean_member_default)

    # std_member_default = np.nanstd(std_members)
    # std_members = np.nan_to_num(std_members, nan=std_member_default)

    # mean_nonmember_default = np.nanmean(mean_nonmembers)
    # mean_nonmembers = np.nan_to_num(mean_nonmembers, nan=mean_nonmember_default)

    # std_nonmember_default = np.nanstd(std_nonmembers)
    # std_nonmembers = np.nan_to_num(std_nonmembers, nan=std_nonmember_default)

    # pr_in = scipy.stats.norm.logpdf(target_logits, mean_members, std_members + 1e-30)
    # pr_out = scipy.stats.norm.logpdf(target_logits, mean_nonmembers, std_nonmembers + 1e-30)

    # likelihood_scores = pr_in - pr_out
    # likelihood_scores = likelihood_scores.flatten()
    # likelihood_file = os.path.join(args.basefolder, f"likelihood_scores_{args.score_method}.npy")
    # np.save(arr=likelihood_scores, file=likelihood_file)

    # return likelihood_scores


def main(args):
    non_target_logits, target_logits=load_logit_scores(args=args)
    pp_budgets=load_pp_budgets(args=args)
    assignment=load_assignments_target(args=args)
    train_labels, test_labels=get_labels(args=args)
    print("train_labels shape: ", train_labels.shape)
    print("test_labels shape: ", test_labels.shape)

    print("pp_budgets shape: ", pp_budgets.shape)


    # Get the indices of the groups
    group_1 = np.where(pp_budgets == 1)[0]
    group_2 = np.where(pp_budgets == 2)[0]
    group_3 = np.where(pp_budgets == 3)[0]

    # Extract the training logits and labels that were used during the training process
    tr_target_logits = target_logits[0][assignment]
    # Add one axis to the tr_target_logits
    tr_target_logits = tr_target_logits.reshape(1,tr_target_logits.shape[0], tr_target_logits.shape[1])
    tr_train_labels = train_labels[assignment]

    
    print("tr_target_logits shape: ", tr_target_logits.shape)
    print("tr_train_labels shape: ", tr_train_labels.shape)

    train_target_class_logits = np.take_along_axis(tr_target_logits, tr_train_labels[np.newaxis, :, np.newaxis], axis=2)
    test_target_class_logits = np.take_along_axis(non_target_logits, test_labels[np.newaxis, :, np.newaxis], axis=2)

    print("train_target_class_logits shape: ", train_target_class_logits.shape, train_target_class_logits)
    print("test_target_class_logits shape: ", test_target_class_logits.shape,test_target_class_logits)

    group_1_logits = train_target_class_logits[0][group_1]
    group_2_logits = train_target_class_logits[0][group_2]
    group_3_logits = train_target_class_logits[0][group_3]

    print("group_1_logits shape: ", group_1_logits.shape)
    print("group_2_logits shape: ", group_2_logits.shape)
    print("group_3_logits shape: ", group_3_logits.shape)

    group_1_sample = np.random.choice(group_1_logits.shape[0], size=10000, replace=False)
    group_2_sample = np.random.choice(group_2_logits.shape[0], size=10000, replace=False)
    group_3_sample = np.random.choice(group_3_logits.shape[0], size=10000, replace=False)

    test_sample = np.random.choice(test_target_class_logits.shape[1], size=10000, replace=False)
    
    group_1_sample_logits = group_1_logits[group_1_sample]
    group_2_sample_logits = group_2_logits[group_2_sample]
    group_3_sample_logits = group_3_logits[group_3_sample]
    test_sample_logits = test_target_class_logits[0][test_sample]
   

    y_member = np.zeros(10000)
    y_non_member = np.ones(10000)
    y_true = np.concatenate((y_member, y_non_member))

    likelihood_scores_1 = np.concatenate((group_1_sample_logits.flatten(), test_sample_logits.flatten()))
    likelihood_scores_2 = np.concatenate((group_2_sample_logits.flatten(), test_sample_logits.flatten()))
    likelihood_scores_3 = np.concatenate((group_3_sample_logits.flatten(), test_sample_logits.flatten()))
    
    # group_1_logits = target_logits[0][group_1]
    # group_2_logits = target_logits[0][group_2]
    # group_3_logits = target_logits[0][group_3]

    # likelihood_scores = compute_likelihood(args=args)
    # print("likelihood_scores shape: ", likelihood_scores.shape)
    # target_assignment = load_assignments_target(args=args)
    # print("target_assignment shape: ", target_assignment.shape)
    # y_true = np.zeros(likelihood_scores.shape[0])
    # y_true[target_assignment] = 1

    fpr_1, tpr_1, _ = sklearn.metrics.roc_curve(y_true, likelihood_scores_1, pos_label=1)
    fpr_2, tpr_2, _ = sklearn.metrics.roc_curve(y_true, likelihood_scores_2, pos_label=1)
    fpr_3, tpr_3, _ = sklearn.metrics.roc_curve(y_true, likelihood_scores_3, pos_label=1)

    auc1 = sklearn.metrics.auc(fpr_1, tpr_1)
    auc2 = sklearn.metrics.auc(fpr_2, tpr_2)
    auc3 = sklearn.metrics.auc(fpr_3, tpr_3)

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
    plt.show()
    save_path = os.path.join(args.basefolder, f"roc_curve_per_privacy_group.pdf")
    plt.savefig(save_path)
    print(f"ROC curve saved at {save_path}")

    
if __name__ == "__main__":
    args=parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args=args)
