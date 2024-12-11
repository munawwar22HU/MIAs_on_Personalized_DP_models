import numpy as np
run1 =  np.load("../results1/sampling/MNIST/epochs_5_batch_1024_lr_0.7_max_grad_norm_0.4_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0/run1/assignment.npy")
run2 =  np.load("../results1/sampling/MNIST/epochs_5_batch_1024_lr_0.7_max_grad_norm_0.4_budgets_1.0_2.0_3.0_ratios_0.54_0.37_0.09_seeds_0/run2/assignment.npy")

print(len(np.setdiff1d(run1, run2)))
assert np.array_equal(run1, run2), "The two runs are not equal"