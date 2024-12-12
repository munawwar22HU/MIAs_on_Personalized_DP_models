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
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from dpsgd_algos.individual_dp_sgd import initialize_training
from training_utils.models import CIFAR10_CNN
from training_utils.models import MNIST_CNN

individualize = 'sampling'
# individualize = 'clipping'

if individualize == 'sampling':
    # default_path = '/mfsnic/adam/idpsgd/sampling/CIFAR10/epochs_30_batch_1024_lr_0.3_max_grad_norm_0.9_budgets_1.0_2.0_ratios_0.5_0.5'
    default_path = "/mfsnic/adam/idpsgd/mia_eps_10_20/sampling/CIFAR10/epochs_60_batch_1024_lr_0.5_max_grad_norm_1.5_budgets_10.0_20.0_ratios_0.5_0.5"
elif individualize == 'clipping':
    # default_path = '/mfsnic/adam/idpsgd/clipping/CIFAR10/epochs_30_batch_1024_lr_0.3_max_grad_norm_0.9_budgets_1.0_2.0_ratios_0.5_0.5'
    default_path = '/mfsnic/adam/idpsgd/mia_eps_10_20/clipping/CIFAR10/epochs_60_batch_1024_lr_0.5_max_grad_norm_1.5_budgets_10.0_20.0_ratios_0.5_0.5'
else:
    raise ValueError('No such individualization')

parser = argparse.ArgumentParser(description='MIA Parser')
parser.add_argument('--basefolder', type=str,
                    default=default_path,
                    help='location of the run folders',
                    )
parser.add_argument('--num_shadow_models', type=int,
                    # default=256,
                    default=698,
                    help='how many shadow models should be used for MIA',
                    )
parser.add_argument('--target_model_name', type=str,
                    default='target_model',
                    # default='None',
                    help='name of the folder that contains the target model',
                    )
parser.add_argument('--dname', type=str,
                    default='CIFAR10',
                    help='dataset to be learned',
                    )
parser.add_argument('--seed', type=int,
                    default=0,
                    help='keys for reproducible pseudo-randomness',
                    )
parser.add_argument('--individualize', type=str,
                    default=individualize,
                    help='(i)DP-SGD method ("None", "clipping", "sampling")',
                    )


def rename_checkpoint_keys(checkpoint):
    state_dict = checkpoint["module_state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[8:]  # remove `_module.`
        new_state_dict[name] = v

    return new_state_dict


def load_model(args, model_path, cuda: bool = True):
    if args.dname == 'MNIST':
        model = MNIST_CNN()
    elif args.dname == 'CIFAR10' or args.dname == 'SVHN':
        model = CIFAR10_CNN()
    else:
        raise ValueError(f"No such dataset: {args.dname}")

    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    checkpoint = torch.load(model_path)

    state_dict = rename_checkpoint_keys(checkpoint=checkpoint)

    model.load_state_dict(state_dict)
    if cuda:
        model.cuda()
    model.eval()
    return model


def inference(args, loader: DataLoader, cuda: bool = True) -> np.ndarray[float]:
    """Compute logits of a given model on all the datapoints."""

    if args.target_model_name != 'None':
        args.num_shadow_models = 1
    targets = np.asarray(loader.dataset.targets)
    result_logits = torch.zeros(args.num_shadow_models, len(loader.dataset), args.num_classes)
    for index in range(args.num_shadow_models):
        print(f"index={index}")
        if args.target_model_name != 'None':
            run_name = args.target_model_name
        else:
            run_name = 'run' + str(index+1)
        load_path = os.path.join(args.basefolder, run_name)
        model_path = os.path.join(load_path, 'opacus_model.ckpt')
        model = load_model(args, model_path=model_path)

        

        with torch.no_grad():
            end = 0
            for v in loader:
                if args.dname=="CIFAR10" or args.dname=="MNIST" or args.dname=="SVHN":
                    data = v[0]
                    target = v[1]
                # elif args.individualize == 'clipping':
                #     data, target, _ = v
                batch_size = data.shape[0]
                if cuda:
                    data, target = data.cuda(), target.cuda()
                begin = end
                end = begin + batch_size

                output = model(data)
                output = output.detach().cpu()
                result_logits[index, begin:end, :] = output

                if not np.all(target.cpu().numpy() == targets[begin:end]):
                    raise Exception("The targets in the data loader are not ordered in the same way as targets from "
                                    "the dataset.")

    result_logits_arr = result_logits.numpy()
    return result_logits_arr


def main(args):
    print(f"Running MIA on {args.basefolder}")
    if args.dname in ['MNIST', 'SVHN', 'CIFAR10']:
        args.num_classes = 10
    else:
        raise ValueError(f"No such dataset: {args.dname}")

    device, train_loader, test_loader = initialize_training(dataset_name=args.dname,
                                                  cuda=True,
                                                  epochs=-1,
                                                  n_workers=6,
                                                  batch_size=1000,
                                                  seed=args.seed,
                                                  shuffle=False,
                                                  args=args)
    result_logits_arr = inference(args, loader=train_loader)
    test_result_logits_arr = inference(args, loader=test_loader)
    print("Train Loader",len(train_loader.dataset))
    print("Test Loader",len(test_loader.dataset))
    if args.target_model_name != 'None':
        logits_name = 'target_logits.npy'
        
    else:
        logits_name = 'shadow_logits.npy'
    logits_file = os.path.join(args.basefolder, logits_name)

    if args.target_model_name != 'None':
        nom_member_logits_file = os.path.join(args.basefolder, 'non_target_logits.npy')

        np.save(arr=test_result_logits_arr, file=nom_member_logits_file)
        print(test_result_logits_arr.shape)
    

    print(result_logits_arr.shape)
    np.save(arr=result_logits_arr, file=logits_file)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args=args)
