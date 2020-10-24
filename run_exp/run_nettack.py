import argparse
import datetime
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from GIB.util import get_root_dir

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', help='experiment ID. The result files will be saved under the folder "{}_{}/".format(exp_id, date_time).', required=True)
parser.add_argument('--data_type', help='Data type: choose from PROTEINS.', required=True)
parser.add_argument('--model_type', help='Model type', required=True)
parser.add_argument('--beta1', type=float, default=0.001, help='beta1 value for feature IB, set a float value >= 0.')
parser.add_argument('--beta2', type=float, default=0.01, help='beta2 value for structure IB, set a float value >= 0.')
parser.add_argument('--struct_dropout_mode', default="\(standard,0.6\)", help='mode for structure dropout.')
parser.add_argument('--latent_size', type=int, default=16, help='latent size')
parser.add_argument('--lr', type=float, default=-1, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=-1, help="weight_decay.")
parser.add_argument('--threshold', type=float, default=0.05, help='threshold for GCNJaccard')
parser.add_argument('--gamma', type=float, default=0.3, help='gamma for RGCN')
parser.add_argument('--seed', type=int, help='seed', required=True)
parser.add_argument('--gpuid', type=str, default=torch.cuda.is_available(), help='Use integer 0, 1, 2, ... to denote which cuda device to use. If using CPU, pass in "False"')
parser.add_argument('--date_time', default="{0}-{1}".format(datetime.datetime.now().month, datetime.datetime.now().day), help="Today's date and time.")
args = parser.parse_args()

# For explanation of the args, see experiments/GIB_node_attack_exp.py.
shared_args = {
    "direct_attack": True,
    "attacked_nodes": "n:10",
    "num_layers": 2,
    "reparam_mode": "diag",
    "prior_mode": "mixGau-100",
    "reparam_all_layers": '\(-2,\)',
    "epochs": 2000,
    "retrain_iters": 1,
    "is_anneal_beta": True,
    "val_use_mean": True,
}

for n_pert in [1, 2, 3, 4]:
    exec_str = "python {}/experiments/GIB_node_attack_exp.py --n_perturbations={}".format(get_root_dir(), n_pert)
    for param_name, param in args.__dict__.items():
        exec_str += " --{}={}".format(param_name, param)
    for param_name, param in shared_args.items():
        exec_str += " --{}={}".format(param_name, param)
    os.system(exec_str)
    print("\n\n")