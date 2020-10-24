"""Script for running multiple attack experiments with different hyperparameters in a parallel way.

Run this file:

    python run_nettack_grid.py ${Assign_ID} ${GPU_ID}

where the ${Assign_ID} (belonging to [0, M-1] will map to a unique hyperparameter setting, where M is the number of total hyperparameter
combinations. ${GPU_ID} is the CUDA id you want this job to be assigned to (use "False" if wanting to use CPU).
"""

import numpy as np
import torch
import os, sys
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from GIB.util import GIB_PATH, get_root_dir


exp_id = [
"nttack1.0"
]

data_type = [
"Cora-bool",
"Pubmed-bool",
"citeseer-bool",
]

model_type = [
# "GCN",
"GAT",
# "RGCN",
# "GCNJaccard",
]

direct_attack = [
True,
]

attacked_nodes = [
"n:10",
]

n_perturbations = [
1,
2,
3,
4,
]

beta1 = [
0.001,
]

beta2 = [
0.01,
]


latent_size = [
16,
]

num_layers = [
2,
]

reparam_mode = [
"diag",
]

prior_mode = [
"mixGau-100",
]

struct_dropout_mode=[
# '\("standard",0.6\)',
# '\("standard",0.6,2\)',
'\("DNsampling","multi-categorical-sum",1,3,2\)',
'\("DNsampling","Bernoulli",0.1,0.5,"norm",2\)',
]


reparam_all_layers = [
'\(-2,\)',
]

epochs=[
2000,
]

lr = [
-1,
]

seed = [
0,
1,
2,
3,
4,
]

date_time = [
"4-6",
]

retrain_iters = [
1,   
]


def assign_array_id(array_id, param_list):
    if len(param_list) == 0:
        print("redundancy: {0}".format(array_id))
        return []
    else:
        param_bottom = param_list[-1]
        length = len(param_bottom)
        current_param = param_bottom[array_id % length]
        return assign_array_id(int(array_id / length), param_list[:-1]) + [current_param]


array_id = int(sys.argv[1])
param_list = [
    exp_id,
    data_type,
    model_type,
    direct_attack,
    attacked_nodes,
#     n_perturbations,
    beta1,
    beta2,
    latent_size,
    num_layers,
    reparam_mode,
    prior_mode,
    struct_dropout_mode,
    reparam_all_layers,
    epochs,
    lr,
    seed,
    retrain_iters,
    date_time,
]

param_name_list = [
    'exp_id',
    'data_type',
    'model_type',
    'direct_attack',
    'attacked_nodes',
#     'n_perturbations',
    'beta1',
    'beta2',
    'latent_size',
    'num_layers',
    'reparam_mode',
    'prior_mode',
    'struct_dropout_mode',
    'reparam_all_layers',
    'epochs',
    'lr',
    'seed',
    'retrain_iters',
    'date_time',
]
param_chosen = assign_array_id(array_id, param_list)

from shutil import copyfile
current_PATH = os.path.dirname(os.path.realpath(__file__))
def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise

filename = GIB_PATH + "/{}_{}/".format(param_chosen[0], param_chosen[-1])
make_dir(filename)
filename_current = "run_nettack_grid.py"
if not os.path.isfile(filename + filename_current):
    copyfile(current_PATH + "/" + filename_current, filename + filename_current)
make_dir("GIB/{}".format(exp_id))

GPU_ID = sys.argv[2] if len(sys.argv) >= 3 else torch.cuda.is_available()

for n_pert in n_perturbations:
    exec_str = "python {}/experiments/GIB_node_attack_exp.py --n_perturbations={} --gpuid={}".format(get_root_dir(), n_pert, GPU_ID)
    for param, param_name in zip(param_chosen, param_name_list):
        exec_str += " --{}={}".format(param_name, param)
    exec_str += " --idx={}".format(array_id)
    os.system(exec_str)
    print("\n\n")