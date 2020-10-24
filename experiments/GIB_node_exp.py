#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Script for experiments with standard learning with GNNs (including GIB-GAT, GAT, GCN and other baselines.)"""
import argparse
from copy import deepcopy
import datetime
import matplotlib.pylab as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from GIB.experiments.GIB_node_model import GNN, load_model_dict_GNN, get_data, train, test_model, train_baseline
from GIB.pytorch_net.util import plot_matrices, to_np_array, Beta_Function, record_data, str2bool, make_dir, eval_tuple, to_string, filter_filename
from GIB.util import sample_lognormal, add_distant_neighbors, uniform_prior, process_data_for_nettack, GIB_PATH
from GIB.DeepRobust.deeprobust.graph.defense import GCNJaccard
from GIB.DeepRobust.deeprobust.graph.defense import RGCN


# ## Settings:

# In[2]:


"""
Typical GIB-Cat setting: model_type="GAT", beta1=0.001, beta2=0.1, struct_dropout_mode=("Nsampling",'multi-categorical-sum',0.1,3) (or ("DNsampling",'multi-categorical-sum',0.1,3,2));
Typical GIB-Bern setting:model_type="GAT", beta1=0.001, beta2=0.1, struct_dropout_mode=("Nsampling",'Bernoulli',0.1,0.5,"norm") (or ("DNsampling",'Bernoulli',0.1,0.5,"norm",2));
Standard GAT setting:    model_type="GAT", beta1=-1,    beta2=-1,  struct_dropout_mode=("standard",0.6);
Standard GCN setting:    model_type="GCN", beta1=-1,    beta2=-1
RGCN setting:            model_type="RGCN"
GCNJaccard setting:      model_type="GCNJaccard"
"""
exp_id = "exp1.0"           # Experiment id, used for the directory name saving the experiment result files.
data_type = 'Cora'          # Data type. Choose from "Cora", "Pubmed", "citeseer"
model_type = 'GAT'          # Name of the base model. Choose from "GAT", "GCN", 'GCNJaccard', 'RGCN'. 
                            # For GIB-Cat and GIB-Bern, still choose model_type="GAT", but set either beta1 or beta2 nonzero.
beta1 = 0.001               # coefficient for the XIB term. If -1, this term will turn off.
beta2 = 0.1                 # coefficient for the AIB term. If -1, this term will have 0 coefficent (but may still perform sampling, depending on "struct_dropout_mode")
struct_dropout_mode = ("Nsampling", 'multi-categorical-sum', 0.1, 3)  # Mode for how the structural representation is generated. 
                            # For GIB-Cat, choose from ("Nsampling", 'multi-categorical-sum', 0.1, 3) (here 0.1 is temperature, k=3 is the number of sampled neighboring edges with replacement),
                            #    and ("DNsampling", 'multi-categorical-sum', 0.1, 3, 2) (similar as above, but with 2-hop neighbors)
                            # For GIB-Bern, choose from ("Nsampling",'Bernoulli',0.1,0.5,"norm") (here 0.1 is temperature, 0.5 is the prior for the Bernoulli probability)
                            #    and ("DNsampling",'Bernoulli',0.1,0.5,"norm",2) (with 2-hop neighbors)
                            # For standard GAT, choose from ("standard", 0.6) (where standard dropout used on the attention weights in GAT)
                            #    and ("standard", 0.6, 2) (with 2-hop neighbors)
train_fraction = 1.         # Fraction of training labels preserved for the training set. Default 1, meaning using the full training set in the standard split.
added_edge_fraction = 0.    # Fraction of added random edges. Default 0.
feature_noise_ratio = 0.    # Noise ratio for the additive independent Gaussian noise on the features.
latent_size = 16            # Latent dimension for GCN-based or GAT-based models.
sample_size = 1             # How many Z sampled from each node X.
num_layers = 2              # Number of layers for the GNN.
reparam_mode = "diag"       # Reparameterization mode for XIB. Choose from "None", "diag" or "full"
prior_mode = "mixGau-100"   # Prior mode. Choose from "Gaussian" or "mixGau-100" (mixture of 100 Gaussian components)
is_anneal_beta = True       # Whether to anneal beta1 and beta2 from 0 up during training. Default True.
val_use_mean = True         # Whether during evaluation use the parameter value instead of sampling. If True, during evaluation,
                            # XIB will use mean for prediction, and AIB will use the parameter of the categorical distribution for prediction.
reparam_all_layers = (-2,)  # Which layers to use XIB, e.g. (1,2,4). Default (-2,), meaning the second last layer. If True, use XIB for all layers.
epochs = 2000               # Number of epochs. Default 2000
lr = -1                     # Learning rate. If -1, use default learning rate for each model
weight_decay = -1           # weight decay. If -1, use default weight decay for each model
date_time = "{0}-{1}".format(datetime.datetime.now().month, datetime.datetime.now().day)  # Today's month and day. Used for the directory name saving the experiment result files.
seed = 0                    # Random seed.
idx = "0"                   # idx to differentiate different files. Only used if wanting to run the same setting for many times.
save_best_model = True      # Whether to save the model with the best validation accuracy.
skip_previous = False       # If True, will skip the training if the same setting has already been trained.
is_cuda = "cuda:0"          # CUDA device. Choose from False, or "cuda:${NUMBER}", where the ${NUMBER} is the GPU id.

threshold = 0.05            # threshold for GCNJaccard.
gamma = 0.5                 # gamma for RGCN

try:
    # If the current envionrment is Jupyter notebook:
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pylab as plt
    isplot = True
except:
    # If the current envionrment is terminal, pass in settings from the command line:
    import matplotlib
    isplot = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default=exp_id, help='experiment ID')
    parser.add_argument('--data_type', help='Data type: choose from PROTEINS.', required=True)
    parser.add_argument('--model_type', default="GAT", help='Model type: GCN or GAT or GCNJaccard or RGCN')
    parser.add_argument('--train_fraction', type=float, default=1., help='train_fraction')
    parser.add_argument('--added_edge_fraction', type=float, default=0., help='Fraction of added edges.')
    parser.add_argument('--feature_noise_ratio', type=float, default=0., help='Relative amplitude of feature Gaussian noise')
    parser.add_argument('--beta1', type=float, default=0.001, help='beta1 value for feature IB, set a float value >= 0.')
    parser.add_argument('--beta2', type=float, default=0.1, help='beta2 value for structure IB, set a float value >= 0.')
    parser.add_argument('--latent_size', type=int, default=16, help='latent_size')
    parser.add_argument('--sample_size', type=int, default=1, help='sample_size')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    parser.add_argument('--reparam_mode', default="diag", help='diag, diagg, or full')
    parser.add_argument('--prior_mode', default="mixGau-100", help='prior mode for VIB')
    parser.add_argument('--struct_dropout_mode', default="(standard, 0.6)", help='mode for structure dropout.')
    '''
    'Nsampling, categorical/subset/multi-categorical-sum/multi-categorical-max, temperature, sample-neighbor-size'
    'Nsampling, Bernoulli, temperature, prior (0~1)', 'norm'/'none'
    'DNsampling, categorical/subset/multi-categorical-sum/multi-categorical-max, temperature, sample-neighbor-size, hops'
    'DNsampling, Bernoulli, temperature, prior (0~1), 'norm'/'none', hops'
    'standard, 0.6'
    'standard, 0.6, hops'
    '''
    parser.add_argument('--is_anneal_beta', type=str2bool, nargs='?', const=True, default=True, help='Whether to anneal beta.')
    parser.add_argument('--val_use_mean', type=str2bool, nargs='?', const=True, default=True, help='Whether to use mean of Z during validation.')
    parser.add_argument('--reparam_all_layers', type=str, default="\(-2,\)", help='Whether to reparameterize all layers.')
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs.")
    parser.add_argument('--lr', type=float, default=-1, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=-1, help="weight_decay.")
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold for GCNJaccard')
    parser.add_argument('--gamma', type=float, default=0.3, help='gamma for RGCN')
    parser.add_argument('--save_best_model', type=str2bool, nargs='?', const=True, default=True, help='Whether to save the best model.')
    parser.add_argument('--skip_previous', type=str2bool, nargs='?', const=True, default=False, help='Whether to skip previously trained model in the same directory.')
    parser.add_argument('--date_time', default=date_time, help="Current date and time.")
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--gpuid', help='an integer for the accumulator', required=True)
    parser.add_argument('--idx', default="0", help='idx')
    args = parser.parse_args()


if "args" in locals():
    exp_id = args.exp_id
    data_type = args.data_type
    model_type = args.model_type
    train_fraction = args.train_fraction
    added_edge_fraction = args.added_edge_fraction
    feature_noise_ratio = args.feature_noise_ratio
    beta1 = args.beta1
    beta2 = args.beta2
    latent_size = args.latent_size
    sample_size = args.sample_size
    num_layers = args.num_layers
    reparam_mode = args.reparam_mode
    prior_mode = args.prior_mode
    struct_dropout_mode = eval_tuple(args.struct_dropout_mode)
    is_anneal_beta = args.is_anneal_beta
    val_use_mean = args.val_use_mean
    reparam_all_layers = eval_tuple(args.reparam_all_layers)
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    threshold = args.threshold
    gamma = args.gamma
    save_best_model = args.save_best_model
    skip_previous = args.skip_previous
    date_time = args.date_time
    seed = args.seed
    idx = args.idx
    is_cuda = eval(args.gpuid)
    if not isinstance(is_cuda, bool):
        is_cuda = "cuda:{}".format(is_cuda)

baseline = model_type in ['GCNJaccard', 'RGCN']
device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")
# Directory and filename:
dirname = GIB_PATH + "/{0}_{1}/".format(exp_id, date_time)
if baseline:
    filename = dirname + "{0}_{1}_tr_{2}_ed_{3}_{4}_beta_{5}_{6}_lat_{7}_samp_{8}_lay_{9}_anl_{10}_mean_{11}_reall_{12}_epochs_{13}_lr_{14}_l2_{15}_seed_{16}_threshold_{17}_gamma_{18}_{19}_id_{20}".format(
        data_type, model_type, train_fraction, added_edge_fraction, feature_noise_ratio, beta1, beta2, latent_size, sample_size, num_layers,
        is_anneal_beta, val_use_mean, to_string(reparam_all_layers, "-"), epochs, lr, weight_decay, seed, threshold, gamma, is_cuda, idx
    )
else:
    filename = dirname + "{0}_{1}_tr_{2}_ed_{3}_{4}_beta_{5}_{6}_lat_{7}_samp_{8}_lay_{9}_reparam_{10}_prior_{11}_sdrop_{12}_anl_{13}_mean_{14}_reall_{15}_epochs_{16}_lr_{17}_l2_{18}_seed_{19}_{20}_id_{21}".format(
        data_type, model_type, train_fraction, added_edge_fraction, feature_noise_ratio, beta1, beta2, latent_size, sample_size, num_layers, reparam_mode, prior_mode,
        to_string(struct_dropout_mode, "-"), is_anneal_beta, val_use_mean, to_string(reparam_all_layers, "-"), epochs, lr, weight_decay, seed, is_cuda, idx,
    )


# In[3]:


# Setting the seed:
np.random.seed(seed)
torch.manual_seed(seed)

# Setting default hyperparameters:
if struct_dropout_mode[0] is None:
    struct_dropout_mode = ("None",)
if lr == -1:
    lr = None
if weight_decay == -1:
    weight_decay = None
if beta1 == -1:
    beta1 = None
if beta1 is None:
    beta1_list, reparam_mode, prior_mode = None, None, None
else:
    if is_anneal_beta:
        beta_init = 0
        init_length = int(epochs / 4)
        anneal_length = int(epochs / 4)
        beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
        beta1_inter = beta_inter / 4 * (beta_init - beta1) + beta1
        beta1_list = np.concatenate([np.ones(init_length) * beta_init, beta1_inter, 
                                     np.ones(epochs - init_length - anneal_length + 1) * beta1])
    else:
        beta1_list = np.ones(epochs + 1) * beta1
if beta2 == -1:
    beta2_list = None
else:
    if is_anneal_beta:
        beta_init = 0
        init_length = int(epochs / 4)
        anneal_length = int(epochs / 4)
        beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
        beta2_inter = beta_inter / 4 * (beta_init - beta2) + beta2
        beta2_list = np.concatenate([np.ones(init_length) * beta_init, beta2_inter, 
                                     np.ones(epochs - init_length - anneal_length + 1) * beta2])
    else:
        beta2_list = np.ones(epochs + 1) * beta2

# Get Dataset:
data, info = get_data(data_type,
                      train_fraction=train_fraction,
                      added_edge_fraction=added_edge_fraction,
                      feature_noise_ratio=feature_noise_ratio,
                      seed=seed,
                     )

if struct_dropout_mode[0] == 'DNsampling' or (struct_dropout_mode[0] == 'standard' and len(struct_dropout_mode) == 3):
    add_distant_neighbors(data, struct_dropout_mode[-1])
data = process_data_for_nettack(data).to(device)
data = data.to(device)


if model_type == 'GCNJaccard':
    model = GCNJaccard(nfeat=data.features.shape[1], nclass=data.labels.max()+1,
                       num_layers=num_layers,
                       nhid=latent_size, device=device,
                       weight_decay=weight_decay if weight_decay is not None else 5e-4,
                       lr=lr if lr is not None else 0.01,
                      )
elif model_type == 'RGCN':
    model = RGCN(nnodes=data.adj.shape[0], nfeat=data.features.shape[1], nclass=data.labels.max()+1,
                 num_layers=num_layers,
                 nhid=latent_size, device=device,
                 lr=lr if lr is not None else 0.01,
                 gamma=gamma if gamma is not None else 0.5,
                 beta1=beta1 if beta1 is not None else 5e-4,
                 beta2=weight_decay if weight_decay is not None else 5e-4,
                )
else:
    # For GIB-GAT, GAT or GCN:
    model = GNN(
        model_type=model_type,
        num_features=info["num_features"],
        num_classes=info["num_classes"],
        normalize=True,
        reparam_mode=reparam_mode,
        prior_mode=prior_mode,
        latent_size=latent_size,
        sample_size=sample_size,
        num_layers=num_layers,
        struct_dropout_mode=struct_dropout_mode,
        dropout=True,
        val_use_mean=val_use_mean,
        reparam_all_layers=reparam_all_layers,
        is_cuda=is_cuda,
    )


# In[ ]:


print(filename + "\n")

if skip_previous:
    filename_core = "_".join(filename.split("/")[-1].split("_")[:-3])
    if filename_core.endswith("cuda"):
        filename_core = filename_core[:-5]
    cand_filename = filter_filename(dirname, include=filename_core)
    if len(cand_filename) == 0:
        skip_previous = False
if skip_previous:
    print("File already exists at {} with {}".format(dirname, cand_filename))
else:
    if baseline:
        data_record = train_baseline(model, model_type, data, device, threshold, filename, epochs, save_best_model=save_best_model, verbose=True)
    else:
        data_record = train(
            model=model,
            data=data,
            data_type=data_type,
            model_type=model_type,
            loss_type=info['loss'],
            beta1_list=beta1_list,
            beta2_list=beta2_list,
            epochs=epochs,
            inspect_interval=200 if isplot else 20,
            verbose=True,
            isplot=isplot,
            filename=filename,
            compute_metrics=None,
            lr=lr,
            weight_decay=weight_decay,
            save_best_model=save_best_model,
        )

