#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Script for adversarial attacks with GNNs (including GIB-GAT, GAT, GCN and other baselines.)"""
import argparse
from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import os.path as osp
import pickle
from scipy import sparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from GIB.experiments.GIB_node_model import GNN, load_model_dict_GNN, train, get_data, edge_index_2_csr, get_attacked_data_deeprobust
from GIB.pytorch_net.util import to_np_array, to_Variable, filter_filename, record_data, Beta_Function, Early_Stopping, make_dir, str2bool, to_string, eval_tuple, get_list_elements, plot_matrices
from GIB.DeepRobust.deeprobust.graph.defense import GCN
from GIB.DeepRobust.deeprobust.graph.defense import GCNJaccard
from GIB.DeepRobust.deeprobust.graph.defense import RGCN
from deeprobust.graph.data import Dataset
from GIB.util import process_data_for_nettack, select_nodes, add_distant_neighbors, GIB_PATH
import pdb

is_cuda = "cuda:0" if torch.cuda.is_available() else "cpu"


# ## Helper functions:

# In[ ]:


METRICS_RECORD = ["train_acc", "val_acc", "test_acc", "epochs",
                  "train_loss", "val_loss", "test_loss",
                  "train_f1_micro", "best_val_f1_micro", "b_test_f1_micro",
                  "train_f1_macro", "val_f1_macro", "test_f1_macro",
                 ]

def train_baseline_multiple(data, model_type, device, retrain_iters=5, suffix="", verbose=True):
    models = []
    accs = {}
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    for i in range(retrain_iters):
        if verbose:
            print("\nRetrain iter {}:".format(i))
        if model_type == 'GCNJaccard':
            model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                               num_layers=num_layers,
                               nhid=latent_size, device=device,
                               weight_decay=weight_decay if weight_decay is not None else 5e-4,
                               lr=lr if lr is not None else 0.01,
                              )
            model = model.to(device)
            model.fit(features, adj, labels, idx_train, idx_val, train_iters=epochs, threshold=threshold, verbose=verbose)
        elif model_type == 'RGCN':
            model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
                         num_layers=num_layers,
                         nhid=latent_size, device=device,
                         gamma=gamma if gamma is not None else 0.5,
                         beta1=beta1 if beta1 is not None else 5e-4,
                         beta2=weight_decay if weight_decay is not None else 5e-4,
                        )
            model = model.to(device)
            model.fit(features, adj, labels, idx_train, idx_val, train_iters=epochs, verbose=verbose)
        model.eval()
        output = model.test(idx_test)
        model.to(torch.device("cpu"))
        model.device = torch.device("cpu")
        models.append(model)
        record_data(accs, output+[epochs], ['test_loss', 'test_acc', 'epochs'])

    accs = {key + suffix: np.mean(value) for key, value in accs.items()}
    return models, accs


def train_multiple(data, loss_type, retrain_iters=5, suffix="", verbose=True, device=None):
    models = []
    accs = {}
    if struct_dropout_mode[0] == 'DNsampling' or (struct_dropout_mode[0] == 'standard' and len(struct_dropout_mode) == 3):
        data.to(torch.device("cpu"))
        data_core = deepcopy(data)
        add_distant_neighbors(data_core, struct_dropout_mode[-1])
        data.to(device)
        data_core.to(device)
    else:
        data_core = data.to(device)
    for i in range(retrain_iters):
        if verbose:
            print("\nRetrain iter {}:".format(i))

        model = GNN(
            model_type=model_type,
            num_features=info["num_features"],
            num_classes=info["num_classes"],
            reparam_mode=reparam_mode,
            prior_mode=prior_mode,
            struct_dropout_mode=struct_dropout_mode,
            latent_size=latent_size,
            num_layers=num_layers,
            val_use_mean=val_use_mean,
            reparam_all_layers=reparam_all_layers,
            is_cuda=is_cuda,
        )
        record = train(
            model,
            data_core,
            data_type=data_type,
            model_type=model_type,
            loss_type=loss_type,
            epochs=epochs,
            beta1_list=beta1_list,
            beta2_list=beta2_list,
            verbose=verbose,
            isplot=False,
            lr=lr,
            weight_decay=weight_decay,
            save_best_model=True,
        )
        record_data(accs, [record["train_acc"][-1], record["val_acc"][-1], record["test_acc"][-1], record["epoch"][-1], 
                           record["train_loss"][-1], record["val_loss"][-1], record["test_loss"][-1],
                           record["train_f1_micro"][-1], record["best_val_f1_micro"][-1], record["b_test_f1_micro"][-1],
                           record["train_f1_macro"][-1], record["val_f1_macro"][-1], record["test_f1_macro"][-1]],     
                    METRICS_RECORD)
        model_best = load_model_dict_GNN(record["best_model_dict"])
        model.to_device(torch.device("cpu"))
        models.append((model, model_best))
    accs = {key + suffix: np.mean(value) for key, value in accs.items()}
    return models, accs


def analyze(models, data, node_id, baseline, cached=None):
    classification_margins = []
    class_distrs = []
    classification_margins_best = []
    class_distrs_best = []
    cpu_device = torch.device("cpu")
    if struct_dropout_mode[0] == 'DNsampling' or (struct_dropout_mode[0] == 'standard' and len(struct_dropout_mode) == 3):
        data.to(cpu_device)
        data_core = deepcopy(data)
        add_distant_neighbors(data_core, struct_dropout_mode[-1])
    else:
        data_core = data.to(cpu_device)
    for i, model in enumerate(models):
        if isinstance(model, tuple):
            model_last = model[0]
            model_best = model[1]
            model_best = model_best.to(cpu_device)
            model_best.eval()
        else:
            model_last = model
        model_last = model_last.to(cpu_device)
        model_last.device = cpu_device
        model_last.eval()
        if baseline:
            logits = model_last.predict(data_core.features, data_core.adj)
            probs = torch.exp(logits[node_id])
        else:
            if cached is not None and hasattr(model_last, "set_cache"):
                model_last.set_cache(cached)
            logits, _ = model_last(data_core)
            probs = F.softmax(logits[node_id], dim=-1)
        class_distrs.append(probs)
        best_second_class = (probs - 1000 * y_onehot[node_id]).argmax()
        margin = probs[data_core.y[node_id]] - probs[best_second_class]
        classification_margins.append(margin)

        if isinstance(model, tuple):
            if baseline:
                logits_best = model_best.predict(data_core.features, data_core.adj)
                probs_best = torch.exp(logits_best[node_id])
            else:
                if cached is not None and hasattr(model_best, "set_cache"):
                    model_best.set_cache(cached)
                logits_best, _ = model_best(data_core)
                probs_best = F.softmax(logits_best[node_id], dim=-1)
            class_distrs_best.append(probs_best)
            best_second_class_best = (probs_best - 1000 * y_onehot[node_id]).argmax()
            margin_best = probs_best[data_core.y[node_id]] - probs_best[best_second_class_best]
            classification_margins_best.append(margin_best)

    classification_margins = torch.stack(classification_margins)
    class_distrs = torch.stack(class_distrs)
    class_distrs, classification_margins = to_np_array(class_distrs, classification_margins)

    if isinstance(model, tuple):
        classification_margins_best = torch.stack(classification_margins_best)
        class_distrs_best = torch.stack(class_distrs_best)
        class_distrs_best, classification_margins_best = to_np_array(class_distrs_best, classification_margins_best)
    return class_distrs, classification_margins, class_distrs_best, classification_margins_best


# Visualization:
def plot_attack(data, class_distrs_clean, class_distrs_attacked, node_id, retrain_iters):
    def make_xlabel(ix, correct):
        if ix==correct:
            return "Class {}\n(correct)".format(ix)
        return "Class {}".format(ix)

    figure = plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    center_ixs_clean = []
    for ix, block in enumerate(class_distrs_clean.T):
        x_ixs= np.arange(len(block)) + ix*(len(block)+2)
        center_ixs_clean.append(np.mean(x_ixs))
        color = '#555555'
        if ix == data.y[node_id].item():
            color = 'darkgreen'
        plt.bar(x_ixs, block, color=color)

    ax=plt.gca()
    plt.ylim((-.05, 1.05))
    plt.ylabel("Predicted probability")
    ax.set_xticks(center_ixs_clean)
    ax.set_xticklabels([make_xlabel(k, data.y[node_id].item()) for k in range(info["num_classes"])])
    ax.set_title("Predicted class probabilities for node {} on clean data\n({} re-trainings)".format(node_id, retrain_iters))

    fig = plt.subplot(1, 2, 2)
    center_ixs_attacked = []
    for ix, block in enumerate(class_distrs_attacked.T):
        x_ixs= np.arange(len(block)) + ix*(len(block)+2)
        center_ixs_attacked.append(np.mean(x_ixs))
        color = '#555555'
        if ix == data.y[node_id].item():
            color = 'darkgreen'
        plt.bar(x_ixs, block, color=color)


    ax=plt.gca()
    plt.ylim((-.05, 1.05))
    ax.set_xticks(center_ixs_attacked)
    ax.set_xticklabels([make_xlabel(k, data.y[node_id].item()) for k in range(info["num_classes"])])
    ax.set_title("Predicted class probabilities for node {} after {} perturbations\n({} re-trainings)".format(node_id, info_attack["n_perturbations"], retrain_iters))
    plt.tight_layout()
    plt.show()


# ## Settings:

# In[ ]:


"""
Typical GIB-Cat setting: model_type="GAT", beta1=0.001, beta2=0.01, struct_dropout_mode=("DNsampling",'multi-categorical-sum',0.1,3,2);
Typical GIB-Bern setting:model_type="GAT", beta1=0.001, beta2=0.01, struct_dropout_mode=("DNsampling",'Bernoulli',0.1,0.5,"norm",2);
Standard GAT setting:    model_type="GAT", beta1=-1,    beta2=-1,  struct_dropout_mode=("standard", 0.6);
Standard GCN setting:    model_type="GCN", beta1=-1,    beta2=-1
RGCN setting:            model_type="RGCN"
GCNJaccard setting:      model_type="GCNJaccard"
"""
# Attack settings:
attack_type = "nettack"     # Attack type. Currently only implemented Nettack.
direct_attack = True        # Whether to use direct attack
attacked_nodes = "n:10"     # Target nodes to attack. Use "n:10" for the default attack (40 nodes) in the Nettack paper. Alternatively, use "r:1000" for randomly choose 1000 test nodes to attack.
n_perturbations = 1         # Number of feature or structural perturbations. Can be a integer (1, 2, 3,...). When set to -1, for each node will have degree + 2 perturbations.

# Important model settings:
exp_id = "attack"           # Experiment id, used for the directory name saving the experiment result files.
data_type = 'Cora-bool'     # Data type. Should use suffix "-bool" for binarized features. Choose from "Cora-bool", "Pubmed-bool" and "citeseer-bool"
model_type = 'GAT'          # Name of the base model. Choose from "GAT", "GCN", 'GCNJaccard', 'RGCN'. 
                            # For GIB-Cat and GIB-Bern, still choose model_type="GAT", but set either beta1 or beta2 nonzero.
beta1 = 0.001               # coefficient for the XIB term. If -1, this term will turn off.
beta2 = 0.01                # coefficient for the AIB term. If -1, this term will have 0 coefficent (but may still perform sampling, depending on "struct_dropout_mode")
struct_dropout_mode = ("DNsampling",'multi-categorical-sum',1,3,2)  # Mode for how the structural representation is generated.
                            # For GIB-Cat, choose from ("Nsampling", 'multi-categorical-sum', 0.1, 3) (here 0.1 is temperature, k=3 is the number of sampled neighboring edges with replacement),
                            #    and ("DNsampling", 'multi-categorical-sum', 0.1, 3, 2) (similar as above, but with 2-hop neighbors)
                            # For GIB-Bern, choose from ("Nsampling",'Bernoulli',0.1,0.5,"norm") (here 0.1 is temperature, 0.5 is the prior for the Bernoulli probability)
                            #    and ("DNsampling",'Bernoulli',0.1,0.5,"norm",2) (with 2-hop neighbors)
                            # For standard GAT, choose from ("standard", 0.6) (where standard dropout used on the attention weights in GAT)
                            #    and ("standard", 0.6, 2) (with 2-hop neighbors)
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

# Other settings:
retrain_iters = 1           # How many models to train for each setting and given attacked nodes.
is_load = True              # Whether to load previous checkpoint.
threshold = 0.05            # Threshold for GCNJaccard.
gamma = 0.5                 # gamma for RGCN

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pylab as plt
    isplot = False
    verbose = False
except:
    import matplotlib
    isplot = False
    verbose = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default=exp_id, help='experiment ID')
    parser.add_argument('--data_type', help='Data type: choose from PROTEINS.', required=True)
    parser.add_argument('--model_type', default="GAT", help='Model type: GCN or GAT or GCNJaccard or RGCN')
    parser.add_argument('--direct_attack', type=str2bool, nargs='?', const=True, default=True, help='Whether to do direct attack.')
    parser.add_argument('--attacked_nodes', default="::", help='indices of attacked_nodes')
    parser.add_argument('--n_perturbations', type=int, default=5, help='number of perturbations')
    parser.add_argument('--beta1', type=float, default=0.001, help='beta1 value for feature IB, set a float value >= 0.')
    parser.add_argument('--beta2', type=float, default=0.01, help='beta2 value for structure IB, set a float value >= 0.')
    parser.add_argument('--latent_size', type=int, default=16, help='latent_size')
    parser.add_argument('--sample_size', type=int, default=1, help='sample_size')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    parser.add_argument('--reparam_mode', default="diag", help='diag, diagg, or full')
    parser.add_argument('--prior_mode', default="Gaussian", help='prior mode for VIB')
    parser.add_argument('--struct_dropout_mode', default="(standard, 0.6)", help='mode for structure dropout.')
    parser.add_argument('--is_anneal_beta', type=str2bool, nargs='?', const=True, default=True, help='Whether to anneal beta.')
    parser.add_argument('--val_use_mean', type=str2bool, nargs='?', const=True, default=True, help='Whether to use mean of Z during validation.')
    parser.add_argument('--reparam_all_layers', help='Whether to reparameterize all layers.')
    parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs.")
    parser.add_argument('--date_time', default=date_time, help="Current date and time.")
    parser.add_argument('--lr', type=float, default=lr, help='lr')
    parser.add_argument('--weight_decay', type=float, default=-1, help="weight_decay.")
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--gpuid', help='an integer for the accumulator', required=True)
    parser.add_argument('--idx', default="0", help='idx')
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold for GCNJaccard')
    parser.add_argument('--gamma', type=float, default=0.3, help='gamma for RGCN')
    parser.add_argument('--retrain_iters', type=int, default=1, help='retrain_iters')
    parser.add_argument('--is_load', type=str2bool, nargs='?', const=True, default=True, help='Whether to load previous trained instance.')
    
    args = parser.parse_args()


if "args" in locals():
    exp_id = args.exp_id
    data_type = args.data_type
    model_type = args.model_type
    direct_attack = args.direct_attack
    attacked_nodes = args.attacked_nodes
    n_perturbations = args.n_perturbations
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
    date_time = args.date_time
    lr = args.lr
    weight_decay = args.weight_decay
    seed = args.seed
    idx = args.idx
    threshold = args.threshold
    gamma = args.gamma
    retrain_iters = args.retrain_iters
    is_load = args.is_load
    is_cuda = eval(args.gpuid)
    if not isinstance(is_cuda, bool):
        is_cuda = "cuda:{}".format(is_cuda)
baseline = model_type in ['GCNJaccard', 'RGCN']


device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")

dirname = GIB_PATH + "/{0}_{1}/".format(exp_id, date_time)
if baseline: ########## difference is that for baselines we have an additional parameters
    filename = dirname + "{0}_{1}_dire_{2}_nodes_{3}_pert_{4}_lat_{5}_samp_{6}_l_{7}_anl_{8}_mean_{9}_reall_{10}_epochs_{11}_lr_{12}_l2_{13}_seed_{14}_threshold_{15}_gamma_{16}_{17}_id_{18}".format(
        data_type, model_type, direct_attack, attacked_nodes, n_perturbations,
        latent_size, sample_size, num_layers, 
        is_anneal_beta, val_use_mean, reparam_all_layers, epochs, lr, weight_decay, seed, threshold, gamma, is_cuda, idx,
    )
else:
    filename = dirname + "{0}_{1}_dire_{2}_nodes_{3}_pert_{4}_beta_{5}_{6}_lat_{7}_samp_{8}_l_{9}_reparam_{10}_prior_{11}_sdrop_{12}_anl_{13}_mean_{14}_reall_{15}_epochs_{16}_lr_{17}_l2_{18}_seed_{19}_{20}_id_{21}".format(
        data_type, model_type, direct_attack, attacked_nodes, n_perturbations, beta1, beta2,
        latent_size, sample_size, num_layers, reparam_mode, prior_mode, to_string(struct_dropout_mode, "-"), 
        is_anneal_beta, val_use_mean, reparam_all_layers, epochs, lr, weight_decay, seed, is_cuda, idx,
    )
make_dir(filename)
print(filename)

np.random.rand(seed)
torch.manual_seed(seed)

# Set up attacked nodes:
assert attacked_nodes in ["r:1000", "n:10"]
if attacked_nodes.startswith("n:"):
    data, info = get_data(data_type, seed=seed)
else:
    data, info = get_data(data_type, seed=seed, attacked_nodes=attacked_nodes)
data = process_data_for_nettack(data).to(device)
if attacked_nodes.startswith("n:"):
    node_ids = select_nodes(data, info, data_type, num=eval(attacked_nodes.split(":")[-1]), seed=seed)
else:
    node_ids = info['node_ids']
y_onehot = torch.eye(info["num_classes"])[data.y]


if n_perturbations == -1:
    n_perturbations = None
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


# ## Surrogate model for Nettack:

# In[ ]:


if attack_type == "nettack":
    surrogate_model = GCN(nfeat=info["num_features"], nclass=info["num_classes"],
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
    surrogate_model.fit(data.features, data.adj, data.labels, data.idx_train)
    W1 = to_np_array(surrogate_model.gc1.weight)
    W2 = to_np_array(surrogate_model.gc2.weight)
    all_dict = {"params": {"W1": W1, "W2": W2}}
else:
    raise


# ## Perform and evaluate attacks:

# In[ ]:


# Load previous instance:
if is_load:
    try:
        filename_core = "_".join(filename.split("/")[-1].split("_")[:-3])
        cand_filename = filter_filename(dirname, include=filename_core)
        if len(cand_filename) == 0:
            raise Exception("Did not find previous file. Create new.")
        assert cand_filename[0].endswith(".p")
        filename = dirname + cand_filename[0][:-2]
        all_dict_cand = pickle.load(open(filename + ".p", "rb"))
    except Exception as e:
        print(e)
        is_load = False
    if is_load:
        if attack_type == "nettack":
            if not "params" in all_dict_cand:
                is_load = False
            else:
                all_dict = all_dict_cand
                surrogate_model.gc1.weight.data = torch.FloatTensor(all_dict["params"]["W1"]).to(device)
                surrogate_model.gc2.weight.data = torch.FloatTensor(all_dict["params"]["W2"]).to(device)
                print("Load previous trained instance at {}.".format(filename))

is_train = True
if is_load:
    is_all_attacked = True
    for node_id in node_ids:
        if node_id not in all_dict:
            is_all_attacked = False
            break
    if not is_all_attacked:
        if "models_before" in all_dict and isinstance(all_dict["models_before"][0], tuple):
            print("Load models_before.")
            models_before = [(load_model_dict_GNN(model_dict_last, is_cuda=is_cuda), 
                              load_model_dict_GNN(model_dict_best, is_cuda=is_cuda)) for model_dict_last, model_dict_best in all_dict["models_before"]]
            is_train = False
    else:
        is_train = False
        print("all node_ids are attacked. Skip training clean model.")

if is_train:
    print("Training clean model:")
    if baseline:
        models_before, accs_clean = train_baseline_multiple(data, model_type, device, retrain_iters=retrain_iters, suffix="_clean", verbose=verbose)
    else:
        models_before, accs_clean = train_multiple(data, loss_type=info["loss"], retrain_iters=retrain_iters, suffix="_clean", verbose=verbose, device=device)
        all_dict["models_before"] = [(model_last.model_dict, model_best.model_dict) for model_last, model_best in models_before]
else:
    accs_clean = {}
    for key in METRICS_RECORD:
        accs_clean[key + "_clean"] = all_dict[node_ids[0]][key + "_clean"]

# Iterations:
for i, node_id in enumerate(node_ids):
    node_id = int(node_id)
    if is_load and node_id in all_dict:
        print("Skip node {}.".format(node_id))
        continue
    print("\nAttacking the {}th node id={}:".format(i, node_id))
    if attack_type == "nettack":
        data_attacked, info_attack = get_attacked_data_deeprobust(
            data,
            surrogate_model=surrogate_model,
            target_node=node_id,
            direct_attack=direct_attack,
            n_perturbations=n_perturbations,
            verbose=False,
        )
    else:
        raise

    # Train with clean data:
    if verbose:
        print("\nClean data:")
    class_distrs_clean, classification_margins_clean, class_distrs_clean_best, classification_margins_clean_best = analyze(models_before, data, node_id, baseline)
    class_distrs_evasive, classification_margins_evasive, class_distrs_evasive_best, classification_margins_evasive_best = analyze(models_before, data_attacked, node_id, baseline, cached=False)

    # Train with attacked data:
    if verbose:
        print("\nAttacked data:")
    if baseline:
        models_attacked, accs_attacked = train_baseline_multiple(data_attacked, model_type, device, retrain_iters=retrain_iters, suffix="_attacked", verbose=verbose)
    else:
        models_attacked, accs_attacked = train_multiple(data_attacked, loss_type=info["loss"], retrain_iters=retrain_iters, suffix="_attacked", verbose=verbose, device=device)
    class_distrs_attacked, classification_margins_attacked, class_distrs_attacked_best, classification_margins_attacked_best = analyze(models_attacked, data_attacked, node_id, baseline)

    all_dict[node_id] = {
        "class_distrs_clean": class_distrs_clean,
        "class_distrs_evasive": class_distrs_evasive,
        "class_distrs_attacked": class_distrs_attacked,
        "classification_margins_clean": classification_margins_clean,
        "classification_margins_evasive": classification_margins_evasive,
        "classification_margins_attacked": classification_margins_attacked,  
        "class_distrs_clean_best": class_distrs_clean_best,
        "class_distrs_evasive_best": class_distrs_evasive_best,
        "class_distrs_attacked_best": class_distrs_attacked_best,
        "classification_margins_clean_best": classification_margins_clean_best,
        "classification_margins_evasive_best": classification_margins_evasive_best,
        "classification_margins_attacked_best": classification_margins_attacked_best,  
    }
    all_dict[node_id].update(accs_clean)
    all_dict[node_id].update(accs_attacked)
    if len(class_distrs_attacked_best) > 0:
        all_dict[node_id]["models_before"] = [(model_last.model_dict, model_best.model_dict) for model_last, model_best in models_before]
        all_dict[node_id]["models_attacked"] = [(model_last.model_dict, model_best.model_dict) for model_last, model_best in models_attacked]
    pickle.dump(all_dict, open(filename + ".p", "wb"))
    print("margins: clean: {:.4f}   evasive: {:.4f}   poison: {:.4f}   test_acc_clean: {:.4f}   test_acc_poison: {:.4f}     epochs: ({:.0f}, {:.0f})".format(
        classification_margins_clean.mean(),
        classification_margins_evasive.mean(),
        classification_margins_attacked.mean(),
        all_dict[node_id]["test_acc_clean"],
        all_dict[node_id]["test_acc_attacked"],
        all_dict[node_id]["epochs_clean"],
        all_dict[node_id]["epochs_attacked"],
    ))
    try:
        sys.stdout.flush()
    except:
        pass
    if isplot:
        plot_attack(data, class_distrs_clean, class_distrs_attacked, node_id, retrain_iters)

