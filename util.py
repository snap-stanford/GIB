"""Utility functions."""
import torch
from torch_geometric.utils import degree, softmax, subgraph
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops, add_self_loops
import sys, os
import os.path as osp
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from GIB.pytorch_net.util import to_np_array
from GIB.DeepRobust.deeprobust.graph.defense import GCN
import scipy.sparse as sp
import numpy as np
import pdb
import pickle

def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("GIB")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname

GIB_PATH = get_root_dir() + "/results"


COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]
COLOR_LIST = COLOR_LIST * 2

LINESTYLE_LIST = ["-", "--", "-.", ":", (0, (5, 10))]

def get_reparam_num_neurons(out_channels, reparam_mode):
    if reparam_mode is None or reparam_mode == "None":
        return out_channels
    elif reparam_mode == "diag":
        return out_channels * 2
    elif reparam_mode == "full":
        return int((out_channels + 3) * out_channels / 2)
    else:
        raise "reparam_mode {} is not valid!".format(reparam_mode)


def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = torch.randn(mean.shape).to(sigma.device)
    return torch.exp(mean + sigma * sigma0 * e)

def scatter_sample(src, index, temperature, num_nodes=None):
    gumbel = torch.distributions.Gumbel(torch.tensor([0.]).to(src.device), 
            torch.tensor([1.0]).to(src.device)).sample(src.size()).squeeze(-1)
    log_prob = torch.log(src+1e-16)
    logit = (log_prob + gumbel) / temperature
    return softmax(logit, index, num_nodes)

def uniform_prior(index):
    deg = degree(index)
    deg = deg[index]
    return 1./deg.unsqueeze(1)

def add_distant_neighbors(data, hops):
    """Add multi_edge_index attribute to data which includes the edges of 2,3,... hops neighbors."""
    assert hops > 1
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index,
                                   num_nodes=data.x.size(0))
    one_hop_set = set([tuple(x) for x in edge_index.transpose(0, 1).tolist()])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col)
    multi_adj = adj
    for _ in range(hops - 1):
        multi_adj = multi_adj @ adj
    row, col, _ = multi_adj.coo()
    multi_edge_index = torch.stack([row, col], dim=0)
    multi_hop_set = set([tuple(x) for x in multi_edge_index.transpose(0, 1).tolist()])
    multi_hop_set = multi_hop_set - one_hop_set
    multi_edge_index = torch.LongTensor(list(multi_hop_set)).transpose(0, 1)
    data.multi_edge_index = multi_edge_index
    return


def compose_log(metrics, key, spaces=0, tabs=0, newline=False):
    string = "\n" if newline else ""
    return string + "\t" * tabs + " " * spaces + "{}: ({:.4f}, {:.4f}, {:.4f})".format(key, metrics["train_{}".format(key)], metrics["val_{}".format(key)], metrics["test_{}".format(key)])


def edge_index_2_csr(edge_index, size):
    """Edge index (PyG COO format) transformed to csr format."""
    csr_matrix = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), to_np_array(edge_index)),
        shape=(size, size))
    return csr_matrix


def process_data_for_nettack(data):
    data.features = sp.csr_matrix(to_np_array(data.x))
    data.adj = edge_index_2_csr(data.edge_index, size=data.x.shape[0])
    data.labels = to_np_array(data.y)
    data.idx_train = np.where(to_np_array(data.train_mask))[0]
    data.idx_val = np.where(to_np_array(data.val_mask))[0]
    data.idx_test = np.where(to_np_array(data.test_mask))[0]
    return data


def to_tuple_list(edge_index):
    """Transform a coo-format edge_index to a list of edge tuples."""
    return [tuple(item) for item in edge_index.T.cpu().numpy()]


def classification_margin(output, true_label):
    '''probs_true_label - probs_best_second_class'''
    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def select_nodes(data, info, data_type, num=10, seed=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    attack_path = osp.join(get_root_dir(), 'experiments/attack_data', data_type)
    if not os.path.exists(attack_path):
        os.makedirs(attack_path)
    filename = os.path.join(attack_path, "test-node_n:{}_seed_{}.pkl".format(num, seed))
    try:
        with open(filename, 'rb') as f:
            node_ids = pickle.load(f)
            print("Load previous attacked node_ids saved in {}.".format(filename))
    except:
        raise
        device = data.x.device
        gcn = GCN(nfeat=info["num_features"],
                  nclass=info["num_classes"],
                  nhid=16,
                  dropout=0.5, device=device).to(device)
        gcn.fit(data.features, data.adj, data.labels, data.idx_train)
        gcn.eval()
        output = gcn.predict()

        margin_dict = {}
        for idx in data.idx_test:
            margin = classification_margin(output[idx], data.labels[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
        high = [x for x, y in sorted_margins[: num]]
        low = [x for x, y in sorted_margins[-num: ]]
        other = [x for x, y in sorted_margins[num: -num]]
        other = np.random.choice(other, num * 2, replace=False).tolist()
        node_ids = other + low + high
        pickle.dump(node_ids, open(filename, "wb"))
        print("Save attacked node_ids to {}:test-node_n:{}.pkl.".format(attack_path, num))
    return node_ids

def to_inductive(data):
    mask = data.train_mask | data.val_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.val_mask = data.val_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def parse_filename(filename, is_adversarial=False, **kwargs):
    """Parse the filename of the experment result file into a dictionary of settings.
    
    Args:
        filename: a string of filename
        is_adversarial: whether the file is from experiments/GIB_node_adversarial_attack.
    """
    if is_adversarial:
        return parse_filename_adversarial(filename, **kwargs)
    else:
        return parse_filename_standard(filename)


def parse_filename_standard(filename):
    """Parse the filename of the standard experment result file into a dictionary of settings."""
    parse_dict = {}
    filename_split = filename.split("_")
    parse_dict["data_type"] = filename_split[0]
    parse_dict["model_type"] = filename_split[1]

    baseline = parse_dict["model_type"] in ['GCNSVD', 'GCNJaccard', 'RGCN']

    parse_dict["train_fraction"] = eval(filename_split[filename_split.index("tr") + 1]) if "tr" in filename_split else 1
    parse_dict["added_edge_fraction"] = eval(filename_split[filename_split.index("ed") + 1]) if "ed" in filename_split else 0
    parse_dict["feature_noise_ratio"] = eval(filename_split[filename_split.index("ed") + 2]) if filename_split[filename_split.index("ed") + 2] != "beta" else 0
    parse_dict["beta1"] = eval(filename_split[filename_split.index("beta") + 1])
    parse_dict["beta2"] = eval(filename_split[filename_split.index("beta") + 2])
    parse_dict["latent_size"] = int(filename_split[filename_split.index("lat") + 1])
    parse_dict["sample_size"] = int(filename_split[filename_split.index("samp") + 1])
    parse_dict["num_layers"] = int(filename_split[filename_split.index("lay") + 1]) if "lay" in filename_split else 2
    parse_dict["is_anneal_beta"] = eval(filename_split[filename_split.index("anl") + 1])
    parse_dict["val_use_mean"] = eval(filename_split[filename_split.index("mean") + 1])
    parse_dict["reparam_all_layers"] = eval(filename_split[filename_split.index("reall") + 1])
    parse_dict["epochs"] = int(filename_split[filename_split.index("epochs") + 1])
    parse_dict["lr"] = eval(filename_split[filename_split.index("lr") + 1]) if "lr" in filename_split else -1
    parse_dict["weight_decay"] = eval(filename_split[filename_split.index("weight_decay") + 1]) if "weight_decay" in filename_split else -1
    parse_dict["seed"] = int(filename_split[filename_split.index("seed") + 1])
    parse_dict["idx"] = filename_split[filename_split.index("id") + 1][:-2]
    if baseline:
        parse_dict["threshold"] = float(filename_split[filename_split.index("threshold") + 1])
        parse_dict["gamma"] = float(filename_split[filename_split.index("gamma") + 1]) if "gamma" in filename_split else None
    else:
        parse_dict["reparam_mode"] = filename_split[filename_split.index("reparam") + 1]
        parse_dict["prior_mode"] = filename_split[filename_split.index("prior") + 1]
        parse_dict["struct_dropout_mode"] = filename_split[filename_split.index("sdrop") + 1] if "sdrop" in filename_split else "(standard,0.6)"
    return parse_dict


def parse_filename_adversarial(filename, baseline=False):
    """Parse the filename of the adversarial experment result file into a dictionary of settings."""
    parse_dict = {}
    filename_split = filename.split("_")
    parse_dict["data_type"] = filename_split[0]
    parse_dict["model_type"] = filename_split[1]
    parse_dict["direct_attack"] = eval(filename_split[filename_split.index("dire") + 1])
    parse_dict["attacked_nodes"] = filename_split[filename_split.index("nodes") + 1]
    parse_dict["n_perturbations"] = eval(filename_split[filename_split.index("pert") + 1])
    parse_dict["latent_size"] = int(filename_split[filename_split.index("lat") + 1])
    parse_dict["sample_size"] = int(filename_split[filename_split.index("samp") + 1])
    parse_dict["num_layers"] = eval(filename_split[filename_split.index("l") + 1]) if "l" in filename_split else 2
    parse_dict["is_anneal_beta"] = eval(filename_split[filename_split.index("anl") + 1])
    parse_dict["val_use_mean"] = eval(filename_split[filename_split.index("mean") + 1])
    parse_dict["reparam_all_layers"] = filename_split[filename_split.index("reall") + 1]
    parse_dict["lr"] = eval(filename_split[filename_split.index("lr") + 1]) if "lr" in filename_split else -1
    parse_dict["weight_decay"] = eval(filename_split[filename_split.index("l2") + 1]) if "l2" in filename_split else -1
    parse_dict["epochs"] = int(filename_split[filename_split.index("epochs") + 1])
    parse_dict["seed"] = int(filename_split[filename_split.index("seed") + 1])
    parse_dict["cuda"] = filename_split[filename_split.index("id") - 1]
    parse_dict["idx"] = filename_split[filename_split.index("id") + 1][:-2]
    if not baseline:
        parse_dict["beta1"] = eval(filename_split[filename_split.index("beta") + 1])
        parse_dict["beta2"] = eval(filename_split[filename_split.index("beta") + 2])
        parse_dict["reparam_mode"] = filename_split[filename_split.index("reparam") + 1]
        parse_dict["prior_mode"] = filename_split[filename_split.index("prior") + 1]
        parse_dict["struct_dropout_mode"] = filename_split[filename_split.index("sdrop") + 1]
    else:
        parse_dict["gamma"] = eval(filename_split[filename_split.index("gamma") + 1])
        parse_dict["threshold"] = eval(filename_split[filename_split.index("threshold") + 1][:-2])
    return parse_dict