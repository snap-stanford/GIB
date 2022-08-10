import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname("__file__"), ".."))
from copy import deepcopy
import torch
from torch_geometric.datasets import Planetoid, MoleculeNet
import pickle
import torch_geometric.transforms as T


def get_data(
    dataset_name,
    lcc=False,
    added_edge_fraction=0,
    feature_noise_ratio=0,
):
    """Get the pytorch-geometric data object.

    Args:
        dataset_name: Dataset name. Choose from "Cora", "Pubmed", "citeseer".
                       If want the feature to be binarized, include "-bool" in data_type string.
                       If use random splitting with train:val:test=0.1:0.1:0.8 include "-rand" in the data_type string.
        train_fraction: Fraction of training labels preserved for the training set.
        lcc: Largest Connected Component. Only work with the LCC,
        added_edge_fraction: Fraction of added (or deleted) random edges. Use positive (negative) number for randomly adding (deleting) edges.
        feature_noise_ratio: Noise ratio for the additive independent Gaussian noise on the features.

    Returns:
        A pytorch-geometric data object containing the specified dataset.
    """

    def to_mask(idx, size):
        mask = torch.zeros(size).bool()
        mask[idx] = True
        return mask

    data_path = os.path.join(
        os.path.dirname(os.path.realpath("__file__")), "..", "data"
    )

    # Load data:
    info = {}
    if dataset_name in ["Cora", "Pubmed", "citeseer"]:
        ds = Planetoid(
            root=data_path, name=dataset_name, transform=T.NormalizeFeatures()
        )

        ds.loss = "softmax"
    else:
        raise Exception("dataset_name {} is not valid!".format(dataset_name))

    # work with lcc only
    if lcc:
        for idx, data in enumerate(ds):
            ds[idx] = T.LargestConnectedComponents()(data)

    # Add random edges for non-targeted attacks:
    if added_edge_fraction > 0:
        for idx, data in enumerate(ds):
            ds[idx] = add_random_edge(data, added_edge_fraction=added_edge_fraction)
    elif added_edge_fraction < 0:
        for idx, data in enumerate(ds):
            ds[idx] = remove_edge_random(
                data, remove_edge_fraction=-added_edge_fraction
            )

    # Perturb features for non-targeted attacks:
    if feature_noise_ratio > 0:
        for idx, data in enumerate(ds):
            x_max_mean = data.x.max(1)[0].mean()
            data.x = (
                data.x + torch.randn(data.x.shape) * x_max_mean * feature_noise_ratio
            )
            ds[idx] = data

    return ds


def get_list_elements(List, string_idx):
    """Select elements of the list based on string_idx.

    Format of string_idx:
        if starting with "r", means first performs random permutation.
        "100:200": the 100th to 199th elements
        "100:" : the 100th elements and onward
        ":200" : the 0th to 199th elements
        "150" : the 150th element
        "::" : all elements
    """
    # Permute if starting with "r":
    if string_idx.startswith("r"):
        List = np.random.permutation(List).tolist()
        string_idx = string_idx[1:]
    # Select indices:
    if string_idx == "::":
        return List
    elif ":" in string_idx:
        string_split = string_idx.split(":")
        string_split = [string for string in string_split if len(string) != 0]
        if len(string_split) == 2:
            start_idx, end_idx = string_idx.split(":")
            start_idx, end_idx = eval(start_idx), eval(end_idx)
            if end_idx > len(List):
                raise Exception("The end index exceeds the length of the list!")
            list_selected = List[start_idx:end_idx]
        elif len(string_split) == 1:
            if string_idx.startswith(":"):
                list_selected = List[: eval(string_idx[1:])]
            else:
                list_selected = List[eval(string_idx[:-1]) :]
        else:
            raise
    else:
        string_idx = eval(string_idx)
        list_selected = [List[string_idx]]
    return list_selected


def remove_edge_random(data, remove_edge_fraction):
    """Randomly remove a certain fraction of edges."""
    data_c = deepcopy(data)
    num_edges = int(data_c.edge_index.shape[1] / 2)
    num_removed_edges = int(num_edges * remove_edge_fraction)
    edges = [tuple(ele) for ele in np.array(data_c.edge_index.T)]
    for i in range(num_removed_edges):
        idx = np.random.choice(len(edges))
        edge = edges[idx]
        edge_r = (edge[1], edge[0])
        edges.pop(idx)
        try:
            edges.remove(edge_r)
        except:
            pass
    data_c.edge_index = torch.LongTensor(np.array(edges).T).to(data.edge_index.device)
    return data_c


def add_random_edge(data, added_edge_fraction=0):
    """Add random edges to the original data's edge_index."""
    if added_edge_fraction == 0:
        return data
    data_c = deepcopy(data)
    num_edges = int(data.edge_index.shape[1] / 2)
    num_added_edges = int(num_edges * added_edge_fraction)
    edges = [tuple(ele) for ele in np.array(data.edge_index.T)]
    added_edges = []
    for i in range(num_added_edges):
        while True:
            added_edge_cand = tuple(
                np.random.choice(data.x.shape[0], size=2, replace=False)
            )
            added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
            if added_edge_cand in edges or added_edge_cand in added_edges:
                if added_edge_cand in edges:
                    assert added_edge_r_cand in edges
                if added_edge_cand in added_edges:
                    assert added_edge_r_cand in added_edges
                continue
            else:
                added_edges.append(added_edge_cand)
                added_edges.append(added_edge_r_cand)
                break

    added_edge_index = torch.LongTensor(np.array(added_edges).T).to(
        data.edge_index.device
    )
    data_c.edge_index = torch.cat([data.edge_index, added_edge_index], 1)
    return data_c


def get_edge_corrupted_data(data, corrupt_fraction, is_original_included=True):
    """Add random edges to the original data's edge_index.

    Args:
        data: PyG data instance
        corrupt_fraction: fraction of edges being removed and then the corresponding random edge added.
        is_original_included: if True, the original edges may be included in the random edges.

    Returns:
        data_edge_corrupted: new data instance where the edge is replaced by random edges.
    """
    data_edge_corrupted = deepcopy(data)
    num_edges = int(data.edge_index.shape[1] / 2)
    num_corrupted_edges = int(num_edges * corrupt_fraction)
    edges = [tuple(item) for item in np.array(data.edge_index.T)]
    removed_edges = []
    num_nodes = data.x.shape[0]

    # Remove edges:
    for i in range(num_corrupted_edges):
        id = np.random.choice(range(len(edges)))
        edge = edges.pop(id)
        try:
            edge_r = edges.remove((edge[1], edge[0]))
        except:
            pass
        removed_edges.append(edge)
        removed_edges.append((edge[1], edge[0]))

    # Setting up excluded edges when adding:
    remaining_edges = list(set(edges).difference(set(removed_edges)))
    if is_original_included:
        edges_exclude = remaining_edges
    else:
        edges_exclude = edges

    # Add edges:
    added_edges = []
    for i in range(num_corrupted_edges):
        while True:
            added_edge_cand = tuple(np.random.choice(num_nodes, size=2, replace=False))
            added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
            if added_edge_cand in edges_exclude or added_edge_cand in added_edges:
                continue
            else:
                added_edges.append(added_edge_cand)
                added_edges.append(added_edge_r_cand)
                break

    added_edge_index = torch.LongTensor(np.array(added_edges + remaining_edges).T).to(
        data.edge_index.device
    )
    data_edge_corrupted.edge_index = added_edge_index
