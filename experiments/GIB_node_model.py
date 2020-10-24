#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict
from copy import deepcopy
import itertools
import matplotlib.pylab as plt
import numpy as np
import os.path as osp
import pickle
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import sklearn
from sklearn.manifold import TSNE
import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch.distributions.normal import Normal

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, softmax, degree, to_undirected
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from GIB.pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
from GIB.pytorch_net.util import sample, to_cpu_recur, to_np_array, to_Variable, record_data, make_dir, remove_duplicates, update_dict, get_list_elements, to_string, filter_filename
from GIB.util import get_reparam_num_neurons, sample_lognormal, scatter_sample, uniform_prior, compose_log, edge_index_2_csr, COLOR_LIST, LINESTYLE_LIST, process_data_for_nettack, parse_filename, add_distant_neighbors
from GIB.DeepRobust.deeprobust.graph.targeted_attack import Nettack


# ## GCNConv:

# In[2]:


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True,
                 reparam_mode=None, prior_mode=None, sample_size=1, val_use_mean=True,
                 **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.reparam_mode = None if reparam_mode == "None" else reparam_mode
        self.prior_mode = prior_mode
        self.val_use_mean = val_use_mean
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, self.out_neurons))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)

        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    

    def set_cache(self, cached):
        self.cached = cached


    def to_device(self, device):
        self.to(device)
        if self.cached and self.cached_result is not None:
            edge_index, norm = self.cached_result
            self.cached_result = edge_index.to(device), norm.to(device)
        return self


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        out = self.propagate(edge_index, x=x, norm=norm)

        if self.reparam_mode is not None:
            # Reparameterize:
            self.dist, _ = reparameterize(model=None, input=out, 
                                          mode=self.reparam_mode, 
                                          size=self.out_channels
                                         )  # [B, Z]
            Z = sample(self.dist, self.sample_size)  # [S, B, Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(x.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(x.size(0), self.out_channels).to(x.device),
                                           )  # [B, Z]

            # Calculate prior loss:
            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1)
            else:
                Z_logit = self.dist.log_prob(Z).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z)  # [S, B]
                prior_logit = self.feature_prior.log_prob(Z).sum(-1)  # [S, B]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)  # [B, Z]
            else:
                out = out[:, :self.out_channels]  # [B, Z]
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)  # [B]

        structure_kl_loss = torch.zeros([]).to(x.device)
        return out, ixz, structure_kl_loss
            

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# ## GATConv:

# In[3]:


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        struct_dropout_mode (tuple, optional): Choose from: None, ("standard", prob), ("info", ${MODE}),
            where ${MODE} chooses from "subset", "lognormal", "loguniform".
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, reparam_mode=None, prior_mode=None,
                 struct_dropout_mode=None, sample_size=1,
                 val_use_mean=True,
                 bias=True,
                 **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * self.out_neurons))
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_neurons))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)
            
        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.skip_editing_edge_index = struct_dropout_mode[0] == 'DNsampling'
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x) and not self.skip_editing_edge_index:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x)

        if self.reparam_mode is not None:
            # Reparameterize:
            out = out.view(-1, self.out_neurons)
            self.dist, _ = reparameterize(model=None, input=out,
                                          mode=self.reparam_mode,
                                          size=self.out_channels,
                                         )  # dist: [B * head, Z]
            Z_core = sample(self.dist, self.sample_size)  # [S, B * head, Z]
            Z = Z_core.view(self.sample_size, -1, self.heads * self.out_channels)  # [S, B, head * Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(out.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(out.size(0), self.out_channels).to(x.device),
                                           )  # feature_prior: [B * head, Z]

            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1).view(-1, self.heads).mean(-1)
            else:
                Z_logit = self.dist.log_prob(Z_core).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z_core)  # [S, B * head]
                prior_logit = self.feature_prior.log_prob(Z_core).sum(-1)  # [S, B * head]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0).view(-1, self.heads).mean(-1)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)
            else:
                out = out[:, :self.out_channels].contiguous().view(-1, self.heads * self.out_channels)
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)

        if "Nsampling" in self.struct_dropout_mode[0]:
            if 'categorical' in self.struct_dropout_mode[1]:
                structure_kl_loss = torch.sum(self.alpha*torch.log((self.alpha+1e-16)/self.prior))
            elif 'Bernoulli' in self.struct_dropout_mode[1]:
                posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
                prior = torch.distributions.bernoulli.Bernoulli(self.prior) 
                structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            else:
                raise Exception("I think this belongs to the diff subset sampling that is not implemented")
        else:
            structure_kl_loss = torch.zeros([]).to(x.device)

        return out, ixz, structure_kl_loss

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_neurons)  # [N_edge, heads, out_channels]
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_neurons:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_neurons)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [N_edge, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Sample attention coefficients stochastically.
        if self.struct_dropout_mode[0] == "None":
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        else:
            if self.struct_dropout_mode[0] == "standard":
                alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
                prob_dropout = self.struct_dropout_mode[1]
                alpha = F.dropout(alpha, p=prob_dropout, training=self.training)
            elif self.struct_dropout_mode[0] == "identity":
                alpha = torch.ones_like(alpha)
                alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            elif self.struct_dropout_mode[0] == "info":
                mode = self.struct_dropout_mode[1]
                if mode == "lognormal":
                    max_alpha = self.struct_dropout_mode[2] if len(self.struct_dropout_mode) > 2 else 0.7
                    alpha = 0.001 + max_alpha * alpha
                    self.kl = -torch.log(alpha/(max_alpha + 0.001))
                    sigma0 = 1. if self.training else 0.
                    alpha = sample_lognormal(mean=torch.zeros_like(alpha), sigma=alpha, sigma0=sigma0)
                else:
                    raise Exception("Mode {} for the InfoDropout is invalid!".format(mode))
            elif "Nsampling" in self.struct_dropout_mode[0]:
                neighbor_sampling_mode = self.struct_dropout_mode[1]
                if 'categorical' in neighbor_sampling_mode:
                    alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
                    self.alpha = alpha
                    self.prior = uniform_prior(edge_index_i)
                    if self.val_use_mean is False or self.training:
                        temperature = self.struct_dropout_mode[2]
                        sample_neighbor_size = self.struct_dropout_mode[3]
                        if neighbor_sampling_mode == 'categorical':
                            alpha = scatter_sample(alpha, edge_index_i, temperature, size_i)
                        elif 'multi-categorical' in neighbor_sampling_mode:
                            alphas = []
                            for _ in range(sample_neighbor_size): #! this can be improved by parallel sampling
                                alphas.append(scatter_sample(alpha, edge_index_i, temperature, size_i))
                            alphas = torch.stack(alphas, dim=0)
                            if 'sum' in neighbor_sampling_mode:
                                alpha = alphas.sum(dim=0)
                            elif 'max' in neighbor_sampling_mode:
                                alpha, _ = torch.max(alphas, dim=0)
                            else:
                                raise
                        else:
                            raise
                elif neighbor_sampling_mode == 'Bernoulli':
                    if self.struct_dropout_mode[4] == 'norm':
                        alpha_normalization = torch.ones_like(alpha)
                        alpha_normalization = softmax(alpha_normalization, edge_index_i, num_nodes=size_i)
                    alpha = torch.clamp(torch.sigmoid(alpha), 0.01, 0.99)
                    self.alpha = alpha
                    self.prior = (torch.ones_like(self.alpha)*self.struct_dropout_mode[3]).to(alpha.device)
                    if not self.val_use_mean or self.training:
                        temperature = self.struct_dropout_mode[2]
                        alpha = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.Tensor([temperature]).to(alpha.device),
                            probs=alpha).rsample()
                    if self.struct_dropout_mode[4] == 'norm':
                        alpha = alpha*alpha_normalization
                else:
                    raise
            else:
                raise

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_neurons)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


    def to_device(self, device):
        self.to(device)
        return self

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# ## GNN:

# In[4]:


class GNN(torch.nn.Module):
    def __init__(
        self,
        model_type,
        num_features,
        num_classes,
        reparam_mode,
        prior_mode,
        latent_size,
        sample_size=1,
        num_layers=2,
        struct_dropout_mode=("standard", 0.6),
        dropout=True,
        with_relu=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
        is_cuda=False,
    ):
        """Class implementing a general GNN, which can realize GAT, GIB-GAT, GCN.
        
        Args:
            model_type:   name of the base model. Choose from "GAT", "GCN".
            num_features: number of features of the data.x.
            num_classes:  number of classes for data.y.
            reparam_mode: reparameterization mode for XIB. Choose from "diag" and "full". Default "diag" that parameterizes the mean and diagonal element of the Gaussian
            prior_mode:   distribution type for the prior. Choose from "Gaussian" or "mixGau-{Number}", where {Number} is the number of components for mixture of Gaussian.
            latent_size:  latent size for each layer of GNN. If model_type="GAT", the true latent size is int(latent_size/2)
            sample_size=1:how many Z to sample for each feature X.
            num_layers=2: number of layers for the GNN
            struct_dropout_mode: Mode for how the structural representation is generated. Only effective for model_type=="GAT"
                          Choose from ("Nsampling", 'multi-categorical-sum', 0.1, 3) (here 0.1 is temperature, k=3 is the number of sampled edges with replacement), 
                          ("DNsampling", 'multi-categorical-sum', 0.1, 3, 2) (similar as above, with the local dependence range T=2) 
                          ("standard", 0.6) (standard dropout used on the attention weights in GAT)
            dropout:      whether to use dropout on features.
            with_relu:    whether to use nonlinearity for GCN.
            val_use_mean: Whether during evaluation use the parameter value instead of sampling. If True, during evaluation,
                          XIB will use mean for prediction, and AIB will use the parameter of the categorical distribution for prediction.
            reparam_all_layers: Which layers to use XIB, e.g. (1,2,4). Default (-2,), meaning the second last layer. If True, use XIB for all layers.
            normalize:    whether to normalize for GCN (only effective for GCN)
            is_cuda:      whether to use CUDA, and if so, which GPU to use. Choose from False, True, "CUDA:{GPU_ID}", where {GPU_ID} is the ID for the CUDA.
        """
        super(GNN, self).__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.normalize = normalize
        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.struct_dropout_mode = struct_dropout_mode
        self.dropout = dropout
        self.latent_size = latent_size
        self.sample_size = sample_size
        self.num_layers = num_layers
        self.with_relu = with_relu
        self.val_use_mean = val_use_mean
        self.reparam_all_layers = reparam_all_layers
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")

        self.init()


    def init(self):
        """Initialize the layers for the GNN."""
        self.reparam_layers = []
        if self.model_type == "GCN":
            for i in range(self.num_layers):
                if self.reparam_all_layers is True:
                    is_reparam = True
                elif isinstance(self.reparam_all_layers, tuple):
                    reparam_all_layers = tuple([kk + self.num_layers if kk < 0 else kk for kk in self.reparam_all_layers])
                    is_reparam = i in reparam_all_layers
                else:
                    raise
                if is_reparam:
                    self.reparam_layers.append(i)
                setattr(self, "conv{}".format(i + 1),
                        GCNConv(self.num_features if i == 0 else self.latent_size,
                                self.latent_size if i != self.num_layers - 1 else self.num_classes,
                                cached=True,
                                reparam_mode=self.reparam_mode if is_reparam else None,
                                prior_mode=self.prior_mode if is_reparam else None,
                                sample_size=self.sample_size,
                                bias=True if self.with_relu else False,
                                val_use_mean=self.val_use_mean,
                                normalize=self.normalize,
                ))
            # self.conv1 = ChebConv(self.num_features, 16, K=2)
            # self.conv2 = ChebConv(16, self.num_features, K=2)

        elif self.model_type == "GAT":
            latent_size = int(self.latent_size / 2)  # Under the default setting, latent_size = 8
            for i in range(self.num_layers):
                if i == 0:
                    input_size = self.num_features
                else:
                    if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                        input_size = latent_size * 8 * 2
                    else:
                        input_size = latent_size * 8
                if self.reparam_all_layers is True:
                    is_reparam = True
                elif isinstance(self.reparam_all_layers, tuple):
                    reparam_all_layers = tuple([kk + self.num_layers if kk < 0 else kk for kk in self.reparam_all_layers])
                    is_reparam = i in reparam_all_layers
                else:
                    raise
                if is_reparam:
                    self.reparam_layers.append(i)
                setattr(self, "conv{}".format(i + 1), GATConv(
                    input_size,
                    latent_size if i != self.num_layers - 1 else self.num_classes,
                    heads=8 if i != self.num_layers - 1 else 1, concat=True,
                    reparam_mode=self.reparam_mode if is_reparam else None,
                    prior_mode=self.prior_mode if is_reparam else None,
                    val_use_mean=self.val_use_mean,
                    struct_dropout_mode=self.struct_dropout_mode,
                    sample_size=self.sample_size,
                ))
                if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                    setattr(self, "conv{}_1".format(i + 1), GATConv(
                        input_size,
                        latent_size if i != self.num_layers - 1 else self.num_classes,
                        heads=8 if i != self.num_layers - 1 else 1, concat=True,
                        reparam_mode=self.reparam_mode if is_reparam else None,
                        prior_mode=self.prior_mode if is_reparam  else None,
                        val_use_mean=self.val_use_mean,
                        struct_dropout_mode=self.struct_dropout_mode,
                        sample_size=self.sample_size,
                    ))
            # On the Pubmed dataset, use heads=8 in conv2.
        
        else:
            raise Exception("Model_type {} is not valid!".format(self.model_type))

        self.reparam_layers = sorted(self.reparam_layers)
   
        if self.model_type == "GCN":
            if self.with_relu:
                reg_params = [getattr(self, "conv{}".format(i+1)).parameters() for i in range(self.num_layers - 1)]
                self.reg_params = itertools.chain(*reg_params)
                self.non_reg_params = getattr(self, "conv{}".format(self.num_layers)).parameters()
            else:
                self.reg_params = OrderedDict()
                self.non_reg_params = self.parameters()
        else:
            self.reg_params = self.parameters()
            self.non_reg_params = OrderedDict()
        self.to(self.device)


    def set_cache(self, cached):
        """Set cache for GCN."""
        for i in range(self.num_layers):
            if hasattr(getattr(self, "conv{}".format(i+1)), "set_cache"):
                getattr(self, "conv{}".format(i+1)).set_cache(cached)


    def to_device(self, device):
        """Send all the layers to the specified device."""
        for i in range(self.num_layers):
            getattr(self, "conv{}".format(i+1)).to_device(device)
        self.to(device)
        return self


    def forward(self, data, record_Z=False, isplot=False):
        """Main forward function.
        
        Args:
            data: the pytorch-geometric data class.
            record_Z: whether to record the standard deviation for the representation Z.
            isplot:   whether to plot.
        
        Returns:
            x: output
            reg_info: other information or metrics.
        """
        reg_info = {}
        if self.model_type == "GCN":
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            for i in range(self.num_layers - 1):
                layer = getattr(self, "conv{}".format(i + 1))
                x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
                # Record:
                record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
                if layer.reparam_mode is not None:
                    record_data(reg_info, [layer.Z_std], ["Z_std"])
                if record_Z:
                    record_data(reg_info, [to_np_array(x)], ["Z_{}".format(i)], nolist=True)
                if self.with_relu:
                    x = F.relu(x)
                    self.plot(x, data.y, titles="Layer{}".format(i + 1), isplot=isplot)
                    if self.dropout is True:
                        x = F.dropout(x, training=self.training)
            layer = getattr(self, "conv{}".format(self.num_layers))
            x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
            # Record:
            record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
            if layer.reparam_mode is not None:
                record_data(reg_info, [layer.Z_std], ["Z_std"])
            if record_Z:
                record_data(reg_info, [to_np_array(x)], ["Z_{}".format(self.num_layers - 1)], nolist=True)
            self.plot(x, data.y, titles="Layer{}".format(self.num_layers), isplot=isplot)

        elif self.model_type == "GAT":
            x = F.dropout(data.x, p=0.6, training=self.training)

            for i in range(self.num_layers - 1):
                if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                    x_1, ixz_1, structure_kl_loss_1 = getattr(self, "conv{}_1".format(i + 1))(x, data.multi_edge_index)
                layer = getattr(self, "conv{}".format(i + 1))
                x, ixz, structure_kl_loss = layer(x, data.edge_index)
                # Record:
                record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
                if layer.reparam_mode is not None:
                    record_data(reg_info, [layer.Z_std], ["Z_std"])
                if record_Z:
                    record_data(reg_info, [to_np_array(x)], ["Z_{}".format(i)], nolist=True)
                # Multi-hop:
                if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                    x = torch.cat([x, x_1], dim=-1)
                    record_data(reg_info, [ixz_1, structure_kl_loss_1], ["ixz_DN_list", "structure_kl_DN_list"])
                x = F.elu(x)
                self.plot(x, data.y, titles="Layer{}".format(i + 1), isplot=isplot)
                x = F.dropout(x, p=0.6, training=self.training)

            if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                x_1, ixz_1, structure_kl_loss_1 = getattr(self, "conv{}_1".format(self.num_layers))(x, data.multi_edge_index)
            layer = getattr(self, "conv{}".format(self.num_layers))
            x, ixz, structure_kl_loss = layer(x, data.edge_index)
            # Record:
            record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
            if layer.reparam_mode is not None:
                record_data(reg_info, [layer.Z_std], ["Z_std"])
            if record_Z:
                record_data(reg_info, [to_np_array(x)], ["Z_{}".format(self.num_layers - 1)], nolist=True)
            # Multi-hop:
            if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                x = x + x_1
                record_data(reg_info, [ixz_1, structure_kl_loss_1], ["ixz_DN_list", "structure_kl_DN_list"])
            self.plot(x, data.y, titles="Layer{}".format(self.num_layers), isplot=isplot)

        return x, reg_info


    def compute_metrics_fun(self, data, metrics, mask=None, mask_id=None):
        """Compute metrics for measuring clustering performance.
        Choices: "Silu", "CH", "DB".
        """
        _, info_dict = self(data, record_Z=True)
        y = to_np_array(data.y)
        info_metrics = {}
        if mask is not None:
            mask = to_np_array(mask)
            mask_id += "_"
        else:
            mask_id = ""
        for k in range(self.num_layers):
            if mask is not None:
                Z_i = info_dict["Z_{}".format(k)][mask]
                y_i = y[mask]
            else:
                Z_i = info_dict["Z_{}".format(k)]
                y_i = y
            for metric in metrics:
                if metric == "Silu":
                    score = sklearn.metrics.silhouette_score(Z_i, y_i, metric='euclidean')
                elif metric == "DB":
                    score = sklearn.metrics.davies_bouldin_score(Z_i, y_i)
                elif metric == "CH":
                    score = sklearn.metrics.calinski_harabasz_score(Z_i, y_i)
                info_metrics["{}{}_{}".format(mask_id, metric, k)] = score
        return info_metrics


    def plot(self, Z_list, y, titles=None, isplot=False):
        """Plot the intermediate representation Z."""
        if isplot:
            if not isinstance(Z_list, list):
                Z_list = [Z_list]
            if titles is not None and not isinstance(titles, list):
                titles = [titles]
            length = len(Z_list)
            tsne = TSNE(n_components=2, init='pca')
            plt.figure(figsize=(8 * length, 6))
            for k, Z in enumerate(Z_list):
                plt.subplot(1, length, k+1)
                for i in range(self.num_classes):
                    idx = y == i
                    Z_embed_i = tsne.fit_transform(to_np_array(Z[idx]))
                    plt.plot(Z_embed_i[:, 0], Z_embed_i[:, 1], ".", c=COLOR_LIST[i], label=str(i))
                if titles is not None:
                    plt.title(titles[k], fontsize=18)
            plt.legend(bbox_to_anchor=[1,1])
            plt.show()


    @property
    def model_dict(self):
        """Record model_dict for saving."""
        model_dict = {}
        model_dict["model_type"] = self.model_type
        model_dict["num_features"] = self.num_features
        model_dict["num_classes"] = self.num_classes
        model_dict["normalize"] = self.normalize
        model_dict["reparam_mode"] = self.reparam_mode
        model_dict["prior_mode"] = self.prior_mode
        model_dict["struct_dropout_mode"] = self.struct_dropout_mode
        model_dict["dropout"] = self.dropout
        model_dict["latent_size"] = self.latent_size
        model_dict["sample_size"] = self.sample_size
        model_dict["num_layers"] = self.num_layers
        model_dict["with_relu"] = self.with_relu
        model_dict["val_use_mean"] = self.val_use_mean
        model_dict["reparam_all_layers"] = self.reparam_all_layers
        model_dict["state_dict"] = to_cpu_recur(self.state_dict())
        return model_dict


def load_model_dict_GNN(model_dict, is_cuda=False):
    """Load the GNN model."""
    model = GNN(
        model_type=model_dict["model_type"],
        num_features=model_dict["num_features"],
        num_classes=model_dict["num_classes"],
        normalize=model_dict["normalize"],
        reparam_mode=model_dict["reparam_mode"],
        prior_mode=model_dict["prior_mode"],
        struct_dropout_mode=model_dict["struct_dropout_mode"],
        dropout=model_dict["dropout"],
        latent_size=model_dict["latent_size"],
        sample_size=model_dict["sample_size"],
        num_layers=model_dict["num_layers"],
        with_relu=model_dict["with_relu"],
        val_use_mean=model_dict["val_use_mean"],
        reparam_all_layers=model_dict["reparam_all_layers"],
        is_cuda=is_cuda,
    )
    if "state_dict" in model_dict:
        model.load_state_dict(model_dict["state_dict"])
    return model


# ## Training and testing:

# In[5]:


# Train and test functions:
def train_model(model, data, optimizer, loss_type, beta1=None, beta2=None):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()
    logits, reg_info = model(data)
    if loss_type == 'sigmoid':
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logits[data.train_mask], data.y[data.train_mask])
    elif loss_type == 'softmax':
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits[data.train_mask], data.y[data.train_mask])
    else:
        raise
    # Add IB loss:
    if beta1 is not None and beta1 != 0:
        ixz = torch.stack(reg_info["ixz_list"], 1).mean(0).sum()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            ixz = ixz + torch.stack(reg_info["ixz_DN_list"], 1).mean(0).sum()
        loss = loss + ixz * beta1
    if beta2 is not None and beta2 != 0:
        structure_kl_loss = torch.stack(reg_info["structure_kl_list"]).mean()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            structure_kl_loss = structure_kl_loss + torch.stack(reg_info["structure_kl_DN_list"]).mean()
        loss = loss + structure_kl_loss * beta2
    loss.backward()
    optimizer.step()


def get_test_metrics(model, data, loss_type, beta1=None, beta2=None, compute_metrics=None, isplot=False):
    """Obtain the metrics of the data evaluated by the model."""
    logits, info_dict = model(data, isplot=isplot)
    metrics = OrderedDict()

    # Record structure kl:
    structure_kl_list = [value.item() for value in info_dict["structure_kl_list"]]
    record_data(metrics, structure_kl_list, ["structure_kl_{}".format(i+1) for i in range(model.num_layers)], nolist=True)
    record_data(metrics, [np.sum(structure_kl_list)], ["structure_kl"], nolist=True)
    if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
        structure_kl_DN_list = [value.item() for value in info_dict["structure_kl_DN_list"]]
        record_data(metrics, structure_kl_DN_list, 
                    ["structure_kl_DN_{}".format(i+1) for i in range(model.num_layers)], nolist=True)
        record_data(metrics, [np.sum(structure_kl_DN_list)], ["structure_kl_DN"], nolist=True)
    if compute_metrics is not None:
        info_metrics = model.compute_metrics_fun(data, compute_metrics)
        record_data(metrics, list(info_metrics.values()), list(info_metrics.keys()), nolist=True)
    for mask_id, mask in data('train_mask', 'val_mask', 'test_mask'):
        mask_id = mask_id.split("_")[0]
        # Record Ixz:
        ixz_list = to_np_array(torch.stack(info_dict["ixz_list"], 1)[mask].mean(0)).tolist()
        record_data(metrics, ixz_list, ["{}_ixz{}".format(mask_id, i+1) for i in range(model.num_layers)], nolist=True)
        record_data(metrics, [np.sum(ixz_list)], ["{}_ixz".format(mask_id)], nolist=True)
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            ixz_DN_list = to_np_array(torch.stack(info_dict["ixz_DN_list"], 1)[mask].mean(0)).tolist()
            record_data(metrics, ixz_DN_list, ["{}_ixz{}_DN".format(mask_id, i+1) for i in range(model.num_layers)], nolist=True)
            record_data(metrics, [np.sum(ixz_DN_list)], ["{}_ixz_DN".format(mask_id)], nolist=True)
        if "Z_std" in info_dict and "Z_std" not in metrics:
            record_data(metrics, info_dict["Z_std"], ["Z_std_{}".format(kk) for kk in model.reparam_layers])
        # Record contrastive loss:
        if compute_metrics is not None:
            info_metrics = model.compute_metrics_fun(data, compute_metrics, mask=mask, mask_id=mask_id)
            record_data(metrics, list(info_metrics.values()), list(info_metrics.keys()), nolist=True)
        if loss_type == 'softmax':
            pred = logits[mask].max(1)[1]
            loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits[mask], data.y[mask]).item()
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            f1_micro = sklearn.metrics.f1_score(data.y[mask].tolist(), pred.tolist(), average='micro')
            f1_macro = sklearn.metrics.f1_score(data.y[mask].tolist(), pred.tolist(), average='macro')
        elif loss_type == 'sigmoid':
            pred = torch.sigmoid(logits[mask])
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
            loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logits[mask], data.y[mask]).item()
            acc = 0
            f1_micro = sklearn.metrics.f1_score(data.y[mask].tolist(), pred.tolist(), average='micro')
            f1_macro = sklearn.metrics.f1_score(data.y[mask].tolist(), pred.tolist(), average='macro')

        record_data(metrics, [acc, loss, f1_micro, f1_macro],
                    ["{}_acc".format(mask_id), "{}_loss".format(mask_id), "{}_f1_micro".format(mask_id), "{}_f1_macro".format(mask_id)], nolist=True)
    return metrics


@torch.no_grad()
def test_model(
    model,
    data,
    loss_type,
    beta1=None,
    beta2=None,
    compute_metrics=None,
    isplot=False,
):
    model.eval()
    metrics_all = get_test_metrics(model, data, loss_type=loss_type, beta1=beta1, beta2=beta2,
                                   compute_metrics=compute_metrics, isplot=isplot,
                                  )
    return metrics_all


def train_baseline(model, model_type, data, device, threshold, filename, epochs, save_best_model=False, verbose=True):
    """Train the baseline model for the specified number of epochs."""
    data_record = {}
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if model_type == 'GCNJaccard':
        model = model.to(device)
        model.fit(features, adj, labels, idx_train, idx_val, train_iters=epochs, threshold=threshold, verbose=verbose)
    elif model_type == 'RGCN':
        model = model.to(device)
        model.fit(features, adj, labels, idx_train, idx_val, train_iters=epochs, verbose=verbose)
    model.eval()

    output = model.predict()
    output_train = output[idx_train].max(1)[1]
    output_val = output[idx_val].max(1)[1]
    output_test = output[idx_test].max(1)[1]
    train_f1_micro = sklearn.metrics.f1_score(data.y[idx_train].tolist(), output_train.tolist(), average='micro')
    train_f1_macro = sklearn.metrics.f1_score(data.y[idx_train].tolist(), output_train.tolist(), average='macro')
    val_f1_micro   = sklearn.metrics.f1_score(data.y[idx_val].tolist(), output_val.tolist(), average='micro')
    val_f1_macro   = sklearn.metrics.f1_score(data.y[idx_val].tolist(), output_val.tolist(), average='macro')
    test_f1_micro  = sklearn.metrics.f1_score(data.y[idx_test].tolist(), output_test.tolist(), average='micro')
    test_f1_macro  = sklearn.metrics.f1_score(data.y[idx_test].tolist(), output_test.tolist(), average='macro')
    record_data(data_record, [train_f1_micro, train_f1_macro, val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro, test_f1_micro, test_f1_macro, epochs],
                ['train_f1_micro', 'train_f1_macro', 'val_f1_micro', 'val_f1_macro', 'test_f1_micro', 'test_f1_macro', 'b_test_f1_micro', 'b_test_f1_macro', 'epochs'])
    if save_best_model:
        data_record["best_model_dict"] = model.state_dict()
    if filename is not None:
        make_dir(filename)
        pickle.dump(data_record, open(filename + ".p", "wb"))

    return data_record


def train(
    model,
    data,
    data_type,
    model_type,
    loss_type,
    beta1_list,
    beta2_list,
    epochs,
    verbose=True,
    inspect_interval=10,
    isplot=True,
    filename=None,
    compute_metrics=None, # "Silu", "DB", "CH"
    lr=None,
    weight_decay=None,
    save_best_model=False,
):
    """Training multiple epochs."""
    if lr is None:
        if model_type == "GCN":
            lr = 0.01
        elif model_type == "GAT":
            lr = 0.01 if data_type.startswith("Pubmed") else 0.005
        else:
            lr = 0.01
    if weight_decay is None:
        if model_type == "GCN":
            weight_decay = 5e-4
        elif model_type == "GAT":
            weight_decay = 1e-3 if data_type.startswith("Pubmed") else 5e-4
        else:
            weight_decay = 5e-4 

    # Training:
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=lr)
    best_val_f1_micro = 0
    b_test_f1_micro = 0
    data_record = {"num_layers": model.num_layers}

    # Train:
    for epoch in range(1, epochs + 1):
        beta1 = beta1_list[epoch] if beta1_list is not None else None
        beta2 = beta2_list[epoch] if beta2_list is not None else None
        train_model(
            model,
            data,
            optimizer,
            loss_type,
            beta1=beta1,
            beta2=beta2,
        )
        metrics = test_model(model, data, loss_type, beta1=beta1, beta2=beta2,
                             compute_metrics=compute_metrics if epoch % inspect_interval == 0 else None,
                             isplot=isplot >= 2 if epoch % inspect_interval == 0 else False,
                            )
        if metrics["val_f1_micro"] > best_val_f1_micro:
            best_val_f1_micro = metrics["val_f1_micro"]
            b_test_f1_micro = metrics["test_f1_micro"]
            data_record["best_epoch"] = epoch
            if save_best_model:
                data_record["best_model_dict"] = deepcopy(model.model_dict)
        record_data(data_record, [epoch, best_val_f1_micro, b_test_f1_micro], ["epoch", "best_val_f1_micro", "b_test_f1_micro"])
        record_data(data_record, list(metrics.values()), list(metrics.keys()))
        if verbose and epoch % inspect_interval == 0:
            record_data(data_record, [epoch], ["inspect_epoch"])
            log = 'Epoch: {:03d}:'.format(epoch) + '\tF1 micro: ({:.4f}, {:.4f}, {:.4f})'.format(metrics["train_f1_micro"], best_val_f1_micro, b_test_f1_micro)
            log += compose_log(metrics, "f1_macro", 2)
            log += compose_log(metrics, "acc", tabs=2, newline=True) + compose_log(metrics, "loss", 7)
            if beta1 is not None:
                log += "\n\t\tixz: ({:.4f}, {:.4f}, {:.4f})".format(metrics["train_ixz"], metrics["val_ixz"], metrics["test_ixz"])
                if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
                    log += " " * 7 + "ixz_DN: ({:.4f}, {:.4f}, {:.4f})".format(metrics["train_ixz_DN"], metrics["val_ixz_DN"], metrics["test_ixz_DN"])
                if "Z_std" in metrics:
                    log += "\n\t\tZ_std: {}".format(to_string(metrics["Z_std"], connect=", ", num_digits=4))
            if beta2 is not None:
                log += "\n\t\tstruct_kl: {:.4f}".format(metrics["structure_kl"])
            if compute_metrics is not None:
                for metric in compute_metrics:
                    log += "\n\t"
                    for kk in range(model.num_layers):
                        List = [metrics["{}{}_{}".format(id, metric, kk)] for id in ["", "train_", "val_", "test_"]]
                        log += "\t{}_{}:\t({})".format(metric, kk, "{:.4f}; ".format(List[0]) + to_string(List[1:], connect=", ", num_digits=4))
            log += "\n"
            print(log)
            try:
                sys.stdout.flush()
            except:
                pass

        # Saving:
        if epoch % 200 == 0:
            data_record["model_dict"] = model.model_dict
            if filename is not None:
                make_dir(filename)
                pickle.dump(data_record, open(filename + ".p", "wb"))

    # Plotting:
    if isplot:
        plot(data_record, compute_metrics=compute_metrics)
    return data_record


def plot(data_record, compute_metrics=None):
    def plot_metrics(
        metric_list,
        title=None,
        X_source="epoch",
        linestyle_list=None,
        color_list=None,
        y_scale="standard",
        ax=None,
        figsize=(8, 6),
        is_legend=True,
    ):
        import matplotlib.pylab as plt
        if ax is None:
            plt.figure(figsize=figsize)
            plt_save = plt
        else:
            plt_save = plt
            plt = ax
        fontsize = 15
        for i, metric in enumerate(metric_list):
            if y_scale == "standard":
                plt.plot(data_record[X_source], data_record[metric], label=metric, alpha=0.8,
                    linestyle=linestyle_list[i] if linestyle_list is not None else "-",
                    c=color_list[i] if color_list is not None else None,
                )
            elif y_scale == "log":
                plt.semilogy(data_record[X_source], data_record[metric], label=metric, alpha=0.8,
                    linestyle=linestyle_list[i] if linestyle_list is not None else "-",
                    c=color_list[i] if color_list is not None else None,
                )
            else:
                raise
        if is_legend:
            plt.legend(bbox_to_anchor=[1,1])
        plt.tick_params(labelsize=fontsize)
        if ax is None:
            plt.xlabel("epoch", fontsize=fontsize)
        else:
            plt.set_xlabel("epoch", fontsize=fontsize)
        if title is not None:
            if ax is None:
                plt.title(title, fontsize=fontsize)
            else:
                plt.set_title(title, fontsize=fontsize)
        if ax is None:
            plt_save.show()

    plot_metrics(["train_f1_micro", "best_val_f1_micro", "b_test_f1_micro", "test_f1_micro"], "f1_micro")
    plot_metrics(["train_acc", "val_acc", "test_acc"], "acc")
    plot_metrics(["train_loss", "val_loss", "test_loss"], "loss")
    if "train_ixz" in data_record:
        plot_metrics(["train_ixz", "val_ixz", "test_ixz"], "I(X;Z)")
        if "train_ixz_DN" in data_record:
            plot_metrics(["train_ixz_DN", "val_ixz_DN", "test_ixz_DN"], "I_DN(X;Z)")
        if "Z_std" in data_record:
            plot_metrics(["Z_std"])
    if "structure_kl" in data_record:
        plot_metrics(["structure_kl"], "structure_IB", y_scale="log")
        if "structure_kl_DN" in data_record:
            plot_metrics(["structure_kl_DN"], "structure_IB_DN", y_scale="log")
    if compute_metrics is not None:
        for metric in compute_metrics:
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(24, 6))
            for i, mask_id in enumerate(["train_", "val_", "test_"]):
                color_list = [COLOR_LIST[kk] for kk in range(data_record["num_layers"])]
                plot_metrics(["{}{}_{}".format(mask_id, metric, kk) for kk in range(data_record["num_layers"])],
                             X_source="inspect_epoch",
                             color_list=color_list,
                             title="{}{}".format(mask_id, metric),
                             ax=axs[i],
                             is_legend=i==2,
                            )
            plt.show()


# ## Dataset:

# In[6]:


def get_data(
    data_type,
    train_fraction=1,
    added_edge_fraction=0,
    feature_noise_ratio=0,
    **kwargs):
    """Get the pytorch-geometric data object.
    
    Args:
        data_type: Data type. Choose from "Cora", "Pubmed", "citeseer". If want the feature to be binarized, include "-bool" in data_type string.
                   if want to use largest connected components, include "-lcc" in data_type. If use random splitting with train:val:test=0.1:0.1:0.8,
                   include "-rand" in the data_type string.
        train_fraction: Fraction of training labels preserved for the training set.
        added_edge_fraction: Fraction of added (or deleted) random edges. Use positive (negative) number for randomly adding (deleting) edges.
        feature_noise_ratio: Noise ratio for the additive independent Gaussian noise on the features.

    Returns:
        A pytorch-geometric data object containing the specified dataset.
    """
    def to_mask(idx, size):
        mask = torch.zeros(size).bool()
        mask[idx] = True
        return mask
    path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data', data_type)
    # Obtain the mode if given:
    data_type_split = data_type.split("-")
    
    data_type_full = data_type
    data_type = data_type_split[0]
    mode = "lcc" if "lcc" in data_type_split else None
    boolean = True if "bool" in data_type_split else False
    split = "rand" if "rand" in data_type_split else None
    
    # Load data:
    info = {}
    if data_type in ["Cora", "Pubmed", "citeseer"]:
        dataset = Planetoid(path, data_type, transform=T.NormalizeFeatures())
        data = dataset[0]
        info["num_features"] = dataset.num_features
        info["num_classes"] = dataset.num_classes
        info['loss'] = 'softmax'
    else:
        raise Exception("data_type {} is not valid!".format(data_type))

    # Process the dataset according to the mode given:
    if mode is not None:
        if mode == "lcc":
            data = get_data_lcc(dataset.data)
        else:
            raise

    if boolean:
        data.x = data.x.bool().float()
    
    if split == "rand":
        unlabeled_share = 0.8
        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share

        split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(data.x.shape[0]),
                                                                               train_size=train_share,
                                                                               val_size=val_share,
                                                                               test_size=unlabeled_share,
                                                                               stratify=to_np_array(data.y),
                                                                               random_state=kwargs["seed"] if "seed" in kwargs else None,
                                                                              )
        data.train_mask = to_mask(split_train, data.x.shape[0])
        data.val_mask = to_mask(split_val, data.x.shape[0])
        data.test_mask = to_mask(split_unlabeled, data.x.shape[0])

    # Reduce the number of training examples by randomly choosing some of the original training examples:
    if train_fraction != 1:
        try:
            train_mask_file = "../attack_data/{}/train_mask_tr_{}_seed_{}.p".format(data_type_full, train_fraction, kwargs["seed"] % 10)
            new_train_mask = pickle.load(open(train_mask_file, "rb"))
            data.train_mask = torch.BoolTensor(new_train_mask).to(data.y.device)
            print("Load train_mask at {}".format(train_mask_file))
        except:
            raise
            ids_chosen = []
            n_per_class = int(to_np_array(data.train_mask.sum()) * train_fraction / info["num_classes"])
            train_ids = torch.where(data.train_mask)[0]
            for i in range(info["num_classes"]):
                class_id_train = to_np_array(torch.where(((data.y == i) & data.train_mask))[0])
                ids_chosen = ids_chosen + np.random.choice(class_id_train, size=n_per_class, replace=False).tolist()
            new_train_mask = torch.zeros(data.train_mask.shape[0]).bool().to(data.y.device)
            new_train_mask[ids_chosen] = True
            data.train_mask = new_train_mask
            make_dir("../attack_data/{}/".format(data_type_full))
            pickle.dump(to_np_array(new_train_mask), open("../attack_data/{}/train_mask_tr_{}_seed_{}.p".format(data_type_full, train_fraction, kwargs["seed"] % 10), "wb"))

    # Add random edges for untargeted attacks:
    if added_edge_fraction > 0:
        data = add_random_edge(data, added_edge_fraction=added_edge_fraction)
    elif added_edge_fraction < 0:
        data = remove_edge_random(data, remove_edge_fraction=-added_edge_fraction)

    # Perturb features for untargeted attacks:
    if feature_noise_ratio > 0:
        x_max_mean = data.x.max(1)[0].mean()
        data.x = data.x + torch.randn(data.x.shape) * x_max_mean * feature_noise_ratio

    # For adversarial attacks:
    data.data_type = data_type
    if "attacked_nodes" in kwargs:
        attack_path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'attack_data', data_type_full) 
        if not os.path.exists(attack_path):
            os.makedirs(attack_path)
        try:
            with open(os.path.join(attack_path, "test-node.pkl"), 'rb') as f:
                node_ids = pickle.load(f)
                info['node_ids'] = node_ids
                print("Load previous attacked node_ids saved in {}.".format(attack_path))
        except:
            test_ids = to_np_array(torch.where(data.test_mask)[0])
            node_ids = get_list_elements(test_ids, kwargs['attacked_nodes'])
            with open(os.path.join(attack_path, "test-node.pkl"), 'wb') as f:
                pickle.dump(node_ids, f)
            info['node_ids'] = node_ids
            print("Save attacked node_ids into {}.".format(attack_path))
    return data, info


def remove_edge_random(data, remove_edge_fraction):
    """Randomly remove a certain fraction of edges."""
    data_c = deepcopy(data)
    num_edges = int(data_c.edge_index.shape[1] / 2)
    num_removed_edges = int(num_edges * remove_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data_c.edge_index.T)]
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
    edges = [tuple(ele) for ele in to_np_array(data.edge_index.T)]
    added_edges = []
    for i in range(num_added_edges):
        while True:
            added_edge_cand = tuple(np.random.choice(data.x.shape[0], size=2, replace=False))
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

    added_edge_index = torch.LongTensor(np.array(added_edges).T).to(data.edge_index.device)
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
    edges = [tuple(item) for item in to_np_array(data.edge_index.T)]
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

    added_edge_index = torch.LongTensor(np.array(added_edges + remaining_edges).T).to(data.edge_index.device)
    data_edge_corrupted.edge_index = added_edge_index
    return data_edge_corrupted


# ## Nettack:

# In[7]:


def get_attacked_data_deeprobust(
    data,
    surrogate_model,
    target_node,
    direct_attack,
    n_perturbations,
    verbose=False,
):
    """Nettack implemented by DeepRobust."""
    data_attacked = deepcopy(data)
    features, adj, labels = data.features, data.adj, data.labels
    device = data.x.device
    nettack = Nettack(surrogate_model, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device).to(device)
    if direct_attack:
        nettack.attack(features, adj, labels, target_node, n_perturbations, direct=True, verbose=verbose)
    else:
        nettack.attack(features, adj, labels, target_node, n_perturbations, direct=False, n_influencers=5, verbose=verbose)
    modified_adj = nettack.modified_adj
    modified_features = nettack.modified_features

    adj_coo = nettack.modified_adj.tocoo()
    data_attacked.edge_index = torch.stack(
        list(to_Variable(adj_coo.row, adj_coo.col))).long().to(data.x.device)
    data_attacked.x = to_Variable(nettack.modified_features.toarray()).to(data.x.device)
    data_attacked = process_data_for_nettack(data_attacked)
    info = {"structure_perturbations": nettack.structure_perturbations,
            "feature_perturbations": nettack.feature_perturbations,
           }
    return data_attacked, info


# ## Evasive edge attack:

# In[8]:


def get_evasive_dict(
    dirname,
    filename,
    perturb_mode="edge",
    best_model=None,
    feature_perturb_target=None,
    structure_perturb_target=None,
    verbose=False,
    n_repeats=1,
    device_name="cpu",
):
    """Get evasive attack metrics on feature or structure attacks."""
    from GIB.DeepRobust.deeprobust.graph.defense import GCNJaccard, RGCN
    data_record = pickle.load(open(dirname + filename, "rb"))
    parse_dict = parse_filename(filename)
    data, _ = get_data(parse_dict["data_type"])
    baseline = parse_dict["model_type"] in ["GCNJaccard", "RGCN"]
    # Load model:
    if best_model is None:
        if parse_dict["model_type"] == 'GCNJaccard':
            data = process_data_for_nettack(data)
            best_model = GCNJaccard(nfeat=data.features.shape[1], nclass=data.labels.max()+1,
                               nhid=parse_dict["latent_size"], #device=device,
                               weight_decay=parse_dict["weight_decay"],
                               lr=parse_dict["lr"],
                              )
            best_model.load_state_dict(data_record['best_model_dict'])
        elif parse_dict["model_type"] == 'RGCN':
            data = process_data_for_nettack(data)
            best_model = RGCN(nnodes=data.adj.shape[0], nfeat=data.features.shape[1], nclass=data.labels.max()+1,
                         nhid=parse_dict["latent_size"], #device=device,
                         lr=parse_dict["lr"],
                         gamma=parse_dict["gamma"],
                         beta1=parse_dict["beta1"],
                         beta2=parse_dict["weight_decay"],
                        )
            best_model.load_state_dict(data_record['best_model_dict'])
        else:
            best_model = load_model_dict_GNN(data_record["best_model_dict"])
            best_model.set_cache(False)
    
    if device_name != "cpu":
        best_model.to(torch.device(device_name))

    assert parse_dict["added_edge_fraction"] == 0

    best_model.eval()
    df_dict_list = []
    if perturb_mode == "edge":
        if structure_perturb_target is None:
            perturb_target = [-0.75, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]
        else:
            perturb_target = structure_perturb_target
    elif perturb_mode == "feature":
        if feature_perturb_target is None:
            perturb_target = [0.5, 1., 1.5]
        else:
            perturb_target = feature_perturb_target
    else:
        raise

    for perturb_ratio in perturb_target:
        if verbose:
            print("seed: {}    {}: {}".format(parse_dict["seed"], perturb_mode, perturb_ratio))
        for k in range(n_repeats):
            df_dict = deepcopy(parse_dict)
            df_dict["seed_evasive"] = parse_dict["seed"]
            df_dict["best_epoch"] = data_record["best_epoch"] if "best_epoch" in data_record else np.NaN
            df_dict["b_test_f1_micro"] = data_record["b_test_f1_micro"][-1]
            df_dict["repeat_id"] = k
            # Get perturbed data:
            if perturb_mode == "edge":
                df_dict["added_edge_fraction_evasive"] = perturb_ratio
                if perturb_ratio != 0:
                    data_c = deepcopy(data)
                    edge_index_c = torch.LongTensor(pickle.load(open("data_evasive/{}_{}_seed_{}.p".format(parse_dict["data_type"], perturb_ratio, parse_dict["seed"]), "rb"))).to(data.x.device)
                    data_c.edge_index = edge_index_c
                else:
                    data_c = deepcopy(data)
            elif perturb_mode == "feature":
                df_dict["feature_noise_ratio_evasive"] = perturb_ratio
                if perturb_ratio != 0:
                    data_c = deepcopy(data)
                    x_max_mean = data_c.x.max(1)[0].mean()
                    data_c.x = data.x + torch.randn(data.x.shape) * x_max_mean * perturb_ratio
                else:
                    data_c = deepcopy(data)

            if "struct_dropout_mode" in parse_dict:
                struct_dropout_mode = parse_dict["struct_dropout_mode"]
                if 'DNsampling' in struct_dropout_mode or ("standard" in struct_dropout_mode and len(struct_dropout_mode.split("-")) == 3):
                    add_distant_neighbors(data_c, int(struct_dropout_mode.split("-")[-1]))

            # Evaluate:
            data_c = process_data_for_nettack(data_c)
            if device_name != "cpu":
                data_c.to(torch.device(device_name))
            if baseline:
                output = best_model.predict(features=data_c.features, adj=data_c.adj)
                output_test = output[data_c.idx_test].max(1)[1]
                test_f1_micro_best = sklearn.metrics.f1_score(data_c.y[data_c.idx_test].tolist(), output_test.tolist(), average='micro')
            else:
                test_f1_micro_best = get_test_metrics(best_model, data_c, loss_type="softmax")['test_f1_micro']
            df_dict["test_f1_micro_evasive_best"] = test_f1_micro_best
            df_dict_list.append(df_dict)
    return df_dict_list


# ## Helper functions:

# In[9]:


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    
    Adapted from https://github.com/danielzuegner/nettack/blob/master/nettack/utils.py

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return nodes_to_keep


def get_data_lcc(data):
    """Return a new data object consisting of the largest connected component."""
    data_lcc = deepcopy(data)
    edge_index_sparse = edge_index_2_csr(data.edge_index, data.num_nodes)
    lcc = largest_connected_components(edge_index_sparse)
    edge_index_lcc_sparse = edge_index_sparse[lcc][:, lcc].tocoo()
    data_lcc.edge_index = torch.stack(list(to_Variable(edge_index_lcc_sparse.row, edge_index_lcc_sparse.col))).long()

    data_lcc.x = data.x[lcc]
    data_lcc.y = data.y[lcc]
    data_lcc.train_mask = data.train_mask[lcc]
    data_lcc.val_mask = data.val_mask[lcc]
    data_lcc.test_mask = data.test_mask[lcc]
    return data_lcc


# In[ ]:


if __name__ == "__main__":
    # Test dataset:
    data, info = get_data("Cora")
    A_sparse = edge_index_2_csr(data.edge_index, data.num_nodes)
    lcc = largest_connected_components(A_sparse)
    data, info = get_data("Cora-lcc")
    data, info = get_data("Cora-lcc-bool")

