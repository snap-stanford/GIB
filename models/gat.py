from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import torch_geometric.nn as gnn
from torch_geometric.utils import degree
from torch_scatter import gather_csr, scatter, segment_csr


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, torch.Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def uniform_prior(index):
    deg = degree(index)
    deg = deg[index]
    return 1.0 / deg.unsqueeze(1)


def softmax(
    src: torch.Tensor,
    index: Optional[torch.Tensor] = None,
    ptr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> torch.Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce="max"), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce="sum"), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce="max")
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce="sum")
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)


class GATConv(gnn.conv.GATConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        reparametrize_mode=None,
        prior_mode=None,
        struct_dropout_mode=None,
        sample_size=1,
        val_use_mean=True,
        bias=True,
        **kwargs,
    ):
        """Graph Attention Convolution layer with GIB principle

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            heads (int, optional): number of attention heads. Defaults to 1.
            concat (bool, optional): concatenate heads and output channels. Defaults to True.
            negative_slope (float, optional): Defaults to 0.2.
            reparametrize_mode (string, optional): reparametrization mode for latent space. Defaults to None == diagonal.
            prior_mode (string, optional): feature prior. Defaults to None.
            struct_dropout_mode (List[string], optional): structural dropout: first item should be sampling mode and second distribution. Defaults to None.
            sample_size (int, optional): sample size of latent space. Defaults to 1.
            val_use_mean (bool, optional): use latent space mean as layer';s output. Defaults to True.
            bias (bool, optional): _description_. Defaults to True.
        """
        super(GATConv, self).__init__(
            aggr="add", in_channels=in_channels, out_channels=out_channels, **kwargs
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.reparametrize_mode = (
            reparametrize_mode if reparametrize_mode != "None" else "diag"
        )
        self.prior_mode = prior_mode
        # self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean
        self._alpha = None

    def reparameterize(self, encoder_out, size=None):
        # mode = diag
        mean_logit = encoder_out
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]

        size = math.ceil(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, -size:], beta=1) + 1e-10
        dist = Normal(mean, std)
        return dist, (mean, std)

    def forward(self, x, edge_index, edge_attr=None, size=None, **kwargs):
        out = super().forward(x, edge_index, edge_attr=edge_attr, size=size)

        # Reparameterize:
        out = out.view(-1, self.out_channels)
        dist, _ = self.reparameterize(
            encoder_out=out, size=self.out_channels
        )  # dist: [B * head, Z]

        Z_core = Z = dist.rsample((self.sample_size,))  # [B * head, Z]
        Z = Z_core
        self.feature_prior = Normal(
            loc=torch.zeros(Z.shape).to(x.device),
            scale=torch.ones(Z.shape).to(x.device),
        )  # [B, Z]

        ixz = torch.distributions.kl.kl_divergence(dist, self.feature_prior).view(
            x.shape[0], -1, self.heads
        )
        self.Z_std = Z.std((0, 1))
        self.Z_std = self.Z_std.cpu().data.mean()

        out = (
            out[:, : self.out_channels]
            .contiguous()
            .view(-1, self.heads * self.out_channels)
        )

        if "Nsampling" in self.struct_dropout_mode[0]:
            structure_kl_loss = self.alpha * torch.log(
                (self.alpha + 1e-16) / self.prior
            )

        else:
            structure_kl_loss = torch.zeros(ixz.shape).to(x.device)
        return out, ixz, structure_kl_loss

    def message(self, x_j, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha  # Save for later use.
        self.alpha = alpha

        # Sample attention coefficients stochastically.
        if self.struct_dropout_mode[0] == "standard":
            prob_dropout = self.struct_dropout_mode[1]
            alpha = F.dropout(alpha, p=prob_dropout, training=self.training)

        elif "Nsampling" in self.struct_dropout_mode[0]:
            # multicategorical-sum only
            self.prior = uniform_prior(index)
            temperature = self.struct_dropout_mode[2]
            sample_neighbor_size = self.struct_dropout_mode[3]

            alphas = []
            for _ in range(
                sample_neighbor_size
            ):  #! this can be improved by parallel sampling
                alphas.append(scatter_sample(alpha, index, temperature, size_i))
            alphas = torch.stack(alphas, dim=0)
            alpha = alphas.sum(dim=0)
        else:
            pass

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


def scatter_sample(src, index, temperature, num_nodes=None):
    gumbel = (
        torch.distributions.Gumbel(
            torch.tensor([0.0]).to(src.device), torch.tensor([1.0]).to(src.device)
        )
        .sample(src.size())
        .squeeze(-1)
    )
    log_prob = torch.log(src + 1e-16)
    logit = (log_prob + gumbel) / temperature
    return softmax(logit, index=index, num_nodes=num_nodes)


class GIBGAT(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        latent_size,
        reparam_mode="diag",
        prior_mode=None,
        sample_size=1,
        struct_dropout_mode=("standard", 0.6),
        dropout=True,
        with_relu=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
        **kwargs,
    ):
        super(GIBGAT, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.latent_size = latent_size
        self.sample_size = sample_size
        self.struct_dropout_mode = struct_dropout_mode
        self.dropout = dropout
        self.with_relu = with_relu
        self.val_use_mean = val_use_mean
        self.reparam_all_layers = reparam_all_layers
        self.normalize = normalize

        self.reparam_layers = []

        # Under the default setting, latent_size = 8
        # latent_size = int(self.latent_size / 2)
        if self.reparam_all_layers is True:
            is_reparam = True

        self.conv1 = GATConv(
            self.num_features,
            self.latent_size,
            heads=1,
            concat=True,
            reparametrize_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            val_use_mean=self.val_use_mean,
            struct_dropout_mode=self.struct_dropout_mode,
            sample_size=self.sample_size,
            **kwargs,
        )
        if self.struct_dropout_mode[0] == "DNsampling" or (
            self.struct_dropout_mode[0] == "standard"
            and len(self.struct_dropout_mode) == 3
        ):
            input_size = latent_size * 8 * 2
        else:
            input_size = latent_size * 8
        self.conv2 = GATConv(
            latent_size,
            self.num_classes,
            heads=1,
            concat=True,
            reparametrize_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            val_use_mean=self.val_use_mean,
            struct_dropout_mode=self.struct_dropout_mode,
            sample_size=self.sample_size,
            **kwargs,
        )

    def forward(self, data):
        out_dict = {"latent_out": [], "ixz_list": [], "structure_kl_list": []}
        x, edge_index, edge_attr = (
            data.x.float(),
            data.edge_index,
            data.edge_attr,
        )

        x = F.dropout(x, p=0.4, training=self.training)

        x, ixz, structure_kl_loss = self.conv1(x, edge_index, edge_attr)
        out_dict["latent_out"] = out_dict["latent_out"] + [x]
        out_dict["ixz_list"] = out_dict["ixz_list"] + [ixz]
        out_dict["structure_kl_list"] = out_dict["structure_kl_list"] + [
            structure_kl_loss
        ]

        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x, ixz, structure_kl_loss = self.conv2(x, data.edge_index, edge_attr)
        out_dict["latent_out"] = out_dict["latent_out"] + [x]
        out_dict["ixz_list"] = out_dict["ixz_list"] + [ixz]
        out_dict["structure_kl_list"] = out_dict["structure_kl_list"] + [
            structure_kl_loss
        ]

        return x, out_dict
