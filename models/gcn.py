import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch.distributions.normal import Normal


class GCNConv(gnn.conv.GCNConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        reparam_mode=None,
        prior_mode=None,
        sample_size=1,
        val_use_mean=True,
        **kwargs
    ):
        """Graph Convolution layer with GIB principle

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels

            reparam_mode (string, optional): reparametrization mode for latent space. Defaults to None == diagonal.
            prior_mode (string, optional): feature prior. Defaults to None.
            struct_dropout_mode (List[string], optional): structural dropout: first item should be sampling mode and second distribution. Defaults to None.
            sample_size (int, optional): sample size of latent space. Defaults to 1.
            val_use_mean (bool, optional): use latent space mean as layer';s output. Defaults to True.
            bias (bool, optional): _description_. Defaults to True.
        """

        super(GCNConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr="add",
            normalize=False,
            **kwargs
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.sample_size = sample_size
        self.val_use_mean = False

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # self.feature_prior = Mixture_Gaussian_reparam(
        #     is_reparam=False, Z_size=self.out_channels, n_components=100
        # )

        self.reset_parameters()

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
        out = super().forward(x, edge_index, edge_weight=edge_attr)
        # Reparameterize:
        self.dist, _ = self.reparameterize(
            encoder_out=out, size=self.out_channels
        )  # [B, Z]

        Z = self.dist.rsample((self.sample_size,))[0]  # [S, B, Z]

        if self.prior_mode == "Gaussian":
            self.feature_prior = Normal(
                loc=torch.zeros(Z.shape).to(x.device),
                scale=torch.ones(Z.shape).to(x.device),
            )  # [B, Z]

        # Calculate prior loss:
        if self.prior_mode == "Gaussian":
            ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior)
        else:
            Z_logit = (
                self.dist.log_prob(Z).sum(-1)
                # if self.reparam_mode.startswith("diag")
                # else self.dist.log_prob(Z)
            )  # [S, B]
            prior_logit = self.feature_prior.log_prob(Z).sum(-1)  # [S, B]
            # upper bound of I(X; Z):
            # print(ixz.shape, "non-gaussian")
            ixz = (Z_logit - prior_logit).mean(0)  # [B]

        self.Z_std = Z.std((0, 1))
        self.Z_std = self.Z_std.cpu().data.mean()
        # if self.val_use_mean is False or self.training:
        #     out = Z.mean(0)  # [B, Z]
        # else:
        #     out = out[:, : self.out_channels]  # [B, Z]

        structure_kl_loss = torch.zeros(ixz.shape).to(x.device)

        return out, ixz, structure_kl_loss

    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class GIBGCN(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        latent_size,
        reparam_mode="diag",
        prior_mode="Gaussian",
        sample_size=1,
        struct_dropout_mode=("standard", 0.6),
        dropout=True,
        with_relu=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
    ):
        super(GIBGCN, self).__init__()
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
        self.conv1 = GCNConv(
            in_channels=self.num_features,
            out_channels=self.latent_size,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            sample_size=self.sample_size,
            val_use_mean=self.val_use_mean,
            normalize=self.normalize,
        )
        self.conv2 = GCNConv(
            in_channels=self.latent_size,
            out_channels=self.num_classes,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            sample_size=self.sample_size,
            val_use_mean=self.val_use_mean,
            normalize=self.normalize,
        )

    def forward(self, data, save_latent=False):
        out_dict = {"latent_out": [], "ixz_list": [], "structure_kl_list": []}

        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr

        # if self.use_relu? x = F.relu(x)
        # if self.dropout: x = F.dropout(x, training=self.training)

        x, ixz, structure_kl_loss = self.conv1(x, edge_index, edge_attr)
        out_dict["latent_out"] = out_dict["latent_out"] + [x]
        out_dict["ixz_list"] = out_dict["ixz_list"] + [ixz]
        out_dict["structure_kl_list"] = out_dict["structure_kl_list"] + [
            structure_kl_loss
        ]
        # save latent ==> torch.save(z)?

        x, ixz, structure_kl_loss = self.conv2(x, edge_index, edge_attr)
        out_dict["latent_out"] = out_dict["latent_out"] + [x]
        out_dict["ixz_list"] = out_dict["ixz_list"] + [ixz]
        out_dict["structure_kl_list"] = out_dict["structure_kl_list"] + [
            structure_kl_loss
        ]

        return x, out_dict


class Mixture_Gaussian_reparam(nn.Module):
    def __init__(
        self,
        # Use as reparamerization:
        mean_list=None,
        scale_list=None,
        weight_logits=None,
        # Use as prior:
        Z_size=None,
        n_components=None,
        mean_scale=0.1,
        scale_scale=0.1,
        # Mode:
        is_reparam=True,
        reparam_mode="diag",
        is_cuda=False,
    ):
        super(Mixture_Gaussian_reparam, self).__init__()
        self.is_reparam = is_reparam
        self.reparam_mode = reparam_mode
        self.is_cuda = is_cuda
        self.device = torch.device(
            self.is_cuda
            if isinstance(self.is_cuda, str)
            else "cuda"
            if self.is_cuda
            else "cpu"
        )

        if self.is_reparam:
            self.mean_list = mean_list  # size: [B, Z, k]
            self.scale_list = scale_list  # size: [B, Z, k]
            self.weight_logits = weight_logits  # size: [B, k]
            self.n_components = self.weight_logits.shape[-1]
            self.Z_size = self.mean_list.shape[-2]
        else:
            self.n_components = n_components
            self.Z_size = Z_size
            self.mean_list = nn.Parameter(
                (torch.rand(1, Z_size, n_components) - 0.5) * mean_scale
            )
            self.scale_list = nn.Parameter(
                torch.log(
                    torch.exp(
                        (torch.rand(1, Z_size, n_components) * 0.2 + 0.9) * scale_scale
                    )
                    - 1
                )
            )
            self.weight_logits = nn.Parameter(torch.zeros(1, n_components))
            if mean_list is not None:
                self.mean_list.data = torch.tensor(mean_list)
                self.scale_list.data = torch.tensor(scale_list)
                self.weight_logits.data = torch.tensor(weight_logits)

        self.to(self.device)

    def log_prob(self, input):
        """Obtain the log_prob of the input."""
        input = input.unsqueeze(-1)  # [S, B, Z, 1]
        if self.reparam_mode == "diag":
            if self.is_reparam:
                # logits: [S, B, Z, k]
                logits = -(
                    (input - self.mean_list) ** 2
                ) / 2 / self.scale_list**2 - torch.log(
                    self.scale_list * np.sqrt(2 * np.pi)
                )
            else:
                scale_list = F.softplus(self.scale_list, beta=1)
                logits = -(
                    (input - self.mean_list) ** 2
                ) / 2 / scale_list**2 - torch.log(scale_list * np.sqrt(2 * np.pi))
        else:
            raise
        # log_softmax(weight_logits): [B, k]
        # logits: [S, B, Z, k]
        # log_prob: [S, B, Z]
        log_prob = torch.logsumexp(
            logits + F.log_softmax(self.weight_logits, -1).unsqueeze(-2), axis=-1
        )  # F(...).unsqueeze(-2): [B, 1, k]
        return log_prob

    def prob(self, Z):
        return torch.exp(self.log_prob(Z))

    def sample(self, n=None):
        if n is None:
            n_core = 1
        else:
            assert isinstance(n, tuple)
            n_core = n[0]
        weight_probs = F.softmax(self.weight_logits, -1)  # size: [B, m]
        idx = (
            torch.multinomial(weight_probs, n_core, replacement=True)
            .unsqueeze(-2)
            .expand(-1, self.mean_list.shape[-2], -1)
        )  # multinomial result: [B, S]; result: [B, Z, S]
        mean_list = torch.gather(self.mean_list, dim=-1, index=idx)  # [B, Z, S]
        if self.is_reparam:
            scale_list = torch.gather(self.scale_list, dim=-1, index=idx)  # [B, Z, S]
        else:
            scale_list = F.softplus(
                torch.gather(self.scale_list, dim=-1, index=idx), beta=1
            )  # [B, Z, S]
        Z = torch.normal(mean_list, scale_list).permute(2, 0, 1)
        if n is None:
            Z = Z.squeeze(0)
        return Z

    def rsample(self, n=None):
        return self.sample(n=n)

    def __repr__(self):
        return "Mixture_Gaussian_reparam({}, Z_size={})".format(
            self.n_components, self.Z_size
        )

    @property
    def model_dict(self):
        model_dict = {"type": "Mixture_Gaussian_reparam"}
        model_dict["is_reparam"] = self.is_reparam
        model_dict["reparam_mode"] = self.reparam_mode
        model_dict["Z_size"] = self.Z_size
        model_dict["n_components"] = self.n_components
        model_dict["mean_list"] = np.array(self.mean_list)
        model_dict["scale_list"] = np.array(self.scale_list)
        model_dict["weight_logits"] = np.array(self.weight_logits)
        return model_dict
