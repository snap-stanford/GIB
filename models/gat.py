from turtle import forward
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.distributions.normal import Normal

import torch_geometric.nn as nn
from torch_geometric.nn.dense.linear import Linear

class GATConv(nn.conv.GATConv):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        heads=1, 
        concat=True,
        negative_slope=0.2, 
        reparam_mode=None, 
        prior_mode=None,
        struct_dropout_mode=None, 
        sample_size=1,
        val_use_mean=True,
        bias=True,
        **kwargs
    ):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        # self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean


        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reparameterize(self, encoder_out, size=None):
        # mode = diag
        mean_logit = encoder_out
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]
        size = int(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, size:], beta=1) + 1e-10
        dist = Normal(mean, std)

        return dist, (mean, std)

    def forward(self, x, edge_index, size=None):
        out = super().forward(x, edge_index)

        # Reparameterize:
        # out = out.view(-1, self.out_neurons)
        self.dist, _ = self.reparameterize(model=None, input=out,
                                        mode=self.reparam_mode,
                                        size=self.out_channels,
                                        )  # dist: [B * head, Z]
        Z_core = Z = self.dist.rsample((self.sample_size,)) # [S, B * head, Z]
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

        self.Z_std = Z.std((0, 1))
        self.Z_std = self.Z_std.cpu().data.mean()
        if self.val_use_mean is False or self.training:
            out = Z.mean(0)
        else:
            out = out[:, :self.out_channels].contiguous().view(-1, self.heads * self.out_channels)

        if "Nsampling" in self.struct_dropout_mode[0]:
            if 'categorical' in self.struct_dropout_mode[1]:
                structure_kl_loss = torch.sum(self.alpha*torch.log((self.alpha+1e-16)/self.prior))
            elif 'Bernoulli' in self.struct_dropout_mode[1]:
                posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
                prior = torch.distributions.bernoulli.Bernoulli(self.prior) 
                structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            else:
                raise Exception("I think this belongs to the diff subset sampling that is not implemented")
        
        return out, ixz, structure_kl_loss

    def message(self, edge_index_i, x_i, x_j, size_i):
        #TODO
        ...

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GIBGAT(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        latent_size,
        reparam_mode=None,
        prior_mode=None,
        sample_size=1,
        struct_dropout_mode=("standard", 0.6),
        dropout=True,
        with_relu=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
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

        latent_size = int(self.latent_size / 2)  # Under the default setting, latent_size = 8
        if self.reparam_all_layers is True:
            is_reparam = True

        self.conv1 = GATConv(
            self.num_features,
            latent_size,
            heads=8, 
            concat=True,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            val_use_mean=self.val_use_mean,
            struct_dropout_mode=self.struct_dropout_mode,
            sample_size=self.sample_size,
        )
        if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
            input_size = latent_size * 8 * 2
        else:
            input_size = latent_size * 8
        self.conv2 = GATConv(
            input_size,
            self.num_classes,
            heads=1, 
            concat=True,
            reparam_mode=self.reparam_mode,
            prior_mode=self.prior_mode,
            val_use_mean=self.val_use_mean,
            struct_dropout_mode=self.struct_dropout_mode,
            sample_size=self.sample_size,
        )

        # another sublayer?
        # if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
        #     setattr(self, "conv{}_1".format(i + 1), GATConv(
        #         input_size,
        #         latent_size if i != self.num_layers - 1 else self.num_classes,
        #         heads=8 if i != self.num_layers - 1 else 1, concat=True,
        #         reparam_mode=self.reparam_mode if is_reparam else None,
        #         prior_mode=self.prior_mode if is_reparam  else None,
        #         val_use_mean=self.val_use_mean,
        #         struct_dropout_mode=self.struct_dropout_mode,
        #         sample_size=self.sample_size,
        #     ))

    def forward(self, data):
        out_dict = {
            'latent_out': [],
            'ixz_list' : [],
            'structure_kl_list': []
        }

        x = F.dropout(data.x, p=0.6, training=self.training)


        # another sublayer?
        # if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
        #     x_1, ixz_1, structure_kl_loss_1 = getattr(self, "conv{}_1".format(i + 1))(x, data.multi_edge_index)
                
            
        x, ixz, structure_kl_loss = self.conv1(x, data.edge_index)
        out_dict['latent_out'] = out_dict['latent_out'] + [x]
        out_dict['ixz_list'] = out_dict['ixz_list'] + [ixz]
        out_dict['structure_kl_list'] = out_dict['structure_kl_list'] + [structure_kl_loss]
        # Multi-hop:
        # if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
        #     x = torch.cat([x, x_1], dim=-1)

        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # another sublayer?
        # if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
        #     x_1, ixz_1, structure_kl_loss_1 = getattr(self, "conv{}_1".format(self.num_layers))(x, data.multi_edge_index)

        x, ixz, structure_kl_loss = self.conv2(x, data.edge_index)
        out_dict['latent_out'] = out_dict['latent_out'] + [x]
        out_dict['ixz_list'] = out_dict['ixz_list'] + [ixz]
        out_dict['structure_kl_list'] = out_dict['structure_kl_list'] + [structure_kl_loss]

        # Multi-hop:
        # if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
        #     x = x + x_1

        return x, out_dict