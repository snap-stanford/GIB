import torch.nn as nn

class GIBGCN(nn.Module):
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

        self.conv1 = GCNConv(self.num_features, self.latent_size)
        self.conv2 = GCNConv(self.latent_size, self.num_classes, re)

    def forward()

    # def ...