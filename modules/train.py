import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from GIB.models.gcnn import GIBGCN

class NodeLevelGNN(pl.LightningModule):
    def __init__(self, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        super(NodeLevelGNN, self).__init__()

        self.model = config.model
        self.loss = self.cross_entropy_loss if config.loss_type == "softmax" else self.sigmoid_loss

        self.lr = config.lr 
        self.weight_decay = config.weight_decay

    def forward(self, data):
        out, ixz, structure_kl_loss = self.model(data)
        return out, ixz, structure_kl_loss

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction="mean")

    def sigmoid_loss(self, logits, labels):
        return F.sigmoid(logits, labels, reduction="mean")

    def loss_acc(self, logits, data, mask):
        loss = self.loss(logits[mask], data.y[mask])
        acc = (logits[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def logger(self, loss, acc, ixz, structure_kl_loss, mode):
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        self.log(f"{mode}_ixz", ixz)
        self.log(f"{mode}_structure_kl_loss", structure_kl_loss)

    def training_step(self, batch, batch_idx):
        data = batch
        logits, ixz, structure_kl_loss = self.forward(data)

        # Only calculate the loss on the nodes corresponding to the mask
        loss, acc = self.loss_acc(logits, data, data.train_mask)
        self.logger(loss, acc, ixz, structure_kl_loss, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        logits, ixz, structure_kl_loss = self.forward(data)

        # Only calculate the loss on the nodes corresponding to the mask
        loss, acc = self.loss_acc(logits, data, data.val_mask)
        self.logger(loss, acc, ixz, structure_kl_loss, mode="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

# GAT 
# lr = 0.01 if data_type.startswith("Pubmed") else 0.005
# weight_decay = 1e-3 if data_type.startswith("Pubmed") else 5e-4

# GCN 
# lr = 0.01, 
# weight_decay = 5e-4
#TODO
# def train_node_level