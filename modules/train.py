import os

import torch
from torch.nn import functional as F

import torch_geometric.data as geom_data

import pytorch_lightning as pl
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from GIB.models.gcnn import GIBGCN


AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class NodeLevelGNN(pl.LightningModule):
    def __init__(self, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        super(NodeLevelGNN, self).__init__()

        self.model = config.model
        self.loss = self.cross_entropy_loss if config.loss_type == "softmax" else self.sigmoid_loss
        self.config = config

        if config.lr is None:
            if config.model_name == "GCN":
                self.lr = 0.01
            elif config.model_name == "GAT":
                self.lr = 0.01 if data_type.startswith("Pubmed") else 0.005
            else:
                self.lr = 0.01
        if config.weight_decay is None:
            if config.model_name == "GCN":
                self.weight_decay = 5e-4
            elif config.model_name == "GAT":
                self.weight_decay = 1e-3 if data_type.startswith("Pubmed") else 5e-4
            else:
                self.weight_decay = 5e-4

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
        logits, out_dict = self.forward(data)
        
        ixz, structure_kl_loss = out_dict['ixz_list'], out_dict['structure_kl_list']
        ixz =  torch.stack(ixz, 1).mean(0).sum()
        structure_kl_loss = torch.stack(structure_kl_loss).mean()

        # Only calculate the loss on the nodes corresponding to the mask
        loss, acc = self.loss_acc(logits, data, data.train_mask)

         # IB Loss
        is_dnsampling = self.model.struct_dropout_mode[0] == 'DNsampling'
        is_dropoutstandard = self.model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3
        if self.config.beta1 is not None and self.config.beta1 != 0:
            if is_dnsampling  or is_dropoutstandard:
                ... # ixz = ixz + torch.stack(reg_info["ixz_DN_list"], 1).mean(0).sum()
            loss = loss + ixz * self.config.beta1
        
        if self.config.beta2 is not None and self.config.beta2 != 0:
            
            if is_dnsampling  or is_dropoutstandard:
                ... # structure_kl_loss = structure_kl_loss + torch.stack(reg_info["structure_kl_DN_list"]).mean()
            loss = loss + structure_kl_loss * self.config.beta2

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
# config:
# lr=None, weight_decay=None, beta1=None, beta2=None
def train_node_level(config, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=200,
        progress_bar_refresh_rate=0,
    )  # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

     # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        pl_model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        pl_model = NodeLevelGNN(
            model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        pl_model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return pl_model, result


# # Path to the folder where the datasets are/should be downloaded
# DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# # Path to the folder where the pretrained models are saved
# CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

# # Setting the seed
# pl.seed_everything(42)

# # Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False