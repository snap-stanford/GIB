import os
import sys

sys.path.append(os.path.join(os.path.dirname("__file__"), ".."))
import torch
from torch.nn import functional as F

from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 1 if AVAIL_GPUS else 64


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, config):
        super(NodeLevelGNN, self).__init__()
        self.model = config.model
        self.loss = (
            self.cross_entropy_loss
            if config.loss_type == "softmax"
            else self.sigmoid_loss
        )

        if config.lr is None:
            if config.model_name == "GCN":
                config.lr = 0.01
            elif config.model_name == "GAT":
                config.lr = 0.01 if config.dataset_name.startswith("Pubmed") else 0.005
            else:
                config.lr = 0.01
        if config.weight_decay is None:
            if config.model_name == "GCN":
                config.weight_decay = 5e-4
            elif config.model_name == "GAT":
                config.weight_decay = (
                    1e-3 if config.dataset_name.startswith("Pubmed") else 5e-4
                )
            else:
                self.weight_decay = 5e-4

        self.config = config
        self.save_hyperparameters()

    def get_mask(self, data, mode):
        try:
            # Only calculate the loss on the nodes corresponding to the mask
            if mode == "train":
                mask = data.train_mask
            elif mode == "val":
                mask = data.val_mask
            elif mode == "test":
                mask = data.test_mask
            else:
                assert False, "Unknown forward mode: %s" % mode
        except AttributeError:
            # Not all datasets use masks
            print(data)
            mask = torch.tensor([True] * BATCH_SIZE)
        return mask

    def forward(self, data, mode="train"):
        logits, out_dict = self.model(data)
        mask = self.get_mask(data, mode)

        loss, acc = self.loss_acc(logits, data, mask)
        return logits, out_dict, loss, acc

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction="none")

    def sigmoid_loss(self, logits, labels):
        return F.sigmoid(logits, labels, reduction="none")

    def loss_acc(self, logits, data, mask):

        loss = self.loss(logits[mask], data.y[mask])

        acc = (logits[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def logger_fun(
        self, loss, acc, ixz=None, iyz=None, structure_kl_loss=None, mode="train"
    ):
        root_dir = "../saved_models"
        base_path = f"{root_dir}/NodeLevel_{self.config.exp_name}"
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{mode}/acc", acc, on_step=False, on_epoch=True)

        if ixz is not None:
            self.log(f"{mode}/ixz_conv1", ixz.mean(0), on_step=False, on_epoch=True)
            self.log(f"{mode}/ixz_conv2", ixz.mean(1), on_step=False, on_epoch=True)
            self.log(f"{mode}/ixz", ixz.mean(), on_step=False, on_epoch=True)
            for idx, tensor in enumerate(ixz):
                torch.save(tensor, f"{base_path}/{mode}/ixz_conv{idx}.pt")

        if iyz is not None:
            self.log(f"{mode}/iyz", iyz.mean(), on_step=False, on_epoch=True)
            torch.save(iyz.view(-1, 1), f"{base_path}/{mode}/iyz.pt")

        if structure_kl_loss is not None:
            self.log(
                f"{mode}/structure_kl_loss",
                structure_kl_loss.mean(),
                on_step=False,
                on_epoch=True,
            )
            for idx, tensor in enumerate(structure_kl_loss):
                torch.save(tensor, f"{base_path}/{mode}/structure_kl_loss_conv{idx}.pt")

    def training_step(self, batch, batch_idx):
        data = batch
        out, out_dict, loss, acc = self.forward(data, mode="train")

        mask = self.get_mask(data, "train")
        ixz, structure_kl_loss = self.get_ixz_kl_loss(out_dict, mask)
        iyz = loss

        # IB Loss
        ib_loss = loss.mean()
        if self.config.beta1 is not None and self.config.beta1 != 0:
            ib_loss += ixz.mean() * self.config.beta1

        if self.config.beta2 is not None and self.config.beta2 != 0:
            ib_loss += structure_kl_loss.mean() * self.config.beta2

        self.logger_fun(ib_loss, acc, ixz, iyz, structure_kl_loss, mode="train")
        out = {"loss": ib_loss, "acc": acc}
        return out

    def get_ixz_kl_loss(self, out_dict, mask):
        if "ixz_list" in out_dict.keys():
            ixz = torch.stack(out_dict["ixz_list"])[:, mask, ...]
        else:
            ixz = None
        if "ixz_list" in out_dict.keys():
            structure_kl_loss = torch.stack(out_dict["structure_kl_list"])
        else:
            structure_kl_loss = None
        return ixz, structure_kl_loss

    def validation_step(self, batch, batch_idx):
        data = batch
        out, out_dict, loss, acc = self.forward(data, mode="val")

        mask = self.get_mask(data, "train")
        ixz, structure_kl_loss = self.get_ixz_kl_loss(out_dict, mask)
        iyz = loss

        # IB Loss
        ib_loss = loss.mean()
        if self.config.beta1 is not None and self.config.beta1 != 0:
            ib_loss += ixz.mean() * self.config.beta1

        if self.config.beta2 is not None and self.config.beta2 != 0:
            ib_loss += structure_kl_loss.mean() * self.config.beta2

        self.logger_fun(ib_loss, acc, ixz, iyz, structure_kl_loss, mode="val")

        out = {"val_loss": ib_loss, "val_acc": acc}
        return out

    def test_step(self, batch, batch_idx):
        data = batch
        out, out_dict, loss, acc = self.forward(data, mode="test")

        mask = self.get_mask(data, "train")
        ixz, structure_kl_loss = self.get_ixz_kl_loss(out_dict, mask)
        iyz = loss

        # IB Loss
        ib_loss = loss.mean()
        if self.config.beta1 is not None and self.config.beta1 != 0:
            ib_loss += ixz.mean() * self.config.beta1

        if self.config.beta2 is not None and self.config.beta2 != 0:
            ib_loss += structure_kl_loss.mean() * self.config.beta2

        self.logger_fun(ib_loss, acc, ixz, iyz, structure_kl_loss, mode="test")
        out = {"test_loss": ib_loss, "test_acc": acc}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        return optimizer


def train_node_level(config, dataset, val_dataset=None, test_dataset=None):
    pl.seed_everything(42)
    node_data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=12)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(config.CHECKPOINT_PATH, "NodeLevel_" + config.exp_name)
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(f"{root_dir}/train", exist_ok=True)
    os.makedirs(f"{root_dir}/val", exist_ok=True)
    os.makedirs(f"{root_dir}/test", exist_ok=True)
    wandb_logger = WandbLogger(project="gib", name=config.exp_name)
    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=root_dir,
        callbacks=[
            # EarlyStopping(monitor="val/loss", mode="min", min_delta=1e-2, patience=6),
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val/acc"),
        ],
        gpus=AVAIL_GPUS,
        max_epochs=100,  # GIB GCN acc plateaus at 70~ epochs, GAT at 20~
        progress_bar_refresh_rate=1,
    )  # 0 because epoch size is 1

    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        config.CHECKPOINT_PATH, "NodeLevel_%s.ckpt" % config.model_name
    )
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        pl_model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        pl_model = NodeLevelGNN(
            config=config,
        )
        trainer.fit(
            model=pl_model,
            train_dataloaders=node_data_loader,
            val_dataloaders=node_data_loader
            if val_dataset is None
            else DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=12),
        )
        pl_model = NodeLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Test best model on the test set
    test_result = trainer.test(
        model=pl_model,
        dataloaders=node_data_loader
        if val_dataset is None
        else DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=12),
        verbose=False,
    )
    batch = next(iter(node_data_loader))
    batch = batch.to(pl_model.device)
    train_out, train_out_dict, train_loss, train_acc = pl_model.forward(
        batch,
        mode="train",
    )
    val_out, val_out_dict, val_loss, val_acc = pl_model.forward(
        batch,
        mode="val",
    )
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test/acc"]}
    return trainer, train_out_dict, val_out_dict, pl_model, result, test_result
