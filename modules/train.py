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
BATCH_SIZE = 256 if AVAIL_GPUS else 64


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

    def forward(self, data, mode="train"):
        logits, out_dict = self.model(data)
        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

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

    def logger_fun(self, loss, acc, ixz, iyz, structure_kl_loss, mode):
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        self.log(f"{mode}_ixz", ixz.mean())
        self.log(f"{mode}_iyz", iyz.mean())
        self.log(f"{mode}_structure_kl_loss", structure_kl_loss.mean())

        root_dir = "../saved_models"
        base_path = f"{root_dir}/NodeLevel_{self.config.exp_name}"
        for idx, tensor in enumerate(ixz):
            torch.save(tensor, f"{base_path}/{mode}_ixz_conv{idx}.pt")

        torch.save(iyz.view(-1, 1), f"{base_path}/{mode}_iyz.pt")

        for idx, tensor in enumerate(structure_kl_loss):
            torch.save(tensor, f"{base_path}/{mode}_structure_kl_loss_conv{idx}.pt")

        self.logger.experiment.log(
            {
                # f"{mode}_ixz_conv1": wandb.Table(
                #     data=ixz[0].tolist(), columns=["D1", "D2", "D3", "D4"]
                # ),
                # f"{mode}_ixz_conv2": wandb.Table(
                #     data=ixz[1].tolist(), columns=["D1", "D2", "D3", "D4"]
                # ),
                # f"{mode}_iyz": wandb.Table(
                #     data=iyz.view(-1, 1).tolist(), columns=["D1"]
                # ),
                # f"{mode}_structure_kl_loss_conv1": wandb.Table(
                #     data=structure_kl_loss[0].tolist(), columns=["D1", "D2", "D3", "D4"]
                # ),
                # f"{mode}_structure_kl_loss_conv2": wandb.Table(
                #     data=structure_kl_loss[1].tolist(), columns=["D1", "D2", "D3", "D4"]
                # ),
            },
            # commit=False,
        )

    def training_step(self, batch, batch_idx):
        data = batch
        out, out_dict, loss, acc = self.forward(data, mode="train")

        ixz = torch.stack(out_dict["ixz_list"])[:, data.train_mask, ...]
        structure_kl_loss = torch.stack(out_dict["structure_kl_list"])[
            :, data.train_mask, ...
        ]
        iyz = loss

        # IB Loss
        ixz_mean = ixz.mean()
        structure_kl_loss_mean = structure_kl_loss.mean()
        ib_loss = loss.mean()

        if self.config.beta1 is not None and self.config.beta1 != 0:
            ib_loss += ixz_mean * self.config.beta1

        if self.config.beta2 is not None and self.config.beta2 != 0:
            ib_loss += structure_kl_loss_mean * self.config.beta2

        self.logger_fun(ib_loss, acc, ixz, iyz, structure_kl_loss, mode="train")
        out = {
            "loss": ib_loss,
            "acc": acc,
        }
        return out

    def validation_step(self, batch, batch_idx):
        data = batch
        out, out_dict, loss, acc = self.forward(data, mode="val")

        ixz = torch.stack(out_dict["ixz_list"])[:, data.train_mask, ...]
        structure_kl_loss = torch.stack(out_dict["structure_kl_list"])[
            :, data.val_mask, ...
        ]
        iyz = loss

        # IB Loss
        ixz_mean = ixz.mean()
        structure_kl_loss_mean = structure_kl_loss.mean()
        ib_loss = loss.mean()

        if self.config.beta1 is not None and self.config.beta1 != 0:
            ib_loss += ixz_mean * self.config.beta1

        if self.config.beta2 is not None and self.config.beta2 != 0:
            ib_loss += structure_kl_loss_mean * self.config.beta2

        self.logger_fun(ib_loss, acc, ixz, iyz, structure_kl_loss, mode="val")
        out = {"val_loss": ib_loss, "val_acc": acc}
        return out

    def test_step(self, batch, batch_idx):
        data = batch
        out, out_dict, loss, acc = self.forward(data, mode="test")

        ixz = torch.stack(out_dict["ixz_list"])[:, data.train_mask, ...]
        structure_kl_loss = torch.stack(out_dict["structure_kl_list"])[
            :, data.test_mask, ...
        ]
        iyz = loss

        # IB Loss
        ixz_mean = ixz.mean()
        structure_kl_loss_mean = structure_kl_loss.mean()
        ib_loss = loss.mean()

        if self.config.beta1 is not None and self.config.beta1 != 0:
            ib_loss += ixz_mean * self.config.beta1

        if self.config.beta2 is not None and self.config.beta2 != 0:
            ib_loss += structure_kl_loss_mean * self.config.beta2

        self.logger_fun(ib_loss, acc, ixz, iyz, structure_kl_loss, mode="test")
        out = {"test_loss": ib_loss, "test_acc": acc}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        return optimizer


def train_node_level(config, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = DataLoader(dataset, batch_size=1, num_workers=12)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(config.CHECKPOINT_PATH, "NodeLevel_" + config.exp_name)
    os.makedirs(root_dir, exist_ok=True)
    wandb_logger = WandbLogger(project="gib", name=config.exp_name)
    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=root_dir,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-2),
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
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
            **model_kwargs,
        )
        trainer.fit(pl_model, node_data_loader, node_data_loader)
        pl_model = NodeLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Test best model on the test set
    test_result = trainer.test(pl_model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(pl_model.device)
    _, _, train_loss, train_acc = pl_model.forward(
        batch,
        mode="train",
    )
    _, _, val_loss, val_acc = pl_model.forward(
        batch,
        mode="val",
    )
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return pl_model, result


# # Setting the seed
# pl.seed_everything(42)

# # Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
