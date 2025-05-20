import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AveragePrecision

from src.models import MLP
from utils.hash import decode_id, generate_id


# variable parameters
def generate_params(seed=None):
    # Set seeds if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    hdim = random.choice([16, 32, 64])
    n_layers = random.choice([2, 3])
    activation = random.choice(["relu", "tanh", "swish"])
    mbsize = random.choice([32, 128])
    positive_ratio = random.choice([0.05, 0.2, 0.5])
    dropout = random.choice([0, 0.1, 0.2, 0.5])

    hyperparameters = {
        "hd": hdim,
        "n": n_layers,
        "mb": mbsize,
        "pr": positive_ratio,
        "act": activation,
        "dr": dropout,
    }
    experiment_id = generate_id(hyperparameters)

    return (hdim, n_layers, activation, mbsize, positive_ratio, dropout), experiment_id


class MLPClassifierPrior(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        val_loss_fn,
        lr,
        factor=0.2,
        patience=2,
        min_lr=1e-6,
    ):
        super().__init__()

        # Create sequential model
        self.model = model

        # Loss and metrics
        self.loss = loss_fn
        self.auc_metric = val_loss_fn

        # Hyperparameters
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.learning_rate = lr
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.softmax(logits, dim=1)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)

        # Compute AUC metric
        auc = self.auc_metric(preds, y)
        self.log("Perf Val/Mean", auc, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            verbose=True,
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "Perf Val/Mean"}


class MLPClassifierFull(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        val_loss_fn,
        lr,
        factor=0.2,
        patience=2,
        min_lr=1e-6,
    ):
        super().__init__()

        # Create sequential model
        self.model = model

        # Loss and metrics
        self.loss = loss_fn
        self.auc_metric = val_loss_fn

        # Hyperparameters
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.learning_rate = lr
        # self.save_hyperparameters()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        c, p, y = batch
        x = torch.cat([p, c], dim=1)
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        c, p, y = batch
        x = torch.cat([p, c], dim=1)
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.softmax(logits, dim=1)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)

        # Compute AUC metric
        auc = self.auc_metric(preds, y)
        self.log("Perf Val/Mean", auc, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            verbose=True,
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "Perf Val/Mean"}


# Example usage
# def train_classifier(train_loader, val_loader, input_size=10, max_epochs=50):
#     # Initialize the model
#     model = MLPClassifier(
#         input_size=input_size,
#         hidden_layers=[64, 32],
#         dropout_rate=0.2,
#         learning_rate=1e-3
#     )

#     # Initialize Lightning Trainer
#     trainer = pl.Trainer(
#         max_epochs=max_epochs,
#         accelerator='cpu',
#         enable_checkpointing=False,
#         logger=False
#     )

#     # Train the model
#     trainer.fit(model, train_loader, val_loader)

#     return model


def load_model(filepath: str, n_prior_features, n_costly_features: int = 0, type: str = "MLPClassifierPrior"):
    filename = filepath.split("/")[-1][5:-5]
    hyperparameters = decode_id(filename)

    hdim = hyperparameters["hd"]
    n_layers = hyperparameters["n"]
    mbsize = hyperparameters["mb"]
    positive_ratio = hyperparameters["pr"]
    activation = hyperparameters["act"]
    dropout = hyperparameters["dr"]

    mlp = MLP(idim=n_prior_features + n_costly_features if type != "MLPClassifierPrior" else n_prior_features, hdim=hdim, n_layers=n_layers, dropout=dropout, activation=activation)

    auc_metric = AveragePrecision(task="multiclass", num_classes=2, average="macro")
    lr = 1e-3

    # load checkpoint from file
    if type == "MLPClassifierPrior":
        model = MLPClassifierPrior(mlp, loss_fn=nn.CrossEntropyLoss(), val_loss_fn=auc_metric, lr=lr)
    elif type == "MLPClassifierFull":
        model = MLPClassifierFull(mlp, loss_fn=nn.CrossEntropyLoss(), val_loss_fn=auc_metric, lr=lr)

    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, mbsize
