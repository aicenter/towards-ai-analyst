import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision

from src import RESULTS_DIR
from src.cmi_estimator_prior import CMIEstimatorPrior
from src.datasets import AmarettoDataset
from src.dime_utils import MaskLayer
from src.models import Encoder, PredictorPrior, ValueNetworkPrior
from utils.hash import generate_id
from utils.samplers import WeightedSampler


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="DIME")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


### Experiment setup
experiment_name = "amaretto/dime"

# fixed parameters
lr = 1e-3
min_lr = 1e-6
max_num_samples = 100_000
dropout = 0


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

    hyperparameters = {
        "hd": hdim,
        "n": n_layers,
        "mb": mbsize,
        "pr": positive_ratio,
        "act": activation,
    }
    experiment_id = generate_id(hyperparameters)
    return (hdim, n_layers, activation, mbsize, positive_ratio), experiment_id


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Make results directory.
    for k in range(20):
        (hdim, n_layers, activation, mbsize, positive_ratio), experiment_id = generate_params(args.seed)
        results_dir = RESULTS_DIR / experiment_name / experiment_id
        if os.path.exists(results_dir):
            continue
        else:
            os.makedirs(results_dir)
            print(f"Experiment set up with experiment_id = {experiment_id}.")
            break

    ## Load datasets

    train_dataset = AmarettoDataset(split="train")
    val_dataset = AmarettoDataset(split="val")
    # test_dataset = AmarettoDataset(split="test")

    ## Get dimensions

    sampler = WeightedSampler(labels=train_dataset.labels, positive_ratio=0.25, max_num_samples=100)
    dl = DataLoader(train_dataset, batch_size=32, sampler=sampler, drop_last=True)

    for batch in dl:
        break

    c, p, y = batch
    n_prior_features = p.shape[1]
    n_costly_features = c.shape[1]

    ## Initialize models, masks and other stuff

    auc_metric = AveragePrecision(task="multiclass", num_classes=2, average="macro")
    mask_layer = MaskLayer(mask_size=n_costly_features)
    device = torch.device("cpu")

    # Models
    prior_encoder = Encoder(
        idim=n_prior_features,
        hdim=hdim,
        dropout=dropout,
        activation=activation,
        n_layers=n_layers,
    )
    feature_encoder = Encoder(
        idim=n_costly_features * 2,
        hdim=hdim,
        dropout=dropout,
        activation=activation,
        n_layers=n_layers,
    )

    predictor = PredictorPrior(prior_encoder, feature_encoder)
    value_network = ValueNetworkPrior(prior_encoder, feature_encoder, out_dim=n_costly_features)

    # For masking unobserved features.
    mask_layer = MaskLayer(mask_size=n_costly_features, append=True)

    # pretrain = MaskingPretrainerPrior(
    #     predictor,
    #     mask_layer,
    #     lr=1e-5,
    #     loss_fn=nn.CrossEntropyLoss(),
    #     val_loss_fn=auc_metric,
    # )

    ### Set-up dataloaders

    # subsampled train dataloader
    sampler = WeightedSampler(labels=train_dataset.labels, positive_ratio=positive_ratio, max_num_samples=max_num_samples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=mbsize,
        sampler=sampler,
        pin_memory=False,
    )

    # subsampled validation dataloader
    # step10_indices = np.arange(0, len(val_dataset), step=10)
    step100_indices = np.arange(0, len(val_dataset), step=100)
    val_subset = torch.utils.data.Subset(val_dataset, step100_indices)
    val_dataloader = DataLoader(
        val_subset,
        batch_size=mbsize,
        pin_memory=False,
    )

    ## Set up the experimental things

    logger = TensorBoardLogger("logs", name=f"{experiment_name}")
    checkpoint_callback = best_hard_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="Perf Val/Mean",
        mode="max",
        filename=f"best_{experiment_id}",
        dirpath=results_dir,
        verbose=False,
    )

    # Jointly train predictor and value networks
    greedy_cmi_estimator = CMIEstimatorPrior(
        value_network,
        predictor,
        mask_layer,
        lr=lr,
        min_lr=min_lr,
        max_features=n_costly_features,
        eps=0.05,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        # val_loss_fn=AUROC(task="multiclass", num_classes=2),
        val_loss_fn=AveragePrecision(task="multiclass", num_classes=2, average="macro"),
        eps_decay=0.2,
        eps_steps=10,
        patience=30,
        feature_costs=None,
    )

    ## Training

    # pretrain
    # trainer = Trainer(
    #     max_epochs=10,
    #     num_sanity_val_steps=2,
    # )
    # trainer.fit(pretrain, train_dataloader, val_dataloader)

    # costly features training
    trainer = Trainer(
        max_epochs=100,
        precision=32,
        logger=logger,
        num_sanity_val_steps=10,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(greedy_cmi_estimator, train_dataloader, val_dataloader)
