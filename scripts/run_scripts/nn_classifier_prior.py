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
from src.datasets import AmarettoDataset
from src.models import MLP
from src.nn_classifier import MLPClassifierPrior, generate_params
from utils.samplers import WeightedSampler


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Classifier with Prior')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return parser.parse_args()


### Experiment setup
experiment_name = 'amaretto/nn_classifier_prior'

# fixed parameters
lr = 1e-3
min_lr = 1e-6
max_num_samples = 100_000

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Make results directory.
    (hdim, n_layers, activation, mbsize, positive_ratio, dropout), experiment_id = generate_params(args.seed)
    results_dir = RESULTS_DIR / experiment_name / experiment_id
    if os.path.exists(results_dir):
        raise RuntimeError('Could not find correct hyperparameter combination for maximum number of tries. Abort operation.')
    else:
        os.makedirs(results_dir)
        print(f"Experiment set up with experiment_id = {experiment_id}.")

    ## Load datasets

    train_dataset = AmarettoDataset(split="train")
    val_dataset = AmarettoDataset(split="val")
    # test_dataset = AmarettoDataset(split="test")
    print('Data are loaded.')

    ## Get dimensions

    sampler = WeightedSampler(labels=train_dataset.labels, positive_ratio=0.5, max_num_samples=100)
    dl = DataLoader(train_dataset, batch_size=32, sampler=sampler, drop_last=True)

    for batch in dl:
        break

    c, p, y = batch
    n_prior_features = p.shape[1]
    n_costly_features = c.shape[1]

    ## Initialize models, masks and other stuff

    auc_metric = AveragePrecision(task="multiclass", num_classes=2, average="macro")
    device = torch.device("cpu")

    ### Set-up dataloaders

    # subsampled train dataloader
    sampler = WeightedSampler(
        labels=train_dataset.labels,
        positive_ratio=positive_ratio,
        max_num_samples=max_num_samples
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=mbsize,
        sampler=sampler,
        pin_memory=False,
    )

    # subsampled validation dataloader
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

    # Models
    mlp = MLP(
        idim=n_prior_features,
        hdim=hdim,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation
    )
    model = MLPClassifierPrior(mlp, loss_fn=nn.CrossEntropyLoss(), val_loss_fn=auc_metric, lr=lr)

    ## Training
    trainer = Trainer(
        max_epochs=100,
        precision=32,
        logger=logger,
        num_sanity_val_steps=10,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
