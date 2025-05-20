import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision

from src import RESULTS_DIR
from src.cmi_estimator_prior import CMIEstimatorPrior
from src.datasets import AmarettoDataset
from src.dime_utils import MaskLayer
from src.models import Encoder, PredictorPrior, ValueNetworkPrior
from utils.hash import decode_id
from utils.samplers import WeightedSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Results")
    parser.add_argument("--index", type=int, default=None, help="Index of the file.")
    return parser.parse_args()

args = parse_args()
index = args.index

for split in ['val', 'test']:
    dataset = AmarettoDataset(split=split)
    sampler = WeightedSampler(labels=dataset.labels, positive_ratio=0.25, max_num_samples=100)
    dl = DataLoader(dataset, batch_size=32, sampler=sampler, drop_last=True)

    for batch in dl:
        break

    c, prior, y = batch
    n_prior_features = prior.shape[1]
    n_costly_features = c.shape[1]

    path = RESULTS_DIR / "amaretto/dime_old"

    path_list = os.listdir(path)
    p = path_list[index]

    checkpoint_path = f"{path}/{p}/best_{p}.ckpt"

    # extract hyperparameters
    hyperparameters = decode_id(p)
    hdim = hyperparameters["hd"]
    n_layers = hyperparameters["n"]
    activation = hyperparameters["act"]
    mbsize = hyperparameters["mb"]
    positive_ratio = hyperparameters["pr"]

    dropout = 0
    lr = 1e-3
    min_lr = 1e-6

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
    mask_layer = MaskLayer(mask_size=n_costly_features, append=True)

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
        val_loss_fn=AveragePrecision(task="multiclass", num_classes=2, average="macro"),
        eps_decay=0.2,
        eps_steps=10,
        patience=30,
        feature_costs=None,
    )
    checkpoint = torch.load(checkpoint_path)
    greedy_cmi_estimator.load_state_dict(checkpoint["state_dict"])
   
    # Trainer
    trainer = Trainer(
        max_epochs=10,
        precision=32,
        num_sanity_val_steps=10,
        # no logging, no progress bar
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=0,
    )

    dataloader = DataLoader(dataset, batch_size=mbsize)
    results_dict = greedy_cmi_estimator.inference(
        trainer, dataloader,
        feature_costs=None, budget=n_costly_features, only_intermediate=True
    )
    
    np.savez_compressed(
        f"{path}/{p}/{split}.npz",
        confidences=results_dict['confidences'],
        probabilities=results_dict['probabilities'],
        queried_features=results_dict['queried_features']
    )
        
    # delete some stuff to free up some space
    del results_dict, dataset, dataloader
    import gc
    gc.collect()
    
    print(f'Data for {split} is saved.')

