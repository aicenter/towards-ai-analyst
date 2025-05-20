import argparse
import os

import cloudpickle
import torch
import torch.nn as nn
from src import RESULTS_DIR
from utils.evaluation import f1_smart
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import AmarettoDataset
from src.nn_classifier import load_model


# Choose the model with parameter
def parse_args():
    parser = argparse.ArgumentParser(description="Results for NN classifiers")
    parser.add_argument("--model", type=str, default="prior", help="Classifier model: either `prior` of `full`")
    return parser.parse_args()


args = parse_args()
model_type = args.model
model_name = "MLPClassifierPrior" if model_type == "prior" else "MLPClassifierFull"

# Load data
val_dataset = AmarettoDataset(split="val")
test_dataset = AmarettoDataset(split="test")

path = RESULTS_DIR / f"amaretto/nn_classifier_{model_type}"
print(path)
path_list = os.listdir(path)

for p in path_list:
    # For each model, calculate the results and save.

    checkpoint_path = f"{path}/{p}/best_{p}.ckpt"

    for batch in DataLoader(val_dataset):
        break

    c, prior, y = batch
    n_prior_features = prior.shape[1]
    n_costly_features = c.shape[1]

    model, mbsize = load_model(checkpoint_path, n_prior_features, n_costly_features, type=model_name)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=mbsize,
        pin_memory=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=mbsize,
        pin_memory=False,
    )

    model.eval()

    val_probabilities = []
    test_probabilities = []

    if model_type == "prior":
        with torch.no_grad():
            for c, prior, y in tqdm(val_dataloader):
                out = model(prior)
                probs = nn.Softmax(dim=1)(out)[:, 1].numpy()
                val_probabilities.extend(probs)

        with torch.no_grad():
            for c, prior, y in tqdm(test_dataloader):
                out = model(prior)
                probs = nn.Softmax(dim=1)(out)[:, 1].numpy()
                test_probabilities.extend(probs)

    else:
        with torch.no_grad():
            for c, prior, y in tqdm(val_dataloader):
                x = torch.cat([prior, c], dim=1)
                out = model(x)
                probs = nn.Softmax(dim=1)(out)[:, 1].numpy()
                val_probabilities.extend(probs)

        with torch.no_grad():
            for c, prior, y in tqdm(test_dataloader):
                x = torch.cat([prior, c], dim=1)
                out = model(x)
                probs = nn.Softmax(dim=1)(out)[:, 1].numpy()
                test_probabilities.extend(probs)

    # calculate the results
    val_f1, thr = f1_smart(val_dataset.labels, val_probabilities)
    test_f1 = f1_score(test_dataset.labels, test_probabilities >= thr)

    print("validation F1:", val_f1)
    print("test F1:", test_f1)

    # save everything to the correct path
    with open(f"{path}/{p}/probabilities.pkl", "wb") as f:
        cloudpickle.dump(
            {
                "thr": thr,
                "val_f1": val_f1,
                "test_f1": test_f1,
                "val_probabilities": val_probabilities,
                "test_probabilities": test_probabilities,
            },
            f,
        )

    # print(f'Results saved to {path}/{p}/probabilities.pkl')
