import os

import cloudpickle
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision
from tqdm import tqdm

from src import RESULTS_DIR
from src.datasets import AmarettoDataset
from src.models import MLP
from src.nn_classifier import MLPClassifierFull, MLPClassifierPrior
from utils.evaluation import f1_smart
from utils.hash import decode_id

val_dataset = AmarettoDataset(split="val")
test_dataset = AmarettoDataset(split="test")


def load_model(filepath: str, n_prior_features, n_costly_features: int = 0, type: str = "MLPClassifierPrior"):
    filename = filepath.split("/")[-1][5:-5]
    hyperparameters = decode_id(filename)

    hdim = hyperparameters["hd"]
    n_layers = hyperparameters["n"]
    mbsize = hyperparameters["mb"]
    positive_ratio = hyperparameters["pr"]
    activation = hyperparameters["act"]
    dropout = hyperparameters["dr"]

    mlp = MLP(idim=n_prior_features + n_costly_features, hdim=hdim, n_layers=n_layers, dropout=dropout, activation=activation)

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


path = RESULTS_DIR / "amaretto/nn_classifier_full"
path_list = os.listdir(path)

for p in path_list:
    checkpoint_path = f"{path}/{p}/best_{p}.ckpt"

    for batch in DataLoader(val_dataset):
        break

    c, prior, y = batch
    n_prior_features = prior.shape[1]
    n_costly_features = c.shape[1]

    # load checkpoint from file
    model, mbsize = load_model(checkpoint_path, n_prior_features, n_costly_features, "MLPClassifierFull")

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

    # evaluation
    model.eval()

    val_probabilities = []
    with torch.no_grad():
        for c, prior, y in tqdm(val_dataloader):
            x = torch.cat([prior, c], dim=1)
            out = model(x)
            probs = nn.Softmax(dim=1)(out)[:, 1].numpy()
            val_probabilities.extend(probs)

    test_probabilities = []
    with torch.no_grad():
        for c, prior, y in tqdm(test_dataloader):
            x = torch.cat([prior, c], dim=1)
            out = model(x)
            probs = nn.Softmax(dim=1)(out)[:, 1].numpy()
            test_probabilities.extend(probs)

    val_f1, thr = f1_smart(val_dataset.labels, val_probabilities)
    test_f1 = f1_score(test_dataset.labels, test_probabilities >= thr)

    print("validation F1:", val_f1)
    print("test F1:", test_f1)

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
# %%
