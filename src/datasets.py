import pickle

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src import DATA_DIR

###########################################################################
###                              Amaretto                               ###
###########################################################################


class AmarettoDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        self.datapath = DATA_DIR / "amaretto"
        self.prior = pl.read_parquet(self.datapath / f"{split}_prior.pq")
        self.costly_features = pl.read_parquet(self.datapath / f"{split}_costly_features.pq")

        self.indexes = self.prior["id"].to_numpy()
        self.anomaly = self.prior["Anomaly"].to_numpy()
        self.labels = np.array(
            self.prior["Anomaly"].to_numpy() != 0,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple:
        p = np.array(self.prior.row(index)[3:])
        c = np.array(self.costly_features.row(index)[3:])
        y = self.labels[index]
        return (
            torch.tensor(c, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64),
        )


class AmarettoDataset2F(Dataset):
    """Most important 2 costly features dataset."""

    def __init__(self, split: str = "train") -> None:
        self.datapath = DATA_DIR / "amaretto"
        self.prior = pl.read_parquet(self.datapath / f"{split}_prior.pq")
        self.costly_features = pl.read_parquet(self.datapath / f"{split}_costly_features.pq")

        self.indexes = self.prior["id"].to_numpy()
        self.anomaly = self.prior["Anomaly"].to_numpy()
        self.labels = np.array(
            self.prior["Anomaly"].to_numpy() != 0,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple:
        p = np.array(self.prior.row(index)[3:])
        c = np.array(self.costly_features.row(index)[3:])[[1, 89]]
        y = self.labels[index]
        return (
            torch.tensor(c, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64),
        )


class AmarettoLabels:
    def __init__(self, split: str = "train") -> None:
        self.datapath = DATA_DIR / "amaretto"
        self.prior = pl.read_parquet(self.datapath / f"{split}_prior.pq", columns=["id", "Anomaly"])

        self.indexes = self.prior["id"].to_numpy()
        self.anomaly = self.prior["Anomaly"].to_numpy()
        self.labels = np.array(
            self.prior["Anomaly"].to_numpy() != 0,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.labels)


###########################################################################
###                              Private                                ###
###########################################################################


class CostlyFeaturesDataset(Dataset):
    def __init__(self, costly_features, prior, labels) -> None:
        self.costly_features = costly_features
        self.prior = prior
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple:
        x = self.costly_features[index, :]
        p = self.prior[index, :]
        y = self.labels[index]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(y).long(),
        )


class PrivateDataset:
    def __init__(self) -> None:
        self.datapath = DATA_DIR / "matrix_data.pkl"

        # load dataset and get values
        with open(self.datapath, "rb") as f:
            data = pickle.load(f)

        # prior features
        self._Xp_train = data["X_train_prior"]
        self._Xp_val = data["X_val_prior"]
        self._Xp_test = data["X_test_prior"]
        self._prior_features = data["prior_features"]

        # costly features
        self._Xc_train = data["X_train_costly"]
        self._Xc_val = data["X_val_costly"]
        self._Xc_test = data["X_test_costly"]
        self._costly_features = data["costly_features"]

        # labels
        self._y_train = data["y_train"]
        self._y_val = data["y_val"]
        self._y_test = data["y_test"]

    @property
    def features_count(self):
        return {
            "n_prior": len(self._prior_features),
            "n_costly": len(self._costly_features),
        }

    @property
    def costly_features_names(self):
        return self._costly_features

    @property
    def prior_features_names(self):
        return self._prior_features

    def create_datasets(
        self,
    ) -> tuple[CostlyFeaturesDataset, CostlyFeaturesDataset, CostlyFeaturesDataset]:
        return (
            CostlyFeaturesDataset(
                costly_features=self._Xc_train,
                prior_features=self._Xp_train,
                labels=self._y_train,
            ),
            CostlyFeaturesDataset(
                costly_features=self._Xc_val,
                prior_features=self._Xp_val,
                labels=self._y_val,
            ),
            CostlyFeaturesDataset(
                costly_features=self._Xc_test,
                prior_features=self._Xp_test,
                labels=self._y_test,
            ),
        )
