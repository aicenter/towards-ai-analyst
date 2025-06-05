# This code builds upon original DIME implementation.
# Original source: [https://github.com/suinleelab/DIME](https://github.com/suinleelab/DIME).

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, *, idim, hdim, n_layers, dropout, activation):
        super().__init__()
        self.idim = idim
        self.hdim = hdim
        self.n_layers = n_layers
        self.dropout = dropout

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"activation {activation} not supported")

        layers = []
        layers.append(nn.Linear(idim, hdim))
        layers.append(self.activation)
        if dropout != 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hdim, hdim))
            layers.append(self.activation)
            if dropout != 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, idim, hdim, n_layers, dropout, activation):
        super().__init__()
        self.idim = idim
        self.hdim = hdim
        self.n_layers = n_layers
        self.dropout = dropout

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"activation {activation} not supported")

        if self.n_layers < 2:
            raise ValueError(f"n_layers needs to be at least 2, but is only {self.n_layers}")

        layers = []
        layers.append(nn.Linear(idim, hdim))
        layers.append(self.activation)
        if dropout != 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hdim, hdim))
            layers.append(self.activation)
            if dropout != 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hdim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x


class PredictorPrior(nn.Module):
    def __init__(self, prior_encoder, feature_encoder, num_classes=2, hdim=128, dropout=0.1):
        super().__init__()
        self.prior_encoder = prior_encoder
        self.feature_encoder = feature_encoder
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(prior_encoder.hdim + feature_encoder.hdim, hdim)
        self.linear2 = nn.Linear(hdim, num_classes)

    def forward(self, x, prior):
        x = self.feature_encoder(x)
        prior = self.prior_encoder(prior)

        x_cat = torch.cat((x, prior), dim=1)
        x_cat = self.dropout(self.linear1(x_cat).relu())
        x_cat = self.linear2(x_cat)
        return x_cat


class ValueNetworkPrior(nn.Module):
    def __init__(self, prior_encoder, feature_encoder, out_dim, hdim=128, dropout=0.1):
        super().__init__()
        self.prior_encoder = prior_encoder
        self.feature_encoder = feature_encoder
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(prior_encoder.hdim + feature_encoder.hdim, hdim)
        self.linear2 = nn.Linear(hdim, out_dim)

    def forward(self, x, prior):
        x = self.feature_encoder(x)
        prior = self.prior_encoder(prior)

        x_cat = torch.cat((x, prior), dim=1)
        x_cat = self.dropout(self.linear1(x_cat).relu())
        x_cat = self.linear2(x_cat).squeeze()

        return x_cat
