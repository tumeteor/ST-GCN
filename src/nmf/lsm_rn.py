from torch.nn.parameter import Parameter
from collections import defaultdict
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.graph_utils import partition_graph_by_lonlat
from src.datasets import TensorIterableDataset


class LSM_RN(pl.LightningModule):
    def __init__(self, t_steps, n, k, λ, adj_mat, datasets):
        super().__init__()
        self.datasets = datasets
        self.λ = λ
        self.n = n
        self.k = k
        self.t_steps = k
        self.L = self.get_laplacian(adj_mat)
        self.U = Parameter(torch.Tensor(t_steps, n, k))
        self.B = Parameter(torch.Tensor(k, k))
        # TODO initalize U, B

    def get_laplacian(self, adj_mat):  # TODO sparse
        # degree matrix
        degree_mat = np.diag(np.sum(adj_mat, 1))
        return degree_mat - adj_mat

    def forward(self, t):
        U_t = self.U[t]
        return torch.chain_matmul(U_t, B, U_t.t)

    def loss(U_t, G_hat, G):
        # TODO sparse matrixes
        mask = torch.zeros(G.size())
        mask[G.nonzero()] = 1

        mse = torch.norm(torch.pow(mask * (G_hat - G)), p="fro", dim=(1, 2))
        laplacian_term = np.einsum("ii", torch.chain_matmul(U_t.t, self.L, U_t))

        return mse + self.λ * laplacian_term

    def training_step(self, batch, batch_nb):
        t, G = batch
        G_hat = self.forward(t)
        U_t = self.U[t]

        return {"loss": self.loss(U_t, G_hat, G)}

    def validation_step(self, batch, batch_nb):
        t, G = batch
        G_hat = self.forward(t)
        U_t = self.U[t]

        return {"val_loss": self.loss(U_t, G_hat, G)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(TensorIterableDataset(self.datasets["tng"]), batch_size=8)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(TensorIterableDataset(self.datasets["val"]), batch_size=8)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(TensorIterableDataset(self.datasets["tst"]), batch_size=8)
