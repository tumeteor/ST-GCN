from torch.nn.parameter import Parameter
from collections import defaultdict
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.graph_utils import partition_graph_by_lonlat

torch.manual_seed(0)


class LSM_RN(pl.LightningModule):
    def __init__(self, t_steps, n, k, λ, adj_mat, datasets, batch_size=8):
        super(LSM_RN, self).__init__()

        self.n = n
        self.k = k
        self.λ = λ
        self.datasets = datasets
        self.batch_size = batch_size
        self.t_steps = k
        self.L = self.get_laplacian(adj_mat)
        self.U = Parameter(torch.Tensor(t_steps, n, k))
        self.B = Parameter(torch.Tensor(k, k))
        torch.nn.init.normal_(self.U, mean=0, std=0.2)
        torch.nn.init.normal_(self.B, mean=0, std=0.2)

    def get_laplacian(self, adj_mat):  # TODO sparse
        # degree matrix
        degree_mat = np.diag(np.sum(adj_mat, 1))
        return torch.Tensor(degree_mat - adj_mat)

    def forward(self, t):
        U_t = self.U[t]
        # chain_matmul does not support batched
        return U_t.matmul(self.B).matmul(U_t.transpose(1, 2))

    def loss(self, U_t, G_hat, G):
        # TODO sparse matrixes
        mask = torch.zeros(G.size())
        mask[G != 0] = 1

        fro = torch.norm(mask * (G_hat - G), p="fro", dim=(1, 2))
        laplacian_term = torch.einsum(
            "...ii", U_t.transpose(1, 2).matmul(self.L).matmul(U_t)
        )

        return (fro + self.λ * laplacian_term).mean()

    def training_step(self, batch, batch_nb):
        t, G = batch
        G_hat = self.forward(t)
        U_t = self.U[t]

        return {"loss": self.loss(U_t, G_hat, G)}

    def validation_step(self, batch, batch_nb):
        t, G = batch
        G_hat = self.forward(t)

        mask = torch.zeros(G.size())
        mask[G != 0] = 1

        mae = (mask * (G_hat - G)).abs().sum() / mask.sum()

        return {"val_mae": mae}

    def validation_end(self, outputs):
        avg_mae = sum([o["val_mae"] for o in outputs]) / len(outputs)
        return {"avg_val_mae": avg_mae}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def _dataloader_from_tensor(self, t):
        return DataLoader(
            torch.utils.data.TensorDataset(torch.arange(0, len(t)), t),
            shuffle=True,
            batch_size=self.batch_size,
        )

    @pl.data_loader
    def tng_dataloader(self):
        return self._dataloader_from_tensor(self.datasets["tng"])

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader_from_tensor(self.datasets["val"])

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader_from_tensor(self.datasets["tst"])
