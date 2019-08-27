from torch.nn.parameter import Parameter
from collections import defaultdict
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.graph_utils import partition_graph_by_lonlat
from src.utils.memory import mem_report

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
        torch.nn.init.uniform_(self.U, 0, 0.2)
        torch.nn.init.uniform_(self.B, 0, 0.2)

    def get_laplacian(self, adj_mat):  # TODO sparse
        # degree matrix
        degree_mat = np.diag(np.sum(adj_mat, 0).A1)
        return torch.Tensor(degree_mat - adj_mat)

    def forward(self, t):
        mem_report()
        U_t = self.U[t]
        # chain_matmul does not support batched
        return U_t.matmul(self.B).matmul(U_t.transpose(1, 2))

    def loss(self, U_t, G_hat, G):
        # TODO sparse matrixes
        mask = torch.zeros(G.size())
        mask[G != 0] = 1

        fro = torch.norm(mask * (G_hat - G), p="fro", dim=(1, 2)).mean()
        laplacian_term = (
            self.λ
            * torch.einsum(
                "...ii", U_t.transpose(1, 2).matmul(self.L).matmul(U_t)
            ).mean()
        )

        return {
            "fro": fro,
            "laplacian_term": laplacian_term,
            "loss": fro + laplacian_term,
        }

    def training_step(self, batch, batch_nb):
        t, G = batch
        G_hat = self.forward(t)
        U_t = self.U[t]

        return self.loss(U_t, G_hat, G)

    def validation_step(self, batch, batch_nb):
        t, G = batch
        G_hat = self.forward(t)
        U_t = self.U[t]

        mask = torch.zeros(G.size())
        mask[G != 0] = 1

        mae = (mask * (G_hat - G)).abs().sum() / mask.sum()

        metrics = self.loss(U_t, G_hat, G)

        #  print("gt", G[G != 0][:10])
        #  print("pred", G_hat[G != 0][:10])
        return {
            "val_mae": mae,
            "val_loss": metrics["loss"],
            "val_fro": metrics["fro"],
            "val_laplacian_term": metrics["laplacian_term"],
        }

    def validation_end(self, outputs):
        # TODO refactor
        return {
            "avg_val_mae": sum([o["val_mae"] for o in outputs]) / len(outputs),
            "avg_val_loss": sum([o["val_loss"] for o in outputs]) / len(outputs),
            "avg_val_fro": sum([o["val_fro"] for o in outputs]) / len(outputs),
            "avg_laplacian_term": sum([o["val_laplacian_term"] for o in outputs])
            / len(outputs),
        }

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.0001)]

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
