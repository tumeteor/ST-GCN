from torch import optim

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import scipy.sparse as sp

from torch.utils.data import DataLoader, TensorDataset

from src.metrics.measures import rmse, smape
from src.modules.layers.block import STGCNBlock, TimeBlock
from src.utils.sparse import sparse_scipy2torch


class STGCN(pl.LightningModule):
    """
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes=6163, num_features=29, num_timesteps_input=9,
                 num_timesteps_output=1, adj=None, datasets=None, targets=None, mask=None, normalized=False, scaler=None):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear(num_timesteps_input * 64,
                               num_timesteps_output)

        self.datasets = datasets
        self.targets = targets
        self.mask = mask

        self.adj = self._transform_adj(adj) if normalized else adj
        self.scaler = scaler

    def _transform_adj(self, adj):
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = self._normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_scipy2torch(adj)
        return adj

    def _normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        print(f"Out3 shape: {out3.shape}")
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-3)

    def training_step(self, batch, batch_nb):
        # (batch_size, num_nodes, num_input_time_steps, num_features)
        # [batch_size, seq_length, input_dim(features)] 1, 6163, 9, 27
        x, y = batch
        x = x.permute(0, 1, 3, 2).float()

        y = y.squeeze(dim=0)
        y_hat = self.forward(A_hat=self.adj, X=x).squeeze(dim=0)

        _mask = self.mask['train'][batch_nb, :, :]
        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        return {'loss': torch.sum(torch.abs(y_hat - y.float()))}

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['train'], self.targets['train']), batch_size=1, shuffle=False)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['valid'], self.targets['valid']), batch_size=1, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['test'], self.targets['test']), batch_size=1, shuffle=False)

    def validation_step(self, batch, batch_nb):
        # batch shape: torch.Size([1, 6163, 26, 10])
        # [batch_size, seq_length, input_dim(features)] 1, 6163, 9, 27
        x, y = batch
        x = x.permute(0, 1, 3, 2).float()

        y = y.squeeze(dim=0)

        y_hat = self.forward(A_hat=self.adj, X=x).squeeze(dim=0)
        _mask = self.mask['valid'][batch_nb, :, :]

        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        # convert to np.array for inverse transformation
        y_hat = self.scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
        y = self.scaler.inverse_transform(np.array(y).reshape(-1, 1))

        _mae = torch.FloatTensor(np.abs(y_hat - y)).sum() / _mask.sum()
        _rmse = torch.FloatTensor(rmse(actual=y, predicted=y_hat))
        _smape = torch.FloatTensor(smape(actual=y, predicted=y_hat))
        return {'val_mae': _mae,
                'val_rmse': _rmse,
                'val_smape': _smape}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_mae_loss = torch.stack([x['val_mae'] for x in outputs]).mean()
        avg_rmse_loss = torch.stack([x['val_rmse'] for x in outputs]).mean()
        avg_smape_loss = torch.stack([x['val_smape'] for x in outputs]).mean()
        return {'avg_val_mae': avg_mae_loss,
                'avg_rmse_loss': avg_rmse_loss,
                'avg_smape_loss': avg_smape_loss
                }

