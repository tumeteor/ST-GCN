from torch.utils.data import DataLoader
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.sparse as sp

from src.data_loader.tensor_dataset import CustomTensorDataset
from src.modules.layers.lstmcell import GCLSTMCell
from src.utils.sparse import sparse_scipy2torch, dense_to_sparse
from src.metrics.measures import rmse, smape

torch.manual_seed(0)


class TGCN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, adjs, adj_norm=True,
                 datasets=None, cluster_idx_ids=None, dropout=0.5):
        super(TGCN, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gc_lstm = GCLSTMCell(input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.datasets = datasets
        self.adjs = [self._transform_adj(_adj) for _adj in adjs] if adj_norm else adjs
        self.cluster_idx_ids = cluster_idx_ids
        self.dropout = nn.Dropout(dropout)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.015)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=3, verbose=True
        )

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

    def forward(self, x, adj):
        # shape x: [batch_size, seq_length, input_dim(features)] 6163, 9, 27
        if torch.cuda.is_available():
            h0 = nn.Parameter(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = nn.Parameter(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = nn.Parameter(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = nn.Parameter(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        for i in range(self.layer_dim):
            cn = c0[i, :, :]
            hn = h0[i, :, :]
            x = torch.squeeze(x, dim=0)
            for seq in range(x.size(2)):
                hn, cn = self.gc_lstm(x=x[:, :, seq].float(), hx=hn, cx=cn, adj=adj)

                outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        # out.size() --> 100, 10
        return out

    def configure_optimizers(self):
        return [self.opt]

    def training_step(self, batch, batch_nb):
        x, y, adj, mask = batch
        adj = dense_to_sparse(adj.to_dense().squeeze(dim=0))
        # x: torch.Size([1, 6163, 29, 9])
        # y: torch.Size([1, 6163, 1])
        x = x.squeeze(dim=0)
        y = y.squeeze(dim=0)
        x = x.permute(0, 2, 1)
        y_hat = self.forward(x, adj).squeeze(dim=0)

        y_hat = y_hat.masked_select(mask.bool())
        y = y.masked_select(mask.bool())
        # use l1 loss
        return {'loss': torch.sum(torch.abs(y_hat - y.float()))}

    @pl.data_loader
    def tng_dataloader(self):
        ds = CustomTensorDataset(self.datasets, adj_list=self.adjs,
                                 mode='train',
                                 cluster_idx_ids=self.cluster_idx_ids,
                                 time_steps=251)
        return DataLoader(ds, batch_size=1, shuffle=False)

    @pl.data_loader
    def val_dataloader(self):
        ds = CustomTensorDataset(self.datasets, adj_list=self.adjs,
                                 mode='valid',
                                 cluster_idx_ids=self.cluster_idx_ids,
                                 time_steps=51)
        return DataLoader(ds, batch_size=1, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        ds = CustomTensorDataset(self.datasets, adj_list=self.adjs,
                                 mode='test',
                                 cluster_idx_ids=self.cluster_idx_ids,
                                 time_steps=11)
        return DataLoader(ds, batch_size=1, shuffle=False)

    def validation_step(self, batch, batch_nb):
        # batch shape: torch.Size([1, 6163, 26, 10])
        print(len(batch))
        x, y, adj, mask = batch
        print(x.shape)
        print(y.shape)
        print(adj.shape)
        print(mask.shape)

        adj = dense_to_sparse(adj.to_dense().squeeze(dim=0))
        x = x.squeeze(dim=0)
        y = y.squeeze(dim=0).float()
        x = x.permute(0, 2, 1)

        y_hat = self.forward(x, adj).squeeze(dim=0)
        y_hat = y_hat.masked_select(mask.bool())
        y = y.masked_select(mask.bool())
        print(f"y_hat: {y_hat}")
        print(f"y: {y}")
        # convert to np.array for inverse transformation
        # y_hat = scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
        # y = scaler.inverse_transform(np.array(y).reshape(-1, 1))
        no_gt = False
        if mask.sum().item() == 0:
            no_gt = True
        _mae = torch.FloatTensor(np.abs(y_hat - y)).sum() / mask.sum()
        _rmse = torch.FloatTensor([rmse(actual=y.numpy(), predicted=y_hat.numpy())])
        _smape = torch.FloatTensor([smape(actual=y.numpy(), predicted=y_hat.numpy())])
        _no_gt = torch.BoolTensor([no_gt])
        return {'val_mae': _mae,
                'val_rmse': _rmse,
                'val_smape': _smape,
                'val_no_gt': _no_gt}

    def validation_end(self, outputs):
        # OPTIONAL
        outputs = [x for x in outputs if not x['val_no_gt'].item()]
        if len(outputs) == 0:
            return {'avg_val_mae': torch.FloatTensor([-1]),
                    'avg_rmse_loss':torch.FloatTensor([-1]),
                    'avg_smape_loss': torch.FloatTensor([-1])
                    }
        avg_mae_loss = torch.stack([x['val_mae'].float() for x in outputs]).mean()
        avg_rmse_loss = torch.stack([x['val_rmse'].float() for x in outputs]).mean()
        avg_smape_loss = torch.stack([x['val_smape'].float() for x in outputs]).mean()
        return {'avg_val_mae': avg_mae_loss,
                'avg_rmse_loss': avg_rmse_loss,
                'avg_smape_loss': avg_smape_loss
                }
