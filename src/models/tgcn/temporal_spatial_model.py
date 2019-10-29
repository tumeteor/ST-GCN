from torch.utils.data import DataLoader
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.sparse as sp
import sys
import logging
from src.data_loader.tensor_dataset import GraphTensorDataset
from src.modules.layers.lstmcell import GCLSTMCell
from src.utils.ops import clear_parameters
from src.utils.sparse import sparse_scipy2torch, dense_to_sparse
from src.metrics.measures import rmse, smape
from src.configs.configs import TGCN as TGCNConfig
from src.configs.configs import Data as DataConfig

torch.manual_seed(0)
np.random.seed(0)


class TGCN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, adjs, adj_norm=True,
                 datasets=None, cluster_idx_ids=None, dropout=0.5, device=None):
        super(TGCN, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gc_lstm = GCLSTMCell(input_dim, hidden_dim, dropout).to(device)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.datasets = datasets
        self.adjs = [self._transform_adj(_adj) for _adj in adjs] if adj_norm else adjs
        self.cluster_idx_ids = cluster_idx_ids

        self.opt = torch.optim.Adam(self.parameters(), lr=TGCNConfig.lr, weight_decay=TGCNConfig.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=TGCNConfig.lr_scheduler_patience, verbose=True
        )
        self.device = device

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
        h0 = nn.Parameter(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        # Initialize cell state
        c0 = nn.Parameter(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)

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
        batch = [b.to(self.device) for b in batch]
        x, y, adj, mask = batch
        # squeeze sparse tensor using sum
        adj = torch.sparse.sum(adj, dim=0).float()
        # x: torch.Size([1, 6163, 29, 9])
        # y: torch.Size([1, 6163, 1])
        x = x.squeeze(dim=0)
        y = y.squeeze(dim=0)
        x = x.permute(0, 2, 1)
        try:
            y_hat = self.forward(x, adj).squeeze(dim=0)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logging.warning('| WARNING: ran out of memory, retrying batch', sys.stdout)
                clear_parameters(self)
                y_hat = self.forward(x, adj).squeeze(dim=0)
            else:
                raise e

        y_hat = y_hat.masked_select(mask.bool())
        y = y.masked_select(mask.bool())
        # use l1 loss
        # NOTE: need to ignore batch when mask is full (all missing values)
        # batch iteration is currently handled by pytorch-lightning
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        return {'loss': loss_fn(y_hat, y.float())}

    def on_batch_start(self, batch):
        """
        We want to skip batch with all NaNs. Return -1 will skip the batch
        Args:
            batch:

        Returns:
            int:

        """
        if batch[-1].sum().item() == 0:
            return -1

    @pl.data_loader
    def train_dataloader(self):
        ds = GraphTensorDataset(self.datasets, adj_list=self.adjs,
                                mode='train',
                                cluster_idx_ids=self.cluster_idx_ids,
                                time_steps=DataConfig.train_num_steps)
        # increase the shared memory for the docker container to use more workers for prefetching
        # otherwise use single
        # batch_size in the logic of DataLoader needs to be equal to the number of gpus in our setting
        # NOTE: the implicit (variable-length) batch size is the number of nodes in the subgraph
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    @pl.data_loader
    def val_dataloader(self):
        ds = GraphTensorDataset(self.datasets, adj_list=self.adjs,
                                mode='valid',
                                cluster_idx_ids=self.cluster_idx_ids,
                                time_steps=DataConfig.valid_num_steps)
        # increase the shared memory for the docker container to use more workers for prefetching
        # otherwise use single thread
        # batch_size in the logic of DataLoader needs to be equal to the number of gpus in our setting
        # NOTE: the implicit (variable-length) batch size is the number of nodes in the subgraph
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    @pl.data_loader
    def test_dataloader(self):
        ds = GraphTensorDataset(self.datasets, adj_list=self.adjs,
                                mode='test',
                                cluster_idx_ids=self.cluster_idx_ids,
                                time_steps=DataConfig.test_num_steps)
        # batch_size in the logic of DataLoader needs to be equal to the number of gpus in our setting
        # NOTE: the implicit (variable-length) batch size is the number of nodes in the subgraph
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    def validation_step(self, batch, batch_nb):
        batch = [b.to(self.device) for b in batch]
        # batch shape: torch.Size([1, num_nodes, num_features, look_back])
        x, y, adj, mask = batch
        # squeeze sparse tensor using sum
        adj = torch.sparse.sum(adj, dim=0).float()
        x = x.squeeze(dim=0)
        y = y.squeeze(dim=0).float()
        x = x.permute(0, 2, 1)

        try:
            y_hat = self.forward(x, adj).squeeze(dim=0)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, retrying batch', sys.stdout)
                clear_parameters(self)
                y_hat = self.forward(x, adj).squeeze(dim=0)
            else:
                raise e

        y_hat = y_hat.masked_select(mask.bool())
        y = y.masked_select(mask.bool())
        logging.debug(f"y_hat: {y_hat}")
        logging.debug(f"y: {y}")
        # convert to np.array for inverse transformation
        # y_hat = scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
        # y = scaler.inverse_transform(np.array(y).reshape(-1, 1))
        no_gt = False
        if mask.sum().item() == 0:
            no_gt = True
        _mae = torch.abs(y_hat - y).mean().float()
        _rmse = torch.FloatTensor([rmse(actual=y.cpu().numpy(), predicted=y_hat.cpu().numpy())])
        _smape = torch.FloatTensor([smape(actual=y.cpu().numpy(), predicted=y_hat.cpu().numpy())])
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
