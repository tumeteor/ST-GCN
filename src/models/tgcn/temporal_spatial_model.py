from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Sampler, Subset
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.sparse as sp
from src.utils.sparse import sparse_scipy2torch
from src.models.tgcn.layers.lstmcell import GCLSTMCell
from src.metrics.measures import rmse, smape

torch.manual_seed(0)


class CustomSampler(Sampler):

    def __init__(self, data_source, cum_indices, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.cum_indices = [0] + cum_indices
        self.shuffle = shuffle

    def __iter__(self):
        batch = []
        for prev, curr in zip(self.cum_indices, self.cum_indices[1:]):
            for idx in range(prev, curr):
                batch.append(idx)
            yield batch
            batch = []

    def __len__(self):
        return len(self.data_source)

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


class CustomTensorDataset(TensorDataset):
    def __init__(self, *tensors, adj_tensor):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.adj_tensor = adj_tensor

    def __getitem__(self, index):
        return tuple((tensor[index], self.adj_tensor) for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class TGCN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, adjs, adj_norm=False,
                 datasets=None, targets=None, dropout=0.5, mask=None, scaler=None):
        super(TGCN, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gc_lstm = GCLSTMCell(input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.datasets = datasets
        self.adjs = [self._transform_adj(_adj) for _adj in adjs] if adj_norm else adjs
        self.dropout = nn.Dropout(dropout)
        self.targets = targets
        self.scaler = scaler

        self.mask = mask

        self.opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.015)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=3, verbose=True
        )

        self.a = torch.split(self.datasets['valid'], 4000, dim=1)[0].permute(1, 0, 2, 3)
        self.b = torch.split(self.targets['valid'], 4000, dim=1)[0].permute(1, 0, 2)
        self.c = torch.split(self.mask['valid'], 4000, dim=1)[0]


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
        print(f"batch length: {len(batch)}")
        (x, adj), (y, _) = batch
        adj = to_sparse(adj.to_dense().squeeze(dim=0))
        # x: torch.Size([1, 6163, 29, 9])
        # y: torch.Size([1, 6163, 1])

        y = y.squeeze(dim=0)
        y_hat = self.forward(x, adj).squeeze(dim=0)

        _mask = self.mask['train'][batch_nb, :, :] if batch_nb < 91 else self.c[batch_nb-91, :, :]
        print(f"CCCC: {y_hat.shape}, {_mask.shape}")
        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)
        # use l1 loss
        return {'loss': torch.sum(torch.abs(y_hat - y.float()))}

    @pl.data_loader
    def tng_dataloader(self):
        datasets = ConcatDataset(
            [TensorDataset(self.datasets['train'].permute(1, 0, 2, 3), self.targets['train'].permute(1, 0, 2)),
             TensorDataset(self.a, self.b)])

        dl = DataLoader(datasets, batch_sampler=CustomSampler(datasets, datasets.cumulative_sizes), shuffle=False)
        batches = list()
        for batch_nb, batch in enumerate(dl):
            x, y = batch
            print(f"AAA: {type(self.adjs[batch_nb])}, batch_nb: {batch_nb}")
            print(f'batch_nb: {batch_nb}, adj: {self.adjs[batch_nb].shape}')
            x = x.permute(1, 0, 2, 3)
            y = y.permute(1, 0, 2)
            print(f'BBB: {y.shape}, {x.shape}')
            batches.append(CustomTensorDataset(x, y, adj_tensor=self.adjs[batch_nb]))
        return DataLoader(ConcatDataset(batches), batch_size=1, shuffle=False)
        # return DataLoader(TensorDataset(self.datasets['train'], self.targets['train']), batch_size=1, shuffle=False)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['valid'], self.targets['valid']), batch_size=1, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['test'], self.targets['test']), batch_size=1, shuffle=False)

    def validation_step(self, batch, batch_nb):
        # batch shape: torch.Size([1, 6163, 26, 10])
        x, y = batch
        y = y.squeeze(dim=0)

        y_hat = self.forward(x, self.adjs[0]).squeeze(dim=0)
        _mask = self.mask['valid'][batch_nb, :, :]

        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        # convert to np.array for inverse transformation
        y_hat = self.scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
        y = self.scaler.inverse_transform(np.array(y).reshape(-1, 1))

        _mae = torch.FloatTensor(np.abs(y_hat - y)).sum() / _mask.sum()
        _rmse = torch.FloatTensor([rmse(actual=y, predicted=y_hat)])
        _smape = torch.FloatTensor([smape(actual=y, predicted=y_hat)])
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
