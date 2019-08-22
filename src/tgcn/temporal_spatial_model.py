from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.sparse as sp
from src.utils.sparse import sparse_scipy2torch
from src.tgcn.layers.lstmcell import GCLSTMCell

torch.manual_seed(0)


class TGCN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, adj, adj_norm=False,
                 datasets=None, targets=None, dropout=0.5, mask=None):
        super(TGCN, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gc_lstm = GCLSTMCell(input_dim, hidden_dim, adj=adj)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.datasets = datasets
        self.adj = self._transform_adj(adj) if adj_norm else adj
        self.dropout = nn.Dropout(dropout)
        self.targets = targets

        self.mask = mask

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

    def forward(self, x):
        # shape x: [batch_size, seq_length, input_dim(features)] 6163, 9, 27
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        cn = c0[0, :, :]
        hn = h0[0, :, :]
        x = torch.squeeze(x, dim=0)
        for seq in range(x.size(2)):
            hn, cn = self.gc_lstm(x=x[:, :, seq].float(), hx=hn, cx=cn)
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        # out.size() --> 100, 10
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y = y.squeeze(dim=0)
        y_hat = self.forward(x).squeeze(dim=0)

        _mask = self.mask['train'][batch_nb, :, :]
        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        return {'loss': F.mse_loss(y_hat, y.float())}

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
        x, y = batch
        y = y.squeeze(dim=0)

        y_hat = self.forward(x).squeeze(dim=0)
        _mask = self.mask['valid'][batch_nb, :, :]

        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        print(f"y_hat: {y_hat}")
        mae = (y_hat - y.float()).abs().sum() / _mask.sum()
        return {'val_mae': mae}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_mae'] for x in outputs]).mean()
        return {'avg_val_mae': avg_loss}
