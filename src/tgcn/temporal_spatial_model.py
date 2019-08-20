from torch.autograd import Variable
from torch.utils.data import DataLoader
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
                 datasets=None, dropout=0.5, indices=None):
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

        self.indices = torch.LongTensor(indices)

    def _transform_adj(self, adj):
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = self._normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_scipy2torch(adj)

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
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        x, y = batch[:, :, :, :9], batch[:, :, 0, 9:]

        y_hat = self.forward(x).squeeze()

        return {'loss': F.mse_loss(torch.index_select(y_hat, dim=0, index=self.indices),
                                   torch.index_select(y, dim=0, index=self.indices).float())}

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=1, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.datasets['valid'], batch_size=1, shuffle=True)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=1, shuffle=True)

    def validation_step(self, batch, batch_nb):
        x, y = batch[:, :, :, :9], batch[:, :, 0, 9:]
        y = y.squeeze(dim=0)
        print(f"y: {y.shape}")
        y_hat = self.forward(x).squeeze()
        print(f"y_hat: {y_hat.shape}")
        print(self.indices)
        return {'val_loss': F.mse_loss(torch.index_select(y_hat, dim=0, index=self.indices),
                                       torch.index_select(y, dim=0, index=self.indices).float())}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
