from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F

import pytorch_lightning as pl
import torch
torch.manual_seed(0)
from src.tgcn.layers.gcn import GCN
from src.tgcn.layers.lstm import LSTMs


class TGCN(pl.LightningModule):
    def __init__(self, gcn_in, gcn_out, adj, dropout=0.5, datasets=None):
        super(TGCN, self).__init__()

        self.net = GCN(in_feats=6163, hidden_size=32, out_feats=6163, dropout=dropout)
        self.model = LSTMs(gcn_out, hidden_dim=32, layers=1, dropout=dropout)
        self.datasets = datasets
        self.adj = adj

    def forward(self, emb):
        gcn_emb = self.net(x=emb, adj=self.adj)
        _, x = self.model(gcn_emb)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        x, y = batch[:, :, :, 9].squeeze(dim=0), batch[:, :, :, 9:10]
        y = torch.squeeze(y, dim=2)
        _, y_hat = self.forward(x)
     
        return {'loss': F.mse_loss(y_hat, y)}

    def _tensor_rolling_window(self, dataset, window_size, step_size=1):
        # unfold dimension to make our rolling window
        return dataset.unfold(0, window_size, step_size)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self._tensor_rolling_window(dataset=self.datasets['train'], window_size=10),
                          batch_size=1, shuffle=True)







