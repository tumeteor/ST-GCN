from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F

import pytorch_lightning as pl
import torch

torch.manual_seed(0)
from src.tgcn.layers.gcn import GCN
from src.tgcn.layers.tempcells import TempCells


class TGCN(pl.LightningModule):
    def __init__(self, gcn_in, gcn_out, adj, dropout=0.5, datasets=None):
        super(TGCN, self).__init__()

        self.net = GCN(in_feats=gcn_in, hidden_size=32, out_feats=gcn_out, dropout=dropout)
        self.model = TempCells(gcn_out, hidden_dim=32)
        self.datasets = datasets
        self.adj = adj

    def forward(self, emb):
        # shape of emb: [batch_size, num_nodes, num_features, num_timesteps]
        # simplify with batch size = 1
        for i in range(len(emb)):
            gcn_embs = []
            for j in range(len(emb[0, 0, 0, :])):
                gcn_emb = self.net(x=emb[i, :, :, j].float(), adj=self.adj)
                gcn_embs.append(gcn_emb)
            # size of gcn_embs: [num_timesteps, num_nodes, num_features]
            gcn_embs = torch.stack(gcn_embs).permute(1, 2, 0).float()
            print(f"shape of gcn_embs: {gcn_embs.shape}")
            _, x = self.model(gcn_embs)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        x, y = batch[:, :, :, :9], batch[:, :, :, 9:]
        y = torch.squeeze(y, dim=2)
        _, y_hat = self.forward(x)

        return {'loss': F.mse_loss(y_hat, y)}

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=1, shuffle=True)
