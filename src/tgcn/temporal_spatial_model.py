from torch.utils.data import DataLoader
from torch import optim
import pytorch_lightning as pl
import torch
from src.tgcn.layers.gcn import GCN
from src.tgcn.layers.lstm import LSTMs


class TGCN(pl.LightningModule):
    def __init__(self, gcn_in, gcn_hid, gcn_out, ex_in, lstm_hid, datasets):
        self.net = GCN(gcn_in, gcn_hid, gcn_out)
        self.model = LSTMs(gcn_out + ex_in, lstm_hid)
        self.datasets = datasets

    def forward(self, adj, ex_emb):
        gcn_emb = self.net(adj)
        _, x = self.model(torch.cat((gcn_emb, ex_emb), 0))
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def training_step(self, *args, **kwargs):
        pass

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




