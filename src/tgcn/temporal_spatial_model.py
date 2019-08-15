from torch.utils.data import DataLoader

from src.tgcn.layers.gcn import GCN
from src.tgcn.layers.lstm import LSTMs
import pytorch_lightning as pl
import torch


class TGCN(pl.LightningModule):
    def __init__(self, gcn_in, gcn_hid, gcn_out, lstm_hid, output_pred, lstm_layers, lstm_drop, batch_size=8, lr=0.0001):
        self.net = GCN(gcn_in, gcn_hid, gcn_out)
        self.model = LSTMs(gcn_out, lstm_hid, output_pred, lstm_layers, lstm_drop)
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.lr)]

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





