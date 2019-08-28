import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorch_lightning as pl
from src.gcn_lstm.gcn_lstm import GCN_LSTM

torch.manual_seed(0)


class GCNLSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        adj_normalized,
        ts_dataset,
        batch_size=8,
    ):
        super().__init__()

        # unsqueeze to batch and time for further broadcasting
        size = len(ts_dataset)
        tng_size = int(0.7 * size)
        tst_size = int(0.2 * size)
        self.ts_tng_dataset = Subset(ts_dataset, list(range(0, tng_size)))
        self.ts_val_dataset = Subset(ts_dataset, list(range(tng_size, size - tst_size)))
        self.ts_tst_dataset = Subset(ts_dataset, list(range(size - tst_size, size)))

        self.batch_size = batch_size
        self.A = adj_normalized
        self.lstm = GCN_LSTM(input_size, hidden_size, num_layers, self.A)

        TIME_STEPS = 9
        self.decoder = nn.Conv2d(TIME_STEPS, 1, kernel_size=(1, hidden_size))

    def normalize_adj(self, adj_mat):
        # TODO
        raise NotImplemented

    def forward(self, input_seq):
        return self.decoder(self.lstm(input_seq))

    def loss(self, speed, mask, speed_hat):
        speed = speed.squeeze()
        mask = mask.squeeze()
        speed_hat = speed_hat.squeeze()
        fro = torch.norm(mask * (speed_hat - speed), p="fro", dim=(1)).mean()
        return {"loss": fro}

    def training_step(self, batch, batch_nb):
        # we use last timestep as the target
        speed, mask, static = batch

        target_speed = speed[:, :, -1].unsqueeze(3)
        target_mask = mask[:, :, -1].unsqueeze(3)
        # TODO static does not change, no need to recompute
        # Our dimensionality is batch x time x edges x features,
        # so we want to concat along features.
        input = torch.cat(
            (speed[:, :, :-1, :], mask[:, :, :-1, :], static[:, :, :-1, :]), dim=3
        )
        # Now we need to provide (seq_len, batch, edges, features) dimensions,
        # in line with LSTM in PyTorch
        input = input.permute(2, 0, 1, 3)

        speed_hat = self.forward(input)

        return self.loss(target_speed, target_mask, speed_hat)

    def validation_step(self, batch, batch_nb):
        # TODO add more metrics
        return self.training_step(batch, batch_nb)

    def validation_end(self, outputs):
        # TODO refactor
        return {
            #  "avg_val_mae": sum([o["val_mae"] for o in outputs]) / len(outputs),
            "avg_val_loss": sum([o["loss"] for o in outputs])
            / len(outputs)
        }

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.0015)]

    @pl.data_loader
    def tng_dataloader(self):
        try:
            return DataLoader(
                self.ts_tng_dataset, shuffle=True, batch_size=self.batch_size
            )
        except Exception as e:
            print(e)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.ts_val_dataset, shuffle=False, batch_size=self.batch_size
        )

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            self.ts_tst_dataset, shuffle=False, batch_size=self.batch_size
        )
