import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorch_lightning as pl
from src.models.gcn_lstm.gcn_lstm import GCN_LSTM

torch.manual_seed(0)


class GCNLSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        adj_normalized,
        ts_dataset,
        speed_transform,
        timesteps,
        batch_size=8,
    ):
        super().__init__()

        # unsqueeze to batch and time for further broadcasting
        size = len(ts_dataset)
        tng_size = int(0.7 * size)
        tst_size = int(0.1 * size)
        self.speed_transform = speed_transform
        self.ts_tng_dataset = Subset(ts_dataset, list(range(0, tng_size)))
        self.ts_val_dataset = Subset(ts_dataset, list(range(tng_size, size - tst_size)))
        self.ts_tst_dataset = Subset(ts_dataset, list(range(size - tst_size, size)))

        self.batch_size = batch_size
        self.A = adj_normalized
        self.lstm = GCN_LSTM(input_size, hidden_size, num_layers, self.A)

        self.decoder = nn.Sequential(
            nn.Conv2d(timesteps, 1, kernel_size=(1, hidden_size)),
            #  nn.Conv2d(timesteps, timesteps // 2, kernel_size=(1, hidden_size)),
            #  nn.BatchNorm2d(timesteps // 2),
            #  nn.Sigmoid(),
            #  nn.Conv2d(timesteps // 2, 1, kernel_size=(1, 1)),
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.015)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=3, verbose=True
        )

    def normalize_adj(self, adj_mat):
        # TODO
        raise NotImplemented

    def forward(self, input_seq):
        embedding = self.lstm(input_seq)
        #  print(embedding)
        return self.decoder(embedding)

    def loss(self, speed, mask, speed_hat, validate=False):
        speed = speed.squeeze()
        mask = mask.squeeze()
        speed_hat = speed_hat.squeeze()
        #  fro = torch.norm(mask * (speed_hat - speed), p="fro", dim=(1)).mean()
        mae = (mask * (speed_hat - speed)).abs().sum() / mask.sum()
        metrics = {"loss": mae}
        if validate:
            speed_hat = self.speed_transform.inverse_transform(
                speed_hat.detach().numpy()
            )
            speed = self.speed_transform.inverse_transform(speed.detach().numpy())
            mask = mask.detach().numpy()
            mae = np.abs(mask * (speed_hat - speed)).sum() / mask.sum()
            metrics["mae"] = mae

        return metrics

    def training_step(self, batch, batch_nb, validate=False):
        # we use last timestep as the target
        speed, mask, static = batch

        # it's batch x edge x time x features at this point
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

        return self.loss(target_speed, target_mask, speed_hat, validate)

    def validation_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb, True)

    def validation_end(self, outputs):
        # TODO refactor
        avg_val_loss = sum([o["loss"] for o in outputs]) / len(outputs)
        self.lr_scheduler.step(avg_val_loss)
        return {
            "avg_val_loss": avg_val_loss,
            "avg_val_mae": sum([o["mae"] for o in outputs]) / len(outputs),
        }

    def configure_optimizers(self):
        return [self.opt]

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
