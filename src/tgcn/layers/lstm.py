from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl


class LSTMs(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, data=None, validation_data=None, out_dim=213, layers=2, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.x = data
        self.valid = validation_data

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        return last_hidden_out, self.linear(last_hidden_out)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.x, batch_size=10, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=10, shuffle=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        x, y = batch[:, :, :9], batch[:, :, 9:10]
        y = torch.squeeze(y, dim=2)
        _, y_hat = self.forward(x)
        print(y.shape)
        print(y_hat.shape)
        print(y)
        print(y_hat)
        return {'loss': F.mse_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch[:, :, :9], batch[:, :, 9:10]
        y = torch.squeeze(y, dim=2)
        _, y_hat = self.forward(x)
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
