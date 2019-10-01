from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
torch.manual_seed(0)
import pytorch_lightning as pl


class LSTMs(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=32, layers=2,
                 datasets=None, targets=None, mask=None):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.look_back = input_dim
        self.targets = targets
        self.datasets = datasets
        self.mask = mask

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        return last_hidden_out, self.linear(last_hidden_out)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y = y.squeeze(dim=0)
        x = x.squeeze(dim=0).float()
        _, y_hat = self.forward(x)
        y_hat = y_hat.squeeze(dim=0)

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
        x = x.squeeze(dim=0).float()
        _, y_hat = self.forward(x)
        y_hat = y_hat.squeeze(dim=0)
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
