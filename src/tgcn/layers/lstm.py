from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class LSTMs(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, data, out_dim=1, layers=2, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.x = data

    def forward(self, x):
        out, hidden = self.lstm(x)

        last_hidden_out = out[:, -1, :].squeeze()

        return last_hidden_out, self.linear(last_hidden_out)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.x, batch_size=10, shuffle=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_nb):
        x, y = batch[:, :, :29], batch[:, :, 29:30]
        y = y.view(1, -1)
        _, y_hat = self.forward(x)
        print(y.shape)
        print(y_hat.shape)
        return {'loss': F.mse_loss(y_hat, y)}
