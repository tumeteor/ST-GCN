from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class LSTMs(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, data, batch_size=20, out_dim=1, layers=2, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.x = data
        self.batch_size = batch_size

    def forward(self, x):
        lstm_out, hidden = self.lstm(x.view(len(x), self.batch_size, -1))

        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.x, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.mse_loss(y_hat, y)}
