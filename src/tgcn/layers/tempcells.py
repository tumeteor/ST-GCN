from torch import nn
import torch

torch.manual_seed(0)


class TempCells(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, out_dim=6163):
        super(TempCells, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm_cell1 = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, input, future=0):
        # shape of input: [input_size, time_step, batch]
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.float)
        c_t = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.float)
        h_t2 = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.float)
        c_t2 = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.float)

        for i, input_t in enumerate(input.chunk(input.size(2), dim=2)):
            input_t = input_t[0, :, :]
            h_t, c_t = self.lstm_cell1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm_cell2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm_cell1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm_cell2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, dim=1).squeeze(2)
        return outputs
